"""
weather_core.py  -  the shared "brain" of the project
======================================================

Both `train.ipynb` (which BUILDS the models) and `main.py` (the Streamlit app that
USES the models) import this file.  Why a shared file?

    If the notebook and the app each had their own copy of the feature code, a tiny
    difference between them would silently give wrong forecasts ("train/serve skew").
    One file = one source of truth.

What is inside (in reading order)
---------------------------------
1.  Settings & the list of things we forecast
2.  load_and_clean()      -> turns the raw CSV into a clean hourly table
3.  build_base_features() -> turns the clean table into model inputs (lags, trends ...)
4.  horizon_matrix()      -> adds the "how many hours ahead?" columns
5.  make_forecast()       -> the function the app calls to produce a forecast
6.  Helpers: alerts, comfort score, daily summary, unit conversion, metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
DISPLAY_TZ = "Europe/Budapest"          # timestamps in the file carry +01:00/+02:00 (Central Europe)
MAX_HORIZON = 72                        # we forecast 1..72 hours ahead
EVAL_HORIZONS = [1, 3, 6, 12, 24, 48, 72]   # horizons used in the accuracy tables
LOOKBACK_HOURS = 24 * 7                 # longest look-back window used by any feature (7 days)
MIN_HISTORY_HOURS = LOOKBACK_HOURS + 1  # history needed before a forecast can start
INTERVAL_LEVEL = 0.80                   # prediction bands cover ~80% of outcomes (10th..90th percentile)

# Everything the regression models forecast.  `kind` is used for unit conversion.
TARGETS = {
    "temp_c":   dict(label="Temperature",  unit="°C",   color="#D9480F", lo=-45.0, hi=50.0,   kind="temp"),
    "feels_c":  dict(label="Feels-like",   unit="°C",   color="#F08C00", lo=-50.0, hi=55.0,   kind="temp"),
    "hum":      dict(label="Humidity",     unit="%",    color="#1C7ED6", lo=0.0,   hi=100.0,  kind="pct"),
    "wind_kmh": dict(label="Wind speed",   unit="km/h", color="#0CA678", lo=0.0,   hi=150.0,  kind="wind"),
    "pres_mb":  dict(label="Pressure",     unit="hPa",  color="#7048E8", lo=900.0, hi=1085.0, kind="pres"),
    "vis_km":   dict(label="Visibility",   unit="km",   color="#5C7080", lo=0.0,   hi=16.1,   kind="dist"),
}

# Sky condition classes (the raw "Summary" column has 27 messy labels -> we simplify to 5)
COND_CLASSES = ["Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Foggy"]
COND_CODE = {name: i for i, name in enumerate(COND_CLASSES)}
COND_EMOJI = {"Clear": "☀️", "Partly Cloudy": "🌤️", "Mostly Cloudy": "⛅", "Overcast": "☁️", "Foggy": "🌫️"}
COND_COLORS = {"Clear": "#F59F00", "Partly Cloudy": "#74C0FC", "Mostly Cloudy": "#4C8DD0",
               "Overcast": "#868E96", "Foggy": "#B197FC"}

# Raw CSV column  ->  short name used everywhere in the code
RAW_TO_SHORT = {
    "Formatted Date": "time",
    "Summary": "summary",
    "Temperature (C)": "temp_c",
    "Apparent Temperature (C)": "feels_c",
    "Humidity": "hum",
    "Wind Speed (km/h)": "wind_kmh",
    "Wind Bearing (degrees)": "wind_dir",
    "Visibility (km)": "vis_km",
    "Pressure (millibars)": "pres_mb",
}
NUMERIC_COLS = ["temp_c", "feels_c", "hum", "wind_kmh", "wind_dir", "vis_km", "pres_mb"]

GROUP_LABELS = {
    "temp": "Temperature (now + history)",
    "feels": "Feels-like temperature",
    "hum": "Humidity",
    "wind": "Wind",
    "pres": "Pressure & pressure trend",
    "vis": "Visibility",
    "dew": "Dew point / air saturation",
    "cond": "Sky-condition history",
    "time": "Time of day / season (now)",
    "h": "Forecast horizon & target time",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def simplify_summary(s) -> object:
    """Map the 27 raw 'Summary' labels to our 5 sky-condition classes."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    if "Fog" in s:
        return "Foggy"
    if "Overcast" in s:
        return "Overcast"
    if "Mostly Cloudy" in s:
        return "Mostly Cloudy"
    if "Partly Cloudy" in s:
        return "Partly Cloudy"
    if "Clear" in s:
        return "Clear"
    if "Rain" in s or "Drizzle" in s:
        return "Overcast"      # rain almost always falls from overcast skies (only ~0.1% of rows)
    return "Clear"             # 'Dry', 'Breezy', 'Windy', 'Humid' alone = no cloud descriptor


def load_and_clean(source) -> tuple[pd.DataFrame, dict]:
    """
    Read the weather CSV (path or file-like object) and return (clean_df, report).

    clean_df : one row per hour, continuous, sorted, UTC index named 'time'
    report   : dict describing every problem we found and fixed (shown in the app)
    """
    raw = pd.read_csv(source)
    missing = [c for c in RAW_TO_SHORT if c not in raw.columns]
    if missing:
        raise ValueError(
            "The file is missing required column(s): " + ", ".join(missing)
            + ".  Expected the same columns as weatherHistory.csv."
        )

    report: dict = {"raw_rows": int(len(raw))}
    df = raw[list(RAW_TO_SHORT)].rename(columns=RAW_TO_SHORT)
    report["dropped_columns"] = [c for c in raw.columns if c not in RAW_TO_SHORT]

    # -- time: parse with the UTC offset, so daylight-saving changes cannot create duplicates --
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    report["unparseable_time_rows"] = int(df["time"].isna().sum())
    df = df.dropna(subset=["time"])
    report["was_unsorted"] = bool(not df["time"].is_monotonic_increasing)

    n = len(df)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    report["duplicate_timestamps_removed"] = int(n - len(df))

    # -- units: humidity 0..1  ->  0..100 % --
    df["hum"] = df["hum"] * 100.0

    # -- impossible values are sensor errors -> mark as missing --
    bad_pres = (df["pres_mb"] < 900) | (df["pres_mb"] > 1100)   # e.g. exactly 0 hPa
    bad_hum = df["hum"] <= 0                                    # 0 % humidity does not occur here
    bad_vis = df["vis_km"] <= 0                                 # mostly logged under CLEAR skies = placeholder
    report["pressure_invalid"] = int(bad_pres.sum())
    report["humidity_invalid"] = int(bad_hum.sum())
    report["visibility_invalid"] = int(bad_vis.sum())
    df.loc[bad_pres, "pres_mb"] = np.nan
    df.loc[bad_hum, "hum"] = np.nan
    df.loc[bad_vis, "vis_km"] = np.nan

    # -- simplify the text label into 5 sky conditions --
    mapping = {s: simplify_summary(s) for s in df["summary"].dropna().unique()}
    df["cond"] = df["summary"].map(mapping)

    # -- make the time axis perfectly regular (1 row per hour) and fill tiny gaps --
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    report["missing_hours_filled"] = int(len(full_index) - len(df))
    df = df.reindex(full_index)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].interpolate(method="linear", limit=6)
    df["cond"] = df["cond"].ffill(limit=6)
    df["summary"] = df["summary"].ffill(limit=6)
    df["cond_code"] = df["cond"].map(COND_CODE)

    report["final_rows"] = int(len(df))
    report["start"] = str(df.index.min())
    report["end"] = str(df.index.max())
    report["remaining_missing"] = {c: int(df[c].isna().sum()) for c in NUMERIC_COLS + ["cond_code"]}
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (everything here only looks at the PAST and the PRESENT)
# ══════════════════════════════════════════════════════════════════════════════
LAGS = [1, 2, 3, 6, 12, 24, 48, 72]      # "what was it N hours ago?"
LAG_VARS = ["temp_c", "pres_mb", "hum", "wind_kmh", "vis_km"]


def dew_point(temp_c, hum_pct):
    """Dew point (Magnus formula).  Temp minus dew point = how close the air is to saturation (fog/cloud)."""
    a, b = 17.62, 243.12
    rh = np.clip(hum_pct, 1.0, 100.0) / 100.0
    g = np.log(rh) + a * temp_c / (b + temp_c)
    return b * g / (a - g)


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the clean hourly table into model inputs.  Row t contains ONLY information
    that is known at time t (no future values) - this is verified by a test in train.ipynb.
    """
    f = pd.DataFrame(index=df.index)

    # A. current conditions ------------------------------------------------------------
    for c in ["temp_c", "feels_c", "hum", "wind_kmh", "pres_mb", "vis_km"]:
        f[c] = df[c]
    rad = np.deg2rad(df["wind_dir"])
    f["wind_u"] = -df["wind_kmh"] * np.sin(rad)      # east-west wind component
    f["wind_v"] = -df["wind_kmh"] * np.cos(rad)      # north-south wind component
    f["wind_dir_sin"] = np.sin(rad)                  # direction is a circle: 359 deg is next to 1 deg
    f["wind_dir_cos"] = np.cos(rad)
    f["dew_c"] = dew_point(df["temp_c"], df["hum"])
    f["dew_spread"] = df["temp_c"] - f["dew_c"]
    f["cond_code"] = df["cond_code"]

    # B. history: value N hours ago ---------------------------------------------------
    for c in LAG_VARS:
        for k in LAGS:
            f[f"{c}_lag{k}"] = df[c].shift(k)
    for k in (3, 6, 24):
        f[f"cond_code_lag{k}"] = df["cond_code"].shift(k)

    # C. trends: how fast is it changing? (pressure tendency is a classic storm signal) --
    f["temp_c_d3"] = df["temp_c"] - df["temp_c"].shift(3)
    f["temp_c_d24"] = df["temp_c"] - df["temp_c"].shift(24)
    f["pres_mb_d3"] = df["pres_mb"] - df["pres_mb"].shift(3)
    f["pres_mb_d6"] = df["pres_mb"] - df["pres_mb"].shift(6)
    f["pres_mb_d24"] = df["pres_mb"] - df["pres_mb"].shift(24)
    f["hum_d3"] = df["hum"] - df["hum"].shift(3)

    # D. rolling summaries of the last 24 h and 7 days ---------------------------------
    r24 = lambda s: s.rolling(24, min_periods=12)                       # noqa: E731
    r7d = lambda s: s.rolling(LOOKBACK_HOURS, min_periods=LOOKBACK_HOURS // 2)  # noqa: E731
    f["temp_c_mean24"] = r24(df["temp_c"]).mean()
    f["temp_c_min24"] = r24(df["temp_c"]).min()
    f["temp_c_max24"] = r24(df["temp_c"]).max()
    f["temp_c_std24"] = r24(df["temp_c"]).std()
    f["pres_mb_std24"] = r24(df["pres_mb"]).std()
    f["hum_mean24"] = r24(df["hum"]).mean()
    f["wind_kmh_mean24"] = r24(df["wind_kmh"]).mean()
    f["wind_kmh_max24"] = r24(df["wind_kmh"]).max()
    f["vis_km_min24"] = r24(df["vis_km"]).min()
    f["vis_km_mean24"] = r24(df["vis_km"]).mean()
    f["cond_code_mean24"] = r24(df["cond_code"]).mean()
    f["temp_c_mean7d"] = r7d(df["temp_c"]).mean()
    f["pres_mb_mean7d"] = r7d(df["pres_mb"]).mean()
    f["wind_kmh_mean7d"] = r7d(df["wind_kmh"]).mean()

    # E. clock of the ORIGIN time (cyclical encoding so 23:00 is next to 00:00) --------
    hour = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear
    f["time_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["time_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["time_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["time_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


# ══════════════════════════════════════════════════════════════════════════════
# 4. THE "HOW FAR AHEAD?" COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
H_COLS = ["h", "h_hour_sin", "h_hour_cos", "h_doy_sin", "h_doy_cos"]


def horizon_matrix(x_rows: np.ndarray, origin_ts: pd.DatetimeIndex, h: np.ndarray) -> np.ndarray:
    """
    Build model input rows.  Each row = (weather features at the origin time) + (how many hours ahead)
    + (clock of the TARGET time, i.e. origin + h).  Knowing the target's hour-of-day lets the model
    draw the daily temperature swing correctly.

    x_rows    : (n, n_features) base features, one row per (origin, horizon) pair
    origin_ts : DatetimeIndex of length n, the origin timestamps
    h         : integer array of length n, hours ahead
    """
    h = np.asarray(h)
    tgt = origin_ts + pd.to_timedelta(h, unit="h")
    hour = np.asarray(tgt.hour) + np.asarray(tgt.minute) / 60.0
    doy = np.asarray(tgt.dayofyear)
    extra = np.column_stack([
        h.astype(float),
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ])
    return np.hstack([np.asarray(x_rows, dtype=float), extra])


def valid_origins(n_rows: int, warmup: int = LOOKBACK_HOURS, max_h: int = MAX_HORIZON) -> np.ndarray:
    """Row numbers that can serve as a forecast start: enough history behind, full 72h of truth ahead."""
    return np.arange(warmup, n_rows - max_h)


def sample_pairs(origin_idx: np.ndarray, k: int, rng: np.random.Generator, max_h: int = MAX_HORIZON,
                 log_uniform: bool = False):
    """
    For every origin pick k random horizons in 1..max_h  ->  (origin_row_numbers, horizons).

    log_uniform=False : every horizon 1..max_h is equally likely.
    log_uniform=True  : short horizons are drawn MORE often (h = e^U, U uniform).  Sky condition
                        changes slowly, so the model must be very sharp at +1..+12 h to beat "no change".
    """
    o = np.repeat(origin_idx, k)
    if log_uniform:
        h = np.rint(np.exp(rng.uniform(0, np.log(max_h + 0.5), size=o.size))).astype(int)
        h = np.clip(h, 1, max_h)
    else:
        h = rng.integers(1, max_h + 1, size=o.size)
    return o, h


def horizon_sigma(delta: np.ndarray, h: np.ndarray, max_h: int = MAX_HORIZON) -> np.ndarray:
    """
    'How big is a typical change after h hours?'  (standard deviation of the change, per horizon 1..max_h)

    After 1 hour temperature moves ~1 degC, after 24 hours ~5 degC.  If we ask the model to predict raw changes,
    the big long-range errors dominate the learning and the short range is neglected.  Dividing by this number
    puts every horizon on the same footing (1.0 = 'a typical change').
    """
    sd = pd.Series(np.asarray(delta, float)).groupby(np.asarray(h)).std()
    sd = sd.reindex(range(1, max_h + 1)).interpolate(limit_direction="both")
    sd = sd.rolling(5, center=True, min_periods=1).mean()          # smooth out sampling noise
    return np.maximum(sd.to_numpy(), 1e-3)


class ScaledDeltaModel(RegressorMixin, BaseEstimator):
    """
    Wraps a boosting model that was trained on *scaled* changes so that callers can simply do

        change_in_real_units = model.predict(X)

    It reads the horizon h from column `h_col` of X, asks the inner model for the scaled change,
    and multiplies by the typical change size sigma[h].  Saved inside models/regressors.joblib.
    """

    def __init__(self, model, sigma: np.ndarray, h_col: int):
        self.model, self.sigma, self.h_col = model, np.asarray(sigma, float), int(h_col)

    def fit(self, X=None, y=None):
        """Does nothing - the inner model is already trained. Exists only so scikit-learn tools (permutation_importance) accept this object."""
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        h = np.clip(np.rint(X[:, self.h_col]).astype(int), 1, len(self.sigma))
        return self.model.predict(X) * self.sigma[h - 1]


def feature_group(name: str) -> str:
    """Which family does a feature belong to?  (used for the 'what drives the forecast' chart)"""
    return GROUP_LABELS.get(name.split("_")[0], "Other")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FORECASTING (used by the Streamlit app and by the notebook demo)
# ══════════════════════════════════════════════════════════════════════════════
def load_bundle(model_dir: str | Path = "models") -> dict:
    """Load the trained models + metadata written by train.ipynb."""
    import joblib
    import sklearn

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    bundle = {
        "regressors": joblib.load(model_dir / "regressors.joblib"),
        "classifier": joblib.load(model_dir / "classifier.joblib"),
        "meta": meta,
        "version_warning": None,
    }
    trained_with = meta.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        bundle["version_warning"] = (
            f"Models were trained with scikit-learn {trained_with}, but {sklearn.__version__} is installed. "
            "It usually still works; if you see errors, re-run train.ipynb to rebuild the models."
        )
    return bundle


def make_forecast(features: pd.DataFrame, origin_ts, bundle: dict, horizon: int = MAX_HORIZON):
    """
    Forecast `horizon` hours ahead starting from `origin_ts`.

    features : output of build_base_features() (index = UTC hourly timestamps)
    returns  : (forecast_df, current_dict)
       forecast_df is indexed by the future timestamps and has, for every variable v:
           v, v_lo, v_hi         -> best guess and the 80% band
       plus  p_<condition>  (probabilities)  and  cond (most likely sky condition).
    """
    meta = bundle["meta"]
    model_cols = meta["model_columns"]
    base_cols = meta["base_columns"]
    horizon = int(min(max(horizon, 1), MAX_HORIZON))

    origin_ts = pd.Timestamp(origin_ts)
    if origin_ts.tzinfo is None:
        origin_ts = origin_ts.tz_localize("UTC")
    if origin_ts not in features.index:
        raise ValueError(f"Origin time {origin_ts} is not in the data.")
    pos = features.index.get_loc(origin_ts)
    if pos < MIN_HISTORY_HOURS - 1:
        raise ValueError(f"Need at least {MIN_HISTORY_HOURS} hours of history before the forecast start.")

    x0 = features.iloc[pos][base_cols].to_numpy(dtype=float)
    hs = np.arange(1, horizon + 1)
    x_rows = np.repeat(x0[None, :], horizon, axis=0)
    origin_index = pd.DatetimeIndex([origin_ts] * horizon)
    M = horizon_matrix(x_rows, origin_index, hs)
    assert M.shape[1] == len(model_cols), "feature layout does not match the trained model"

    target_times = origin_ts + pd.to_timedelta(hs, unit="h")
    out = pd.DataFrame(index=pd.DatetimeIndex(target_times, name="time"))
    current: dict = {}

    for var, model in bundle["regressors"].items():
        cur = float(features.iloc[pos][var])
        if np.isnan(cur):
            raise ValueError(f"The latest value of '{var}' is missing, so a forecast cannot be made from this time.")
        spec = TARGETS[var]
        delta = model.predict(M)                       # the model predicts the CHANGE from now
        iv = meta["intervals"][var]
        pred = np.clip(cur + delta, spec["lo"], spec["hi"])
        out[var] = pred
        out[f"{var}_lo"] = np.clip(pred + np.array(iv["lo"])[hs - 1], spec["lo"], spec["hi"])
        out[f"{var}_hi"] = np.clip(pred + np.array(iv["hi"])[hs - 1], spec["lo"], spec["hi"])
        current[var] = cur

    clf = bundle["classifier"]
    proba = clf.predict_proba(M)
    for j, code in enumerate(clf.classes_):
        out[f"p_{COND_CLASSES[int(code)]}"] = proba[:, j]
    for name in COND_CLASSES:                          # make sure every class column exists
        if f"p_{name}" not in out:
            out[f"p_{name}"] = 0.0
    pcols = [f"p_{n}" for n in COND_CLASSES]
    out["cond"] = [COND_CLASSES[i] for i in out[pcols].to_numpy().argmax(axis=1)]
    out["cond_conf"] = out[pcols].max(axis=1)

    code_now = features.iloc[pos].get("cond_code", np.nan)
    current["cond"] = COND_CLASSES[int(code_now)] if not np.isnan(code_now) else None
    current["time"] = origin_ts
    return out, current


# ══════════════════════════════════════════════════════════════════════════════
# 6. HELPERS: local time, units, alerts, comfort, daily summary, metrics
# ══════════════════════════════════════════════════════════════════════════════
def to_local(index_or_series):
    """UTC-aware timestamps -> naive local wall-clock time (what humans read and what charts show)."""
    if isinstance(index_or_series, pd.Series):
        return index_or_series.dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None)
    return pd.DatetimeIndex(index_or_series).tz_convert(DISPLAY_TZ).tz_localize(None)


def local_to_utc(naive_local: pd.Timestamp) -> pd.Timestamp:
    """A local wall-clock time -> UTC (handles daylight-saving edge cases)."""
    return pd.Timestamp(naive_local).tz_localize(DISPLAY_TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


WIND_FACTORS = {"km/h": 1.0, "mph": 0.621371, "m/s": 0.277778}


def convert_units(var: str, values, temp_unit: str = "°C", wind_unit: str = "km/h"):
    """Return (converted_values, unit_label) for display."""
    kind = TARGETS[var]["kind"]
    if kind == "temp" and temp_unit == "°F":
        return values * 9.0 / 5.0 + 32.0, "°F"
    if kind == "wind" and wind_unit != "km/h":
        return values * WIND_FACTORS[wind_unit], wind_unit
    return values, TARGETS[var]["unit"]


def cond_emoji(cond: str, local_hour: int) -> str:
    if cond == "Clear" and (local_hour >= 20 or local_hour < 6):
        return "🌙"
    return COND_EMOJI.get(cond, "❓")


def build_alerts(fc: pd.DataFrame, current: dict) -> list[dict]:
    """Simple, transparent, rule-based weather warnings on top of the forecast."""
    alerts: list[dict] = []
    when = lambda ts: to_local(pd.DatetimeIndex([ts]))[0].strftime("%a %H:%M")   # noqa: E731

    tmin, tmax = fc["temp_c"].min(), fc["temp_c"].max()
    if tmin <= 0:
        alerts.append(dict(level="warning", title="Freezing temperatures expected",
                           detail=f"Lowest forecast {tmin:.1f} °C around {when(fc['temp_c'].idxmin())}. Watch for ice."))
    elif fc["temp_c_lo"].min() <= 0:
        alerts.append(dict(level="info", title="Possible frost",
                           detail=f"Forecast low is {tmin:.1f} °C, but the uncertainty band dips below 0 °C."))
    if tmax >= 30:
        alerts.append(dict(level="warning", title="Hot weather",
                           detail=f"Highest forecast {tmax:.1f} °C around {when(fc['temp_c'].idxmax())}. Stay hydrated."))
    wmax = fc["wind_kmh"].max()
    if wmax >= 40:
        alerts.append(dict(level="warning", title="Strong wind",
                           detail=f"Peak forecast {wmax:.0f} km/h around {when(fc['wind_kmh'].idxmax())}."))
    if (fc["p_Foggy"] >= 0.40).any() or fc["vis_km"].min() <= 2:
        t = fc["p_Foggy"].idxmax()
        alerts.append(dict(level="warning", title="Fog risk",
                           detail=f"Fog probability peaks at {fc['p_Foggy'].max():.0%} around {when(t)}; "
                                  f"lowest visibility {fc['vis_km'].min():.1f} km."))
    first12 = fc.iloc[:12]
    if "pres_mb" in current and len(first12) and current["pres_mb"] - first12["pres_mb"].min() >= 6:
        alerts.append(dict(level="info", title="Falling pressure",
                           detail=f"Pressure is forecast to fall by {current['pres_mb'] - first12['pres_mb'].min():.0f} hPa "
                                  "within 12 h - unsettled weather often follows."))
    return alerts


def comfort_score(fc: pd.DataFrame) -> pd.Series:
    """0-100 'how nice is it to be outside?' score from feels-like temp, wind, visibility and humidity."""
    t, w, v, h = fc["feels_c"], fc["wind_kmh"], fc["vis_km"], fc["hum"]
    t_pen = np.where(t < 18, 18 - t, np.where(t > 24, t - 24, 0.0))
    s_t = 100 - np.clip(t_pen * 6, 0, 100)
    s_w = 100 - np.clip((w - 15) * 3, 0, 100)
    s_v = np.clip(v / 8.0, 0, 1) * 100
    s_h = 100 - np.clip((h - 70) * 3, 0, 100)
    return pd.Series(0.55 * s_t + 0.20 * s_w + 0.15 * s_v + 0.10 * s_h, index=fc.index, name="comfort")


def daily_summary(fc: pd.DataFrame) -> pd.DataFrame:
    """One row per local calendar day: high/low, wind, most common sky condition."""
    loc = to_local(fc.index)
    g = fc.assign(day=loc.normalize(), hour=loc.hour).groupby("day")
    rows = []
    for day, part in g:
        mode = part["cond"].mode().iloc[0]
        rows.append(dict(day=day, hours=len(part), tmax=part["temp_c"].max(), tmin=part["temp_c"].min(),
                         wind_max=part["wind_kmh"].max(), hum_mean=part["hum"].mean(), cond=mode))
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2 (NaN-safe)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    err = y_pred - y_true
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return dict(mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))),
                r2=float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"))


def smooth(a: np.ndarray, window: int = 7) -> np.ndarray:
    """Moving average with edge padding (keeps the length)."""
    a = np.asarray(a, float)
    pad = window // 2
    p = np.pad(a, pad, mode="edge")
    return np.convolve(p, np.ones(window) / window, mode="valid")

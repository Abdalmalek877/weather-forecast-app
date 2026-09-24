"""
main.py  -  the Weather Forecast web app  (run with:  streamlit run main.py)
============================================================================

What this file does
-------------------
It is only the "face" of the project.  All the real logic (cleaning, features, forecasting)
lives in `weather_core.py`, and the trained models live in the `models/` folder (created by
running `train.ipynb`).  This file just:

    1. loads the data + models once (cached, so the app stays fast),
    2. lets you choose a start time, a forecast length, units and a mode,
    3. asks weather_core.make_forecast() for the forecast,
    4. draws it (Plotly charts, cards, alerts ...).

Two modes (sidebar)
-------------------
    Simple    : the essentials - now, alerts, temperature chart, day cards, best time outside.
    Advanced  : the "full-power" view - all 6 variables with uncertainty bands, sky-condition
                probabilities, a check against what really happened, model accuracy, feature
                importance, data-cleaning report.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import weather_core as wc

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "weatherHistory.csv"
MODEL_DIR = APP_DIR / "models"

st.set_page_config(page_title="Weather Forecast Lab", page_icon="🌦️", layout="wide")

# A little CSS for the hourly strip and the day cards (everything else uses Streamlit's own look)
st.markdown(
    """
    <style>
      .strip {display:flex; gap:8px; overflow-x:auto; padding-bottom:6px;}
      .hcard {min-width:78px; text-align:center; padding:10px 6px; border-radius:12px;
              background:rgba(128,128,128,0.10); border:1px solid rgba(128,128,128,0.18);}
      .hcard .t {font-size:0.78rem; opacity:0.75;}
      .hcard .e {font-size:1.6rem; line-height:1.5;}
      .hcard .v {font-weight:600;}
      .dcard {padding:14px 10px; border-radius:14px; text-align:center;
              background:rgba(128,128,128,0.10); border:1px solid rgba(128,128,128,0.18);}
      .dcard .d {font-weight:600;} .dcard .e {font-size:2.2rem;}
      .dcard .hi {font-size:1.3rem; font-weight:700;} .dcard .lo {opacity:0.7;}
      .small {font-size:0.82rem; opacity:0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOADING (cached: runs once, not on every click)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading the trained models ...")
def get_bundle():
    return wc.load_bundle(MODEL_DIR)


@st.cache_resource(show_spinner="Cleaning the data and building features (first time only) ...")
def get_data(file_bytes: bytes | None):
    """Returns (clean_table, feature_table, cleaning_report).  file_bytes=None -> the bundled CSV."""
    source = io.BytesIO(file_bytes) if file_bytes is not None else DATA_PATH
    df, report = wc.load_and_clean(source)
    feats = wc.build_base_features(df)
    return df, feats, report


# ══════════════════════════════════════════════════════════════════════════════
# 2. SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def conv(var: str, values, units: dict):
    """Convert to the units chosen in the sidebar.  Returns (values, unit_label)."""
    return wc.convert_units(var, np.asarray(values, dtype=float), units["temp"], units["wind"])


def fmt_val(var: str, value: float, units: dict) -> str:
    v, u = conv(var, np.array([value]), units)
    digits = 0 if var in ("hum", "pres_mb") else 1
    return f"{v[0]:.{digits}f} {u}"


def local_label(ts) -> str:
    return wc.to_local(pd.DatetimeIndex([ts]))[0].strftime("%a %d %b, %H:%M")


def base_layout(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10), title=title,
        hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def variable_chart(fc, var, current, units, actual=None, band=True, height=380) -> go.Figure:
    """Forecast line + shaded 80 % band (+ the real observations, if we have them)."""
    spec = wc.TARGETS[var]
    x_now = wc.to_local(pd.DatetimeIndex([current["time"]]))[0]
    x = [x_now] + list(wc.to_local(fc.index))

    def series(col):
        """[value now] + forecast values, converted to the chosen units (starting at 'now' joins the lines up)."""
        return conv(var, np.r_[current[var], fc[col].to_numpy()], units)

    y, unit = series(var)
    fig = go.Figure()
    if band:
        lo, _ = series(f"{var}_lo")
        hi, _ = series(f"{var}_hi")
        fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=hex_to_rgba(spec["color"], 0.18),
                                 name=f"{int(wc.INTERVAL_LEVEL * 100)}% likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Forecast",
                             line=dict(color=spec["color"], width=3),
                             hovertemplate="%{y:.1f} " + unit))
    if actual is not None and actual[var].notna().any():
        ay, _ = conv(var, actual[var].to_numpy(), units)
        fig.add_trace(go.Scatter(x=wc.to_local(actual.index), y=ay, mode="lines", name="What really happened",
                                 line=dict(color="#212529", width=2, dash="dot"),
                                 hovertemplate="%{y:.1f} " + unit))
    fig.add_shape(type="line", x0=x_now, x1=x_now, y0=0, y1=1, yref="paper",
                  line=dict(color="gray", width=1, dash="dash"))
    fig.add_annotation(x=x_now, y=1, yref="paper", text="start", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color="gray"))
    fig.update_yaxes(title_text=f"{spec['label']} ({unit})")
    return base_layout(fig, height)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR  ->  returns all the user's choices
# ══════════════════════════════════════════════════════════════════════════════
def sidebar():
    sb = st.sidebar
    sb.title("🌦️ Weather Forecast Lab")

    mode = sb.radio("Mode", ["Simple", "Advanced"], horizontal=True,
                    help="Simple = the essentials.  Advanced = all variables, probabilities, "
                         "accuracy checks and model internals (the full-power view).")

    sb.markdown("### 1 · Data")
    up = sb.file_uploader("Use your own CSV (same columns as weatherHistory.csv)", type=["csv"])
    return sb, mode, up


def pick_start(sb, feats, uploaded: bool):
    """Let the user pick the moment the forecast starts.  Returns a UTC timestamp."""
    # Only hours that have (a) 7 days of history before them and (b) a current value for every variable
    ok = feats[list(wc.TARGETS)].notna().all(axis=1).to_numpy(copy=True)
    ok[: wc.MIN_HISTORY_HOURS] = False
    candidates = feats.index[ok]
    lo_d = wc.to_local(pd.DatetimeIndex([candidates[0]]))[0].date()
    hi_d = wc.to_local(pd.DatetimeIndex([candidates[-1]]))[0].date()

    sb.markdown("### 2 · When does the forecast start?")
    live = sb.checkbox("Use the very last hour in the data", value=False,
                       help="A real forecast beyond the data - there is no 'answer key' to compare with.")
    if live:
        return candidates[-1]

    default_day = pd.Timestamp("2016-07-14").date() if not uploaded else (pd.Timestamp(hi_d) - pd.Timedelta(days=3)).date()
    default_day = min(max(default_day, lo_d), hi_d)
    day = sb.date_input("Date", value=default_day, min_value=lo_d, max_value=hi_d)
    hour = sb.select_slider("Hour (local time)", options=list(range(24)), value=9, format_func=lambda h: f"{h:02d}:00")
    target = wc.local_to_utc(pd.Timestamp(day) + pd.Timedelta(hours=int(hour)))
    i = candidates.get_indexer([target], method="nearest")[0]          # snap to the closest usable hour
    return candidates[i]


# ══════════════════════════════════════════════════════════════════════════════
# 4. TABS
# ══════════════════════════════════════════════════════════════════════════════
def render_now_cards(fc, current, units, horizon):
    """Top row: what it is like at the start + how each value changes by the end of the horizon."""
    cond = current.get("cond") or "Clear"
    hour = wc.to_local(pd.DatetimeIndex([current["time"]]))[0].hour
    left, right = st.columns([1, 3])
    with left:
        st.markdown(f"<div style='font-size:4rem;line-height:1'>{wc.cond_emoji(cond, hour)}</div>", unsafe_allow_html=True)
        st.markdown(f"**{cond}**")
        st.caption(f"Start: {local_label(current['time'])}")
    with right:
        cols = st.columns(6)
        for col, var in zip(cols, wc.TARGETS):
            end = fc[var].iloc[-1]
            d, unit = conv(var, np.array([end - current[var]]), units)
            if wc.TARGETS[var]["kind"] == "temp" and units["temp"] == "°F":
                d = d - 32.0                                       # a *difference* in °F is only x1.8, not +32
            digits = 0 if var in ("hum", "pres_mb") else 1
            col.metric(wc.TARGETS[var]["label"], fmt_val(var, current[var], units),
                       f"{d[0]:+.{digits}f} by +{horizon}h", delta_color="off")


def render_alerts(fc, current):
    alerts = wc.build_alerts(fc, current)
    if not alerts:
        st.success("No weather warnings for this period.")
        return
    for a in alerts:
        (st.warning if a["level"] == "warning" else st.info)(f"**{a['title']}** — {a['detail']}")


def render_hourly_strip(fc, units, horizon):
    step = max(1, horizon // 12)
    part = fc.iloc[step - 1::step]
    loc = wc.to_local(part.index)
    cards = []
    for ts, (_, row) in zip(loc, part.iterrows()):
        t, u = conv("temp_c", np.array([row["temp_c"]]), units)
        cards.append(
            f"<div class='hcard'><div class='t'>{ts.strftime('%a %H:%M')}</div>"
            f"<div class='e'>{wc.cond_emoji(row['cond'], ts.hour)}</div>"
            f"<div class='v'>{t[0]:.0f}{u}</div>"
            f"<div class='t'>{row['wind_kmh'] * wc.WIND_FACTORS[units['wind']]:.0f} {units['wind']}</div></div>"
        )
    st.markdown(f"<div class='strip'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_day_cards(fc, units):
    days = wc.daily_summary(fc)
    cols = st.columns(len(days))
    for col, (_, d) in zip(cols, days.iterrows()):
        hi, u = conv("temp_c", np.array([d["tmax"]]), units)
        lo, _ = conv("temp_c", np.array([d["tmin"]]), units)
        partial = "" if d["hours"] >= 24 else f"<div class='small'>({int(d['hours'])} h of the day)</div>"
        col.markdown(
            f"<div class='dcard'><div class='d'>{d['day'].strftime('%a %d %b')}</div>"
            f"<div class='e'>{wc.cond_emoji(d['cond'], 12)}</div>"
            f"<div class='hi'>{hi[0]:.0f}{u}</div><div class='lo'>low {lo[0]:.0f}{u}</div>"
            f"<div class='small'>{d['cond']} · wind up to {d['wind_max'] * wc.WIND_FACTORS[units['wind']]:.0f} {units['wind']}</div>"
            f"{partial}</div>",
            unsafe_allow_html=True,
        )


def render_comfort(fc):
    comfort = wc.comfort_score(fc)
    st.markdown("#### 🧘 Comfort score")
    st.caption("0–100: how pleasant it is to be outside (feels-like temperature 55 %, wind 20 %, visibility 15 %, humidity 10 %).")
    if len(comfort) >= 3:
        best = comfort.rolling(3, center=True).mean().dropna()
        mid = best.idxmax()
        a, b = mid - pd.Timedelta(hours=1), mid + pd.Timedelta(hours=1)
        st.success(f"Best 3-hour window: **{local_label(a)} → {wc.to_local(pd.DatetimeIndex([b]))[0].strftime('%H:%M')}** "
                   f"(score {best.max():.0f}/100)")
    fig = go.Figure(go.Scatter(x=wc.to_local(comfort.index), y=comfort.to_numpy(), mode="lines", fill="tozeroy",
                               line=dict(color="#2F9E44", width=2), fillcolor="rgba(47,158,68,0.15)",
                               hovertemplate="%{y:.0f}/100", name="Comfort"))
    fig.update_yaxes(range=[0, 100], title_text="Comfort (0–100)")
    st.plotly_chart(base_layout(fig, 260))


def render_forecast_tab(fc, current, units, horizon, advanced):
    render_now_cards(fc, current, units, horizon)
    st.markdown("#### ⚠️ Alerts")
    render_alerts(fc, current)

    st.markdown("#### 🌡️ Temperature")
    st.caption("Line = best guess. Shaded area = the range where the real value should fall about 8 times out of 10.")
    st.plotly_chart(variable_chart(fc, "temp_c", current, units))

    st.markdown("#### 🕒 Hour by hour")
    render_hourly_strip(fc, units, horizon)

    st.markdown("#### 📅 Day by day")
    render_day_cards(fc, units)
    render_comfort(fc)

    with st.expander("Download the forecast as CSV"):
        out = fc.copy()
        out.index = wc.to_local(out.index)
        out.index.name = "local_time"
        st.download_button("Download CSV", out.round(3).to_csv().encode("utf-8"), "forecast.csv", "text/csv")
        st.dataframe(out.round(2), width="stretch")


def render_all_variables_tab(fc, current, units):
    st.markdown("Every variable the model forecasts, each with its uncertainty band.")
    labels = [wc.TARGETS[v]["label"] for v in wc.TARGETS]
    tabs = st.tabs(labels)
    for tab, var in zip(tabs, wc.TARGETS):
        with tab:
            st.plotly_chart(variable_chart(fc, var, current, units), key=f"var_{var}")

    st.markdown("#### ☁️ Sky condition — probabilities")
    st.caption("The classifier does not say “it will be overcast”; it says how likely each sky type is. "
               "Far ahead the probabilities spread out — the model honestly admits it is less sure.")
    x = wc.to_local(fc.index)
    fig = go.Figure()
    for name in wc.COND_CLASSES:
        fig.add_trace(go.Scatter(x=x, y=fc[f"p_{name}"] * 100, mode="lines", stackgroup="one", name=name,
                                 line=dict(width=0.5, color=wc.COND_COLORS[name]),
                                 fillcolor=hex_to_rgba(wc.COND_COLORS[name], 0.75),
                                 hovertemplate="%{y:.0f}%"))
    fig.update_yaxes(title_text="Probability (%)", range=[0, 100])
    st.plotly_chart(base_layout(fig, 340), key="sky_prob")

    st.markdown("#### 🌫️ Chance of fog")
    fig = go.Figure(go.Bar(x=x, y=fc["p_Foggy"] * 100, marker_color=wc.COND_COLORS["Foggy"], hovertemplate="%{y:.0f}%"))
    fig.update_yaxes(title_text="Fog probability (%)", range=[0, 100])
    st.plotly_chart(base_layout(fig, 240), key="fog")


def render_check_tab(fc, df, current, units):
    st.markdown("Because the data is historical, we can compare the forecast with **what really happened**.")
    actual = df.reindex(fc.index)
    if actual["temp_c"].notna().sum() == 0:
        st.info("This forecast starts at the very end of the data, so there is no answer key yet. "
                "Untick “Use the very last hour” in the sidebar and pick an earlier start to see a comparison.")
        return

    rows = []
    for var, spec in wc.TARGETS.items():
        ok = actual[var].notna()
        if not ok.any():
            continue
        err_model = np.abs(fc.loc[ok, var] - actual.loc[ok, var]).mean()
        err_naive = np.abs(current[var] - actual.loc[ok, var]).mean()
        inside = ((actual.loc[ok, var] >= fc.loc[ok, f"{var}_lo"]) & (actual.loc[ok, var] <= fc.loc[ok, f"{var}_hi"])).mean()
        rows.append({"Variable": spec["label"], "Unit": spec["unit"],
                     "Our error (MAE)": round(err_model, 2), "“No change” error": round(err_naive, 2),
                     "Better than no-change": f"{(1 - err_model / err_naive) * 100:+.0f}%" if err_naive > 0 else "–",
                     "Inside the band": f"{inside * 100:.0f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("MAE = average size of the mistake (smaller is better). “No change” = assume the weather stays exactly as it is "
               "at the start — the hard-to-beat baseline. A well-calibrated 80 % band should contain about 80 % of the real values.")

    ok_c = actual["cond"].notna()
    if ok_c.any() and current.get("cond"):
        acc = (fc.loc[ok_c, "cond"] == actual.loc[ok_c, "cond"]).mean()
        acc_naive = (current["cond"] == actual.loc[ok_c, "cond"]).mean()
        c1, c2 = st.columns(2)
        c1.metric("Sky condition: hours guessed right", f"{acc * 100:.0f}%")
        c2.metric("… if we just repeated the starting sky", f"{acc_naive * 100:.0f}%")

    var = st.selectbox("Plot variable", list(wc.TARGETS), format_func=lambda v: wc.TARGETS[v]["label"], key="check_var")
    st.plotly_chart(variable_chart(fc, var, current, units, actual=actual), key="check_chart")


def render_insights_tab(bundle):
    meta = bundle["meta"]
    st.markdown("How good is the model **on data it never saw** (test years "
                f"{meta['splits']['test'][0][:4]}–{meta['splits']['test'][1][:4]})? All numbers come from `train.ipynb`.")
    dfm = pd.DataFrame(meta["metrics"])

    var = st.selectbox("Variable", list(wc.TARGETS), format_func=lambda v: wc.TARGETS[v]["label"], key="ins_var")
    spec = wc.TARGETS[var]

    st.markdown("#### Error vs. how far ahead we forecast")
    fig = go.Figure()
    styles = {"Persistence": ("#adb5bd", "dot"), "Same time yesterday": ("#74c0fc", "dash"), "Boosting (ours)": (spec["color"], "solid")}
    hs = list(range(1, meta["max_horizon"] + 1))
    for name, vals in meta["curves"][var].items():
        colr, dash = styles.get(name, ("#495057", "solid"))
        fig.add_trace(go.Scatter(x=hs, y=vals, mode="lines", name=name, line=dict(color=colr, dash=dash, width=3 if "ours" in name else 2)))
    fig.update_xaxes(title_text="Hours ahead"); fig.update_yaxes(title_text=f"Average error (MAE, {spec['unit']})")
    st.plotly_chart(base_layout(fig, 360), key="ins_mae")

    piv = dfm[dfm["var"] == var].pivot(index="model", columns="horizon", values="mae").round(2)
    piv.columns = [f"+{c}h" for c in piv.columns]
    st.dataframe(piv, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What drives this forecast?")
        st.caption("Importance = how much the error grows when a group of inputs is scrambled.")
        imp = pd.Series(meta["importance_groups"][var]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=spec["color"]))
        fig.update_xaxes(title_text="Error increase when scrambled")
        st.plotly_chart(base_layout(fig, 380), key="ins_imp")
    with c2:
        st.markdown("#### Is the uncertainty band honest?")
        st.caption("Share of real values that fell inside the band. Target: the dashed line.")
        cov = np.array(meta["coverage"][var]) * 100
        fig = go.Figure(go.Scatter(x=hs, y=cov, mode="lines", line=dict(color=spec["color"], width=3), name="Measured"))
        fig.add_hline(y=meta["interval_level"] * 100, line_dash="dash", line_color="gray")
        fig.update_xaxes(title_text="Hours ahead"); fig.update_yaxes(title_text="Coverage (%)", range=[40, 100])
        st.plotly_chart(base_layout(fig, 380), key="ins_cov")

    st.markdown("#### Sky-condition classifier")
    cm = pd.DataFrame(meta["condition_metrics"])
    fig = go.Figure()
    for name, g in cm.groupby("model"):
        fig.add_trace(go.Scatter(x=g["horizon"], y=g["accuracy"] * 100, mode="lines+markers", name=name))
    fig.update_xaxes(title_text="Hours ahead", type="category"); fig.update_yaxes(title_text="Accuracy (%)")
    st.plotly_chart(base_layout(fig, 320), key="ins_cond")
    st.caption("At short range “the sky stays the same” is about as good as our model; ours is only modestly better further out — the sky type is genuinely hard to predict from a single station.")

    conf = np.array(meta["condition_confusion_24h"]) * 100
    fig = go.Figure(go.Heatmap(z=conf, x=meta["cond_classes"], y=meta["cond_classes"], colorscale="Blues",
                               text=np.round(conf).astype(int), texttemplate="%{text}%", showscale=False))
    fig.update_xaxes(title_text="Predicted"); fig.update_yaxes(title_text="Actual", autorange="reversed")
    st.markdown("**Confusion matrix at +24 h** (each row sums to 100 %: “when the sky really was X, what did we predict?”)")
    st.plotly_chart(base_layout(fig, 380), key="ins_conf")

    with st.expander("Training details"):
        st.json({"trained_at": meta["created_at"], "libraries": meta["versions"], "data_split": meta["splits"],
                 "hyperparameters": meta["hyperparameters"], "best_boosting_rounds": meta["best_iterations"]})


def render_data_tab(df, report, uploaded):
    st.markdown("Real data is messy. This is everything that was found and fixed **before** any model saw the data.")
    rows = [
        ("Rows in the file", report["raw_rows"], "–"),
        ("Duplicate timestamps", report["duplicate_timestamps_removed"], "Kept the first, dropped the rest"),
        ("Rows not sorted by time", "yes" if report["was_unsorted"] else "no", "Sorted chronologically"),
        ("Missing hours in the timeline", report["missing_hours_filled"], "Added and filled by interpolation"),
        ("Pressure = 0 hPa (impossible)", report["pressure_invalid"], "Treated as missing, interpolated"),
        ("Humidity = 0 % (sensor error)", report["humidity_invalid"], "Treated as missing, interpolated"),
        ("Visibility = 0 km (placeholder)", report["visibility_invalid"], "Treated as missing, interpolated"),
        ("Columns dropped", ", ".join(report["dropped_columns"]) or "none",
         "Cloud cover is all zeros; the daily summary describes future hours (a data leak)"),
        ("Rows after cleaning", report["final_rows"], f"{report['start'][:10]} → {report['end'][:10]}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["What we found", "How many", "What we did"]).astype(str), hide_index=True, width="stretch")

    st.markdown("#### Daily average temperature (whole history)")
    daily = df["temp_c"].resample("D").mean()
    fig = go.Figure(go.Scatter(x=wc.to_local(daily.index), y=daily.to_numpy(), mode="lines",
                               line=dict(color=wc.TARGETS["temp_c"]["color"], width=1.5)))
    fig.update_yaxes(title_text="°C")
    st.plotly_chart(base_layout(fig, 300), key="data_hist")
    with st.expander("Peek at the cleaned table (last 48 hours)"):
        st.dataframe(df.tail(48).round(2), width="stretch")


def render_about_tab(bundle):
    st.markdown(
        """
### How this works, in plain English

**The idea.** Look at the weather *right now* and at the last 7 days, then learn from 10 years of history how
the weather usually changes over the next 1–72 hours.

**The steps** (the same ones you can follow in `train.ipynb`)

1. **Clean** the raw data (duplicates, impossible zeros, gaps).
2. **Build features** — about 80 numbers describing the situation: current values, what they were 1, 3, 6, 12, 24 … hours ago,
   trends, wind direction as a circle, dew point, time of day and season. *Only the past is used — never the future.*
3. **Train** one gradient-boosting model per variable. It is told *“how many hours ahead?”* as an extra input and predicts the
   **change** from now, which is easier than predicting the absolute value.
4. **Test honestly** — train on 2006–2013, tune on 2014, and score on 2015–2016, years the model has never seen.
5. **Add uncertainty** — the 80 % band comes from the model's real mistakes on the tuning year, so it is measured, not guessed.
6. **Forecast** — this app feeds today's situation into the trained models.

**Why not just “assume nothing changes”?** That is the *persistence* baseline. It is hard to beat for the first hour or two; from about 3–6 hours
on our models are clearly better for most variables (see *Model insights* in Advanced mode). A plain linear model is a surprisingly
strong rival for wind and pressure, so the extra complexity of boosting does not pay off everywhere — that is reported honestly too.

**Limitations** — a single weather station; no rainfall or cloud-cover measurements in the data; the data ends in 2016; and this is a
statistical model of one place, not a physics-based simulation like a national weather service runs. Use it to learn and to
compare, not to plan a mountain expedition.
        """
    )
    if bundle["version_warning"]:
        st.warning(bundle["version_warning"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not (MODEL_DIR / "meta.json").exists():
        st.error("No trained models found. Open **train.ipynb**, run all cells (Kernel → Restart & Run All), "
                 "then refresh this page. The models are saved to the `models/` folder.")
        st.stop()
    bundle = get_bundle()

    sb, mode, upload = sidebar()
    advanced = mode == "Advanced"

    try:
        df, feats, report = get_data(upload.getvalue() if upload is not None else None)
    except Exception as e:                                       # bad upload -> friendly message
        st.error(f"Could not read that file: {e}")
        st.stop()
    if len(feats) <= wc.MIN_HISTORY_HOURS + 24:
        st.error("The file has too little data — at least about 8 days of hourly rows are needed.")
        st.stop()

    origin = pick_start(sb, feats, uploaded=upload is not None)

    sb.markdown("### 3 · How far ahead?")
    horizon = sb.slider("Forecast length (hours)", 6, wc.MAX_HORIZON, 24, step=6,
                        help="Short forecasts are more accurate. Beyond ~48 h treat them as a trend, not a promise.")

    sb.markdown("### 4 · Units")
    units = {"temp": sb.radio("Temperature", ["°C", "°F"], horizontal=True),
             "wind": sb.radio("Wind", ["km/h", "mph", "m/s"], horizontal=True)}

    with sb.expander("ℹ️ Quick guide"):
        st.markdown("Pick a start time and a length. Forecast lines come with a shaded band = the likely range. "
                    "Switch to **Advanced** for every variable, sky probabilities, a comparison with reality and model accuracy.")

    try:
        fc, current = wc.make_forecast(feats, origin, bundle, horizon)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.title("Weather forecast")
    st.caption(f"From **{local_label(origin)}** · next **{horizon} hours** · local time ({wc.DISPLAY_TZ})")

    if advanced:
        names = ["🌤️ Forecast", "📈 All variables", "🎯 Check vs reality", "🧠 Model insights", "🧹 Data & cleaning", "📖 How it works"]
        t = st.tabs(names)
        with t[0]: render_forecast_tab(fc, current, units, horizon, advanced)
        with t[1]: render_all_variables_tab(fc, current, units)
        with t[2]: render_check_tab(fc, df, current, units)
        with t[3]: render_insights_tab(bundle)
        with t[4]: render_data_tab(df, report, upload is not None)
        with t[5]: render_about_tab(bundle)
    else:
        t = st.tabs(["🌤️ Forecast", "📖 How it works"])
        with t[0]: render_forecast_tab(fc, current, units, horizon, advanced)
        with t[1]: render_about_tab(bundle)


main()

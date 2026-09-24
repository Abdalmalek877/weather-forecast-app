# 🌦️ Weather Forecast Lab

A complete machine-learning project: **10 years of hourly weather → trained models → interactive forecast app.**

* Forecasts **1–72 hours ahead** for temperature, feels-like temperature, humidity, wind speed, pressure and visibility
* Predicts the **sky condition** as probabilities (Clear / Partly Cloudy / Mostly Cloudy / Overcast / Foggy)
* Every forecast comes with an **80 % uncertainty band** measured from the model's real mistakes
* **Simple mode** for the essentials, **Advanced (full-power) mode** for everything: all variables, sky probabilities, a comparison with what really happened, accuracy tables, feature importance, data-cleaning report
* Honest evaluation on two years the model never saw, against three baselines

---

## 1 · Project structure

```
weather-forecast-project/
├── train.ipynb          ← STEP 1: builds the models (already run; outputs are saved inside)
├── main.py              ← STEP 2: the Streamlit web app
├── weather_core.py      ← shared logic used by BOTH files (cleaning, features, forecasting)
├── data/weatherHistory.csv
├── models/              ← created by train.ipynb: regressors.joblib, classifier.joblib, meta.json, metrics CSVs
├── requirements.txt
└── .streamlit/config.toml
```

**Why does `weather_core.py` exist?** If the notebook and the app each had their own copy of the feature code, one tiny difference would silently
give wrong forecasts (“train/serve skew”). One shared file means the app computes features *exactly* the way the models were trained.

---

## 2 · Setup and run

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. launch the app  (the trained models are already included in models/)
streamlit run main.py
```

To **re-train** the models (about 8–10 minutes on a laptop CPU): `jupyter notebook train.ipynb` → *Kernel → Restart & Run All*.
The notebook overwrites the `models/` folder; then refresh the app.

**Using your own data:** upload a CSV in the app's sidebar. It needs the same columns as `weatherHistory.csv`
(`Formatted Date, Summary, Temperature (C), Apparent Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees), Visibility (km), Pressure (millibars)`)
and at least ~8 days of hourly rows. Note that the models were trained on *this* station, so forecasts for a different city are only a rough guide.

---

## 3 · How it works (the story for your viva)

**Goal:** given the weather now and over the past week, predict the next 72 hours.

1. **Data cleaning.** The file had duplicates (24), unsorted rows, 3 missing hours, and impossible values (pressure = 0 hPa in 1,288 rows, humidity = 0 %, visibility = 0 km placeholders).
   We fixed these. We also *dropped* two columns on purpose: `Loud Cover` (all zeros) and `Daily Summary` — it describes the whole day *including future hours*, which would be **data leakage** (the model would “cheat”).
2. **Feature engineering (~80 inputs).** Current values; values 1, 2, 3, 6, 12, 24, 48, 72 hours ago; trends; 24-hour and 7-day statistics; wind as x/y components (direction is a circle);
   dew point; time of day and season encoded as sine/cosine. A **leakage test** in the notebook proves features at time *t* never use anything after *t*.
3. **Direct multi-horizon forecasting.** One model per variable receives *“how many hours ahead?”* as an input. This avoids the error build-up of predicting one hour at a time and feeding predictions back in.
4. **Predict the change, not the value.** The model predicts “how much will temperature change” (easier than the absolute value), scaled by the typical change at that horizon so short and long horizons are learned equally carefully.
5. **Algorithm: gradient boosting** (`HistGradientBoosting` from scikit-learn — same family as XGBoost/LightGBM). The number of trees is chosen on a validation year, not guessed.
6. **Time-based split — never shuffled.** Train 2006–2013 · Validation 2014 · Test 2015–2016, with 72-hour gaps between them so no future information leaks across.
7. **Uncertainty.** The 80 % band for each hour ahead is the 10th–90th percentile of the model's errors on the validation year.
8. **Sky condition.** A classifier outputs the probability of each of 5 sky types. The 27 messy raw labels were simplified to these 5.

---

## 4 · Results on unseen test years (2015–2016)

Average error (MAE) — lower is better. “No change” = assume the weather stays as it is now.

| Temperature (°C)       | +1 h | +6 h | +24 h | +72 h |
|------------------------|-----:|-----:|------:|------:|
| “No change”            | 0.87 | 4.03 | 2.22  | 3.40  |
| Linear model (Ridge)   | 0.65 | 1.70 | 2.06  | 2.84  |
| **Gradient boosting**  | 0.72 | **1.47** | **1.97** | 2.84 |

What the results really say (and you should say them too):

* Boosting is the most accurate model for **temperature, feels-like temperature and humidity** from about +3 h to +48 h; at +6 h it cuts the temperature error by ~64 % versus “no change”.
* At **+1 h** “no change” or the simple linear model is as good or better for several variables — the weather barely changes in one hour.
* For **wind and pressure** the simple linear model is about as good as boosting. More complex is not automatically better.
* **Sky condition** is hard to predict from one station: at +24 h accuracy is 45 % vs 45 % for “the sky stays the same” and 32 % for “always guess the most common class”. That is why the app shows *probabilities*.
* The 80 % bands actually contained **79 %** of real outcomes on the test years (temperature and pressure slightly lower, ~78 %).

---

## 5 · Limitations (honest ones)

* **One weather station.** The model cannot see weather approaching from elsewhere. Real forecasts use satellites and physics-based simulations; ours learns statistical patterns from one place, so skill fades after 1–2 days.
* **No rainfall or cloud-cover data** in the file, so there is no “chance of rain” — only sky condition.
* Bands are calibrated on one year and have the same width in all seasons.
* The data ends in December 2016: the app is a demonstration on historical data, not a live service. The “Check vs reality” tab works *because* the future is known for historical start times.
* Times are shown in Central European local time (the station's timezone).

---

## 6 · Troubleshooting

| Problem | Fix |
|---|---|
| “No trained models found” | Run all cells of `train.ipynb`, then refresh the app. |
| Warning about a scikit-learn version | The models were trained with scikit-learn 1.8. Usually harmless; if you see errors, re-run `train.ipynb` to rebuild the models on your version. |
| Chart or parameter errors in the app | Update Streamlit: `pip install -U streamlit plotly` (needs Streamlit ≥ 1.50). |
| Notebook can't find `weather_core` | Start Jupyter from inside the project folder. |

---

## 7 · Ideas for the next level

LSTM / Transformer models · live data from a weather API · several cities · season-specific uncertainty bands · SHAP explanations · comparison with XGBoost / LightGBM / CatBoost.

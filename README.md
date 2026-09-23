# Renewable Energy & German Electricity Price Analysis

## From Prediction to Causal Effects: XGBoost, SHAP and Double Machine Learning

This project was developed as part of my **Master's Thesis** and investigates how **solar and wind electricity generation influence German day-ahead electricity prices**.

The project combines three perspectives of modern data analysis:

- **Prediction:** XGBoost is used to predict German day-ahead electricity prices.
- **Explainability:** SHAP is applied to identify the variables that drive the model's predictions.
- **Causal Analysis:** Double Machine Learning with Neighbors-Left-Out cross-fitting (NLO-DML) is used to estimate the effects of solar and wind generation on electricity prices.

The main objective is to distinguish between variables that are important for **predicting electricity prices** and variables that can be interpreted as having an estimated **causal effect** under the assumptions of the DML model.

---

## Project Details

- **Data Source:** [Open Power System Data – Time Series](https://data.open-power-system-data.org/time_series/2020-10-06/)
- **Country:** Germany
- **Frequency:** Hourly time-series data
- **Analysis Period:** October 2018 – September 2020
- **Final Dataset:** 17,376 hourly observations
- **Target Variable:** `DE_LU_price_day_ahead`
- **Main Treatments:** Solar generation and wind generation
- **Programming Language:** Python

### Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- Statsmodels
- Jupyter Notebook

---

## Project Workflow

The project is divided into two main notebooks:

**1. `Preprocessing_version.ipynb`**

The original dataset `time_series_60min_singleindex.csv` is cleaned, transformed and prepared for the empirical analysis.

The resulting dataset is exported as:

`df_final_60min.csv`

**2. `ML_version.ipynb`**

The preprocessed dataset is used for:

- XGBoost electricity-price prediction
- Model evaluation
- SHAP explainability
- NLO Double Machine Learning for solar generation
- NLO Double Machine Learning for wind generation

---

# 1. Data Preprocessing

The original Open Power System Data dataset contains electricity-system information for several European countries.

Since this project focuses on Germany, the dataset was reduced to variables related to:

- German electricity demand
- Electricity-demand forecasts
- Solar generation
- Wind generation
- Installed solar and wind capacity
- German-Luxembourg day-ahead electricity prices

The original dataset contained approximately **50,401 hourly observations and 300 variables**.

After selecting the relevant period and variables, the time series was checked for missing timestamps, duplicate observations and missing values.

### Missing Data Treatment

Four missing hourly timestamps were identified and reintroduced to preserve a continuous hourly time series.

Missing values were handled using:

- **Forward and backward filling** for installed renewable capacity
- **Linear interpolation** for missing day-ahead electricity prices

After restoring the time series, the dataset contained **17,544 consecutive hourly observations**.

---

## Autocorrelation Analysis

Before feature engineering, the autocorrelation structure of the electricity-price series was examined.

![Autocorrelation of Electricity Prices](images/acf_electricity_price.png)

The ACF shows strong positive autocorrelation between consecutive electricity prices.

Recurring patterns around daily lags indicate that historical electricity prices contain important information for predicting current prices.

This motivated the creation of lagged and rolling price features.

---

## Feature Engineering

Several time-series features were created to allow XGBoost to capture temporal patterns.

### Calendar Features

- `hour`
- `dayofweek`
- `hour_sin`
- `hour_cos`
- `dayofweek_sin`
- `dayofweek_cos`

Sine and cosine transformations were used because hours and weekdays are cyclical variables.

For example, hour 23 and hour 0 are chronologically close even though their numerical values are far apart.

### Price Lag Features

- `price_lag_1h`
- `price_lag_24h`
- `price_lag_168h`

These variables represent:

- previous-hour electricity price
- same-hour price on the previous day
- same-hour price one week earlier

### Rolling Price Features

- `price_rolling_mean_24h`
- `price_rolling_std_24h`

These variables capture the recent electricity-price level and short-term market volatility.

All lagged and rolling features were constructed using information available **before the current observation** to avoid target leakage.

After feature engineering, the final modelling dataset contained:

**17,376 observations and 20 columns.**

---

# 2. Exploratory Data Analysis

## Electricity Price Development

![Electricity Price Over Time](images/electricity_price_over_time.png)

German day-ahead electricity prices exhibit considerable short-term volatility.

The dataset contains both positive and negative price spikes. Negative prices were retained because they represent actual electricity-market behaviour rather than data errors.

They may occur when electricity supply is high relative to demand, particularly during periods of strong renewable generation.

---

## Electricity Demand and Renewable Generation

![Demand and Renewable Generation](images/demand_solar_wind_over_time.png)

Electricity demand generally remains above combined solar and wind generation, but the difference varies significantly over time.

Solar and wind also show different temporal patterns:

- **Solar generation** follows a strong daily and seasonal pattern.
- **Wind generation** is more irregular and strongly dependent on weather conditions.

For this reason, solar and wind generation were analysed separately.

---

# 3. XGBoost Electricity Price Prediction

An **XGBoost Regressor** was trained to predict the German-Luxembourg day-ahead electricity price.

The dataset was divided chronologically into:

- **80% training data**
- **20% test data**

The observations were not randomly shuffled because the dataset represents a time series.

### XGBoost Configuration

The main model parameters were:

| Parameter | Value |
|---|---:|
| Number of estimators | 350 |
| Learning rate | 0.03 |
| Maximum tree depth | 4 |
| Minimum child weight | 5 |
| Subsample | 0.80 |
| Column sample by tree | 0.80 |
| L1 regularization | 0.0 |
| L2 regularization | 1.0 |

---

## Model Performance

The predictive performance was evaluated using **R²** and **RMSE**.

| Metric | Test Result |
|---|---:|
| R² | **0.9172** |
| RMSE | **4.7732 EUR/MWh** |

The model explains approximately **91.72% of the observed variation** in electricity prices in the test dataset.

---

## Actual vs. Predicted Electricity Prices

![Actual vs Predicted Electricity Prices](images/actual_vs_predicted_prices.png)

The predicted electricity-price series follows the observed price development closely.

The model successfully captures:

- regular short-term price fluctuations
- negative-price periods
- most medium-sized price movements

However, the model tends to underestimate some rare extreme positive price spikes.

---

# 4. Model Explainability with SHAP

Machine-learning models can achieve strong predictive performance while remaining difficult to interpret.

To better understand the XGBoost predictions, **SHAP (SHapley Additive exPlanations)** was used.

SHAP quantifies how strongly each feature contributes to the model prediction.

---

## Global Feature Importance

![SHAP Global Feature Importance](images/shap_feature_importance.png)

The most important predictor is:

**`price_lag_1h`**

The electricity price from the previous hour therefore contains the strongest predictive information about the current electricity price.

Solar and wind generation are also among the most important non-price predictors.

Other relevant variables include:

- hour of the day
- weekly electricity-price lag
- 24-hour rolling mean
- electricity demand
- electricity-demand forecast

---

## SHAP Summary Plot

![SHAP Summary Plot](images/shap_summary_plot.png)

The SHAP summary plot shows both the magnitude and direction of each feature's contribution.

High previous-hour electricity prices generally contribute to **higher predicted current prices**.

In contrast:

- Higher **solar generation** tends to reduce the predicted electricity price.
- Higher **wind generation** also tends to reduce the predicted electricity price.

These results indicate a negative **predictive relationship** between renewable generation and electricity prices.

However, SHAP explains the behaviour of the prediction model and **does not establish causality**.

For this reason, the causal effects are analysed separately using Double Machine Learning.

---

# 5. Neighbors-Left-Out Double Machine Learning

Standard random cross-fitting is not ideal for time-series data because observations close in time may be strongly dependent.

This project therefore applies **Neighbors-Left-Out Double Machine Learning (NLO-DML)**.

The chronological observations are divided into blocks. When one block is used for evaluation, that block and its neighbouring blocks are excluded from nuisance-model training.

This creates a temporal separation between training and evaluation observations.

### NLO-DML Configuration

- **Number of folds:** 8
- **Neighbouring folds left out:** 1
- **Nuisance learner:** XGBoost
- **HAC maximum lag:** 168 hours
- **Outcome:** German day-ahead electricity price

Two separate treatment specifications were estimated:

1. Solar generation
2. Wind generation

Heteroskedasticity- and autocorrelation-consistent (**HAC**) standard errors were used to account for remaining serial dependence.

---

# 6. Causal Effect of Solar Generation

![NLO-DML Solar Generation](images/nlo_dml_solar.png)

The NLO-DML estimate for solar generation is:

**−0.2351 EUR/MWh per additional GW of solar generation**

| Measure | Result |
|---|---:|
| Estimated effect | **−0.2351 EUR/MWh per GW** |
| HAC Standard Error | **0.0180** |
| 95% Confidence Interval | **[−0.2704, −0.1999]** |
| p-value | **4.69 × 10⁻³⁹** |

Under the maintained identification assumptions of the DML model, an additional **1 GW of solar generation** is associated with an average reduction of approximately **0.235 EUR/MWh** in the German day-ahead electricity price.

---

# 7. Causal Effect of Wind Generation

![NLO-DML Wind Generation](images/nlo_dml_wind.png)

The NLO-DML estimate for wind generation is:

**−0.1022 EUR/MWh per additional GW of wind generation**

| Measure | Result |
|---|---:|
| Estimated effect | **−0.1022 EUR/MWh per GW** |
| HAC Standard Error | **0.0098** |
| 95% Confidence Interval | **[−0.1215, −0.0830]** |
| p-value | **2.46 × 10⁻²⁵** |

Under the maintained identification assumptions, an additional **1 GW of wind generation** is associated with an average reduction of approximately **0.102 EUR/MWh** in the German day-ahead electricity price.

---

# 8. Key Findings

The analysis produces three main findings.

**Prediction:**  
XGBoost achieved strong out-of-sample predictive performance with an **R² of 0.9172** and an **RMSE of 4.7732 EUR/MWh**.

**Explainability:**  
SHAP identified the **previous-hour electricity price** as the strongest predictor. Solar and wind generation were also important predictors, with higher renewable generation generally associated with lower predicted electricity prices.

**Causal Analysis:**  
Under the maintained identification assumptions, NLO-DML estimated negative price effects for both renewable technologies:

| Renewable Source | Estimated Effect per Additional GW |
|---|---:|
| Solar Generation | **−0.2351 EUR/MWh** |
| Wind Generation | **−0.1022 EUR/MWh** |

The findings are consistent with the **merit-order mechanism**, according to which electricity generated from low-marginal-cost renewable technologies can displace more expensive conventional generation in the market-clearing process.

---

# Conclusion

This project demonstrates the importance of distinguishing between **prediction, model explanation and causal inference**.

XGBoost provides accurate electricity-price predictions, while SHAP helps explain which variables drive those predictions.

The NLO-DML analysis goes one step further by estimating the effects of renewable electricity generation while accounting for high-dimensional controls and the temporal dependence of the data.

Overall, the results indicate that both **solar and wind generation are associated with lower German day-ahead electricity prices**, while demonstrating how machine learning, explainable AI and causal inference can be combined within a single time-series data-analysis workflow.

---

## Repository Structure

```text
Project/
│
├── README.md
├── Preprocessing_version.ipynb
├── ML_version.ipynb
├── df_final_60min.csv
│
└── images/
    ├── acf_electricity_price.png
    ├── electricity_price_over_time.png
    ├── demand_solar_wind_over_time.png
    ├── actual_vs_predicted_prices.png
    ├── shap_feature_importance.png
    ├── shap_summary_plot.png
    ├── nlo_dml_solar.png
    └── nlo_dml_wind.png

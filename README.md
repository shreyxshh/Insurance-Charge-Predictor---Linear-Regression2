# Insurance Charge Predictor

Streamlit app for the Linear / Ridge / Lasso regression notebook (`LinearRegressionM2.ipynb`).

## Run it

```bash
pip install -r requirements.txt
# put insurance.csv in this folder (or upload it from the sidebar)
streamlit run app.py
```

Folder layout:

```
insurance_app/
├── app.py
├── insurance.csv          <- your dataset (not included)
├── requirements.txt
└── .streamlit/config.toml
```

## What's inside

- **Predict**: profile inputs, estimated yearly charge with a typical range, percentile, driver breakdown, and an age what-if curve.
- **Explore data**: distributions, group comparisons, scatter plots, correlation heatmap.
- **Model performance**: R², RMSE, MAE for all three models, actual vs predicted, residuals, feature effects.
- **About**: the pipeline and how to read the numbers.

The app retrains from `insurance.csv` on start (a second or two, cached), reproducing the notebook's preprocessing so the scaler always matches the model.

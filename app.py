"""
Insurance Charge Predictor
--------------------------
Streamlit front-end for the LinearRegressionM2 notebook.

The app re-creates the notebook's preprocessing pipeline from `insurance.csv`
(so the scaler and feature engineering always match the model), trains
Linear / Ridge / Lasso regression on log(charges), and serves predictions.

Run:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#10263B"
TEAL = "#0B7A75"
TEAL_SOFT = "#D3ECEA"
CORAL = "#D9503A"
AMBER = "#E3A23A"
MUTED = "#5F7385"

DATA_PATH = Path(__file__).parent / "insurance.csv"
REQUIRED_COLS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

# Same feature order and scaled columns as the notebook
FEATURES = [
    "age", "is_male", "is_smoker", "region_southeast",
    "bmi_categories_Obesity", "bmi", "children",
]
SCALE_COLS = ["age", "bmi", "children"]

MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge (alpha = 1.0)": lambda: Ridge(alpha=1.0),
    "Lasso (alpha = 0.1)": lambda: Lasso(alpha=0.1),
}

# One fixed color per model, reused across every chart and table so a model
# never changes color between sections.
MODEL_COLORS = {
    "Linear Regression": "#0B7A75",   # TEAL
    "Ridge (alpha = 1.0)": "#5B7C99",  # slate blue
    "Lasso (alpha = 0.1)": "#E3A23A",  # AMBER
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
    .block-container {{ padding-top: 3.5rem; max-width: 1250px; }}
    h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}

    .page-title {{
        font-size: 2.3rem; font-weight: 800; margin: 0 0 .35rem 0;
        color: {INK} !important; line-height: 1.25;
        position: relative; z-index: 999;
    }}
    .page-sub {{ color: {MUTED}; margin: 0 0 1.4rem 0; font-size: 1.05rem; }}

    .result {{
        background: {INK}; color: #fff; border-radius: 14px;
        padding: 1.6rem 1.8rem; border-left: 8px solid {TEAL};
    }}
    .result .label {{ font-size: .95rem; color: #B9C7D3; font-weight: 500; }}
    .result .amount {{ font-size: 3.4rem; font-weight: 800; line-height: 1.1; margin: .2rem 0 .4rem 0; }}
    .result .range {{ font-size: .95rem; color: #DCE6ED; }}

    .stat {{
        background: #fff; border: 1px solid #DCE5E9; border-radius: 10px;
        padding: .8rem 1rem; height: 100%;
    }}
    .stat .k {{ font-size: .8rem; color: {MUTED}; font-weight: 600; }}
    .stat .v {{ font-size: 1.35rem; font-weight: 800; color: {INK}; }}
    .stat .n {{ font-size: .8rem; color: {MUTED}; }}

    .chip {{
        display: inline-block; padding: .15rem .65rem; border-radius: 999px;
        font-size: .82rem; font-weight: 700; background: {TEAL_SOFT}; color: {TEAL};
    }}
    .chip.warn {{ background: #FBE3DE; color: {CORAL}; }}
    .chip.mid {{ background: #FBEBCF; color: #9A6510; }}

    div[data-testid="stTabs"] button p {{ font-weight: 700; font-size: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data + model pipeline (mirrors the notebook)
# --------------------------------------------------------------------------- #
def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> unscaled model features, exactly as in the notebook.

    Note: the notebook called `df.astype(int)` which truncates `bmi` to a whole
    number *before* bmi categories and scaling. It is reproduced here so the
    app's numbers match the notebook's.
    """
    X = pd.DataFrame(
        {
            "age": raw["age"].astype(int),
            "is_male": (raw["sex"] == "male").astype(int),
            "is_smoker": (raw["smoker"] == "yes").astype(int),
            "region_southeast": (raw["region"] == "southeast").astype(int),
            "bmi": raw["bmi"].astype(int),
            "children": raw["children"].astype(int),
        }
    )
    X["bmi_categories_Obesity"] = (X["bmi"] > 29.9).astype(int)
    return X[FEATURES]


def bmi_group(bmi: float) -> str:
    if bmi <= 18.5:
        return "Underweight"
    if bmi <= 24.9:
        return "Normal"
    if bmi <= 29.9:
        return "Overweight"
    return "Obesity"


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Training models on your data…")
def train(raw: pd.DataFrame) -> dict:
    df = raw[REQUIRED_COLS].drop_duplicates().reset_index(drop=True)
    X = engineer(df)
    y = np.log(df["charges"].astype(int))  # notebook: astype(int) then np.log

    scaler = StandardScaler()
    Xs = X.copy()
    Xs[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42
    )

    fitted, rows, preds = {}, [], {}
    for name, make in MODELS.items():
        model = make().fit(X_train, y_train)
        p = model.predict(X_test)
        fitted[name], preds[name] = model, p
        rows.append(
            {
                "Model": name,
                "R²": r2_score(y_test, p),
                "RMSE (log)": float(np.sqrt(mean_squared_error(y_test, p))),
                "MAE (log)": mean_absolute_error(y_test, p),
                "MAE ($)": mean_absolute_error(np.exp(y_test), np.exp(p)),
            }
        )

    return {
        "clean": df,
        "scaler": scaler,
        "models": fitted,
        "metrics": pd.DataFrame(rows).set_index("Model"),
        "y_test": y_test,
        "preds": preds,
        "mean_scaled": Xs.mean(),
        "encoded": pd.concat([X, df["charges"]], axis=1),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_log(bundle: dict, model_name: str, inputs: pd.DataFrame) -> np.ndarray:
    X = engineer(inputs)
    X[SCALE_COLS] = bundle["scaler"].transform(X[SCALE_COLS])
    return bundle["models"][model_name].predict(X[FEATURES]), X


def money(x: float) -> str:
    return f"${x:,.0f}"


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=INK),
        title_font=dict(size=16),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="#E3EAEE", zeroline=False)
    fig.update_yaxes(gridcolor="#E3EAEE", zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: data + model choice
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Settings")

    up = st.file_uploader(
        "Drag and drop insurance.csv here, or browse",
        type="csv",
        help="Needs the columns: age, sex, bmi, children, smoker, region, charges.",
    )
    if up is not None:
        raw = pd.read_csv(up)
        st.caption(f"Using uploaded file: **{up.name}**")
    elif DATA_PATH.exists():
        raw = load_csv(str(DATA_PATH))
        st.caption("Using `insurance.csv` from the app folder. Upload a file to replace it.")
    else:
        raw = None

    model_name = st.radio(
        "Model",
        list(MODELS),
        help="All three are trained on log(charges). Linear Regression is the "
        "one your notebook saved.",
    )
    st.divider()
    st.caption(
        "Estimates come from a simple regression model trained on a public "
        "sample dataset. They are for learning, not a real insurance quote."
    )

if raw is None:
    st.markdown('<p class="page-title">Insurance Charge Predictor</p>', unsafe_allow_html=True)
    st.info(
        "Add your dataset to get started: place `insurance.csv` next to `app.py`, "
        "or upload it from the sidebar. (Kaggle: *Medical Cost Personal Datasets*.)"
    )
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in raw.columns]
if missing:
    st.error(f"The CSV is missing these columns: {', '.join(missing)}.")
    st.stop()

bundle = train(raw)
clean = bundle["clean"]
rmse_log = bundle["metrics"].loc[model_name, "RMSE (log)"]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<p class="page-title">Insurance Charge Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Enter a profile to estimate the yearly medical insurance '
    "charge, see what drives it, and compare the three regression models.</p>",
    unsafe_allow_html=True,
)

tab_predict, tab_explore, tab_models, tab_about = st.tabs(
    ["Predict", "Explore data", "Model performance", "About"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Predict
# --------------------------------------------------------------------------- #
with tab_predict:
    if "bmi" not in st.session_state:
        st.session_state["bmi"] = 27.0

    def apply_bmi():
        h = st.session_state["h_cm"] / 100
        st.session_state["bmi"] = float(
            np.clip(round(st.session_state["w_kg"] / (h * h), 1), 15.0, 55.0)
        )

    left, right = st.columns([1, 1.45], gap="large")

    with left:
        st.subheader("Profile")
        age = st.slider("Age", 18, 64, 35)
        sex = st.radio("Sex", ["female", "male"], horizontal=True, format_func=str.title)
        bmi = st.slider("BMI", 15.0, 55.0, step=0.1, key="bmi")
        grp = bmi_group(bmi)
        chip_cls = {"Normal": "", "Underweight": "mid", "Overweight": "mid", "Obesity": "warn"}[grp]
        st.markdown(f'<span class="chip {chip_cls}">{grp}</span>', unsafe_allow_html=True)
        with st.expander("Don't know your BMI?"):
            c1, c2 = st.columns(2)
            c1.number_input("Height (cm)", 120, 220, 170, key="h_cm")
            c2.number_input("Weight (kg)", 30, 200, 75, key="w_kg")
            st.button("Use this BMI", on_click=apply_bmi)
        children = st.slider("Children covered", 0, 5, 0)
        smoker = st.radio("Smoker", ["no", "yes"], horizontal=True, format_func=str.title)
        region = st.selectbox(
            "Region", ["northeast", "northwest", "southeast", "southwest"],
            format_func=str.title,
        )

    profile = pd.DataFrame(
        [dict(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)]
    )
    log_pred, X_row = predict_log(bundle, model_name, profile)
    pred = float(np.exp(log_pred[0]))
    low, high = float(np.exp(log_pred[0] - rmse_log)), float(np.exp(log_pred[0] + rmse_log))
    pctile = float((clean["charges"] < pred).mean() * 100)
    median = float(clean["charges"].median())
    vs_median = (pred / median - 1) * 100

    with right:
        st.markdown(
            f"""
            <div class="result">
              <div class="label">Estimated yearly insurance charge</div>
              <div class="amount">{money(pred)}</div>
              <div class="range">Typical range {money(low)} to {money(high)}
              &nbsp;·&nbsp; {model_name.split(" (")[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        s1, s2, s3 = st.columns(3)
        s1.markdown(
            f'<div class="stat"><div class="k">Higher than</div>'
            f'<div class="v">{pctile:.0f}%</div>'
            f'<div class="n">of policies in the dataset</div></div>',
            unsafe_allow_html=True,
        )
        s2.markdown(
            f'<div class="stat"><div class="k">Versus median policy</div>'
            f'<div class="v">{vs_median:+.0f}%</div>'
            f'<div class="n">median is {money(median)}</div></div>',
            unsafe_allow_html=True,
        )
        s3.markdown(
            f'<div class="stat"><div class="k">Monthly equivalent</div>'
            f'<div class="v">{money(pred / 12)}</div>'
            f'<div class="n">estimate ÷ 12</div></div>',
            unsafe_allow_html=True,
        )

        # Where this profile sits in the population
        st.write("")
        fig = px.histogram(clean, x="charges", nbins=45, color_discrete_sequence=["#B7C9D3"])
        fig.add_vline(x=pred, line_color=CORAL, line_width=3)
        fig.add_annotation(
            x=pred, y=1, yref="paper", text="This profile", showarrow=False,
            yanchor="bottom", font=dict(color=CORAL, size=13),
        )
        fig.update_layout(title="Where this estimate falls among all policies",
                          xaxis_title="Yearly charge ($)", yaxis_title="Policies", bargap=0.05)
        st.plotly_chart(style(fig, 280), width="stretch")

    st.divider()
    d1, d2 = st.columns(2, gap="large")

    # What drives this estimate (log-linear => multiplicative effects)
    with d1:
        model = bundle["models"][model_name]
        coef = pd.Series(model.coef_, index=FEATURES)
        contrib = coef * (X_row.iloc[0] - bundle["mean_scaled"])
        groups = {
            "Age": ["age"],
            "Smoking": ["is_smoker"],
            "BMI": ["bmi", "bmi_categories_Obesity"],
            "Children": ["children"],
            "Sex": ["is_male"],
            "Region": ["region_southeast"],
        }
        impact = pd.Series(
            {g: (np.exp(contrib[c].sum()) - 1) * 100 for g, c in groups.items()}
        ).sort_values()
        fig = go.Figure(
            go.Bar(
                x=impact.values, y=impact.index, orientation="h",
                marker_color=[CORAL if v > 0 else TEAL for v in impact.values],
                text=[f"{v:+.0f}%" for v in impact.values], textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(title="What moves the estimate (vs. an average policyholder)",
                          xaxis_title="Change in charge (%)", showlegend=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(style(fig, 330), width="stretch")

    # What-if: age curve for smokers vs non-smokers
    with d2:
        ages = np.arange(18, 65)
        curves = []
        for s_flag, label in [("no", "Non-smoker"), ("yes", "Smoker")]:
            sweep = pd.DataFrame(
                dict(age=ages, sex=sex, bmi=bmi, children=children, smoker=s_flag, region=region)
            )
            lp, _ = predict_log(bundle, model_name, sweep)
            curves.append(pd.DataFrame({"Age": ages, "Charge": np.exp(lp), "Group": label}))
        curves = pd.concat(curves)
        fig = px.line(
            curves, x="Age", y="Charge", color="Group",
            color_discrete_map={"Non-smoker": TEAL, "Smoker": CORAL},
        )
        fig.update_traces(line_width=3)
        fig.add_trace(
            go.Scatter(x=[age], y=[pred], mode="markers", name="Your profile",
                       marker=dict(size=13, color=INK, line=dict(width=2, color="#fff")))
        )
        fig.update_layout(title="What-if: charge by age, other inputs held fixed",
                          yaxis_title="Yearly charge ($)")
        st.plotly_chart(style(fig, 330), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2: Explore data
# --------------------------------------------------------------------------- #
with tab_explore:
    k1, k2, k3, k4 = st.columns(4)
    for col, k, v in [
        (k1, "Policies (after de-duplication)", f"{len(clean):,}"),
        (k2, "Average charge", money(clean["charges"].mean())),
        (k3, "Median charge", money(clean["charges"].median())),
        (k4, "Smokers", f"{(clean['smoker'] == 'yes').mean() * 100:.0f}%"),
    ]:
        col.markdown(
            f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    e1, e2 = st.columns(2, gap="large")
    with e1:
        view = st.radio("Charge distribution", ["Raw charges", "Log of charges"], horizontal=True)
        series = clean["charges"] if view == "Raw charges" else np.log(clean["charges"])
        fig = px.histogram(series, nbins=40, color_discrete_sequence=[TEAL], marginal="box")
        fig.update_layout(showlegend=False, xaxis_title=view, yaxis_title="Policies", bargap=0.04)
        st.plotly_chart(style(fig, 380), width="stretch")
        st.caption(
            "Charges are strongly skewed, which is why the model predicts "
            "log(charges) instead."
        )
    with e2:
        by = st.selectbox("Compare charges by", ["smoker", "sex", "region", "children"])
        fig = px.box(
            clean, x=by, y="charges", color=by, points="outliers",
            color_discrete_sequence=[TEAL, CORAL, AMBER, "#5B7C99", "#8E6C8A", "#7A9E7E"],
        )
        fig.update_layout(showlegend=False, yaxis_title="Yearly charge ($)")
        st.plotly_chart(style(fig, 380), width="stretch")
        st.write("")  # spacing to keep both columns aligned

    e3, e4 = st.columns(2, gap="large")
    with e3:
        c_x, c_c = st.columns(2)
        xvar = c_x.selectbox("X axis", ["age", "bmi"])
        cvar = c_c.selectbox("Colour by", ["smoker", "sex", "region"])
        fig = px.scatter(
            clean, x=xvar, y="charges", color=cvar, opacity=0.65,
            color_discrete_sequence=[TEAL, CORAL, AMBER, "#5B7C99"],
        )
        fig.update_layout(yaxis_title="Yearly charge ($)")
        st.plotly_chart(style(fig, 380), width="stretch")
    with e4:
        corr = bundle["encoded"].corr()
        fig = px.imshow(
            corr.round(2), text_auto=True, aspect="auto",
            color_continuous_scale=[[0, TEAL], [0.5, "#F7F9FA"], [1, CORAL]],
            zmin=-1, zmax=1,
        )
        fig.update_layout(title="Correlation between engineered features")
        st.write("")
        st.plotly_chart(style(fig, 420), width="stretch")

    with st.expander("Preview the data"):
        st.dataframe(clean.head(100), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 3: Model performance
# --------------------------------------------------------------------------- #
with tab_models:
    st.caption(
        f"80/20 split (random_state=42): {bundle['n_train']:,} training rows, "
        f"{bundle['n_test']:,} test rows. Metrics marked (log) are on log(charges), "
        "as in your notebook."
    )
    metrics = bundle["metrics"]
    m1, m2, m3 = st.columns(3)
    best = metrics["R²"].idxmax()
    m1.markdown(
        f'<div class="stat"><div class="k">R² ({model_name.split(" (")[0]})</div>'
        f'<div class="v">{metrics.loc[model_name, "R²"]:.3f}</div>'
        f'<div class="n">share of variance explained</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="stat"><div class="k">Average error</div>'
        f'<div class="v">{money(metrics.loc[model_name, "MAE ($)"])}</div>'
        f'<div class="n">mean absolute error on test set</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="stat"><div class="k">Best model by R²</div>'
        f'<div class="v">{best.split(" (")[0]}</div>'
        f'<div class="n">R² {metrics.loc[best, "R²"]:.3f}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    p1, p2 = st.columns([1.1, 1], gap="large")
    with p1:
        best_style = f"background-color: {TEAL_SOFT}; color: {INK}; font-weight: 700;"
        st.dataframe(
            metrics.style.format(
                {"R²": "{:.4f}", "RMSE (log)": "{:.4f}", "MAE (log)": "{:.4f}", "MAE ($)": "${:,.0f}"}
            ).highlight_max(subset=["R²"], props=best_style)
            .highlight_min(subset=["RMSE (log)", "MAE (log)", "MAE ($)"], props=best_style),
            width="stretch",
        )
        st.caption("Highlighted cell = best value in that column (highest R², lowest error).")
        fig = px.bar(
            metrics.reset_index(), x="Model", y="R²", text=metrics["R²"].round(3).values,
            color="Model", color_discrete_map=MODEL_COLORS,
        )
        fig.update_layout(showlegend=False, title="R² by model", yaxis_range=[0, 1])
        st.plotly_chart(style(fig, 300), width="stretch")

    with p2:
        y_true = np.exp(bundle["y_test"].values)
        y_hat = np.exp(bundle["preds"][model_name])
        model_color = MODEL_COLORS[model_name]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_hat, mode="markers",
                                 marker=dict(color=model_color, opacity=0.6, size=7), name="Test policies"))
        top = max(y_true.max(), y_hat.max())
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines",
                                 line=dict(color=CORAL, dash="dash"), name="Perfect prediction"))
        fig.update_layout(title=f"Actual vs predicted ({model_name.split(' (')[0]})",
                          xaxis_title="Actual charge ($)", yaxis_title="Predicted charge ($)")
        st.plotly_chart(style(fig, 400), width="stretch")

    r1, r2 = st.columns(2, gap="large")
    with r1:
        resid = bundle["y_test"].values - bundle["preds"][model_name]
        fig = px.histogram(resid, nbins=35, color_discrete_sequence=[MODEL_COLORS[model_name]])
        fig.update_layout(title="Residuals (log scale)", showlegend=False,
                          xaxis_title="Actual - predicted", yaxis_title="Policies")
        st.plotly_chart(style(fig, 320), width="stretch")
    with r2:
        model = bundle["models"][model_name]
        scale = dict(zip(SCALE_COLS, bundle["scaler"].scale_))
        rows = []
        for f, c in zip(FEATURES, model.coef_):
            per_unit = c / scale[f] if f in scale else c
            unit = {"age": "each extra year", "bmi": "each extra BMI point",
                    "children": "each extra child"}.get(f, "yes vs no")
            label = {"is_male": "Male", "is_smoker": "Smoker",
                     "region_southeast": "Southeast region",
                     "bmi_categories_Obesity": "BMI in obesity range"}.get(f, f.title() if f != "bmi" else "BMI")
            rows.append({"Feature": label, "Compared by": unit,
                         "Effect on charge": f"{(np.exp(per_unit) - 1) * 100:+.1f}%"})
        st.markdown("**How each feature shifts the charge** (holding the others fixed)")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 4: About
# --------------------------------------------------------------------------- #
with tab_about:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.subheader("How the model works")
        st.markdown(
            """
            1. **Clean:** drop duplicate rows.
            2. **Encode:** sex and smoker become 0/1; region is one-hot encoded.
            3. **Engineer:** BMI is binned (underweight, normal, overweight, obesity).
            4. **Scale:** age, BMI and children are standardised.
            5. **Select:** Pearson correlation and chi-squared tests kept
               age, sex, smoker, southeast region, obesity flag, BMI and children.
            6. **Transform the target:** the model predicts `log(charges)`; the app
               converts back with `exp()`.
            7. **Train:** Linear, Ridge and Lasso regression on an 80/20 split.
            """
        )
    with a2:
        st.subheader("Reading the results")
        st.markdown(
            """
            - The **typical range** is the estimate multiplied and divided by the
              model's test error (one RMSE on the log scale). It is a rough guide,
              not a confidence interval.
            - Because the target is logged, each feature changes the charge by a
              **percentage**, not a fixed dollar amount.
            - Only the southeast region was kept as a feature, so the other three
              regions give identical estimates.
            - The dataset is a public US sample in dollars. Treat outputs as an
              educational estimate, not a quote.
            """
        )

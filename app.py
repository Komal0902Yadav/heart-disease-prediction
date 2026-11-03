import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl

warnings.filterwarnings("ignore")
st.set_page_config(page_title="CardioCheck", page_icon="❤️", layout="wide")

DATA_FILE = Path(__file__).parent / "processed.cleveland.data"
COLORS = {"No Disease": "#2E86AB", "Disease": "#D1495B"}

# ---------- labels for the dropdowns ----------
SEX = {0: "Female", 1: "Male"}
CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
YESNO = {0: "No", 1: "Yes"}
RESTECG = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

DEFAULTS = dict(age=55, sex=1, cp=4, trestbps=130, chol=240, fbs=0, restecg=0,
                thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=3)
EXAMPLES = {
    "Low-risk patient": dict(age=35, sex=0, cp=3, trestbps=115, chol=190, fbs=0, restecg=0,
                             thalach=185, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3),
    "Borderline patient": dict(age=55, sex=1, cp=3, trestbps=135, chol=240, fbs=0, restecg=1,
                               thalach=140, exang=0, oldpeak=1.5, slope=2, ca=1, thal=3),
    "High-risk patient": dict(age=65, sex=1, cp=4, trestbps=160, chol=290, fbs=1, restecg=2,
                              thalach=105, exang=1, oldpeak=3.0, slope=3, ca=3, thal=7),
}


# ---------- load data + train (cached so it only runs once) ----------
@st.cache_data
def get_data():
    return pl.load_data(DATA_FILE)


@st.cache_resource
def get_bundle():
    return pl.train_all(get_data())


if not DATA_FILE.exists():
    st.error("Dataset not found. Put `processed.cleveland.data` in the same folder as app.py.")
    st.stop()

df = get_data()
bundle = get_bundle()
models = bundle["models"]

# ---------- sidebar: patient input ----------
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example():
    ex = EXAMPLES.get(st.session_state["example"])
    if ex:
        st.session_state.update(ex)


sb = st.sidebar
sb.header("Patient details")
sb.selectbox("Quick example", ["Custom"] + list(EXAMPLES), key="example", on_change=load_example)
model_name = sb.selectbox("Model", list(models), index=1)
sb.divider()
age = sb.slider("Age", 20, 80, key="age")
sex = sb.selectbox("Sex", list(SEX), format_func=SEX.get, key="sex")
cp = sb.selectbox("Chest pain type", list(CP), format_func=CP.get, key="cp")
trestbps = sb.slider("Resting blood pressure (mm Hg)", 90, 200, key="trestbps")
chol = sb.slider("Cholesterol (mg/dl)", 120, 570, key="chol")
fbs = sb.selectbox("Fasting blood sugar > 120 mg/dl", list(YESNO), format_func=YESNO.get, key="fbs")
restecg = sb.selectbox("Resting ECG", list(RESTECG), format_func=RESTECG.get, key="restecg")
thalach = sb.slider("Maximum heart rate", 70, 205, key="thalach")
exang = sb.selectbox("Exercise-induced angina", list(YESNO), format_func=YESNO.get, key="exang")
oldpeak = sb.slider("ST depression (oldpeak)", 0.0, 6.5, step=0.1, key="oldpeak")
slope = sb.selectbox("ST segment slope", list(SLOPE), format_func=SLOPE.get, key="slope")
ca = sb.selectbox("Major vessels coloured (0-3)", [0, 1, 2, 3], key="ca")
thal = sb.selectbox("Thalassemia", list(THAL), format_func=THAL.get, key="thal")

patient = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
               thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

# ---------- header ----------
st.title("❤️ CardioCheck – Heart Disease Risk Assistant")
st.caption("Decision Tree + Random Forest + Naïve Bayes trained on the UCI Cleveland Heart Disease dataset")

tab1, tab2, tab3, tab4 = st.tabs(["🩺 Risk Prediction", "📊 Data Exploration",
                                  "🤖 Model Performance", "ℹ️ About"])

# =====================================================================
# TAB 1: PREDICTION
# =====================================================================
with tab1:
    probs = {m: pl.predict_proba(bundle, m, patient) for m in models}
    p = probs[model_name]

    if p >= 0.6:
        level, color = "High risk", "#D1495B"
    elif p >= 0.4:
        level, color = "Borderline", "#E9A820"
    else:
        level, color = "Low risk", "#2E9E5B"

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Prediction by {model_name}")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Probability of heart disease"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 40], "color": "#DDF3E4"},
                             {"range": [40, 60], "color": "#FFF3CD"},
                             {"range": [60, 100], "color": "#F8D7DA"}]}))
        gauge.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge)
        label = "Heart disease likely" if p >= 0.5 else "No heart disease predicted"
        if level == "High risk":
            st.error(f"**{level}: {label}.** Further medical examination is recommended.")
        elif level == "Borderline":
            st.warning(f"**{level}: {label}.** Mixed signals, so further tests would help decide.")
        else:
            st.success(f"**{level}: {label}.** Routine monitoring and preventive care.")

    with c2:
        st.subheader("What do all 3 models say?")
        comp = pd.DataFrame({"Model": list(probs), "Probability (%)": [v * 100 for v in probs.values()]})
        fig = px.bar(comp, x="Model", y="Probability (%)", text=comp["Probability (%)"].round(1),
                     range_y=[0, 100], color_discrete_sequence=["#2E86AB"])
        fig.add_hline(y=50, line_dash="dash", line_color="grey")
        fig.update_layout(height=300, margin=dict(t=20, b=10))
        st.plotly_chart(fig)
        votes = sum(v >= 0.5 for v in probs.values())
        st.write(f"**{votes} of 3 models** predict heart disease.")

    st.subheader("How this patient compares with the dataset")
    num_labels = {"age": "Age", "trestbps": "Resting BP", "chol": "Cholesterol",
                  "thalach": "Max heart rate", "oldpeak": "ST depression"}
    grp = df.groupby("target")[pl.NUM_COLS].mean()
    cmp_df = pd.DataFrame({
        "This patient": [patient[c] for c in pl.NUM_COLS],
        "Avg. of patients WITHOUT disease": grp.loc[0].values,
        "Avg. of patients WITH disease": grp.loc[1].values,
    }, index=[num_labels[c] for c in pl.NUM_COLS]).round(1)
    st.dataframe(cmp_df)
    st.caption("⚠️ Educational project. This is a decision-support tool trained on 300 patients "
               "and is not a medical diagnosis.")

# =====================================================================
# TAB 2: EDA
# =====================================================================
with tab2:
    st.subheader("Exploratory Data Analysis")
    d = df.copy()
    d["Diagnosis"] = d["target"].map({0: "No Disease", 1: "Disease"})
    d["Chest pain"] = d["cp"].map(CP)

    pct = d["target"].mean() * 100
    a, b = st.columns(2)
    with a:
        st.markdown("**Target distribution**")
        st.plotly_chart(px.histogram(d, x="Diagnosis", color="Diagnosis", color_discrete_map=COLORS)
                        .update_layout(showlegend=False, height=320))
        st.caption(f"{pct:.0f}% of the {len(d)} patients have heart disease.")
    with b:
        st.markdown("**Age distribution**")
        st.plotly_chart(px.histogram(d, x="age", color="Diagnosis", nbins=20, barmode="overlay",
                                     opacity=0.7, color_discrete_map=COLORS).update_layout(height=320))
        st.caption(f"Mean age: {d[d.target == 0].age.mean():.1f} (no disease) vs "
                   f"{d[d.target == 1].age.mean():.1f} (disease).")

    a, b = st.columns(2)
    with a:
        st.markdown("**Cholesterol by diagnosis (box plot)**")
        st.plotly_chart(px.box(d, x="Diagnosis", y="chol", color="Diagnosis",
                               color_discrete_map=COLORS).update_layout(showlegend=False, height=320))
        st.caption("Dots outside the whiskers are the outliers flagged by the IQR method.")
    with b:
        st.markdown("**Age vs maximum heart rate (scatter)**")
        st.plotly_chart(px.scatter(d, x="age", y="thalach", color="Diagnosis",
                                   color_discrete_map=COLORS).update_layout(height=320))
        st.caption(f"Average max heart rate: {d[d.target == 0].thalach.mean():.0f} (no disease) vs "
                   f"{d[d.target == 1].thalach.mean():.0f} (disease).")

    a, b = st.columns(2)
    with a:
        st.markdown("**Chest pain type vs diagnosis**")
        st.plotly_chart(px.histogram(d, x="Chest pain", color="Diagnosis", barmode="group",
                                     color_discrete_map=COLORS).update_layout(height=360))
    with b:
        st.markdown("**Correlation heatmap**")
        corr = df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto").update_layout(height=360))
        top = corr["target"].drop("target").abs().sort_values(ascending=False).head(3).index.tolist()
        st.caption("Features most correlated with the target: " + ", ".join(top) + ".")

# =====================================================================
# TAB 3: MODEL PERFORMANCE
# =====================================================================
with tab3:
    st.subheader("Model comparison (20% held-out test set)")
    res = bundle["results"]
    st.dataframe(res.set_index("Model").round(3))
    best = res.sort_values("Recall", ascending=False).iloc[0]
    st.info(f"Highest recall: **{best['Model']}** ({best['Recall']:.3f}). In healthcare, recall matters "
            "most because a false negative is a patient with disease who gets overlooked.")

    melt = res.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                    var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(melt, x="Model", y="Score", color="Metric", barmode="group",
                           range_y=[0, 1]).update_layout(height=380))

    a, b = st.columns(2)
    with a:
        st.markdown("**Confusion matrix**")
        cm_model = st.selectbox("Choose model", list(models), index=1, key="cm_model")
        cm = bundle["cms"][cm_model]
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  x=["Predicted No", "Predicted Disease"],
                                  y=["Actual No", "Actual Disease"]).update_layout(height=340))
        st.caption(f"False negatives (missed patients): {cm[1][0]}")
    with b:
        st.markdown("**Top 10 important features (Random Forest)**")
        imp = bundle["importance"].head(10).iloc[::-1].reset_index()
        imp.columns = ["Feature", "Importance"]
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h")
                        .update_layout(height=340))

# =====================================================================
# TAB 4: ABOUT
# =====================================================================
with tab4:
    st.subheader("About this project")
    st.markdown(f"""
**Problem:** Predict whether a patient has heart disease from routine medical attributes, to help
doctors decide who needs further examination.

**Dataset:** UCI Heart Disease (Cleveland subset): {len(df)} patients after cleaning, 13 features + 1 target.

**Data mining task:** Classification (supervised). Target: 0 = no disease, 1 = disease present.

**Pipeline:**
1. Missing values (`ca`, `thal`) filled with the mode
2. Duplicates removed
3. Target converted from 0-4 to binary
4. Categorical features (`cp`, `restecg`, `slope`, `thal`) one-hot encoded
5. 80/20 stratified train/test split
6. Numeric features Min-Max scaled (fitted on training data only)
7. Decision Tree, Random Forest and Naïve Bayes trained and evaluated

**Limitations:** Small dataset (~300 rows from one hospital), so results are a proof of concept.
It is a decision-support tool and does **not** replace a doctor's diagnosis.
""")

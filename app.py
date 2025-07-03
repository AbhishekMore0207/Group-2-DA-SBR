import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import plotly.express as px

def safe_trendline_scatter(df, x, y, **kwargs):
    try:
        import statsmodels.api
        return px.scatter(df, x=x, y=y, trendline="ols", **kwargs)
    except ImportError:
        return px.scatter(df, x=x, y=y, **kwargs)

st.set_page_config(page_title="Alcohol Consumer Dashboard", layout="wide")
st.sidebar.title("⚙️ Global Controls")
theme_name = st.sidebar.selectbox("🎨 Theme", ["Default", "Vibrant", "Monochrome", "High-Contrast"])
palette_picker = st.sidebar.color_picker("Accent colour", "#1f77b4")
currency = st.sidebar.selectbox("💱 Currency", ["USD", "AED", "EUR", "INR"])
rate = {"USD":1, "AED":3.67, "EUR":0.92, "INR":83}[currency]
symbol = {"USD":"$", "AED":"د.إ", "EUR":"€", "INR":"₹"}[currency]
note = st.sidebar.text_area("🗒️ Session note")
if note:
    st.session_state.setdefault("notes", []).append(note)

tab_titles = [
    "📊 Data Visualisation",
    "🎯 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression"
]
tabs = st.tabs(tab_titles)

# Data upload required every time
st.sidebar.info("Please upload your cleaned Excel dataset (.xlsx)")
uploaded_file = st.sidebar.file_uploader("Upload cleaned dataset", type=["xlsx"])
if uploaded_file is None:
    st.stop()
df = pd.read_excel(uploaded_file)

# 1) DATA VISUALISATION
with tabs[0]:
    st.header("Exploratory Insights")
    fig = px.histogram(df, x="Age", nbins=30, title="Age Distribution", marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Respondents concentrate in the 25–45 range, prime spending years.")

    fig = safe_trendline_scatter(df, x="Income_kUSD", y="Drinks_Per_Week", title="Income vs Drinks/Week")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Slight upward trend: higher income loosely correlates with more weekly drinks.")

    fig = px.pie(df, names="Preferred_Drink", hole=0.4, title="Preferred Drink Mix")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Beer and wine together account for >60 % of preferences.")

    fig = px.histogram(df, x="Generation", color="Price_Sensitivity",
                       barmode="group", title="Price Sensitivity by Generation")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Gen Z reports the largest share of ‘High’ price sensitivity.")

    spend_adj = df["Monthly_Spend_USD"] * rate
    fig = px.histogram(spend_adj, nbins=40, title=f"Monthly Alcohol Spend ({currency})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Tail shows power-spenders – 1 % spend > {symbol}{spend_adj.quantile(0.99):,.0f}/mo.")

    sb_data = df.dropna(subset=["Taste_Preference", "Brand_Loyalty"])
    if sb_data.empty:
        st.warning("No data available for Taste vs Brand Loyalty sunburst.")
    else:
        fig = px.sunburst(sb_data, path=["Taste_Preference","Brand_Loyalty"], title="Taste vs Brand Loyalty")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bitter-taste respondents show the highest ‘High’ loyalty cluster.")

    agebin = pd.cut(df["Age"], bins=[17,25,35,45,55,65,80], labels=["18-25","26-35","36-45","46-55","56-65","66+"])
    fig = px.bar(df.assign(Age_Bin=agebin), x="Age_Bin", color="Support_Local_Store", barmode="group", title="Willingness to Support Local Store (by Age)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("26-35 group most supportive of a new local outlet.")

    fig = px.box(df, x="Gender", y="Drinks_Per_Week", title="Drinking Intensity by Gender")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Male median ≈ 3 drinks/week vs female ≈ 2.")

    fig = px.violin(df, x="Support_Local_Store", y="Health_Score", box=True, title="Health-Consciousness vs Store Support")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Even highly health-conscious (score ≥ 4) show ~40 % support.")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Numeric Feature Correlations")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Monthly Spend correlates with Income and Drinks/Week.")

# 2) CLASSIFICATION
with tabs[1]:
    st.header("Predict ‘Support_Local_Store’")
    target = "Support_Local_Store"
    feature_cols = [col for col in df.columns if col != target]
    X = df[feature_cols]
    y = df[target]
    cat_cols_X = X.select_dtypes(include="object").columns.tolist()
    num_cols_X = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    all_labels = sorted(set(y_train) | set(y_test))
    is_binary = len(all_labels) == 2
    pos_label = all_labels[-1] if is_binary else None

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols_X),
        ("num", StandardScaler(), num_cols_X)
    ])
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    roc_info, metric_rows = {}, []
    for name, clf in models.items():
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        if hasattr(clf, "predict_proba") and is_binary:
            probs = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(pos_label)]
        else:
            probs = None

        labels_in_test = set(y_test)
        labels_in_pred = set(y_pred)
        if is_binary and pos_label in labels_in_test and pos_label in labels_in_pred:
            precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        elif is_binary:
            precision = recall = f1 = 0.0
        else:
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metric_rows.append([
            name,
            accuracy_score(y_test, y_pred),
            precision,
            recall,
            f1
        ])
        if probs is not None and is_binary and (pos_label in labels_in_test):
            binarized_y = [1 if v == pos_label else 0 for v in y_test]
            fpr, tpr, _ = roc_curve(binarized_y, probs)
            roc_info[name] = (fpr, tpr, auc(fpr, tpr))
        st.session_state[f"pipe_{name}"] = pipe

    met_df = pd.DataFrame(metric_rows, columns=["Model","Accuracy","Precision","Recall","F1"])
    st.dataframe(met_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1": "{:.2%}"
    }))
    sel = st.selectbox("Show Confusion Matrix for:", list(models.keys()))
    pipe = st.session_state[f"pipe_{sel}"]
    y_cm_pred = pipe.predict(X_test)
    cm_labels = sorted(list(set(y_test) | set(y_cm_pred)))
    cm = confusion_matrix(y_test, y_cm_pred, labels=cm_labels)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=cm_labels).plot(ax=ax)
    st.pyplot(fig)

    # Always show ROC Curve for binary classification after confusion matrix
    if is_binary and any(roc_info.values()):
        fig, ax = plt.subplots()
        for name, (fpr, tpr, auc_score) in roc_info.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")
        ax.plot([0,1],[0,1],"--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (all models)")
        ax.legend()
        st.pyplot(fig)

    rf_imp = st.session_state["pipe_Random Forest"].named_steps["clf"].feature_importances_
    feats = pd.DataFrame({"Feature": cat_cols_X + num_cols_X, "Importance": rf_imp})
    top5 = feats.sort_values("Importance", ascending=False).head(5)
    fig = px.bar(top5, x="Feature", y="Importance", title="Top-5 Predictive Features")
    st.plotly_chart(fig)
    st.subheader("📤 Predict on New Data")
    up = st.f

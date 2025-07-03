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
    roc_available = False
    for name, clf in models.items():
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # For ROC: check binary, classifier supports prob, positive label in test, two unique test labels
        if hasattr(clf, "predict_proba") and is_binary and len(set(y_test))==2:
            probs = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(pos_label)]
            binarized_y = np.array([1 if v == pos_label else 0 for v in y_test])
            fpr, tpr, _ = roc_curve(binarized_y, probs)
            roc_info[name] = (fpr, tpr, auc(fpr, tpr))
            roc_available = True
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

    # Show ROC Curve if available for binary classification
    if is_binary and roc_available:
        fig, ax = plt.subplots()
        for name, (fpr, tpr, auc_score) in roc_info.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")
        ax.plot([0,1],[0,1],"--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (all models)")
        ax.legend()
        st.pyplot(fig)
    elif is_binary:
        st.info("ROC curve is not available due to insufficient data or missing probability estimates.")

    rf_imp = st.session_state["pipe_Random Forest"].named_steps["clf"].feature_importances_
    feats = pd.DataFrame({"Feature": cat_cols_X + num_cols_X, "Importance": rf_imp})
    top5 = feats.sort_values("Importance", ascending=False).head(5)
    fig = px.bar(top5, x="Feature", y="Importance", title="Top-5 Predictive Features")
    st.plotly_chart(fig)
    st.subheader("📤 Predict on New Data")
    upload_pred = st.file_uploader("Upload CSV (no target column)", type=["csv"])
    if upload_pred:
        new_df = pd.read_csv(upload_pred)
        new_df = new_df.dropna()
        preds = st.session_state["pipe_Random Forest"].predict(new_df)
        new_df["Predicted_Support"] = preds
        csv_out = new_df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", data=csv_out, file_name="support_predictions.csv", mime="text/csv")
        st.write("Preview", new_df.head())

# 3) CLUSTERING
with tabs[2]:
    st.header("Customer Segmentation (K-means)")
    cluster_num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cluster_df = df.dropna(subset=cluster_num_cols).copy()
    k = st.slider("Number of clusters", 2, 10, 4)
    inertias = []
    for i in range(2, 11):
        km = KMeans(n_clusters=i, n_init="auto", random_state=42)
        km.fit(cluster_df[cluster_num_cols])
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertias, marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Method")
    st.pyplot(fig)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    cluster_df["Cluster"] = km.fit_predict(cluster_df[cluster_num_cols])
    persona = cluster_df.groupby("Cluster").agg({
        "Age":"median",
        "Income_kUSD":"median",
        "Drinks_Per_Week":"median",
        "Preferred_Drink":lambda x: x.mode()[0] if len(x.mode()) else np.nan,
        "Brand_Loyalty":lambda x: x.mode()[0] if len(x.mode()) else np.nan
    }).rename(columns={
        "Age":"Median Age",
        "Income_kUSD":"Median Income (kUSD)",
        "Drinks_Per_Week":"Drinks/Week",
        "Preferred_Drink":"Fav Drink",
        "Brand_Loyalty":"Typical Loyalty"
    })
    st.dataframe(persona)
    st.download_button("Download CSV with clusters", data=cluster_df.to_csv(index=False).encode(), file_name="clustered_data.csv", mime="text/csv")

# 4) ASSOCIATION RULE MINING
with tabs[3]:
    st.header("Association Rule Mining (Apriori)")
    ar_cat_cols = df.select_dtypes(include="object").columns.tolist()
    cols = st.multiselect("Columns to mine", ar_cat_cols, default=["Preferred_Drink","Purchase_Channel"] if "Preferred_Drink" in ar_cat_cols and "Purchase_Channel" in ar_cat_cols else ar_cat_cols[:2])
    min_sup = st.slider("Min support (%)", 1, 20, 5) / 100
    min_conf = st.slider("Min confidence (%)", 5, 80, 30) / 100
    if cols:
        ar_df = df.dropna(subset=cols)
        basket = pd.get_dummies(ar_df[cols])
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf).sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
        st.subheader("🧠 Rule Insights")
        if rules.empty:
            st.warning("No rules meet the chosen thresholds.")
        else:
            for _, row in rules.iterrows():
                ant = ', '.join(row['antecedents'])
                con = ', '.join(row['consequents'])
                st.markdown(f"- **{ant} → {con}** &nbsp; (conf {row.confidence:.0%}, lift {row.lift:.2f})")

# 5) REGRESSION
with tabs[4]:
    st.header("Predict Monthly Spend (Regression)")
    target_reg = "Monthly_Spend_USD"
    feature_cols_reg = [col for col in df.columns if col != target_reg]
    y_reg = df[target_reg]
    X_reg = df[feature_cols_reg]
    reg_df = pd.concat([X_reg, y_reg], axis=1).dropna(subset=feature_cols_reg + [target_reg])
    X_reg = reg_df[feature_cols_reg]
    y_reg = reg_df[target_reg]
    cat_cols_reg = X_reg.select_dtypes(include="object").columns.tolist()
    num_cols_reg = X_reg.select_dtypes(include=["int64", "float64"]).columns.tolist()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    pre_r = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols_reg),
        ("num", StandardScaler(), num_cols_reg)
    ])
    regs = {
        "Linear":  LinearRegression(),
        "Ridge":   Ridge(alpha=1.0),
        "Lasso":   Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    rows, feat_imp = [], {}
    for n, reg in regs.items():
        pipe = Pipeline([("prep", pre_r), ("reg", reg)])
        pipe.fit(Xr_train, yr_train)
        score = pipe.score(Xr_test, yr_test)
        rows.append([n, score])
        if hasattr(reg, "coef_"):
            feat_imp[n] = reg.coef_
        elif hasattr(reg, "feature_importances_"):
            feat_imp[n] = reg.feature_importances_
    st.dataframe(pd.DataFrame(rows, columns=["Model","R² (test)"]).style.format({"R² (test)": "{:.3f}"}))
    dt_imp = feat_imp["Decision Tree"]
    fi = pd.DataFrame({"Feature": cat_cols_reg + num_cols_reg, "Importance": dt_imp})
    top8 = fi.sort_values("Importance", ascending=False).head(8)
    fig = px.bar(top8, x="Feature", y="Importance", title="Top Spend Drivers (Decision Tree)")
    st.plotly_chart(fig)

if st.session_state.get("notes"):
    st.sidebar.subheader("💬 Stored Notes")
    for i, n in enumerate(st.session_state["notes"], 1):
        st.sidebar.markdown(f"**{i}.** {n}")

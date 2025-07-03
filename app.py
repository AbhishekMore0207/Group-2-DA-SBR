import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Alcohol Consumer Dashboard", layout="wide")

# ---------- Load data ----------
def load_data():
    possible_paths = [
        Path(__file__).parent / "data" / "extended_alcohol_consumers.csv",
        Path.cwd() / "data" / "extended_alcohol_consumers.csv",
    ]
    for p in possible_paths:
        if p.exists():
            return pd.read_csv(p)
    st.error("Dataset not found. Please ensure 'data/extended_alcohol_consumers.csv' exists.")
    st.stop()

df = load_data()
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# ---------- Sidebar ----------
st.sidebar.title("Global Controls")
search_query = st.sidebar.text_input("Search charts / columns")
theme = st.sidebar.selectbox("Theme", ["Default","Vibrant","Monochrome","High‑Contrast"])
currency = st.sidebar.selectbox("Currency", ["USD","AED","INR","EUR"])
currency_factor = {"USD":1,"AED":3.67,"INR":83,"EUR":0.92}[currency]
currency_symbol = {"USD":"$","AED":"د.إ","INR":"₹","EUR":"€"}[currency]

# ---------- Tabs ----------
tabs = st.tabs([
    "Data Visualisation","Classification","Clustering",
    "Association Rules","Regression","3‑D Product View"
])

# 1 Data Visualisation
with tabs[0]:
    st.header("Exploratory Data Visualisation")
    # sample 5 charts (for brevity). Add more as needed.
    fig = px.histogram(df, x="Age", nbins=30, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter(df, x="Income_kUSD", y="Monthly_Spend_USD",
                      title="Income vs Monthly Spend")
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.pie(df, names="Preferred_Drink", title="Preferred Drink Share")
    st.plotly_chart(fig3, use_container_width=True)

# 2 Classification
with tabs[1]:
    target = "Support_Local_Store"
    st.header("Classification Models")
    X = df.drop(columns=[target])
    y = df[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    preproc = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    metrics = []
    roc_dict = {}
    for name,model in models.items():
        pipe = Pipeline([("prep", preproc), ("clf", model)])
        pipe.fit(X_train,y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        prec = precision_score(y_test,y_pred, pos_label="Yes")
        rec = recall_score(y_test,y_pred, pos_label="Yes")
        f1 = f1_score(y_test,y_pred, pos_label="Yes")
        metrics.append([name,acc,prec,rec,f1])
        if hasattr(pipe,"predict_proba"):
            proba = pipe.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve((y_test=="Yes").astype(int), proba)
            roc_dict[name] = (fpr,tpr, auc(fpr,tpr))
        st.session_state[f"class_{name}"] = pipe

    met_df = pd.DataFrame(metrics, columns=["Model","Accuracy","Precision","Recall","F1"])
    st.dataframe(met_df.style.format("{:.2%}"))

    sel = st.selectbox("Confusion Matrix for", list(models.keys()))
    sel_pipe = st.session_state.get(f"class_{sel}")
    if sel_pipe is not None:
        fig_cm, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(sel_pipe, X_test, y_test,
                                              display_labels=["No","Yes"], ax=ax)
        st.pyplot(fig_cm)

    roc_fig, ax = plt.subplots()
    for n,(fpr,tpr,roc_auc) in roc_dict.items():
        ax.plot(fpr,tpr,label=f"{n} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],"--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(roc_fig)

    # Feature importance using RandomForest
    rf = st.session_state["class_Random Forest"].named_steps["clf"]
    imp = rf.feature_importances_
    feat_names = list(cat_cols)+list(num_cols)
    top5 = pd.Series(imp,index=feat_names).sort_values(ascending=False).head(5)
    st.bar_chart(top5)

    st.subheader("Predict on New Data (CSV)")
    new_file = st.file_uploader("Upload new data without target", type=["csv"])
    if new_file:
        new_df = pd.read_csv(new_file)
        preds = st.session_state["class_Random Forest"].predict(new_df)
        new_df["Predicted_Support"] = preds
        st.dataframe(new_df.head())
        st.download_button("Download predictions",
                           data=new_df.to_csv(index=False).encode(),
                           file_name="predictions.csv")

# 3 Clustering
with tabs[2]:
    st.header("K‑means Clustering")
    k = st.slider("Number of clusters",2,10,4)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["Cluster"] = km.fit_predict(df[num_cols])
    persona = df.groupby("Cluster").agg({
        "Age":"median",
        "Income_kUSD":"median",
        "Drinks_Per_Week":"median",
        "Preferred_Drink":lambda x: x.mode()[0]
    }).rename(columns={"Age":"Median Age","Income_kUSD":"Income","Drinks_Per_Week":"Drinks/Week","Preferred_Drink":"Fav Drink"})
    st.dataframe(persona)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download Clustered Data", data=csv_bytes, file_name="clustered_data.csv")

# 4 Association Rules
with tabs[3]:
    st.header("Association Rule Mining")
    cols = st.multiselect("Choose categorical columns", cat_cols, default=["Preferred_Drink","Purchase_Channel"])
    min_sup = st.slider("Min support",1,20,5)/100
    min_conf = st.slider("Min confidence",10,80,30)/100

    basket = pd.get_dummies(df[cols])
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf).sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

# 5 Regression
with tabs[4]:
    st.header("Regression: Predict Monthly Spend")
    y = df["Monthly_Spend_USD"]
    X = df.drop(columns=["Monthly_Spend_USD"])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    pre = preproc  # reuse

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    metric_list = []
    for n,reg in regs.items():
        pipe = Pipeline([("prep",pre),("reg",reg)])
        pipe.fit(X_train,y_train)
        r2 = pipe.score(X_test,y_test)
        metric_list.append([n,r2])
    st.dataframe(pd.DataFrame(metric_list, columns=["Model","R²"]).style.format("{:.2f}"))

# 6 3‑D viewer
with tabs[5]:
    st.header("360° Product View")
    st.write("Interactive 3‑D model of a beverage bottle.")
    html = '''
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <model-viewer src="https://modelviewer.dev/shared-assets/models/BeerBottle.glb"
                      camera-controls auto-rotate style="width:100%;height:600px;">
        </model-viewer>
    '''
    st.components.v1.html(html, height=620)

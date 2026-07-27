"""
Fitbit : Calorie Burn Prediction & Workout Pattern Clustering
Streamlit app version of the accompanying analysis notebook.

Run with:
    streamlit run streamlit_app.py

Expects `Fitbit_dataset.csv` to be in the same folder as this script
(or upload it from the sidebar).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Fitbit Analysis", layout="wide", page_icon="🏋️")

TARGET = "Calories_Burned (kcal)"


# --------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Data type ensuring
    df["Workout_Type"] = df["Workout_Type"].astype(str).str.lower()
    df["Gender"] = df["Gender"].astype(str).str.lower()

    # Missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Outlier handling (IQR clipping)
    for col in ["Max_BPM", "Avg_BPM", TARGET]:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Categorical encoding
    encoders = {}
    for col in ["Gender", "Workout_Type", "Experience_Level"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc

    return df, encoders


@st.cache_resource
def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "KNN": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(random_state=42)

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }
        fitted[name] = model

    return results, fitted


@st.cache_resource
def run_clustering(df: pd.DataFrame):
    df_cluster = df.drop(columns=["Workout_Type", TARGET])
    scaler2 = StandardScaler()
    X_scaled = scaler2.fit_transform(df_cluster)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    K_range = range(2, 10)
    inertias, sil_scores = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))

    return X_pca, pca, list(K_range), inertias, sil_scores


# --------------------------------------------------------------------------
# Sidebar - data source & navigation
# --------------------------------------------------------------------------
st.sidebar.title("🏋️ Fitbit Analysis")

uploaded = st.sidebar.file_uploader("Upload Fitbit CSV (optional)", type=["csv"])
data_source = uploaded if uploaded is not None else "Fitbit_dataset.csv"

try:
    raw_df = load_data(data_source)
except FileNotFoundError:
    st.error(
        "Couldn't find `Fitbit_dataset.csv` next to this script. "
        "Upload the file using the sidebar to continue."
    )
    st.stop()

df, encoders = preprocess(raw_df)

page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Comparison", "Predict Calories", "Workout Clustering"],
)

# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------
if page == "Overview":
    st.title("Fitbit : Calorie Burn Prediction & Workout Pattern Clustering")
    st.markdown(
        "This app explores the Fitbit workout dataset, predicts **calories burned** "
        "using several regression models, and clusters users into distinct workout patterns."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{raw_df.shape[0]:,}")
    col2.metric("Columns", f"{raw_df.shape[1]:,}")
    col3.metric("Missing values", f"{int(raw_df.isnull().sum().sum()):,}")

    st.subheader("Raw data sample")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)

# --------------------------------------------------------------------------
# Page: EDA
# --------------------------------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Distribution of Calories Burned")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x=TARGET, bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of Calories Burned")
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Workout Type vs Calories Burned")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="Workout_Type", y=TARGET, data=df, ax=ax)
    labels = encoders["Workout_Type"].inverse_transform(sorted(df["Workout_Type"].unique()))
    ax.set_xticklabels(labels)
    ax.set_title("Workout Type vs Calories Burned")
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Page: Model Comparison
# --------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Training & Comparison")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with st.spinner("Training models..."):
        results, fitted = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    st.subheader("Model performance")
    st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)

    best_name = results_df.index[0]
    st.success(f"Best model by R²: **{best_name}**")

    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test_scaled)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.5, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual Calories")
        ax.set_ylabel("Predicted Calories")
        ax.set_title(f"{best_name}: Actual vs Predicted")
        st.pyplot(fig)

    with col2:
        if hasattr(best_model, "feature_importances_"):
            st.subheader("Feature Importance")
            feat_imp = pd.Series(
                best_model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 5))
            feat_imp.plot(kind="bar", ax=ax)
            ax.set_title(f"{best_name} Feature Importance")
            st.pyplot(fig)
        else:
            st.info(f"{best_name} does not expose feature importances directly.")

# --------------------------------------------------------------------------
# Page: Predict Calories
# --------------------------------------------------------------------------
elif page == "Predict Calories":
    st.title("Predict Calories Burned")
    st.markdown("Fill in workout details and get a predicted calorie burn.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with st.spinner("Training models..."):
        results, fitted = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    model_choice = st.selectbox("Model to use", results_df.index.tolist())
    model = fitted[model_choice]

    workout_types = sorted(raw_df["Workout_Type"].astype(str).str.lower().unique())
    genders = sorted(raw_df["Gender"].astype(str).str.lower().unique())

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 10, 100, int(raw_df["Age"].median()))
            gender = st.selectbox("Gender", genders)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, float(raw_df["Weight (kg)"].median()))
            height = st.number_input("Height (m)", 1.2, 2.2, float(raw_df["Height (m)"].median()))
            workout_type = st.selectbox("Workout Type", workout_types)
            experience_level = st.selectbox(
                "Experience Level", sorted(raw_df["Experience_Level"].unique())
            )
        with c2:
            max_bpm = st.number_input("Max BPM", 100.0, 220.0, float(raw_df["Max_BPM"].median()))
            avg_bpm = st.number_input("Avg BPM", 60.0, 200.0, float(raw_df["Avg_BPM"].median()))
            resting_bpm = st.number_input(
                "Resting BPM", 40.0, 120.0, float(raw_df["Resting_BPM"].median())
            )
            session_duration = st.number_input(
                "Session Duration (hours)", 0.1, 5.0, float(raw_df["Session_Duration (hours)"].median())
            )
            workout_frequency = st.number_input(
                "Workout Frequency (days/week)", 1, 7, int(raw_df["Workout_Frequency (days/week)"].median())
            )
        with c3:
            fat_pct = st.number_input(
                "Fat Percentage", 3.0, 60.0, float(raw_df["Fat_Percentage"].median())
            )
            water_intake = st.number_input(
                "Water Intake (liters)", 0.5, 6.0, float(raw_df["Water_Intake (liters)"].median())
            )
            bmi = st.number_input("BMI", 10.0, 60.0, float(weight / (height ** 2)))
            base_met = st.number_input("Base MET", 1.0, 20.0, float(raw_df["Base_MET"].median()))
            hr_intensity = st.number_input(
                "HR Intensity", 0.0, 2.0, float(raw_df["HR_Intensity"].median())
            )
            effective_met = st.number_input(
                "Effective MET", 0.0, 20.0, float(raw_df["Effective_MET"].median())
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        gender_enc = encoders["Gender"].transform([gender])[0]
        workout_enc = encoders["Workout_Type"].transform([workout_type])[0]

        input_row = pd.DataFrame(
            [{
                "Age": age,
                "Gender": gender_enc,
                "Weight (kg)": weight,
                "Height (m)": height,
                "Max_BPM": max_bpm,
                "Avg_BPM": avg_bpm,
                "Resting_BPM": resting_bpm,
                "Session_Duration (hours)": session_duration,
                "Workout_Type": workout_enc,
                "Fat_Percentage": fat_pct,
                "Water_Intake (liters)": water_intake,
                "Workout_Frequency (days/week)": workout_frequency,
                "Experience_Level": experience_level,
                "BMI": bmi,
                "Base_MET": base_met,
                "HR_Intensity": hr_intensity,
                "Effective_MET": effective_met,
            }]
        )[X.columns]

        input_scaled = scaler.transform(input_row)
        prediction = model.predict(input_scaled)[0]
        st.metric("Predicted Calories Burned", f"{prediction:,.1f} kcal")

# --------------------------------------------------------------------------
# Page: Workout Clustering
# --------------------------------------------------------------------------
elif page == "Workout Clustering":
    st.title("Unsupervised Workout Pattern Clustering")

    X_pca, pca, K_range, inertias, sil_scores = run_clustering(df)
    st.write(f"PCA explained variance (2 components): **{pca.explained_variance_ratio_.sum():.2%}**")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA - 2D View")
        st.pyplot(fig)

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(list(K_range), inertias, "bo-")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")

        axes[1].plot(list(K_range), sil_scores, "ro-")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Scores")
        plt.tight_layout()
        st.pyplot(fig)

    best_k = st.slider("Number of clusters (K)", 2, 9, 4)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    df_clustered = df.copy()
    df_clustered["Cluster"] = cluster_labels

    score = silhouette_score(X_pca, cluster_labels)
    st.metric("Silhouette Score", f"{score:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(best_k):
        mask = cluster_labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {i}", alpha=0.6)
    ax.legend()
    ax.set_title("KMeans Clusters (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.subheader("Cluster profiles")
    cluster_summary = df_clustered.groupby("Cluster")[
        ["Avg_BPM", "Session_Duration (hours)", "Workout_Frequency (days/week)", "Experience_Level"]
    ].mean()
    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("Experience Level vs Cluster")
    crosstab = pd.crosstab(df_clustered["Cluster"], df_clustered["Experience_Level"])
    fig, ax = plt.subplots(figsize=(8, 4))
    crosstab.plot(kind="bar", ax=ax)
    ax.set_title("Cluster vs Experience Level")
    st.pyplot(fig)

# Fitbit: Calorie Burn Prediction & Workout Pattern Clustering

An end-to-end Machine Learning project that leverages Fitbit wearable sensor data and user demographics to predict calorie expenditure during workout sessions and uncover behavioral patterns through clustering.

## 📌 Problem Statement
Accurate calorie estimation during workouts is vital for modern fitness tracking applications. While wearable sensors capture core physiological signals like heart rate and session duration, additional contextual factors (e.g., workout type, hydration level, user experience) heavily influence energy expenditure. 

This project tackles two primary tasks:
1. **Supervised Learning (Regression):** Predicts the exact calories burned per workout session.
2. **Unsupervised Learning (Clustering):** Identifies hidden user segments and workout intensity patterns without pre-existing labels.

---

## 🚀 Business Use Cases
* **Wearable Fitness Apps:** Real-time, localized calorie burn predictions during live exercises.
* **Personalized Fitness Coaching:** Tailored workout intensity and duration tracking.
* **Health Monitoring & Nutrition Platforms:** Accurate energy expenditure logging for meal and nutrition planning.
* **User Behavioral Segmentation:** Discovering unique fitness habits across various experience levels without requiring manual labels.

---

## 📊 Dataset Overview
The dataset encompasses user demographics, workout logs, and raw physiological sensor inputs:
* **Demographics:** Age, Gender, Weight, Height, BMI, Experience Level (Beginner/Intermediate/Advanced)
* **Sensor Metrics:** Max BPM, Avg BPM, Resting BPM, Session Duration (hours)
* **Contextual Features:** Workout Type (Cardio/Strength/HIIT/Yoga), Water Intake (liters), Workout Frequency (days/week)
* **Target Variable:** Calories Burned

---

## 🛠️ Project Workflow & Architecture

### Phase 1: Data Preprocessing & EDA
* Imputation of any missing values.
* Outlier detection and distribution capping.
* Categorical encoding (One-Hot / Ordinal Encoding) for features like `Workout_Type` and `Experience_Level`.
* Numerical feature scaling using `StandardScaler` / `MinMaxScaler`.

### Phase 2: Supervised Learning (Regression)
Trained, optimized, and evaluated multiple regression models to predict `Calories_Burned`:
* Linear Regression / Ridge / Lasso
* K-Nearest Neighbors (KNN) Regressor
* Decision Tree & Random Forest Regressors
* XGBoost Regressor
* **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score (Target: $R^2 \ge 0.80$).

### Phase 3: Unsupervised Learning (Clustering)
Discovered core physical patterns focusing on workout intensity and physiological feedback:
* Dropped target labels and compressed features using **Principal Component Analysis (PCA)**.
* Segmented data using **KMeans Clustering** (with optional Hierarchical/DBSCAN comparison).
* **Evaluation Metric:** Silhouette Score (Acceptance threshold $\ge 0.15$ given the high natural overlap in human physiological data).

---

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
* **Deployment/UI (Optional):** Streamlit

---

## 🏃 How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/fitbit-ml-analysis.git](https://github.com/your-username/fitbit-ml-analysis.git)
   cd fitbit-ml-analysis

# ScholarScope: A beginner-friendly Streamlit app for student score prediction and clustering
# ------------------------------------------------------------
# STEP 0: Imports and configuration
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# ------------------------------------------------------------
# STEP 1: Page configuration and custom CSS
st.set_page_config(page_title="ScholarScope", layout="wide")
# Custom styling for a light academic look
st.markdown(
    """
    <style>
        .main {
            background-color: #F8FAFC;
            padding: 2rem;
        }
        .card {
            background-color: white;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        h1, h2, h3 {
            color: #1E293B;
        }
        p, .stMarkdown p {
            color: #64748B;
        }
        .stButton > button {
            background-color: #059669 !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Small note under the title
st.title("ScholarScope")
st.caption("Demo mode: uses synthetic data if data/students.csv is missing.")
st.subheader("Student Performance Prediction & Academic Group Clustering")

# ------------------------------------------------------------
# STEP 2: Helper functions

def load_data() -> pd.DataFrame:
    """Load the dataset, generating a synthetic one if the CSV does not exist."""
    csv_path = "data/students.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Generate synthetic data as described
        rng = np.random.default_rng(42)
        n = 500
        student_ids = [f"S{i:03d}" for i in range(1, n + 1)]
        hours = rng.uniform(0, 8, n).round(1)
        attendance = rng.uniform(40, 100, n).round(1)
        prev = rng.uniform(30, 100, n).round(1)
        sleep = rng.uniform(4, 9, n).round(1)
        assign = rng.integers(0, 21, n)
        noise = rng.normal(0, 4, n)
        final = (
            5
            + 0.30 * prev
            + 1.8 * hours
            + 0.20 * attendance
            + 0.8 * assign
            + 0.8 * sleep
            + noise
        )
        final = np.clip(final, 0, 100).round(1)
        df = pd.DataFrame(
            {
                "Student_ID": student_ids,
                "Hours_Studied": hours,
                "Attendance": attendance,
                "Previous_Score": prev,
                "Sleep_Hours": sleep,
                "Assignments_Done": assign,
                "Final_Score": final,
            }
        )
        # Introduce ~3% NaNs in Attendance and Sleep_Hours
        for col in ["Attendance", "Sleep_Hours"]:
            nan_idx = rng.choice(n, size=int(0.03 * n), replace=False)
            df.loc[nan_idx, col] = np.nan
        # Ensure the data folder exists then save
        import os
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df

@st.cache_data
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with the column median.
    Median imputation is simple, works for both training and inference,
    and does not create outliers as mean imputation might.
    """
    for col in ["Attendance", "Sleep_Hours"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def grade_from_score(score: float) -> str:
    """Return a letter grade based on the numeric score."""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"

def standing_from_grade(grade: str) -> str:
    mapping = {
        "A": "Excellent",
        "B": "Good",
        "C": "Average",
        "D": "Needs Improvement",
        "F": "At Risk",
    }
    return mapping[grade]

# ------------------------------------------------------------
# STEP 3: Load and clean data (cached)
raw_df = load_data()
df = fill_missing_values(raw_df.copy())

FEATURE_COLS = ["Hours_Studied", "Attendance", "Previous_Score", "Sleep_Hours", "Assignments_Done"]
TARGET_COL = "Final_Score"

# ------------------------------------------------------------
# STEP 4: Model training (cached resource)
@st.cache_resource
def train_models():
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    # Predictions on test set
    rf_pred = rf.predict(X_test)
    dt_pred = dt.predict(X_test)
    # Metrics
    results = {
        "rf": {
            "model": rf,
            "r2": r2_score(y_test, rf_pred),
            "mae": mean_absolute_error(y_test, rf_pred),
            "importances": rf.feature_importances_,
        },
        "dt": {
            "model": dt,
            "r2": r2_score(y_test, dt_pred),
            "mae": mean_absolute_error(y_test, dt_pred),
            "importances": None,  # Decision tree importances not visualised here
        },
    }
    return results

model_results = train_models()

# ------------------------------------------------------------
# STEP 5: Build clustering objects (cached resource)
@st.cache_resource
def build_clusters():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    # Compute average final score per cluster to label them meaningfully
    df_cluster = df.copy()
    df_cluster["ClusterLabel"] = kmeans.labels_
    cluster_means = df_cluster.groupby("ClusterLabel")[TARGET_COL].mean()
    # Sort cluster ids by mean score
    sorted_labels = cluster_means.sort_values().index.tolist()
    name_map = {
        sorted_labels[0]: "Needs Support",
        sorted_labels[1]: "Steady",
        sorted_labels[2]: "High Achiever",
    }
    return scaler, kmeans, name_map

scaler, kmeans_model, cluster_name_map = build_clusters()

def predict_cluster(row: pd.Series) -> str:
    """Assign a cluster name to a single student row (features only)."""
    scaled = scaler.transform(row[FEATURE_COLS].values.reshape(1, -1))
    label = kmeans_model.predict(scaled)[0]
    return cluster_name_map[label]

# ------------------------------------------------------------
# STEP 6: UI – Navigation tabs
tab1, tab2, tab3 = st.tabs(["Score Predictor", "Model & Data Lab", "Student Clusters"])

# ------------------------------------------------------------
# TAB 1: Score Predictor
with tab1:
    st.subheader("Predict a student's final score")
    col1, col2 = st.columns(2)
    with col1:
        hours_input = st.slider("Hours Studied per day", 0.0, 8.0, 4.0, 0.5)
        attendance_input = st.slider("Attendance (%)", 40, 100, 80)
        prev_input = st.slider("Previous Score", 30, 100, 70)
    with col2:
        sleep_input = st.slider("Sleep Hours per night", 4.0, 9.0, 6.5, 0.5)
        assign_input = st.slider("Assignments Done", 0, 20, 10)
    model_choice = st.radio("Choose model", ["Random Forest", "Decision Tree"])
    predict_btn = st.button("Predict")
    if predict_btn:
        # Build a one‑row DataFrame in the same order as training features
        input_df = pd.DataFrame(
            {
                "Hours_Studied": [hours_input],
                "Attendance": [attendance_input],
                "Previous_Score": [prev_input],
                "Sleep_Hours": [sleep_input],
                "Assignments_Done": [assign_input],
            }
        )
        model_key = "rf" if model_choice == "Random Forest" else "dt"
        model = model_results[model_key]["model"]
        pred_score = float(model.predict(input_df)[0])
        pred_score = max(0, min(100, round(pred_score, 1)))
        grade = grade_from_score(pred_score)
        standing = standing_from_grade(grade)
        cluster_name = predict_cluster(input_df.iloc[0])
        # Summary card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"**Predicted Final Score:** {pred_score}/100")
        st.write(f"**Letter Grade:** {grade}")
        st.write(f"**Standing:** {standing}")
        st.write(f"**Academic Cluster:** {cluster_name}")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2: Model & Data Lab
with tab2:
    st.subheader("Dataset overview")
    # Metric cards
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Students", len(df))
    col_b.metric("Mean Final Score", f"{df[TARGET_COL].mean():.2f}")
    col_c.metric("Average Attendance (%)", f"{df["Attendance"].mean():.2f}")
    st.write("---")
    st.write("**First 10 rows of the data**")
    st.dataframe(df.head(10))
    # Scatter plot
    fig_scatter = px.scatter(
        df,
        x="Hours_Studied",
        y=TARGET_COL,
        opacity=0.7,
        color_discrete_sequence=["#059669"],
        title="Hours Studied vs Final Score",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    # Histogram of final scores
    fig_hist = px.histogram(
        df,
        x=TARGET_COL,
        nbins=20,
        color_discrete_sequence=["#1E293B"],
        title="Distribution of Final Scores",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    # Model comparison table
    st.subheader("Model performance on test set")
    comp_df = pd.DataFrame(
        {
            "Model": ["Random Forest", "Decision Tree"],
            "R2 Score": [model_results["rf"]["r2"], model_results["dt"]["r2"]],
            "MAE": [model_results["rf"]["mae"], model_results["dt"]["mae"]],
        }
    )
    comp_df["R2 Score"] = comp_df["R2 Score"].round(3)
    comp_df["MAE"] = comp_df["MAE"].round(3)
    st.table(comp_df)
    st.caption("Higher R2 is better. Lower MAE is better.")
    # Bar chart for R2 and MAE
    fig_bar = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Value"),
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Metrics",
        color_discrete_map={"R2 Score": "#059669", "MAE": "#1E293B"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    # Feature importance (Random Forest)
    importances = model_results["rf"]["importances"]
    imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=True)
    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="Importance",
        color_continuous_scale=["#E5E7EB", "#059669"],
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------
# TAB 3: Student Clusters
with tab3:
    st.subheader("Academic clusters based on study behaviour")
    # Cluster distribution bar chart
    cluster_counts = pd.Series(kmeans_model.labels_).value_counts().sort_index()
    dist_df = pd.DataFrame({"Cluster": cluster_counts.index, "Students": cluster_counts.values})
    fig_dist = px.bar(
        dist_df,
        x="Cluster",
        y="Students",
        title="Students per Cluster",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    # Group comparison table
    df_cluster = df.copy()
    df_cluster["ClusterLabel"] = kmeans_model.labels_
    agg = df_cluster.groupby("ClusterLabel").agg(
        Students=("Student_ID", "count"),
        Avg_Hours_Studied=("Hours_Studied", "mean"),
        Avg_Attendance=("Attendance", "mean"),
        Avg_Previous_Score=("Previous_Score", "mean"),
        Avg_Sleep_Hours=("Sleep_Hours", "mean"),
        Avg_Assignments_Done=("Assignments_Done", "mean"),
        Avg_Final_Score=(TARGET_COL, "mean"),
    )
    # Map numeric cluster to readable name
    agg = agg.rename(index=cluster_name_map)
    agg = agg.round(2)
    st.write("**Cluster summary**")
    st.dataframe(agg)
    # Avg final score per cluster chart
    fig_cluster_score = px.bar(
        agg.reset_index(),
        x=agg.reset_index().columns[0],
        y="Avg_Final_Score",
        title="Average Final Score by Cluster",
        color=agg.reset_index().columns[0],
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_cluster_score, use_container_width=True)
    # Cluster checker form
    st.subheader("Check which cluster a new student belongs to")
    with st.form(key="cluster_form"):
        h = st.slider("Hours Studied per day", 0.0, 8.0, 4.0, 0.5)
        a = st.slider("Attendance (%)", 40, 100, 80)
        p = st.slider("Previous Score", 30, 100, 70)
        s = st.slider("Sleep Hours per night", 4.0, 9.0, 6.5, 0.5)
        assign = st.slider("Assignments Done", 0, 20, 10)
        submitted = st.form_submit_button("Check Cluster")
        if submitted:
            row = pd.Series(
                {
                    "Hours_Studied": h,
                    "Attendance": a,
                    "Previous_Score": p,
                    "Sleep_Hours": s,
                    "Assignments_Done": assign,
                }
            )
            cl_name = predict_cluster(row)
            st.success(f"This student falls into the **{cl_name}** cluster.")
            # Short description per cluster
            descriptions = {
                "Needs Support": "Students need extra help and monitoring.",
                "Steady": "Students are performing at an average level.",
                "High Achiever": "Students consistently achieve high scores.",
            }
            st.info(descriptions.get(cl_name, ""))

# End of app
""

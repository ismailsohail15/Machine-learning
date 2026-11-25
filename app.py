# ScholarScope: Student Performance Prediction & Academic Group Clustering
# ------------------------------------------------------------
# STEP 0: Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------------------------------------------------
# STEP 1: Page config and CSS
st.set_page_config(page_title="ScholarScope", layout="wide")
st.markdown(
    """
    <style>
        .main { background-color: #F8FAFC; padding: 2rem; }
        .card {
            background-color: white;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        h1, h2, h3 { color: #1E293B; }
        p, .stMarkdown p { color: #64748B; }
        .stButton > button { background-color: #059669 !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ScholarScope")
st.subheader("Student Performance Prediction & Academic Group Clustering")

# ------------------------------------------------------------
# STEP 2: Helper functions

def load_data() -> pd.DataFrame:
    """Load the real CSV dataset; fall back to synthetic data if not found."""
    csv_path = "data/students.csv"
    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns exist; rename common variants if needed
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower().replace(" ", "_")
            col_map[c] = cl
        df.rename(columns=col_map, inplace=True)
        # Standardise expected column names (case-insensitive match)
        name_fix = {
            "hours_studied": "Hours_Studied",
            "attendance": "Attendance",
            "previous_score": "Previous_Score",
            "sleep_hours": "Sleep_Hours",
            "assignments_done": "Assignments_Done",
            "final_score": "Final_Score",
            "student_id": "Student_ID",
        }
        df.rename(columns=name_fix, inplace=True)
        # If Student_ID column missing, create one
        if "Student_ID" not in df.columns:
            df.insert(0, "Student_ID", [f"S{i:03d}" for i in range(1, len(df) + 1)])
    except FileNotFoundError:
        # Generate 500-row synthetic dataset and save it for future runs
        import os
        rng = np.random.default_rng(42)
        n = 500
        hours = rng.uniform(0, 8, n).round(1)
        attendance = rng.uniform(40, 100, n).round(1)
        prev = rng.uniform(30, 100, n).round(1)
        sleep = rng.uniform(4, 9, n).round(1)
        assign = rng.integers(0, 21, n)
        noise = rng.normal(0, 4, n)
        final = np.clip(5 + 0.30*prev + 1.8*hours + 0.20*attendance + 0.8*assign + 0.8*sleep + noise, 0, 100).round(1)
        df = pd.DataFrame({
            "Student_ID": [f"S{i:03d}" for i in range(1, n + 1)],
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Previous_Score": prev,
            "Sleep_Hours": sleep,
            "Assignments_Done": assign,
            "Final_Score": final,
        })
        for col in ["Attendance", "Sleep_Hours"]:
            nan_idx = rng.choice(n, size=int(0.03 * n), replace=False)
            df.loc[nan_idx, col] = np.nan
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


@st.cache_data
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with the column median."""
    for col in ["Attendance", "Sleep_Hours"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def grade_from_score(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def standing_from_grade(grade: str) -> str:
    return {"A": "Excellent", "B": "Good", "C": "Average", "D": "Needs Improvement", "F": "At Risk"}[grade]


# ------------------------------------------------------------
# STEP 3: Load & clean data
raw_df = load_data()
df = fill_missing_values(raw_df.copy())

FEATURE_COLS = ["Hours_Studied", "Attendance", "Previous_Score", "Sleep_Hours", "Assignments_Done"]
TARGET_COL = "Final_Score"

# ------------------------------------------------------------
# STEP 4: Train models
@st.cache_resource
def train_models():
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    return {
        "rf": {"model": rf, "r2": r2_score(y_test, rf.predict(X_test)),
               "mae": mean_absolute_error(y_test, rf.predict(X_test)),
               "importances": rf.feature_importances_},
        "dt": {"model": dt, "r2": r2_score(y_test, dt.predict(X_test)),
               "mae": mean_absolute_error(y_test, dt.predict(X_test)),
               "importances": None},
    }

model_results = train_models()

# ------------------------------------------------------------
# STEP 5: Build clustering
@st.cache_resource
def build_clusters():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df_c = df.copy()
    df_c["ClusterLabel"] = kmeans.labels_
    sorted_labels = df_c.groupby("ClusterLabel")[TARGET_COL].mean().sort_values().index.tolist()
    name_map = {sorted_labels[0]: "Needs Support", sorted_labels[1]: "Steady", sorted_labels[2]: "High Achiever"}
    return scaler, kmeans, name_map

scaler, kmeans_model, cluster_name_map = build_clusters()


def predict_cluster(row: pd.Series) -> str:
    """Predict the cluster name for a single student feature row."""
    scaled = scaler.transform(row[FEATURE_COLS].values.reshape(1, -1))
    return cluster_name_map[kmeans_model.predict(scaled)[0]]


# ------------------------------------------------------------
# STEP 6: Tabs
tab1, tab2, tab3 = st.tabs(["Score Predictor", "Model & Data Lab", "Student Clusters"])

# ── TAB 1: Score Predictor ───────────────────────────────────
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
    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Hours_Studied": [hours_input],
            "Attendance": [attendance_input],
            "Previous_Score": [prev_input],
            "Sleep_Hours": [sleep_input],
            "Assignments_Done": [assign_input],
        })
        model_key = "rf" if model_choice == "Random Forest" else "dt"
        pred_score = float(model_results[model_key]["model"].predict(input_df)[0])
        pred_score = max(0, min(100, round(pred_score, 1)))
        grade = grade_from_score(pred_score)
        standing = standing_from_grade(grade)
        cluster_name = predict_cluster(input_df.iloc[0])
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"**Predicted Final Score:** {pred_score}/100")
        st.write(f"**Letter Grade:** {grade}")
        st.write(f"**Standing:** {standing}")
        st.write(f"**Academic Cluster:** {cluster_name}")
        st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 2: Model & Data Lab ──────────────────────────────────
with tab2:
    st.subheader("Dataset overview")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Students", len(df))
    col_b.metric("Mean Final Score", f"{df[TARGET_COL].mean():.2f}")
    col_c.metric("Average Attendance (%)", f"{df['Attendance'].mean():.2f}")
    st.write("---")
    st.write("**First 10 rows of the data**")
    st.dataframe(df.head(10))

    # Scatter: Hours Studied vs Final Score
    st.plotly_chart(
        px.scatter(df, x="Hours_Studied", y=TARGET_COL, opacity=0.7,
                   color_discrete_sequence=["#059669"],
                   title="Hours Studied vs Final Score"),
        use_container_width=True,
    )

    # Histogram of Final Scores
    st.plotly_chart(
        px.histogram(df, x=TARGET_COL, nbins=20,
                     color_discrete_sequence=["#1E293B"],
                     title="Distribution of Final Scores"),
        use_container_width=True,
    )

    # Correlation Heatmap (Seaborn)
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", cbar=True, ax=ax)
    st.subheader("Correlation Heatmap")
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # Model comparison
    st.subheader("Model performance on test set")
    comp_df = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree"],
        "R2 Score": [round(model_results["rf"]["r2"], 3), round(model_results["dt"]["r2"], 3)],
        "MAE": [round(model_results["rf"]["mae"], 3), round(model_results["dt"]["mae"], 3)],
    })
    st.table(comp_df)
    st.caption("Higher R² is better. Lower MAE is better.")
    st.plotly_chart(
        px.bar(comp_df.melt(id_vars="Model", var_name="Metric", value_name="Value"),
               x="Model", y="Value", color="Metric", barmode="group",
               title="Model Metrics",
               color_discrete_map={"R2 Score": "#059669", "MAE": "#1E293B"}),
        use_container_width=True,
    )

    # Feature importance
    imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": model_results["rf"]["importances"]})
    imp_df = imp_df.sort_values("Importance", ascending=True)
    st.plotly_chart(
        px.bar(imp_df, x="Importance", y="Feature", orientation="h",
               title="Random Forest Feature Importance",
               color="Importance", color_continuous_scale=["#E5E7EB", "#059669"]),
        use_container_width=True,
    )

# ── TAB 3: Student Clusters ──────────────────────────────────
with tab3:
    st.subheader("Academic clusters based on study behaviour")

    # Cluster distribution bar chart
    cluster_counts = pd.Series(kmeans_model.labels_).value_counts().sort_index()
    dist_df = pd.DataFrame({"Cluster": cluster_counts.index, "Students": cluster_counts.values})
    st.plotly_chart(
        px.bar(dist_df, x="Cluster", y="Students", title="Students per Cluster",
               color="Cluster", color_discrete_sequence=px.colors.qualitative.Pastel),
        use_container_width=True,
    )

    # Cluster summary table
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
    ).rename(index=cluster_name_map).round(2)
    st.write("**Cluster summary**")
    st.dataframe(agg)

    # Avg final score per cluster
    agg_reset = agg.reset_index()
    st.plotly_chart(
        px.bar(agg_reset, x=agg_reset.columns[0], y="Avg_Final_Score",
               title="Average Final Score by Cluster",
               color=agg_reset.columns[0],
               color_discrete_sequence=px.colors.qualitative.Pastel),
        use_container_width=True,
    )

    # ── K-Means Evaluation Charts ────────────────────────────
    st.subheader("K‑Means evaluation")

    # Dynamically limit k so it never exceeds n_samples - 1 (silhouette requirement)
    n_samples = df.shape[0]
    max_k = min(6, n_samples - 1)

    if max_k < 2:
        st.warning("Not enough data points to run K‑Means evaluation (need at least 3 rows).")
    else:
        k_range = range(2, max_k + 1)
        X_scaled_eval = scaler.transform(df[FEATURE_COLS])

        eval_col1, eval_col2, eval_col3 = st.columns(3)

        # 1️⃣ Elbow Method – inertia (WCSS) across k values
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled_eval)
            inertias.append(km.inertia_)
        eval_col1.plotly_chart(
            px.line(x=list(k_range), y=inertias, markers=True,
                    labels={"x": "k", "y": "Inertia (WCSS)"},
                    title="Elbow Method"),
            use_container_width=True,
        )

        # 2️⃣ Silhouette Scores – measures cluster separation quality
        sil_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled_eval)
            sil_scores.append(silhouette_score(X_scaled_eval, km.labels_))
        eval_col2.plotly_chart(
            px.line(x=list(k_range), y=sil_scores, markers=True,
                    labels={"x": "k", "y": "Silhouette Score"},
                    title="Silhouette Scores"),
            use_container_width=True,
        )

        # 3️⃣ PCA 2D Scatter – students projected onto 2 principal components
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled_eval)
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca["Cluster"] = kmeans_model.labels_.astype(str)
        eval_col3.plotly_chart(
            px.scatter(df_pca, x="PC1", y="PC2", color="Cluster",
                       title="PCA of Students (2‑D)",
                       color_discrete_sequence=px.colors.qualitative.Pastel),
            use_container_width=True,
        )

    # ── Cluster Checker Form ─────────────────────────────────
    st.subheader("Check which cluster a new student belongs to")
    with st.form(key="cluster_form"):
        h = st.slider("Hours Studied per day", 0.0, 8.0, 4.0, 0.5)
        a = st.slider("Attendance (%)", 40, 100, 80)
        p = st.slider("Previous Score", 30, 100, 70)
        s = st.slider("Sleep Hours per night", 4.0, 9.0, 6.5, 0.5)
        asgn = st.slider("Assignments Done", 0, 20, 10)
        submitted = st.form_submit_button("Check Cluster")

    # Show result OUTSIDE the form but only after submission
    if submitted:
        row = pd.Series({
            "Hours_Studied": h,
            "Attendance": a,
            "Previous_Score": p,
            "Sleep_Hours": s,
            "Assignments_Done": asgn,
        })
        cl_name = predict_cluster(row)
        descriptions = {
            "Needs Support": "This student needs extra help and close monitoring.",
            "Steady": "This student is performing at an average level.",
            "High Achiever": "This student consistently achieves high scores.",
        }
        st.success(f"This student falls into the **{cl_name}** cluster.")
        st.info(descriptions.get(cl_name, ""))

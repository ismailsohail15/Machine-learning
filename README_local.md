# ScholarScope

## Project Structure
```
.
├── app.py                # Streamlit application (single file)
├── requirements.txt      # Python dependencies
├── data/
│   └── students.csv      # Synthetic dataset (generated if missing)
├── README.md             # Instructions and viva talking points
└── .gitignore            # Files/folders ignored by Git
```

## How to Run the App
1. Make sure you have Python 3.8+ installed.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The app will open in your default web browser.

## Viva Talking Points
### Supervised Learning (Score Prediction)
- **Models**: Random Forest and Decision Tree regressors.
- **Why these models**: Random Forest usually gives higher accuracy (captures non‑linear relations) while Decision Tree is simple to explain and visualise.
- **Train/Test split**: 80 % of the data for training, 20 % reserved for testing (random_state=42 ensures reproducibility).
- **Metrics**:
  - **R² Score** – proportion of variance explained (higher is better).
  - **MAE (Mean Absolute Error)** – average absolute difference between predicted and true scores (lower is better).
- **Feature importance** (shown for Random Forest) helps explain which study behaviours most affect the final score.

### Unsupervised Learning (Student Clustering)
- **No target used**: Clustering groups students based on behaviour (the five input features) without looking at the final score.
- **Scaling**: `StandardScaler` normalises each feature so that differences in units (e.g., hours vs percentages) do not dominate the distance calculations.
- **K‑Means**: Partitions the data into three clusters (`n_clusters=3`). The algorithm is simple, fast, and works well for this low‑dimensional data.
- **Cluster naming**: After fitting, we compute the average `Final_Score` per cluster and map the lowest‑scoring group to **"Needs Support"**, the middle to **"Steady"**, and the highest to **"High Achiever"** – ensuring the names stay consistent regardless of the internal cluster IDs.

### How They Work Together
- The **Score Predictor** tab uses the trained regression model to estimate a single student’s final score based on five inputs.
- The same inputs are fed to the clustering pipeline, assigning the student to an **academic cluster** that describes their overall study pattern.
- Displaying both the predicted score (with grade and standing) **and** the cluster provides a holistic view: the model tells *how well* the student might perform, while the cluster tells *what kind of learner* they are.

---

*Demo mode*: If `data/students.csv` is missing the app automatically generates a synthetic dataset of 500 students.

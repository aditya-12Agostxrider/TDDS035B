# Smart Attendance & Student Engagement Prediction System

A streamlined, production-ready Machine Learning system that combines classroom attendance metrics, learning management system (LMS) digital activity, assignment submission records, and quiz scores to predict student engagement levels, identify learning archetypes, and provide non-punitive early intervention support.

---

## 1. Quick Start

```powershell
cd C:\Users\ADITYA\OneDrive\Desktop\smart_attendance_engagement

# 1. Run automated tests (59 tests across data, models, and predictions)
python -m pytest tests/ -v

# 2. Launch the Streamlit dashboard
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 2. Streamlined Dashboard Architecture ("Less is More")

The user interface is organized into **4 focused, high-impact pages**:

1. **📊 Overview & Attendance**:
   - Executive KPI cards (Enrolled Students, Cohort Attendance %, Avg Engagement, Below-75% Attendance alert).
   - Side-by-side distribution charts (Attendance % vs. Engagement Tier distribution).
   - Searchable cohort roster with attendance status badges (`Eligible >= 75%`, `Warning 60-74%`, `Critical < 60%`).

2. **🔮 Live Prediction & Support**:
   - Quick-load profile presets (*High Engagement, Moderate, Needs Support, Custom*).
   - Live prediction: **Continuous Engagement Score ($0-100$)** + **Engagement Tier Badge** (`Low`, `Medium`, `High`) with class probabilities.
   - Actionable, constructive early warning recommendations (e.g. mentor check-ins, LMS accessibility reviews) without judgmental labels.
   - Batch CSV upload tab for scoring entire classes at once.

3. **🧠 ML Analytics & Patterns**:
   - **Tab 1: Student Patterns (K-Means & PCA)**: Interactive 2D PCA cluster visualization revealing 3 natural student learning archetypes (*Pattern A: Traditional, Pattern B: Digital / Self-directed, Pattern C: Needs Support*).
   - **Tab 2: Model Comparison & Explainability**: Side-by-side performance benchmarks for Regression & Classification, 5-Fold Cross-Validation reliability, and a clean Feature Importance bar chart.

4. **⚖️ Ethics & Principles**:
   - Four core commitments: strictly non-punitive output, demographic neutrality, human-in-the-loop decisions, and transparent explainability.

---

## 3. Machine Learning Architecture

```
Student Records (Attendance, LMS, Quizzes, Assignments)
                          │
                          ▼
            Preprocessing & Scaling Pipeline
            (Median Imputer + StandardScaler)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     Regression     Classification   Clustering
   (Score: 0-100)  (Low/Med/High)   (Patterns A,B,C)
   [Ridge / Lasso]  [Random Forest]  [K-Means + PCA]
          │               │               │
          └───────────────┼───────────────┘
                          ▼
        Explainability & Early Intervention
```

* **Regression**: Linear, Ridge ($L_2$), and Lasso ($L_1$) ($R^2 pprox 0.62$, $	ext{RMSE} pprox 5.96$).
* **Classification**: Logistic Regression, Decision Tree, and Random Forest ($	ext{ROC-AUC} pprox 0.805$, $	ext{Accuracy} pprox 63.3\%$).
* **Clustering**: K-Means ($k=3$) with positive silhouette score ($0.120$) and 2-component PCA ($40.6\%$ explained variance).
* **Validation**: 5-Fold Stratified Cross-Validation + `GridSearchCV` hyperparameter tuning.

---

## 4. Test Verification

```
tests/test_data.py ....................                                  [ 33%]
tests/test_models.py ......................                              [ 71%]
tests/test_prediction.py .................                                [100%]

======================= 59 passed in 17.30s =======================
```

---

## 5. Ethical Governance

* **Non-Punitive Output**: Algorithmic outputs are strictly advisory decision-support tools for mentors. Derogatory labels (*"lazy"*, *"bad student"*) are strictly prohibited.
* **Demographic Privacy**: Attributes such as race, gender, age, socioeconomic background, and student identifiers are strictly excluded from the ML feature matrix.
* **Human-in-the-Loop**: All algorithmic guidance requires human review before any academic intervention takes place.
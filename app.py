"""
app.py
Smart Attendance & Student Engagement Prediction System
Mini Project — Streamlit Dashboard
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preprocessing import (
    validate_data, remove_duplicates, encode_target,
    get_feature_matrix, NUMERIC_FEATURES,
)
from src.eda import (
    plot_attendance_distribution, plot_performance, plot_lms_activity,
    plot_engagement, plot_correlation_heatmap, plot_attendance_vs_engagement,
    get_summary_stats,
)
from src.prediction import (
    predict_from_dict, validate_student_input,
)
from src.explainability import generate_early_warning
from src.classification import CLASS_NAMES

MODEL_DIR = os.path.join(ROOT, 'models')
DATA_DIR  = os.path.join(ROOT, 'data')

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
)

# ── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_student_data():
    path = os.path.join(DATA_DIR, 'student_data.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    from generate_data import generate_dataset
    df = generate_dataset()
    df.to_csv(path, index=False)
    return df


@st.cache_resource
def get_metadata():
    path = os.path.join(MODEL_DIR, 'metadata.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return {}


@st.cache_resource
def get_pipeline(name: str):
    path = os.path.join(MODEL_DIR, f'pipeline_{name}.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


df_all = load_student_data()
metadata = get_metadata()
clf_pipe = get_pipeline('classification')
reg_pipe = get_pipeline('regression')

# ── Sidebar Navigation ───────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "📋 Attendance",
    "📊 Student Data",
    "🔮 Prediction",
    "📈 Regression",
    "🎯 Classification",
    "🔵 Clustering",
    "📉 PCA",
]

with st.sidebar:
    st.title("🎓 Smart Attendance")
    st.caption("Student Engagement Prediction System")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    if clf_pipe is not None and reg_pipe is not None:
        st.success("✅ Models Loaded")
    else:
        st.error("❌ Models Not Loaded — Run train.py")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 🏠 Home
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🎓 Smart Attendance & Engagement System")
    st.caption("AI-powered student engagement monitoring and prediction")

    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df_all))
    c2.metric("Avg Attendance", f"{df_all['attendance_percentage'].mean():.1f}%")
    c3.metric("Avg Engagement Score", f"{df_all['engagement_score'].mean():.1f}")
    c4.metric("Below 75% Attendance", int((df_all['attendance_percentage'] < 75).sum()))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Project Features")
        st.markdown("""
        - **Attendance Tracking** — Check 75% eligibility for hall tickets
        - **Engagement Prediction** — Predict score (0–100) and tier (Low/Medium/High)
        - **Student Clustering** — Group students using K-Means (k=3)
        - **PCA Visualization** — 2D projection of student behavioral patterns
        - **Early Warning Alerts** — Supportive mentorship advice for at-risk students
        """)

    with col2:
        st.subheader("🛠️ Technologies Used")
        st.markdown("""
        - **Language**: Python 3.11
        - **ML Library**: Scikit-Learn (Regression, Classification, Clustering, PCA)
        - **Web Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 📋 Attendance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Attendance":
    st.title("📋 Attendance Calculator")
    st.caption("Check student attendance percentage and 75% eligibility")

    mode = st.radio("Input Mode", ["Select from Dataset", "Manual Entry"], horizontal=True)

    if mode == "Select from Dataset":
        all_ids = df_all['student_id'].tolist()
        stu_id = st.selectbox("Select Student ID", all_ids)
        row = df_all[df_all['student_id'] == stu_id].iloc[0]
        total_c = int(row['total_classes'])
        att_c   = int(row['attended_classes'])
    else:
        stu_id  = st.text_input("Enter Student ID", value="", placeholder="e.g. STU0100")
        total_c = st.number_input("Total Classes", min_value=1, max_value=200, value=50)
        att_c   = st.number_input("Classes Attended", min_value=0, max_value=200, value=42)

    if st.button("Calculate Attendance", use_container_width=True):
        if not stu_id:
            st.error("❌ Please enter a Student ID.")
        elif att_c > total_c:
            st.error("❌ Attended classes cannot be more than total classes.")
        else:
            pct = (att_c / total_c) * 100.0
            absent = total_c - att_c

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Student ID", stu_id)
            m2.metric("Attendance %", f"{pct:.1f}%")
            m3.metric("Classes Attended", f"{att_c} / {total_c}")
            m4.metric("Classes Absent", absent)

            if pct >= 75:
                st.success(f"✅ **{stu_id}** is eligible for hall ticket (Attendance ≥ 75%).")
            elif pct >= 60:
                st.warning(f"⚠️ **{stu_id}** is at risk. Attendance is between 60-74%. Needs improvement.")
            else:
                st.error(f"🔴 **{stu_id}** has critically low attendance ({pct:.1f}%). Not eligible.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 📊 Student Data
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Student Data":
    st.title("📊 Student Data Explorer")
    st.caption("View dataset and exploratory data analysis")

    tab1, tab2 = st.tabs(["📋 Dataset", "📊 EDA Charts"])

    with tab1:
        st.subheader("Student Dataset")
        st.dataframe(df_all.head(30), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(get_summary_stats(df_all), use_container_width=True)

    with tab2:
        st.subheader("Exploratory Visualizations")
        c1, c2 = st.columns(2)

        with c1:
            fig1 = plot_attendance_distribution(df_all)
            st.pyplot(fig1); plt.close(fig1)

            fig2 = plot_lms_activity(df_all)
            st.pyplot(fig2); plt.close(fig2)

            fig3 = plot_attendance_vs_engagement(df_all)
            st.pyplot(fig3); plt.close(fig3)

        with c2:
            fig4 = plot_performance(df_all)
            st.pyplot(fig4); plt.close(fig4)

            fig5 = plot_engagement(df_all)
            st.pyplot(fig5); plt.close(fig5)

            fig6 = plot_correlation_heatmap(df_all)
            st.pyplot(fig6); plt.close(fig6)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 🔮 Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Student Engagement Predictor")
    st.caption("Enter student details to predict engagement score and level")

    pred_mode = st.radio("Input Mode", ["Select from Dataset", "Manual Entry"], horizontal=True, key="pred_mode")

    if pred_mode == "Select from Dataset":
        all_ids = df_all['student_id'].tolist()
        selected_id = st.selectbox("Select Student ID", all_ids, key="pred_stu")
        row = df_all[df_all['student_id'] == selected_id].iloc[0]
        att_v  = float(row.get('attendance_percentage', 75))
        late_v = int(row.get('late_arrivals', 2))
        part_v = float(row.get('class_participation', 55) if pd.notna(row.get('class_participation')) else 55)
        prev_v = float(row.get('previous_score', 60) if pd.notna(row.get('previous_score')) else 60)
        asgn_v = float(row.get('assignment_completion', 70))
        quiz_v = float(row.get('quiz_average', 65))
        lms_v  = int(row.get('LMS_activity', 12) if pd.notna(row.get('LMS_activity')) else 12)
        min_v  = int(row.get('learning_activity_minutes', 160) if pd.notna(row.get('learning_activity_minutes')) else 160)

        st.markdown("---")
        st.markdown(f"**Loaded data for: `{selected_id}`**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Attendance", f"{att_v:.1f}%")
        m2.metric("Assignments", f"{asgn_v:.1f}%")
        m3.metric("Quiz Avg", f"{quiz_v:.1f}")
        m4.metric("LMS Logins", lms_v)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📚 Academic Metrics")
            att_v  = st.slider("Attendance %", 0.0, 100.0, 75.0, 1.0)
            late_v = st.number_input("Late Arrivals", 0, 50, 2)
            part_v = st.slider("Class Participation %", 0.0, 100.0, 55.0, 1.0)
            prev_v = st.slider("Previous Semester Score", 0.0, 100.0, 60.0, 1.0)
        with col2:
            st.subheader("💻 Digital Metrics")
            asgn_v = st.slider("Assignment Completion %", 0.0, 100.0, 70.0, 1.0)
            quiz_v = st.slider("Quiz Average", 0.0, 100.0, 65.0, 1.0)
            lms_v  = st.number_input("LMS Logins (Monthly)", 0, 100, 12)
            min_v  = st.number_input("Learning Minutes (Monthly)", 0, 2000, 160)

    student_dict = {
        'attendance_percentage': att_v,
        'assignment_completion': asgn_v,
        'assignment_average': min(100.0, quiz_v + 5.0),
        'quiz_average': quiz_v,
        'class_participation': part_v,
        'LMS_activity': lms_v,
        'learning_activity_minutes': min_v,
        'previous_score': prev_v,
        'late_arrivals': late_v,
    }

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    if clf_pipe is not None and reg_pipe is not None:
        r_out = predict_from_dict(student_dict, reg_pipe, task='regression')
        c_out = predict_from_dict(student_dict, clf_pipe, task='classification')

        score = r_out['engagement_score']
        level = c_out['engagement_level']
        probs = c_out.get('probabilities', {'Low': 0.33, 'Medium': 0.33, 'High': 0.34})

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Score", f"{score:.1f} / 100")

        if level == "High":
            r2.success(f"🟢 **{level} Engagement**")
        elif level == "Medium":
            r2.warning(f"🟡 **{level} Engagement**")
        else:
            r2.error(f"🔴 **{level} Engagement**")

        st.markdown("**Confidence:**")
        for tier_name in ['High', 'Medium', 'Low']:
            st.progress(probs[tier_name], text=f"{tier_name}: {probs[tier_name]:.0%}")

        # Early Warnings
        warns = generate_early_warning(student_dict)
        if warns:
            st.markdown("---")
            st.markdown("**📌 Mentorship Advice:**")
            for w in warns:
                st.info(w)
        else:
            st.success("✅ Student is performing well across all indicators.")
    else:
        st.error("Models not loaded. Run `python train.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 📈 Regression
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression":
    st.title("📈 Regression Results")
    st.caption("Linear, Ridge (L2), and Lasso (L1) regression model performance")

    reg_res = metadata.get('reg_results', {})
    if reg_res:
        st.subheader("Model Performance Table")
        rdf = pd.DataFrame(reg_res).T[['R2', 'RMSE', 'MAE']]
        rdf.columns = ['R² Score', 'RMSE', 'MAE']
        st.dataframe(rdf.style.format("{:.4f}"), use_container_width=True)

        # Simple bar chart
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        models = list(reg_res.keys())
        r2s = [reg_res[m]['R2'] for m in models]
        bars = ax.bar(models, r2s, color=['#2563EB', '#10B981', '#F59E0B'], edgecolor='white')
        for bar in bars:
            yv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yv + 0.01, f"{yv:.3f}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel("R² Score")
        ax.set_ylim(0, 1.0)
        ax.set_title("Regression Model Comparison")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No regression results found. Run `python train.py` first.")

    st.markdown("---")
    st.subheader("What are these models?")
    st.markdown("""
    - **Linear Regression**: Basic model that fits a straight line to predict engagement score.
    - **Ridge Regression (L2)**: Adds a penalty to prevent overfitting when features are correlated.
    - **Lasso Regression (L1)**: Adds a penalty that can also remove unimportant features automatically.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 🎯 Classification
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Classification":
    st.title("🎯 Classification Results")
    st.caption("Logistic Regression, Decision Tree, and Random Forest classifier performance")

    clf_res = metadata.get('clf_results', {})
    if clf_res:
        st.subheader("Model Performance Table")
        cdf = pd.DataFrame(clf_res).T[['accuracy', 'roc_auc']]
        cdf.columns = ['Accuracy', 'ROC-AUC']
        st.dataframe(cdf.style.format("{:.4f}"), use_container_width=True)

        # Simple bar chart
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        c_models = list(clf_res.keys())
        accs = [clf_res[m]['accuracy'] for m in c_models]
        bars = ax.bar(c_models, accs, color=['#2563EB', '#10B981', '#F59E0B'], edgecolor='white')
        for bar in bars:
            yv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yv + 0.01, f"{yv:.3f}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.set_title("Classification Model Comparison")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No classification results found. Run `python train.py` first.")

    st.markdown("---")
    st.subheader("What are these models?")
    st.markdown("""
    - **Logistic Regression**: A simple classifier that predicts probabilities for each engagement tier.
    - **Decision Tree**: Creates human-readable if-else rules to classify students.
    - **Random Forest**: Combines 100 decision trees together for better accuracy.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 🔵 Clustering
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.title("🔵 Student Clustering (K-Means)")
    st.caption("Grouping students into 3 learning patterns using K-Means algorithm")

    sil = metadata.get('kmeans_silhouette', 0.0)
    st.info(f"**K-Means Configuration**: k = 3 clusters | Silhouette Score = {sil:.4f}")

    # Cluster Summary Table
    sum_path = os.path.join(MODEL_DIR, 'cluster_summary.csv')
    if os.path.exists(sum_path):
        st.subheader("Cluster Averages")
        cs = pd.read_csv(sum_path, index_col=0)
        cs.index = ['Pattern A', 'Pattern B', 'Pattern C']
        st.dataframe(cs.style.format("{:.1f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("What do these clusters mean?")
    st.markdown("""
    - **Pattern A (Traditional)**: High physical attendance, steady quiz scores, moderate portal usage.
    - **Pattern B (Digital)**: Moderate attendance, high LMS activity, high online assignment completion.
    - **Pattern C (Needs Support)**: Low attendance, low LMS logins, missing assignments.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 📉 PCA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 PCA":
    st.title("📉 PCA — 2D Visualization")
    st.caption("Reducing student data from 9 features to 2 dimensions using Principal Component Analysis")

    var_exp = metadata.get('pca_explained_variance', [0.25, 0.16])
    st.info(f"**PC1**: {var_exp[0]*100:.1f}% variance | **PC2**: {var_exp[1]*100:.1f}% variance | **Total**: {(var_exp[0]+var_exp[1])*100:.1f}%")

    pca_path = os.path.join(MODEL_DIR, 'pca_data.npy')
    lbl_path = os.path.join(MODEL_DIR, 'cluster_labels.npy')

    if os.path.exists(pca_path) and os.path.exists(lbl_path):
        X_pca = np.load(pca_path)
        lbls  = np.load(lbl_path)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2563EB', '#10B981', '#F59E0B']
        names  = ['Pattern A', 'Pattern B', 'Pattern C']
        for k in range(3):
            mask = lbls == k
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[k], label=names[k], alpha=0.7, edgecolors='white', s=50)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
        ax.set_title("2D PCA Projection of Student Clusters")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("PCA data not found. Run `python train.py` first.")

    st.markdown("---")
    st.subheader("What is PCA?")
    st.markdown("""
    **Principal Component Analysis (PCA)** is a technique to reduce the number of features 
    while keeping the most important information. Here we compress 9 student features into 
    just 2 dimensions so we can visualize the clusters on a 2D scatter plot.
    """)
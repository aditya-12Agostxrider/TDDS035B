"""
generate_data.py  (run-once script — not imported by the app)
Creates a realistic SYNTHETIC student dataset and saves it to data/student_data.csv.

NOTE: This is synthetic data created for ML experimentation.
      It does NOT represent real students.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N = 600   # large enough for meaningful ML experiments

rng = np.random.default_rng(SEED)


def generate_dataset(n: int = N) -> pd.DataFrame:
    # ── Attendance features ────────────────────────────────────────────────
    total_classes = rng.integers(40, 60, size=n)  # typical semester: 40-60 classes

    # Attendance % drawn from a slightly left-skewed distribution
    attendance_pct = np.clip(rng.normal(loc=72, scale=15, size=n), 0, 100)

    attended_classes = np.round(attendance_pct / 100 * total_classes).astype(int)
    attended_classes = np.clip(attended_classes, 0, total_classes)
    absent_classes = total_classes - attended_classes

    # Late arrivals (roughly correlated with lower attendance)
    late_arrivals = np.clip(
        rng.integers(0, 12, size=n) - (attendance_pct / 100 * 3).astype(int),
        0, 20
    ).astype(int)

    # ── Academic performance ───────────────────────────────────────────────
    assignment_completion = np.clip(
        rng.normal(loc=75, scale=18, size=n), 0, 100
    )
    assignment_average = np.clip(
        0.4 * attendance_pct + 0.4 * assignment_completion + rng.normal(0, 8, n),
        0, 100
    )
    quiz_average = np.clip(
        0.35 * attendance_pct + 0.45 * assignment_average + rng.normal(0, 10, n),
        0, 100
    )
    previous_score = np.clip(
        rng.normal(loc=65, scale=16, size=n), 0, 100
    )

    # ── Learning activity features ─────────────────────────────────────────
    class_participation = np.clip(
        rng.normal(loc=60, scale=20, size=n), 0, 100
    )
    LMS_activity = np.clip(
        rng.integers(0, 30, size=n) + (attendance_pct / 100 * 5).astype(int),
        0, 40
    ).astype(int)
    learning_activity_minutes = np.clip(
        rng.integers(20, 300, size=n) + (LMS_activity * 4).astype(int),
        20, 500
    ).astype(int)

    # ── Engagement score (continuous target) ──────────────────────────────
    # A composite score that naturally reflects multiple input features.
    # Formula chosen so that it is realistic but not perfectly linear.
    # Added wider noise and a wider spread to produce balanced Low/Med/High.
    engagement_score = np.clip(
        0.25 * attendance_pct
        + 0.20 * assignment_completion
        + 0.15 * quiz_average
        + 0.10 * class_participation
        + 0.10 * (LMS_activity / 40 * 100)
        + 0.10 * previous_score
        + 0.10 * (learning_activity_minutes / 500 * 100)
        + rng.normal(0, 6, n),   # wider noise for more spread
        0, 100
    )

    # ── Engagement level (classification target) ───────────────────────────
    # Use quantile-based bins to guarantee roughly balanced classes
    low_thresh  = float(np.percentile(engagement_score, 33))
    high_thresh = float(np.percentile(engagement_score, 67))

    def assign_level(score):
        if score <= low_thresh:
            return 'Low'
        elif score <= high_thresh:
            return 'Medium'
        else:
            return 'High'

    engagement_level = pd.Categorical(
        [assign_level(s) for s in engagement_score],
        categories=['Low', 'Medium', 'High']
    )

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        'student_id':                [f"STU{str(i).zfill(4)}" for i in range(1, n+1)],
        'total_classes':             total_classes,
        'attended_classes':          attended_classes,
        'absent_classes':            absent_classes,
        'attendance_percentage':     attendance_pct.round(2),
        'late_arrivals':             late_arrivals,
        'assignment_completion':     assignment_completion.round(2),
        'assignment_average':        assignment_average.round(2),
        'quiz_average':              quiz_average.round(2),
        'class_participation':       class_participation.round(2),
        'LMS_activity':              LMS_activity,
        'learning_activity_minutes': learning_activity_minutes,
        'previous_score':            previous_score.round(2),
        'engagement_score':          engagement_score.round(2),
        'engagement_level':          engagement_level,
    })

    # ── Introduce realistic missing values (~2-4%) ─────────────────────────
    for col, frac in [
        ('assignment_completion', 0.025),
        ('quiz_average',          0.02),
        ('LMS_activity',          0.015),
        ('learning_activity_minutes', 0.02),
        ('class_participation',   0.03),
    ]:
        missing_idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'student_data.csv')

    df = generate_dataset()
    df.to_csv(out_path, index=False)

    print(f"[generate_data] Saved {len(df)} synthetic student records to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nEngagement level distribution:\n{df['engagement_level'].value_counts()}")
    print(f"\nSample:\n{df.head(3)}")

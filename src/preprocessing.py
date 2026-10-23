"""
Helper functions for the Diabetes 130-US Hospitals readmission project.

All the cleaning logic lives here so every notebook uses exactly the same
steps, and the notebooks themselves stay short and readable.
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

# Discharge codes meaning the patient died or was sent to hospice.
# These patients can't be readmitted, so they would only add noise.
EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]

# The 23 diabetes medication columns in the dataset
MEDICATIONS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
]


def load_raw(path):
    """Read the raw CSV. '?' is how this dataset marks missing values.
    Diagnosis codes are read as text because some start with V or E."""
    diag_types = {'diag_1': str, 'diag_2': str, 'diag_3': str}
    return pd.read_csv(path, na_values='?', dtype=diag_types, low_memory=False)


def group_icd9(code):
    """Map a raw ICD-9 code to one of the 9 groups used by Strack et al. (2014)."""
    if pd.isna(code):
        return 'Missing'
    code = str(code)
    if code.startswith(('V', 'E')):
        return 'Other'
    c = float(code)
    if 390 <= c <= 459 or c == 785:
        return 'Circulatory'
    if 460 <= c <= 519 or c == 786:
        return 'Respiratory'
    if 520 <= c <= 579 or c == 787:
        return 'Digestive'
    if int(c) == 250:
        return 'Diabetes'
    if 800 <= c <= 999:
        return 'Injury'
    if 710 <= c <= 739:
        return 'Musculoskeletal'
    if 580 <= c <= 629 or c == 788:
        return 'Genitourinary'
    if 140 <= c <= 239:
        return 'Neoplasms'
    return 'Other'


def age_to_midpoint(age_band):
    """'[70-80)' -> 75.0, so the model sees age as an ordered number."""
    low, high = re.findall(r'\d+', age_band)
    return (int(low) + int(high)) / 2


# The ID columns are codes, not quantities. These groupings follow IDS_mapping.csv.
def map_admission_type(x):
    if x in (1, 2, 7):           # emergency, urgent, trauma centre
        return 'Emergency/Urgent'
    if x == 3:
        return 'Elective'
    return 'Other/Unknown'


def map_admission_source(x):
    if x == 7:
        return 'Emergency Room'
    if x in (1, 2, 3):           # physician, clinic, HMO referral
        return 'Referral'
    if x in (4, 5, 6, 10, 18, 19, 22, 25, 26):
        return 'Transfer'
    return 'Other/Unknown'


def map_discharge(x):
    if x == 1:
        return 'Home'
    if x in (6, 8):              # home with health service / home IV
        return 'Home with care'
    if x in (2, 3, 4, 5, 10, 15, 22, 23, 24, 27, 28, 29, 30):
        return 'Transferred'
    return 'Other/Unknown'


def clean_data(df, top_n_specialties=10):
    """Run every cleaning step and return (X, y, groups).
    groups = patient_nbr, used only for splitting, never as a feature."""
    df = df.copy()

    # 1. Patients who died or went to hospice can't come back
    df = df[~df['discharge_disposition_id'].isin(EXPIRED_OR_HOSPICE)]

    # 2. A handful of rows have an invalid gender
    df = df[df['gender'] != 'Unknown/Invalid']

    # 3. Drop the ID column, columns with mostly missing data, and two
    #    medications that have the same value for every patient
    df = df.drop(columns=['encounter_id', 'weight', 'payer_code',
                          'examide', 'citoglipton'])

    # 4. Fill the remaining missing categories
    df['race'] = df['race'].fillna('Unknown')
    df['medical_specialty'] = df['medical_specialty'].fillna('Missing')
    top = df['medical_specialty'].value_counts().nlargest(top_n_specialties).index
    df['medical_specialty'] = df['medical_specialty'].where(
        df['medical_specialty'].isin(top), 'Other')

    # 5. Lab results: missing here means the test wasn't done
    df['max_glu_serum'] = df['max_glu_serum'].fillna('Not measured')
    df['A1Cresult'] = df['A1Cresult'].fillna('Not measured')

    # 6. Collapse 700+ ICD-9 codes into 9 clinical groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(group_icd9)

    # 7. Age bands -> numbers, ID codes -> readable groups
    df['age'] = df['age'].apply(age_to_midpoint)
    df['admission_type'] = df.pop('admission_type_id').apply(map_admission_type)
    df['admission_source'] = df.pop('admission_source_id').apply(map_admission_source)
    df['discharge_group'] = df.pop('discharge_disposition_id').apply(map_discharge)

    # 8. Summarise the medication columns, keeping only the two most common ones
    meds = [m for m in MEDICATIONS if m in df.columns]
    df['num_med_changes'] = df[meds].isin(['Up', 'Down']).sum(axis=1)
    df['num_meds_used'] = (df[meds] != 'No').sum(axis=1)
    df = df.drop(columns=[m for m in meds if m not in ('insulin', 'metformin')])

    # 9. Healthcare use in the year before this stay
    df['total_visits'] = (df['number_outpatient'] + df['number_emergency']
                          + df['number_inpatient'])

    # 10. Yes/No columns -> 1/0
    df['change'] = (df['change'] == 'Ch').astype(int)
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    # 11. Target: readmitted within 30 days = 1, everything else = 0
    y = (df.pop('readmitted') == '<30').astype(int)
    groups = df.pop('patient_nbr')

    return (df.reset_index(drop=True), y.reset_index(drop=True),
            groups.reset_index(drop=True))


def split_columns(X):
    """Numeric columns get scaled, the rest get one-hot encoded."""
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def best_f1_threshold(y_true, proba):
    """Find the probability cut-off that gives the best F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = np.argmax(f1[:-1])     # the last point has no matching threshold
    return thresholds[best], f1[best]


def evaluate(y_true, proba, threshold):
    """All the metrics used in the report, in one place."""
    pred = (proba >= threshold).astype(int)
    return {
        'ROC-AUC': roc_auc_score(y_true, proba),
        'PR-AUC': average_precision_score(y_true, proba),
        'Recall': recall_score(y_true, pred),
        'Precision': precision_score(y_true, pred, zero_division=0),
        'F1': f1_score(y_true, pred),
        'Balanced accuracy': balanced_accuracy_score(y_true, pred),
        'Brier score': brier_score_loss(y_true, proba),
        'Threshold': threshold,
    }
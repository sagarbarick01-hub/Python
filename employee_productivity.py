# Employee Productivity Data Transformation Assignment
# Professional solution using Pandas and Scikit-learn

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def make_one_hot_encoder():
    """Compatibility helper for different scikit-learn versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


# Load dataset
df = pd.read_csv("employee_productivity_dataset.csv")

print("\nOriginal Dataset Shape:", df.shape)
print(df.head())


# -------------------------------------------------------------------
# Q1. Handle Missing Values
# Age -> Median
# Salary -> Mean
# Hours_Worked_Per_Week -> Median
# Performance_Score -> Mean
# -------------------------------------------------------------------

print("\nQ1. Missing Values Before Handling:")
print(df.isnull().sum())

df_clean = df.copy()

age_median = df_clean["Age"].median()
salary_mean = df_clean["Salary"].mean()
hours_median = df_clean["Hours_Worked_Per_Week"].median()
performance_mean = df_clean["Performance_Score"].mean()

df_clean["Age"] = df_clean["Age"].fillna(age_median)
df_clean["Salary"] = df_clean["Salary"].fillna(salary_mean)
df_clean["Hours_Worked_Per_Week"] = df_clean["Hours_Worked_Per_Week"].fillna(hours_median)
df_clean["Performance_Score"] = df_clean["Performance_Score"].fillna(performance_mean)

print("\nImputation Values Used:")
print("Age median:", age_median)
print("Salary mean:", salary_mean)
print("Hours_Worked_Per_Week median:", hours_median)
print("Performance_Score mean:", performance_mean)

print("\nMissing Values After Handling:")
print(df_clean.isnull().sum())

print("\nDataset After Handling Missing Values:")
print(df_clean.head(10))


# -------------------------------------------------------------------
# Q2. Label Encoding
# Convert Gender and Department into numeric values
# -------------------------------------------------------------------

df_label = df_clean.copy()

gender_encoder = LabelEncoder()
department_encoder = LabelEncoder()

df_label["Gender"] = gender_encoder.fit_transform(df_label["Gender"])
df_label["Department"] = department_encoder.fit_transform(df_label["Department"])

gender_mapping = dict(zip(gender_encoder.classes_, gender_encoder.transform(gender_encoder.classes_)))
department_mapping = dict(zip(department_encoder.classes_, department_encoder.transform(department_encoder.classes_)))

print("\nQ2. Label Encoding Mappings:")
print("Gender Mapping:", gender_mapping)
print("Department Mapping:", department_mapping)

print("\nUpdated Gender and Department Columns:")
print(df_label[["Gender", "Department"]].head(10))


# -------------------------------------------------------------------
# Q3. One-Hot Encoding
# Apply One-Hot Encoding on Work_Mode and Location
# -------------------------------------------------------------------

columns_before_ohe = df_label.shape[1]

df_ohe = pd.get_dummies(
    df_label,
    columns=["Work_Mode", "Location"],
    dtype=int
)

columns_after_ohe = df_ohe.shape[1]
new_columns_created = [col for col in df_ohe.columns if col not in df_label.columns]

print("\nQ3. One-Hot Encoding:")
print("New one-hot columns created:", new_columns_created)
print("Number of new one-hot columns:", len(new_columns_created))
print("Dataset shape before one-hot encoding:", df_label.shape)
print("Dataset shape after one-hot encoding:", df_ohe.shape)
print("Net increase in columns:", columns_after_ohe - columns_before_ohe)

print("\nDataset After One-Hot Encoding:")
print(df_ohe.head())


# -------------------------------------------------------------------
# Q4. Normalization using MinMaxScaler
# Normalize Salary and Hours_Worked_Per_Week
# -------------------------------------------------------------------

df_minmax = df_ohe.copy()

minmax_scaler = MinMaxScaler()
df_minmax[["Salary", "Hours_Worked_Per_Week"]] = minmax_scaler.fit_transform(
    df_minmax[["Salary", "Hours_Worked_Per_Week"]]
)

print("\nQ4. Min-Max Normalized Columns:")
print(df_minmax[["Salary", "Hours_Worked_Per_Week"]].head(10))


# -------------------------------------------------------------------
# Q5. Standardization using StandardScaler
# Standardize Age and Projects_Completed
# -------------------------------------------------------------------

df_standardized = df_minmax.copy()

standard_scaler = StandardScaler()
df_standardized[["Age", "Projects_Completed"]] = standard_scaler.fit_transform(
    df_standardized[["Age", "Projects_Completed"]]
)

print("\nQ5. Standardized Columns:")
print(df_standardized[["Age", "Projects_Completed"]].head(10))


# -------------------------------------------------------------------
# Q6. Compare Scaling Methods on Salary
# Apply MinMaxScaler and StandardScaler side by side
# -------------------------------------------------------------------

salary_minmax = MinMaxScaler()
salary_standard = StandardScaler()

salary_comparison = pd.DataFrame({
    "Salary": df_clean["Salary"],
    "Salary_MinMax": salary_minmax.fit_transform(df_clean[["Salary"]]).ravel(),
    "Salary_Standard": salary_standard.fit_transform(df_clean[["Salary"]]).ravel()
})

print("\nQ6. Salary Scaling Comparison:")
print(salary_comparison.head(10))


# -------------------------------------------------------------------
# Q7. Build Preprocessing Pipeline
# Uses ColumnTransformer and Pipeline
#
# Notes:
# - Employee_ID and Name are identifiers, so they are excluded from ML preprocessing.
# - Join_Date is excluded here because the assignment focuses on encoding/scaling.
# - Median imputation is used for Age, Hours_Worked_Per_Week, Projects_Completed.
# - Mean imputation is used for Performance_Score and Salary.
# - Gender and Department are ordinal/label encoded.
# - Work_Mode and Location are one-hot encoded.
# -------------------------------------------------------------------

feature_cols = [
    "Department", "Age", "Gender", "Work_Mode",
    "Hours_Worked_Per_Week", "Projects_Completed",
    "Performance_Score", "Salary", "Location"
]

X = df[feature_cols].copy()

median_numeric_cols = ["Age", "Hours_Worked_Per_Week", "Projects_Completed"]
mean_numeric_cols = ["Performance_Score", "Salary"]
ordinal_categorical_cols = ["Gender", "Department"]
onehot_categorical_cols = ["Work_Mode", "Location"]

median_numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

mean_numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

ordinal_categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

onehot_categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", make_one_hot_encoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("median_num", median_numeric_pipeline, median_numeric_cols),
        ("mean_num", mean_numeric_pipeline, mean_numeric_cols),
        ("ordinal_cat", ordinal_categorical_pipeline, ordinal_categorical_cols),
        ("onehot_cat", onehot_categorical_pipeline, onehot_categorical_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

print("\nQ7. Preprocessing Pipeline Created Successfully:")
print(pipeline)


# -------------------------------------------------------------------
# Q8. Apply Pipeline
# Display transformed dataset and final shape
# -------------------------------------------------------------------

X_transformed = pipeline.fit_transform(X)
transformed_columns = pipeline.named_steps["preprocessor"].get_feature_names_out()

transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)

print("\nQ8. Transformed Dataset:")
print(transformed_df.head(10))

print("\nFinal Transformed Dataset Shape:", transformed_df.shape)


# -------------------------------------------------------------------
# Q9. Conceptual Question
# Why is scaling important in Python?
# -------------------------------------------------------------------

print("""
Q9. Why is scaling important in Python?

Scaling is important because many machine learning algorithms are sensitive to
the range of numerical values. Features with larger values, such as Salary,
can dominate features with smaller values, such as Performance_Score. Scaling
brings numerical columns to a comparable range, which improves model training,
distance calculations, convergence speed, and overall performance.
""")


# -------------------------------------------------------------------
# Q10. Conceptual Question
# Why convert categorical data into numerical form?
# -------------------------------------------------------------------

print("""
Q10. Why do we convert categorical data into numerical form?

Machine learning algorithms work with numerical input. Categorical columns such
as Gender, Department, Work_Mode, and Location contain text labels, so they must
be converted into numbers before being used in a model. Label Encoding and
One-Hot Encoding convert categories into machine-readable numeric features.
""")


# Save output files
df_standardized.to_csv("employee_productivity_stepwise_transformed.csv", index=False)
salary_comparison.to_csv("employee_productivity_salary_scaling_comparison.csv", index=False)
transformed_df.to_csv("employee_productivity_pipeline_transformed.csv", index=False)

print("\nOutput files saved:")
print("employee_productivity_stepwise_transformed.csv")
print("employee_productivity_salary_scaling_comparison.csv")
print("employee_productivity_pipeline_transformed.csv")
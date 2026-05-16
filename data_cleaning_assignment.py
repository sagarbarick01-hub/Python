import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("healthcare_data_cleaning_dataset.csv")
total_rows = len(df)

print("Initial shape:", df.shape)

# Q1. Missing values and percentage
missing_count = df.isna().sum()
missing_percent = (missing_count / total_rows * 100).round(2)
missing_summary = pd.DataFrame({
    "Missing Values": missing_count,
    "Missing %": missing_percent
})
print("\nQ1 Missing Summary")
print(missing_summary)

# Q2. Handling missing Age
# Median is chosen because it is robust to unusual age values/outliers.
age_median = df.loc[df["Age"].between(0, 100), "Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print("\nQ2 Age median used:", age_median)

# Q3. Handling missing Treatment_Cost
# Median is chosen because Treatment_Cost is highly skewed by expensive treatments.
cost_median = df["Treatment_Cost"].median()
df["Treatment_Cost"] = df["Treatment_Cost"].fillna(cost_median)
print("Q3 Treatment_Cost median used:", cost_median)

# Q4. Duplicate records
before_dedup = df.shape[0]
duplicate_rows = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
after_dedup = df.shape[0]
print("\nQ4 Duplicate rows:", duplicate_rows)
print("Before:", before_dedup, "After:", after_dedup, "Removed:", before_dedup - after_dedup)

# Q5. Invalid Age values
invalid_age = df[(df["Age"] < 0) | (df["Age"] > 100)]
print("\nQ5 Invalid Age rows:", invalid_age.shape[0])
# No invalid age records were found in this dataset, so no removal/correction is needed.

# Q6. IQR outlier detection for Treatment_Cost
Q1 = df["Treatment_Cost"].quantile(0.25)
Q3 = df["Treatment_Cost"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["Treatment_Cost"] < lower_bound) | (df["Treatment_Cost"] > upper_bound)]
print("\nQ6 IQR values")
print("Q1:", Q1, "Q3:", Q3, "IQR:", IQR)
print("Lower bound:", lower_bound, "Upper bound:", upper_bound)
print("Number of outliers:", outliers.shape[0])

# Q7. Winsorization / capping using 5th and 95th percentile
p5 = df["Treatment_Cost"].quantile(0.05)
p95 = df["Treatment_Cost"].quantile(0.95)
df["Treatment_Cost_Capped"] = df["Treatment_Cost"].clip(lower=p5, upper=p95)
print("\nQ7 Capping limits")
print("5th percentile:", p5, "95th percentile:", p95)
print("Values below 5th percentile:", (df["Treatment_Cost"] < p5).sum())
print("Values above 95th percentile:", (df["Treatment_Cost"] > p95).sum())

# Q8. Log transformation
df["Treatment_Cost_Log"] = np.log1p(df["Treatment_Cost"])
distribution_compare = pd.DataFrame({
    "Before_Treatment_Cost": df["Treatment_Cost"].describe(),
    "After_Log_Transform": df["Treatment_Cost_Log"].describe()
})
print("\nQ8 Distribution comparison")
print(distribution_compare)
print("Skewness before:", df["Treatment_Cost"].skew().round(2))
print("Skewness after:", df["Treatment_Cost_Log"].skew().round(2))

# Q9. Time-based missing handling
df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
missing_dates_before = df["Admission_Date"].isna().sum()
df = df.sort_values("Admission_Date").reset_index(drop=True)

# Forward fill is used because, after chronological sorting, the previous valid date is the nearest past record.
# Backward fill is added only for any leading missing date where no previous date exists.
df["Admission_Date"] = df["Admission_Date"].ffill().bfill()
missing_dates_after = df["Admission_Date"].isna().sum()
print("\nQ9 Admission_Date missing before:", missing_dates_before)
print("Admission_Date missing after:", missing_dates_after)
print("Date range:", df["Admission_Date"].min(), "to", df["Admission_Date"].max())

# Save cleaned output
df.to_csv("healthcare_cleaned_assignment_output.csv", index=False)

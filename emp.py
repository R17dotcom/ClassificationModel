import pandas as pd  
import xgboost as xgb
# Load the dataset  
df = pd.read_csv(r"C:\Sixth Semester\ML\ML Project\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display the first 5 rows  
print(df.head())
# Check for missing values  
print(df.isnull().sum())  

# Check data types of each column  
print(df.dtypes)  

# Basic statistics of numerical columns  
print(df.describe())
# Drop unnecessary columns  
df = df.drop(columns=["EmployeeNumber", "Over18", "StandardHours"])  

# Check dataset shape  
print(df.shape)
from sklearn.preprocessing import LabelEncoder  

# Encode categorical columns  
categorical_cols = ["Attrition", "BusinessTravel", "Department", "EducationField", 
                    "Gender", "JobRole", "MaritalStatus", "OverTime"]  

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])  
# 1️⃣ Experience Ratio (Years at company / Age)
df["Experience_Ratio"] = df["YearsAtCompany"] / df["Age"]

# 2️⃣ Stability Score (Years at company vs. total experience)
df["Stability_Score"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

# 3️⃣ Overtime Workload (Time spent with manager & company vs. total experience)
df["Overtime_Workload"] = (df["YearsWithCurrManager"] + df["YearsAtCompany"]) / (df["TotalWorkingYears"] + 1)

# 4️⃣ Job Role Score (Job level * Monthly Income)
df["JobRole_Score"] = df["JobLevel"] * df["MonthlyIncome"]

# 5️⃣ Travel-Balance (Work-life balance & Business travel effect)
df["BusinessTravel"] = df["BusinessTravel"].replace({"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2})
df["Travel_Balance"] = df["BusinessTravel"] * (4 - df["WorkLifeBalance"])

# 6️⃣ Seniority Ratio (Years in current role vs. years at company)
df["Seniority_Ratio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)

# Check changes  
print(df.head())
from sklearn.model_selection import train_test_split  

# Define features (X) and target variable (y)  
X = df.drop(columns=["Attrition"])  
y = df["Attrition"]  

# Split into 80% training, 20% testing  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train the model
gbm_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbm_model.fit(X_train, y_train)

# Predict on test data
y_pred_gbm = gbm_model.predict(X_test)

# Evaluate accuracy
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
print(f"Gradient Boosting Accuracy: {accuracy_gbm:.2f}")

# Initialize the model  
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  

# Train the model  
rf_model.fit(X_train, y_train)  

# Predict on test data  
y_pred = rf_model.predict(X_test)  

# Evaluate performance  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy Random Forest: {accuracy:.2f}")  

xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

# Detailed classification report  
print("Classification Report:\n", classification_report(y_test, y_pred))

selected_features = [
    "MonthlyIncome", "OverTime", "Age", "MonthlyRate", "DailyRate", "TotalWorkingYears",
    "HourlyRate", "DistanceFromHome", "Experience_Ratio", "Stability_Score",
    "Overtime_Workload", "JobRole_Score", "Travel_Balance", "Seniority_Ratio"
]

# Create a new dataset with only these features  
X_top14 = X[selected_features]
# Split again into train and test sets  
X_train_top14, X_test_top14, y_train, y_test = train_test_split(X_top14, y, test_size=0.2, random_state=42)
# Train Random Forest Model with 8 features
rf_top14 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_top14.fit(X_train_top14, y_train)

# Predict
y_pred_top8 = rf_top14.predict(X_test_top14)

# Evaluate accuracy
accuracy_top14 = accuracy_score(y_test, y_pred_top8)
print(f"Random Forest Accuracy (Top 8 Features): {accuracy_top14:.2f}")
from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train Gradient Boosting with top 8 features
gbm_model_top14 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbm_model_top14.fit(X_train_top14, y_train)

# Predict on test data
y_pred_gbm_top14 = gbm_model_top14.predict(X_test_top14)

# Evaluate accuracy
accuracy_gbm_top14 = accuracy_score(y_test, y_pred_gbm_top14)
print(f"Gradient Boosting Accuracy (Top 14 Features): {accuracy_gbm_top14:.2f}")
from xgboost import XGBClassifier

# Initialize and train XGBoost with top 8 features
xgb_model_top14 = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model_top14.fit(X_train_top14, y_train)

# Predict on test data
y_pred_xgb_top14 = xgb_model_top14.predict(X_test_top14)

# Evaluate accuracy
accuracy_xgb_top14 = accuracy_score(y_test, y_pred_xgb_top14)
print(f"XGBoost Accuracy (Top 8 Features): {accuracy_xgb_top14:.2f}")

import joblib

# Save the trained model
joblib.dump(rf_top14, "employee_attrition_model.pkl")
print("Model saved successfully!")
from sklearn.metrics import classification_report, confusion_matrix

# Assuming y_test is actual values and y_pred is predicted values
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import numpy as np
# Load the trained model
model = joblib.load("employee_attrition_model.pkl")

# Function to take user input and predict attrition based on feature extraction
def predict_attrition():
    print("\nEnter the required values:")
    age = float(input("Enter Age: "))
    years_at_company = float(input("Enter Years at Company: "))
    years_with_manager = float(input("Enter Years with Current Manager: "))
    total_working_years = float(input("Enter Total Working Years: "))
    job_level = float(input("Enter Job Level (1-5): "))
    monthly_income = float(input("Enter Monthly Income: "))
    work_life_balance = float(input("Enter Work-Life Balance (1-4): "))
    business_travel = int(input("Enter Business Travel (0: Non-Travel, 1: Travel_Rarely, 2: Travel_Frequently): "))
    over_time = int(input("Enter OverTime (0: No, 1: Yes): "))
    monthly_rate = float(input("Enter Monthly Rate: "))
    daily_rate = float(input("Enter Daily Rate: "))
    hourly_rate = float(input("Enter Hourly Rate: "))
    distance_from_home = float(input("Enter Distance from Home: "))

    # **Feature Engineering Calculations**
    experience_ratio = years_at_company / age
    stability_score = years_at_company / (total_working_years + 1)
    overtime_workload = (years_with_manager + years_at_company) / (total_working_years + 1)
    job_role_score = job_level * monthly_income
    travel_balance = business_travel * (4 - work_life_balance)
    seniority_ratio = years_at_company / (years_at_company + 1)

    print("\n🔹 **Calculated Feature Ratios** 🔹")
    print(f"📌 Experience Ratio: {experience_ratio:.4f}")
    print(f"📌 Stability Score: {stability_score:.4f}")
    print(f"📌 Overtime Workload: {overtime_workload:.4f}")
    print(f"📌 Job Role Score: {job_role_score:.2f}")
    print(f"📌 Travel Balance: {travel_balance:.2f}")
    print(f"📌 Seniority Ratio: {seniority_ratio:.4f}")

    # Creating input array with all features (original + engineered)
    input_features = np.array([
        monthly_income, over_time, age, monthly_rate, daily_rate, 
        total_working_years, hourly_rate, distance_from_home, 
        experience_ratio, stability_score, overtime_workload, 
        job_role_score, travel_balance, seniority_ratio
    ]).reshape(1, -1)

    # Making the prediction
    prediction = model.predict(input_features)

    # Display result
    if prediction[0] == 1:
        print("\n🚨 The employee is **likely to leave** the company.")
    else:
        print("\n✅ The employee is **likely to stay** in the company.")

# Run the prediction function
predict_attrition()
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance from the trained Random Forest model
feature_importances = rf_top14.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': X_train_top14.columns, 'Importance': feature_importances})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
plt.title("Feature Importance in Employee Attrition Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# ❤️ HEART ATTACK PREDICTION USING MACHINE LEARNING (Beginner Friendly)
# Author: Rajul Luniya
# Objective: Analyze dataset factors and predict user's heart attack risk level.

# -------------------------------------------------
# Step 1: Import Libraries
# -------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("❤️ HEART ATTACK PREDICTION SYSTEM")
print("=====================================\n")

# -------------------------------------------------
# Step 2: Load Dataset
# -------------------------------------------------
# ⚠️ Replace with your actual dataset path
data = pd.read_excel("heart.xlsx")

print("✅ Dataset Loaded Successfully!")
print(f"Total Records: {len(data)}\n")

# -------------------------------------------------
# Step 3: Data Cleaning and Preprocessing
# -------------------------------------------------

# Drop unnecessary columns
drop_cols = ['Patient ID', 'Country', 'Continent', 'Hemisphere']
for col in drop_cols:
    if col in data.columns:
        data = data.drop(columns=col)

# Convert Blood Pressure column (e.g., "120/80") → extract systolic (120)
def extract_systolic(bp):
    try:
        if isinstance(bp, str) and '/' in bp:
            return float(bp.split('/')[0])
        else:
            return float(bp)
    except:
        return np.nan

data['Blood Pressure'] = data['Blood Pressure'].apply(extract_systolic)

# Convert all numeric columns safely
numeric_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Stress Level',
                'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                'Exercise Hours Per Week', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].mean())  # replace NaN with mean

# Encode categorical columns
label_cols = ['Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
              'Alcohol Consumption', 'Diet', 'Previous Heart Problems',
              'Medication Use', 'Heart Attack Risk']

encoder = LabelEncoder()
for col in label_cols:
    if col in data.columns:
        data[col] = encoder.fit_transform(data[col].astype(str))

# -------------------------------------------------
# Step 4: Prepare Features and Target
# -------------------------------------------------
X = data.drop(columns=['Heart Attack Risk'])
y = data['Heart Attack Risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------
# Step 5: Train Model
# -------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained Successfully (Accuracy: {acc*100:.2f}%)\n")

# -------------------------------------------------
# Step 6: Feature Importance
# -------------------------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("📊 Top Factors Influencing Heart Attack Risk:\n")
print(importance.head(10), "\n")

# -------------------------------------------------
# Step 7: User Input
# -------------------------------------------------
print("💬 Please enter your health details for prediction:\n")

def get_input():
    user_data = {}
    user_data['Age'] = float(input("Enter Age: "))
    user_data['Sex'] = input("Sex (Male/Female): ")
    user_data['Cholesterol'] = float(input("Cholesterol Level (mg/dL): "))
    bp_value = input("Blood Pressure (e.g., 120/80 or 130): ")
    user_data['Blood Pressure'] = extract_systolic(bp_value)
    user_data['Heart Rate'] = float(input("Resting Heart Rate (bpm): "))
    user_data['Diabetes'] = input("Do you have Diabetes? (Yes/No): ")
    user_data['Family History'] = input("Family History of Heart Problems? (Yes/No): ")
    user_data['Smoking'] = input("Do you Smoke? (Yes/No): ")
    user_data['Obesity'] = input("Are you Overweight/Obese? (Yes/No): ")
    user_data['Alcohol Consumption'] = input("Do you Consume Alcohol? (Yes/No): ")
    user_data['Exercise Hours Per Week'] = float(input("Exercise Hours Per Week: "))
    user_data['Diet'] = input("Diet (Healthy/Unhealthy): ")
    user_data['Previous Heart Problems'] = input("Previous Heart Problems? (Yes/No): ")
    user_data['Medication Use'] = input("Do you take Regular Medication? (Yes/No): ")
    user_data['Stress Level'] = float(input("Stress Level (1-10): "))
    user_data['Sedentary Hours Per Day'] = float(input("Sedentary Hours Per Day: "))
    user_data['Income'] = float(input("Monthly Income (in thousands): "))
    user_data['BMI'] = float(input("BMI (Body Mass Index): "))
    user_data['Triglycerides'] = float(input("Triglycerides Level (mg/dL): "))
    user_data['Physical Activity Days Per Week'] = float(input("Physical Activity Days/Week: "))
    user_data['Sleep Hours Per Day'] = float(input("Sleep Hours Per Day: "))
    return user_data

user_input = get_input()

# -------------------------------------------------
# Step 8: Prepare User Data
# -------------------------------------------------
user_df = pd.DataFrame([user_input])

# Encode categorical columns in user input
for col in label_cols[:-1]:  # exclude target
    if col in user_df.columns:
        user_df[col] = encoder.fit_transform(user_df[col].astype(str))

# Align with training columns
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# -------------------------------------------------
# Step 9: Prediction
# -------------------------------------------------
pred = model.predict(user_df)[0]
prob = model.predict_proba(user_df)[0][1] * 100

# -------------------------------------------------
# Step 10: Display Results
# -------------------------------------------------
print("\n📊 RESULT:")
if pred == 1:
    if prob > 75:
        print(f"🔴 HIGH RISK of Heart Attack ({prob:.1f}%)")
    elif prob > 50:
        print(f"🟠 MODERATE RISK of Heart Attack ({prob:.1f}%)")
    else:
        print(f"🟡 LOW RISK but some concern ({prob:.1f}%)")
else:
    print(f"🟢 LOW RISK of Heart Attack ({100-prob:.1f}%)")

# -------------------------------------------------
# Step 11: Health Suggestions
# -------------------------------------------------
print("\n💡 HEALTH SUGGESTIONS:")
print("1️⃣ Maintain a balanced diet and regular exercise routine.")
print("2️⃣ Avoid smoking and excessive alcohol.")
print("3️⃣ Manage stress and ensure adequate sleep daily.")
print("4️⃣ Regularly monitor BP, cholesterol, and BMI levels.")
print("5️⃣ Consult your doctor for periodic heart checkups.")

print("\n✅ Prediction Completed Successfully!")

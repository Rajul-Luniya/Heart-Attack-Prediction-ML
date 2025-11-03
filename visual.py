# -------------------------------------------------
# 🧠 HEART ATTACK PREDICTION DATA VISUALIZATION
# -------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated sample data (replace with your real dataset 'heart.xlsx')
data = pd.DataFrame({
    'Age': [25, 35, 45, 55, 65, 40, 50, 60, 70, 48],
    'Blood Pressure': [120, 130, 140, 150, 160, 125, 135, 145, 155, 138],
    'Cholesterol': [180, 210, 230, 250, 270, 200, 220, 240, 260, 225],
    'BMI': [22, 24, 27, 29, 31, 25, 26, 28, 30, 26],
    'Stress Level': [3, 5, 6, 7, 9, 4, 6, 8, 10, 5],
    'Exercise Hours Per Week': [5, 4, 3, 2, 1, 4, 3, 2, 1, 3],
    'Heart Attack Risk': ['Low', 'Low', 'Moderate', 'Moderate', 'High', 'Low', 'Moderate', 'High', 'High', 'Moderate']
})

# Professional plotting style
sns.set(style="whitegrid", context="talk", palette="coolwarm")

# -------------------------------------------------
# 1️⃣ Risk Distribution Pie Chart
# -------------------------------------------------
plt.figure(figsize=(6,6))
risk_counts = data['Heart Attack Risk'].value_counts()
plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("coolwarm", len(risk_counts)))
plt.title("Heart Attack Risk Distribution", fontsize=15, weight='bold')
plt.show()

# -------------------------------------------------
# 2️⃣ Age vs Blood Pressure (Scatter Plot)
# -------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Blood Pressure', hue='Heart Attack Risk', data=data, s=100, edgecolor='black')
plt.title("Age vs Blood Pressure by Heart Attack Risk", fontsize=15, weight='bold')
plt.xlabel("Age (Years)")
plt.ylabel("Blood Pressure (mmHg)")
plt.show()

# -------------------------------------------------
# 3️⃣ Cholesterol vs Risk (Boxplot)
# -------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=data)
plt.title("Cholesterol Levels Across Risk Groups", fontsize=15, weight='bold')
plt.xlabel("Heart Attack Risk Level")
plt.ylabel("Cholesterol (mg/dL)")
plt.show()

# -------------------------------------------------
# 4️⃣ Exercise vs Stress (Scatter Plot)
# -------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x='Exercise Hours Per Week', y='Stress Level', hue='Heart Attack Risk', data=data, s=100, edgecolor='black')
plt.title("Exercise vs Stress Level by Risk Category", fontsize=15, weight='bold')
plt.xlabel("Exercise Hours Per Week")
plt.ylabel("Stress Level (1-10)")
plt.show()

# -------------------------------------------------
# 5️⃣ BMI vs Risk (Bar Plot)
# -------------------------------------------------
plt.figure(figsize=(8,5))
sns.barplot(x='Heart Attack Risk', y='BMI', data=data)
plt.title("Average BMI per Heart Attack Risk Level", fontsize=15, weight='bold')
plt.xlabel("Heart Attack Risk Level")
plt.ylabel("Average BMI")
plt.show()

# -------------------------------------------------
# 6️⃣ Correlation Heatmap
# -------------------------------------------------
plt.figure(figsize=(10,7))
corr = data.drop(columns=['Heart Attack Risk']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Health Factors", fontsize=15, weight='bold')
plt.show()

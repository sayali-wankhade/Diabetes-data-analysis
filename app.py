import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Embedded Dataset (15 rows)
# -----------------------------
data = [
    [6,148,72,35,0,33.6,0.627,50,1],
    [1,85,66,29,0,26.6,0.351,31,0],
    [8,183,64,0,0,23.3,0.672,32,1],
    [1,89,66,23,94,28.1,0.167,21,0],
    [0,137,40,35,168,43.1,2.288,33,1],
    [5,116,74,0,0,25.6,0.201,30,0],
    [3,78,50,32,88,31.0,0.248,26,1],
    [10,115,0,0,0,35.3,0.134,29,0],
    [2,197,70,45,543,30.5,0.158,53,1],
    [8,125,96,0,0,0.0,0.232,54,1],
    [4,110,92,0,0,37.6,0.191,30,0],
    [7,140,85,33,130,29.0,0.350,45,1],
    [2,99,60,17,160,26.0,0.400,24,0],
    [6,160,80,40,200,32.5,0.500,41,1],
    [1,95,70,20,85,27.5,0.300,22,0]
]

columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
           "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Public Health Analytics Dashboard")
st.subheader("📊 Diabetes Dataset Analysis")

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Patients", len(df))
col2.metric("🍬 Avg Glucose", round(df["Glucose"].mean(), 2))
col3.metric("⚠️ High Risk", len(df[df["Glucose"] > 140]))

# -----------------------------
# Missing Values Handling
# -----------------------------
cols = ["Glucose","BloodPressure","BMI","Insulin"]
for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

st.write("🧹 Missing values filled using median")

# -----------------------------
# Descriptive Statistics
# -----------------------------
st.write("📊 Descriptive Statistics")
st.dataframe(df.describe())

# -----------------------------
# Outlier Removal (IQR)
# -----------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# -----------------------------
# Normalization
# -----------------------------
df_norm = (df_clean - df_clean.min()) / (df_clean.max() - df_clean.min())
st.write("📏 Data Normalization Completed")

# -----------------------------
# Heatmap
# -----------------------------
st.write("🔥 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------
# Histogram
# -----------------------------
st.write("📈 Glucose Histogram")
fig, ax = plt.subplots()
ax.hist(df_clean["Glucose"])
st.pyplot(fig)

# -----------------------------
# Boxplot
# -----------------------------
st.write("📦 BMI Box Plot")
fig, ax = plt.subplots()
sns.boxplot(y=df_clean["BMI"], ax=ax)
st.pyplot(fig)

# -----------------------------
# Scatter Plot
# -----------------------------
st.write("📉 Age vs Glucose")
fig, ax = plt.subplots()
ax.scatter(df_clean["Age"], df_clean["Glucose"])
st.pyplot(fig)

# -----------------------------
# Countplot
# -----------------------------
st.write("📊 Diabetic vs Non-Diabetic")
fig, ax = plt.subplots()
sns.countplot(x="Outcome", data=df_clean, ax=ax)
st.pyplot(fig)
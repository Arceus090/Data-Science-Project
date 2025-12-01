import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


st.title("⚽ Football Player Market Value Regression App")


file_path = "footballcw.csv"
df = pd.read_csv(file_path)
st.write("### 📌 Original Dataset Preview")
st.dataframe(df.head())


# data preprocessing
unique_positions = df["position"].unique()
position_mapping = {
    position: encoded_value
    for encoded_value, position in enumerate(unique_positions, start=1)
}
df["position_encoded"] = df["position"].map(position_mapping)

st.write("### 📌 Position Encoding Mapping")
st.dataframe(df[["position", "position_encoded"]].drop_duplicates())

# cleaning
df = df.fillna(0)


columns_to_drop = ["player", "team", "name", "position"]
revised_df = df.drop(columns=columns_to_drop)

st.write("### 📌 Cleaned Dataset")
st.dataframe(revised_df.head())

# charts
st.subheader("📊 Exploratory Data Analysis (EDA)")

# Age disitribution
st.write("### 🎯 Distribution of Player's Age")
fig_age, ax_age = plt.subplots(figsize=(15, 6))
sns.histplot(revised_df["age"], color="red", kde=True, ax=ax_age)
ax_age.set_xticklabels(ax_age.get_xticks(), rotation=90)
ax_age.set_title("Distribution of Player's Age")
st.pyplot(fig_age)

# Top 10 Highest markets values
st.write("### 🏆 Top 10 Highest Market Value Players")
top_market_df = df.nlargest(10, "highest_value").sort_values(
    "highest_value", ascending=False
)
fig_top, ax_top = plt.subplots(figsize=(15, 6))
sns.barplot(
    x="name",
    y="highest_value",
    data=top_market_df,
    palette="hot",
    edgecolor=sns.color_palette("dark", 7),
    ax=ax_top,
)
ax_top.set_xticklabels(ax_top.get_xticklabels(), rotation=90)
ax_top.set_title("Top 10 Highest Market Value Football Players")
st.pyplot(fig_top)


st.write("### 🔥 Correlation Heatmap")
correlation = revised_df.corr()
fig_corr, ax_corr = plt.subplots(figsize=(12, 12))
sns.heatmap(correlation, annot=True, cmap="viridis", ax=ax_corr)
st.pyplot(fig_corr)

#  training
st.subheader("🤖 Regression Modeling")

target = "current_value"
X = revised_df.drop(columns=[target])
y = revised_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_choice = st.selectbox(
    "Select a Regression Model:", ["Linear Regression", "Random Forest"]
)

if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
else:
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Metrics
st.write("### 📈 Model Performance")
st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")
rmse = sqrt(mean_squared_error(y_test, y_pred))
st.write(f"**RMSE:** {rmse:.3f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")

# Actual vs Predicted
st.write("### 🎨 Actual vs Predicted (Red = Actual, Blue = Predicted)")
fig_ap, ax_ap = plt.subplots(figsize=(10, 6))
ax_ap.scatter(y_test, y_test, color="red", label="Actual Values")
ax_ap.scatter(y_test, y_pred, color="blue", label="Predicted Values")
ax_ap.set_xlabel("Actual")
ax_ap.set_ylabel("Market Value")
ax_ap.set_title("Actual (Red) vs Predicted (Blue)")
ax_ap.legend()
st.pyplot(fig_ap)


# player pred

st.subheader("🧮 Predict a Player's Market Value")

user_inputs = {}
for col in X.columns:
    user_inputs[col] = st.number_input(
        f"Enter value for {col}", value=float(X[col].mean())
    )

user_df = pd.DataFrame([user_inputs])

if model_choice == "Linear Regression":
    user_df_scaled = scaler.transform(user_df)
    prediction = model.predict(user_df_scaled)
else:
    prediction = model.predict(user_df)

st.success(f"Predicted Market Value: **€{prediction[0]:,.2f}**")

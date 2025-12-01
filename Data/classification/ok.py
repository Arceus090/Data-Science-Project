import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Cardio Classifier", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("cardio_train.csv", sep=";")
    df = df.head(5000)  # for faster processing
    return df


df = load_data()

st.title("🫀 Cardio Disease Classifier")
st.write("Dataset shape before cleaning:", df.shape)

# data cleaning
df = df.drop_duplicates()
df = df[df["ap_hi"] > 0]
df = df[df["ap_lo"] > 0]

st.write("### ✅ Dataset after cleaning:")
st.dataframe(df.head())


df["age_years"] = (df["age"] / 365).astype(int)
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2


# CHARTS

st.header("📊 Exploratory Data Analysis")

### Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cholesterol Count")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="cholesterol", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution (Years)")
    fig, ax = plt.subplots()
    sns.histplot(df["age_years"], kde=True, ax=ax)
    st.pyplot(fig)

### Row 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("BMI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["bmi"], kde=True, ax=ax)
    st.pyplot(fig)

with col4:
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    df["gender"].replace({1: "Female", 2: "Male"}).value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

### Row 3
col5, col6 = st.columns(2)

with col5:
    st.subheader("Cholesterol vs Cardio")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="cholesterol", hue="cardio", ax=ax)
    st.pyplot(fig)

with col6:
    st.subheader("Blood Pressure Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="ap_hi", y="ap_lo", hue="cardio", ax=ax)
    st.pyplot(fig)

### Row 4
st.subheader("Active vs Cardio")
fig, ax = plt.subplots()
sns.countplot(data=df, x="active", hue="cardio", ax=ax)
st.pyplot(fig)


# testing

st.header("🤖 Model Training")

X = df[
    [
        "age_years",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "bmi",
    ]
]
y = df["cardio"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=500),
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies[name] = acc

# Accuracy chart
st.subheader("Model Accuracy Comparison")
fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values())
ax.set_ylim(0, 1)
st.pyplot(fig)

best_model_name = max(accuracies, key=accuracies.get)
st.success(
    f"Best Model: **{best_model_name}** with accuracy {accuracies[best_model_name]:.2f}"
)

best_model = models[best_model_name]


# USER INPUT

st.header("🧑‍⚕️ Predict Cardio Disease")

age_years = st.number_input("Age (years)", min_value=10, max_value=100, value=50)
gender_text = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender_text == "Male" else 2  # male=1, female=2

height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
ap_hi = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
ap_lo = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
chol = st.selectbox("Cholesterol (1–3)", [1, 2, 3])
gluc = st.selectbox("Glucose (1–3)", [1, 2, 3])
smoke = st.selectbox("Smoke (0/1)", [0, 1])
alco = st.selectbox("Alcohol (0/1)", [0, 1])
active = st.selectbox("Active (0/1)", [0, 1])

bmi = weight / (height / 100) ** 2

if st.button("Predict"):
    user_data = np.array(
        [
            [
                age_years,
                gender,
                height,
                weight,
                ap_hi,
                ap_lo,
                chol,
                gluc,
                smoke,
                alco,
                active,
                bmi,
            ]
        ]
    )

    scaled = scaler.transform(user_data)
    prediction = best_model.predict(scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of cardio disease")
    else:
        st.success("✅ Low risk of cardio disease")

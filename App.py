# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import seaborn as sns

st.set_page_config(page_title="Disease Prediction EDA App", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("🩺 Disease Prediction Dashboard")
st.sidebar.write("Upload dataset and explore EDA with interactive visualization.")

uploaded_file = st.sidebar.file_uploader("C:\Users\user\OneDrive\Desktop\My Projects\Disease_prediction\diabetes_cleaned.csv", type=["csv"])

# ---------- Load dataset ----------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("✅ Dataset successfully loaded!")
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# ---------- Show dataset overview ----------
st.header("📊 Dataset Overview")
st.write("Shape of the dataset:", df.shape)
st.dataframe(df.head())

# Missing values
st.subheader("🧩 Missing Values Overview")
missing = df.isnull().sum().reset_index()
missing.columns = ["Feature", "Missing Count"]
st.dataframe(missing[missing["Missing Count"] > 0])

# ---------- Data Cleaning ----------
st.header("🧹 Data Cleaning and Imputation")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
imputer = SimpleImputer(strategy="median")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
st.write("✅ Missing values handled using median imputation.")

# ---------- Univariate Analysis ----------
st.header("📈 Univariate Analysis")
feature = st.selectbox("Select a numeric column:", numeric_cols)

fig1 = px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}", color_discrete_sequence=['#6A5ACD'])
st.plotly_chart(fig1, use_container_width=True)

fig2, ax = plt.subplots()
sns.boxplot(x=df[feature], color="#87CEEB", ax=ax)
st.pyplot(fig2)

# ---------- Correlation ----------
st.header("🔗 Correlation Heatmap")

# ✅ Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] > 1:
    corr = numeric_df.corr()
    fig3, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)
else:
    st.warning("No numeric columns available for correlation heatmap.")

# ---------- Target-based Analysis ----------
target_col = st.selectbox("Select Target Column (binary outcome):", df.columns)
if df[target_col].nunique() == 2:
    st.success(f"✅ Target column '{target_col}' selected.")
    numeric_features = [col for col in numeric_cols if col != target_col]

    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.histogram(df, x=numeric_features[0], color=target_col, barmode="overlay", title=f"{numeric_features[0]} by Target")
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        fig5 = px.box(df, x=target_col, y=numeric_features[1], title=f"{numeric_features[1]} by Target")
        st.plotly_chart(fig5, use_container_width=True)
else:
    st.error("Please ensure the selected target column is binary (0 and 1).")

# ---------- Modeling Section ----------
st.header("🤖 Quick Disease Prediction Model")

try:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Model Performance")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    fig6, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig6)

except Exception as e:
    st.error(f"Error in model training: {e}")

# ---------- Prediction Form ----------
st.header("🧠 Predict Disease for a New Patient")

if st.checkbox("Show Prediction Form"):
    inputs = {}
    for col in X.columns:
        val = st.number_input(f"{col}:", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        inputs[col] = val

    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write("### 🩸 Prediction Result:")
    st.success(f"Predicted Class: {'Positive (1)' if pred == 1 else 'Negative (0)'} | Probability: {prob:.2f}")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly, Scikit-learn | Disease Prediction Capstone (Healthcare Domain)")

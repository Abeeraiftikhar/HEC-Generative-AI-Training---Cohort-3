import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("🎓 AI-Powered Student Performance Analytics")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv('student_data.csv')

df = load_data()

# --- SIDEBAR: ML GPA PREDICTOR ---
st.sidebar.header("🔮 Predict Student GPA")
input_attendance = st.sidebar.slider("Attendance Rate (%)", 60.0, 100.0, 85.0)
input_hours = st.sidebar.slider("Weekly Study Hours", 1.0, 15.0, 7.0)
input_failures = st.sidebar.selectbox("Past Class Failures", [0, 1, 2, 3])

# Train XGBoost Model
X = df[['Attendance_Rate', 'Study_Hours_Weekly', 'Past_Failures']]
y = df['Current_GPA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
model.fit(X_train, y_train)

# Predict based on sidebar inputs
features = np.array([[input_attendance, input_hours, input_failures]])
predicted_gpa = model.predict(features)[0]
st.sidebar.metric(label="Predicted GPA", value=f"{predicted_gpa:.2f} / 4.0")

# --- MAIN DASHBOARD VISUALS ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Students Tracked", len(df))
col2.metric("Average Attendance", f"{df['Attendance_Rate'].mean():.1f}%")
col3.metric("Average GPA", f"{df['Current_GPA'].mean():.2f}")

st.markdown("---")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("📊 Attendance vs. GPA Trends")
    fig1 = px.scatter(df, x="Attendance_Rate", y="Current_GPA", 
                     trendline="ols", color="Study_Hours_Weekly",
                     labels={"Attendance_Rate": "Attendance (%)", "Current_GPA": "GPA"})
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    st.subheader("🔥 Performance Correlation Heatmap")
    subjects = ['Math_Score', 'Science_Score', 'English_Score', 'Current_GPA']
    corr_matrix = df[subjects].corr()
    fig2 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig2, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📈 Subject Score Distributions")
    df_melted = df.melt(id_vars=['Student_ID'], value_vars=['Math_Score', 'Science_Score', 'English_Score'], 
                        var_name='Subject', value_name='Score')
    fig3 = px.box(df_melted, x='Subject', y='Score', color='Subject')
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    st.subheader("🤖 Model Accuracy (Actual vs. Predicted)")
    y_pred = model.predict(X_test)
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig4 = px.scatter(eval_df, x='Actual', y='Predicted', trendline="ols")
    st.plotly_chart(fig4, use_container_width=True)

if st.checkbox("Show Data Log"):
    st.dataframe(df)

import openai
import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import base64
import requests
from bs4 import BeautifulSoup

# Set OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key

# Hide 'Made with Streamlit' footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Create directories for logs and saved chats
os.makedirs('Chat Logs', exist_ok=True)
os.makedirs('Saved Chats', exist_ok=True)

# Generate log file path
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join('Chat Logs', f'log_{timestamp}.txt')

# Initialize session state for typed query history
if 'typed_query_history' not in st.session_state:
    st.session_state.typed_query_history = []

# Google Scholar scraping function
def scrape_google_scholar(query):
    url = "https://scholar.google.com/scholar?q=" + "+".join(query.split())
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        titles = soup.find_all('h3', class_='gs_rt')
        return [title.get_text() for title in titles]
    else:
        return "Failed to retrieve data"

# Data analysis functions
def generate_bar_chart(df, column):
    fig = px.bar(df, x=column, title=f'Bar Chart of {column}')
    st.plotly_chart(fig)

def generate_histogram(df, column):
    fig = px.histogram(df, x=column, title=f'Histogram of {column}')
    st.plotly_chart(fig)

def show_advanced_stats(df):
    st.write("Skewness of each column:")
    st.write(df.skew())
    st.write("Kurtosis of each column:")
    st.write(df.kurtosis())

def build_linear_regression_model(df):
    st.sidebar.subheader("Linear Regression Model")
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) > 1:
        x_column = st.sidebar.selectbox("Select Feature Column", numeric_columns)
        y_column = st.sidebar.selectbox("Select Target Column", numeric_columns)
        X = df[[x_column]]
        y = df[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Squared Error:", mse)
        st.write("R^2 Score:", r2)
    else:
        st.sidebar.write("Not enough numeric columns for regression analysis.")

def generate_seaborn_plot(df):
    st.sidebar.subheader("Seaborn Plot Options")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Pairplot", "Heatmap", "Boxplot"])

    if plot_type == "Pairplot":
        pairplot = sns.pairplot(df)
        fig = pairplot.fig
    elif plot_type == "Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
    elif plot_type == "Boxplot":
        selected_column = st.sidebar.selectbox("Choose Column for Boxplot", df.columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_column], ax=ax)

    st.pyplot(fig)

def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Main function with multi-page setup
def main():
    st.sidebar.image("ACN_LOG

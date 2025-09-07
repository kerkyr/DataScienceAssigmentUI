import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Heart Disease Dashboard", 
    page_icon="❤️", 
    layout="wide"
)

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("heart_disease.csv")

# ------------------------------
# Title & Introduction
# ------------------------------
st.title("❤️ AuraCare Heart Disease Dashboard")
st.markdown("""
Welcome to the **AuraCare Dashboard**.<br>
This dashboard visualizes the distribution of input attributes and their relationship with **Heart Disease Status**.
""", unsafe_allow_html=True)

# ------------------------------
# Define Attributes
# ------------------------------
numerical = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
    'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
]

categorical = [
    'Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 'Diabetes',
    'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
    'Alcohol Consumption', 'Stress Level', 'Sugar Consumption'
]

attributes = numerical + categorical

# ------------------------------
# Helper Functions
# ------------------------------
def show_numerical(attr):
    """Display histogram and boxplot for numerical attributes."""
    col1, col2 = st.columns(2)
    
    # Histogram
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[attr].hist(bins=10, color='skyblue', edgecolor='black', ax=ax)
        ax.set_title(f'Distribution of {attr}')
        ax.set_xlabel(attr)
        ax.set_ylabel("Frequency")
        st.pyplot(fig, use_container_width=False)
    
    # Boxplot
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Heart Disease Status', y=attr, palette="Set2", ax=ax)
        ax.set_title(f'{attr} vs Heart Disease Status')
        ax.set_xlabel("Heart Disease Status")
        ax.set_ylabel(attr)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        st.pyplot(fig, use_container_width=False)

def show_categorical(attr):
    """Display pie chart and countplot for categorical attributes."""
    col1, col2 = st.columns(2)
    
    # Pie chart
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[attr].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=90, shadow=True,
            textprops={'fontsize': 9}, ax=ax
        )
        ax.set_ylabel("")
        ax.set_title(f"Distribution of {attr}")
        st.pyplot(fig, use_container_width=False)
    
    # Countplot by Heart Disease Status
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=attr, hue="Heart Disease Status", palette="Set3", ax=ax)
        ax.set_title(f"{attr} vs Heart Disease Status")
        ax.set_xlabel(attr)
        ax.set_ylabel("Count")
        ax.legend(title="Heart Disease")
        st.pyplot(fig, use_container_width=False)

# ------------------------------
# Render Tabs for All Attributes
# ------------------------------
tabs = st.tabs(attributes)
for i, attr in enumerate(attributes):
    with tabs[i]:
        st.subheader(f"📊 {attr} Analysis")
        if attr in numerical:
            show_numerical(attr)
        else:
            show_categorical(attr)

# ------------------------------
# Observation Note
# ------------------------------
st.markdown("""
From the attribute visualizations above, there is **no single clear factor** that strongly determines the presence of heart disease.
""")

# ------------------------------
# Overall Target Distribution
# ------------------------------
st.markdown("---")
st.subheader("⚖️ Heart Disease Status Distribution")
st.markdown("""
Since the dataset is **imbalanced**, we applied the **SMOTE (Synthetic Minority Oversampling Technique)**  
to balance the classes for better model training and fairer predictions.
""")

# Compute original distribution
dist = df['Heart Disease Status'].value_counts(normalize=True).mul(100).round(2)

# Shared chart settings
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0.05)
fig_size = (3, 3)

# Create two columns for side-by-side pie charts
col1, col2 = st.columns(2)

# Original Distribution
with col1:
    st.markdown("### Original Distribution")
    fig1, ax1 = plt.subplots(figsize=fig_size)
    ax1.pie(
        dist, labels=dist.index, autopct='%.2f%%',
        colors=colors, explode=explode, startangle=90, shadow=True
    )
    ax1.set_title("Heart Disease Status (Original)", fontsize=11)
    st.pyplot(fig1, use_container_width=False)

# After SMOTE
with col2:
    st.markdown("### After SMOTE")
    fig2, ax2 = plt.subplots(figsize=fig_size)
    smote_dist = [50, 50]
    smote_labels = ["No Heart Disease", "Heart Disease"]
    ax2.pie(
        smote_dist, labels=smote_labels, autopct='%.1f%%',
        colors=colors, explode=explode, startangle=90, shadow=True
    )
    ax2.set_title("Heart Disease Status (After SMOTE)", fontsize=11)
    st.pyplot(fig2, use_container_width=False)


import streamlit as st
from modules.data_processing import DataProcessor
from modules.modeling import PerformancePredictor
from modules.visualization import Dashboard
from modules.utils import DatabaseConnector
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize components
    st.title("Student Performance Prediction System")
    db = DatabaseConnector()
    processor = DataProcessor()
    predictor = PerformancePredictor()
    dashboard = Dashboard()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox(
        "Select Module",
        ["Dashboard", "Data Processing", "Model Training", "Alert System"]
    )
    
    # Main application logic
    if app_mode == "Dashboard":
        show_dashboard(db, dashboard)
    elif app_mode == "Data Processing":
        show_data_processing(processor)
    elif app_mode == "Model Training":
        show_model_training(predictor)
    elif app_mode == "Alert System":
        show_alert_system(db, dashboard)

def show_dashboard(db, dashboard):
    st.header("Performance Analytics Dashboard")
    
    # Fetch data
    student_data = db.get_student_records()
    performance_data = db.get_performance_data()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(student_data))
    with col2:
        st.metric("At-Risk Students", 
                 len(performance_data[performance_data['risk_level'] == 'high']))
    with col3:
        st.metric("Avg. Course Score", 
                 f"{performance_data['score'].mean():.1f}")
    
    # Show visualizations
    tab1, tab2, tab3 = st.tabs(["Performance Trends", "Risk Distribution", "Model Insights"])
    
    with tab1:
        fig = dashboard.plot_performance_trends(performance_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = dashboard.plot_risk_distribution(performance_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = dashboard.plot_feature_importance(performance_data)
        st.plotly_chart(fig, use_container_width=True)

# [Additional functions for other modules...]

if __name__ == "__main__":
    main()
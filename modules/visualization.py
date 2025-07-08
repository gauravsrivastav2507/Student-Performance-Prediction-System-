import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

class Dashboard:
    def plot_performance_trends(self, data):
        fig = px.line(
            data,
            x='week',
            y='score',
            color='student_id',
            title='Student Performance Trends Over Time'
        )
        fig.update_layout(hovermode='x unified')
        return fig
    
    def plot_risk_distribution(self, data):
        fig = px.pie(
            data,
            names='risk_level',
            title='Student Risk Level Distribution'
        )
        return fig
    
    def plot_feature_importance(self, data):
        # Sample feature importance data
        features = ['Assignments', 'Attendance', 'Engagement', 'Previous GPA']
        importance = np.random.rand(len(features))
        
        fig = px.bar(
            x=features,
            y=importance,
            title='Feature Importance for Performance Prediction'
        )
        return fig
    
    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis('off')
        return fig
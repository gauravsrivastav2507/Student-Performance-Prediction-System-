from sqlalchemy import create_engine
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnector:
    def __init__(self):
        self.sql_engine = create_engine(os.getenv('SQL_DB_URI'))
        self.mongo_client = MongoClient(os.getenv('MONGO_URI'))
    
    def get_student_records(self):
        query = "SELECT * FROM students"
        return pd.read_sql(query, self.sql_engine)
    
    def get_performance_data(self):
        db = self.mongo_client['student_performance']
        collection = db['predictions']
        return pd.DataFrame(list(collection.find()))
    
    def save_prediction(self, prediction_data):
        db = self.mongo_client['student_performance']
        collection = db['predictions']
        collection.insert_one(prediction_data)
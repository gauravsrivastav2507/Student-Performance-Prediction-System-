import pandas as pd
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class DataProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Text cleaning pipeline
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join([self.lemmatizer.lemmatize(word) 
                        for word in text.split() 
                        if word not in self.stop_words])
        return text
    
    def prepare_bert_inputs(self, texts, max_length=128):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def extract_numeric_features(self, df):
        # Feature engineering for traditional models
        features = df[['assignment_scores', 'attendance', 'past_gpa']].copy()
        features['score_variance'] = features['assignment_scores'].apply(np.var)
        return features
    
    def reduce_dimensionality(self, features, n_components=5):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(features)

import torch
from transformers import BertForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class PerformancePredictor:
    def __init__(self):
        self.bert_model = None
        self.traditional_model = None
    
    def initialize_bert_model(self, num_labels=2):
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
    
    def train_bert(self, train_loader, val_loader, epochs=3, lr=3e-5):
        optimizer = torch.optim.AdamW(self.bert_model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.bert_model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.bert_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
    
    def train_traditional_model(self, X, y, model_type='xgb'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        if model_type == 'xgb':
            self.traditional_model = XGBClassifier()
        else:
            self.traditional_model = RandomForestClassifier()
            
        self.traditional_model.fit(X_train, y_train)
        preds = self.traditional_model.predict(X_test)
        print(classification_report(y_test, preds))
    
    def predict_risk(self, text_input, numeric_features):
        # Ensemble prediction combining BERT and traditional model
        pass
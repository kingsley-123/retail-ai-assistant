"""
Churn prediction using Random Forest classifier.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json


class ChurnPredictor:
    """Predict customer churn using machine learning."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = [
            'recency_days', 'frequency', 'monetary', 
            'tenure_months', 'avg_order_value'
        ]
        
    def load_data(self, file_path):
        """Load customer data."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} customers")
        print(f"Churned customers: {self.data['churned'].sum()}")
        return self.data
    
    def prepare_data(self):
        """Prepare features and target for training."""
        X = self.data[self.feature_columns]
        y = self.data['churned']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.y_train
    
    def train(self):
        """Train the churn prediction model."""
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate training accuracy
        train_score = self.model.score(self.X_train, self.y_train)
        print(f"\nTraining accuracy: {train_score:.2%}")
        
        return self.model
    
    def evaluate(self):
        """Evaluate model on test set."""
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"Test accuracy: {test_score:.2%}")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Not Churned', 'Churned']))
        
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        return test_score
    
    def predict_churn_probability(self, customer_data):
        """Predict churn probability for a customer."""
        # Convert to DataFrame if dict
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Select required features
        features = customer_data[self.feature_columns]
        
        # Get probability
        prob = self.model.predict_proba(features)[0][1]
        prediction = self.model.predict(features)[0]
        
        return {
            'churn_probability': round(prob, 3),
            'will_churn': bool(prediction),
            'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
        }
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_model(self, path='models/churn_model.json'):
        """Save model parameters."""
        import os
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'feature_columns': self.feature_columns,
            'feature_importance': self.get_feature_importance().to_dict('records')
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"\nModel metadata saved to {path}")


def main():
    """Test churn prediction model."""
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load and prepare data
    predictor.load_data('data/customer_data.csv')
    predictor.prepare_data()
    
    # Train model
    predictor.train()
    
    # Evaluate
    predictor.evaluate()
    
    # Feature importance
    print("\nFeature Importance:")
    print(predictor.get_feature_importance())
    
    # Test prediction
    print("\nTest Prediction:")
    test_customer = {
        'recency_days': 150,
        'frequency': 2,
        'monetary': 600,
        'tenure_months': 10,
        'avg_order_value': 300
    }
    
    result = predictor.predict_churn_probability(test_customer)
    print(f"Customer profile: Recency=150, Frequency=2, Monetary=600")
    print(f"Churn probability: {result['churn_probability']:.1%}")
    print(f"Risk level: {result['risk_level']}")
    
    # Save model
    predictor.save_model()


if __name__ == "__main__":
    main()
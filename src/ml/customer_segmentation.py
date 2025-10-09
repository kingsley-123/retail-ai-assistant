"""
Customer segmentation using RFM analysis and K-means clustering.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json


class CustomerSegmentation:
    """Segment customers based on RFM (Recency, Frequency, Monetary) analysis."""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.segment_labels = {
            0: "At Risk",
            1: "Loyal Customers", 
            2: "Champions"
        }
        
    def load_data(self, file_path):
        """Load customer data."""
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} customers")
        return self.data
    
    def prepare_features(self):
        """Prepare RFM features for clustering."""
        # Use Recency, Frequency, Monetary
        self.features = self.data[['recency_days', 'frequency', 'monetary']].copy()
        
        # Normalize features
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        print("Features prepared for clustering")
        return self.features_scaled
    
    def segment_customers(self):
        """Perform K-means clustering."""
        self.clusters = self.kmeans.fit_predict(self.features_scaled)
        self.data['segment'] = self.clusters
        self.data['segment_label'] = self.data['segment'].map(self.segment_labels)
        
        print(f"\nCustomer Segmentation Results:")
        print(self.data['segment_label'].value_counts())
        
        return self.data
    
    def get_segment_stats(self):
        """Get statistics for each segment."""
        stats = self.data.groupby('segment_label').agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        stats.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Customer Count']
        return stats
    
    def predict_segment(self, recency, frequency, monetary):
        """Predict segment for new customer."""
        features = np.array([[recency, frequency, monetary]])
        features_scaled = self.scaler.transform(features)
        segment = self.kmeans.predict(features_scaled)[0]
        
        return self.segment_labels[segment]
    
    def save_model(self, path='models/segmentation_model.json'):
        """Save model parameters."""
        import os
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'n_clusters': self.n_clusters,
            'cluster_centers': self.kmeans.cluster_centers_.tolist(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'segment_labels': self.segment_labels
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {path}")


def main():
    """Test customer segmentation."""
    # Initialize model
    model = CustomerSegmentation(n_clusters=3)
    
    # Load and process data
    model.load_data('data/customer_data.csv')
    model.prepare_features()
    model.segment_customers()
    
    # Get statistics
    print("\nSegment Statistics:")
    print(model.get_segment_stats())
    
    # Test prediction
    print("\nTest Prediction:")
    segment = model.predict_segment(recency=30, frequency=10, monetary=3000)
    print(f"Customer with Recency=30, Frequency=10, Monetary=3000 → {segment}")
    
    # Save model
    model.save_model()


if __name__ == "__main__":
    main()
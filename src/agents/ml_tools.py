"""
ML tools for the agent to use.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.customer_segmentation import CustomerSegmentation
from ml.churn_prediction import ChurnPredictor
from ml.sales_forecasting import SalesForecaster


class MLToolKit:
    """ML tools for business intelligence."""
    
    def __init__(self):
        """Initialize ML models."""
        print("Loading ML models...")
        
        # Customer segmentation
        self.segmenter = CustomerSegmentation(n_clusters=3)
        self.segmenter.load_data('data/customer_data.csv')
        self.segmenter.prepare_features()
        self.segmenter.segment_customers()
        
        # Churn prediction
        self.churn_predictor = ChurnPredictor()
        self.churn_predictor.load_data('data/customer_data.csv')
        self.churn_predictor.prepare_data()
        self.churn_predictor.train()
        
        # Sales forecasting
        self.forecaster = SalesForecaster()
        self.forecaster.load_data('data/sales_data.csv')
        self.forecaster.prepare_features()
        self.forecaster.train()
        
        print("ML models ready\n")
    
    def analyze_customer_segments(self):
        """Get customer segmentation analysis."""
        stats = self.segmenter.get_segment_stats()
        
        result = "Customer Segmentation Analysis:\n\n"
        for segment, row in stats.iterrows():
            result += f"{segment}:\n"
            result += f"  - Count: {int(row['Customer Count'])} customers\n"
            result += f"  - Avg Recency: {row['Avg Recency']:.0f} days\n"
            result += f"  - Avg Frequency: {row['Avg Frequency']:.1f} purchases\n"
            result += f"  - Avg Monetary: £{row['Avg Monetary']:,.0f}\n\n"
        
        return result
    
    def predict_customer_churn(self, recency, frequency, monetary, tenure, avg_order):
        """Predict churn for a customer."""
        customer_data = {
            'recency_days': recency,
            'frequency': frequency,
            'monetary': monetary,
            'tenure_months': tenure,
            'avg_order_value': avg_order
        }
        
        result = self.churn_predictor.predict_churn_probability(customer_data)
        
        output = f"Churn Prediction:\n"
        output += f"  - Churn Probability: {result['churn_probability']:.1%}\n"
        output += f"  - Risk Level: {result['risk_level']}\n"
        output += f"  - Will Churn: {'Yes' if result['will_churn'] else 'No'}\n"
        
        return output
    
    def forecast_sales(self, periods=3):
        """Forecast future sales."""
        forecast_df = self.forecaster.forecast(periods=periods)
        trend = self.forecaster.get_trend_analysis()
        
        result = f"Sales Forecast (Next {periods} Months):\n\n"
        for _, row in forecast_df.iterrows():
            result += f"  {row['date'].strftime('%Y-%m')}: £{row['forecasted_sales']:,.0f}\n"
        
        result += f"\nTrend Analysis:\n"
        result += f"  - Direction: {trend['trend_direction']}\n"
        result += f"  - Monthly Change: £{trend['monthly_change']:,.0f}\n"
        result += f"  - Growth Rate: {trend['overall_growth_rate']:.1f}%\n"
        
        return result


def main():
    """Test ML tools."""
    toolkit = MLToolKit()
    
    print("="*70)
    print("Testing ML Tools")
    print("="*70)
    
    # Test segmentation
    print("\n1. Customer Segmentation:")
    print(toolkit.analyze_customer_segments())
    
    # Test churn prediction
    print("\n2. Churn Prediction (High-risk customer):")
    print(toolkit.predict_customer_churn(
        recency=150, frequency=2, monetary=600, tenure=10, avg_order=300
    ))
    
    # Test forecasting
    print("\n3. Sales Forecast:")
    print(toolkit.forecast_sales(periods=3))


if __name__ == "__main__":
    main()
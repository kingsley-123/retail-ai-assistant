"""
Sales forecasting using time series analysis.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json


class SalesForecaster:
    """Forecast future sales using time series analysis."""
    
    def __init__(self):
        self.model = LinearRegression()
        
    def load_data(self, file_path):
        """Load sales data."""
        self.data = pd.read_csv(file_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date')
        
        print(f"Loaded {len(self.data)} months of sales data")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        return self.data
    
    def prepare_features(self):
        """Create features for forecasting."""
        # Create time-based features
        self.data['month_index'] = range(len(self.data))
        self.data['month'] = self.data['date'].dt.month
        
        # Calculate moving averages
        self.data['ma_3'] = self.data['sales_amount'].rolling(window=3).mean()
        
        print("\nSales Statistics:")
        print(f"Average monthly sales: £{self.data['sales_amount'].mean():,.0f}")
        print(f"Min sales: £{self.data['sales_amount'].min():,.0f}")
        print(f"Max sales: £{self.data['sales_amount'].max():,.0f}")
        
        return self.data
    
    def train(self, target_column='sales_amount'):
        """Train forecasting model."""
        # Use month_index as primary feature
        X = self.data[['month_index']].values
        y = self.data[target_column].values
        
        # Split: train on all but last 3 months
        split_point = len(X) - 3
        self.X_train = X[:split_point]
        self.y_train = y[:split_point]
        self.X_test = X[split_point:]
        self.y_test = y[split_point:]
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        print(f"\nModel trained on {len(self.X_train)} months")
        print(f"Test set: {len(self.X_test)} months")
        
        return self.model
    
    def evaluate(self):
        """Evaluate model performance."""
        # Predictions on test set
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        print("\nModel Performance:")
        print(f"Mean Absolute Error: £{mae:,.0f}")
        print(f"Root Mean Squared Error: £{rmse:,.0f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Show predictions vs actual
        print("\nTest Set Predictions:")
        for actual, pred in zip(self.y_test, y_pred):
            print(f"Actual: £{actual:,.0f} | Predicted: £{pred:,.0f}")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    def forecast(self, periods=3):
        """Forecast future sales."""
        # Get last month index
        last_index = self.data['month_index'].max()
        
        # Create future indices
        future_indices = np.array([[last_index + i + 1] for i in range(periods)])
        
        # Predict
        predictions = self.model.predict(future_indices)
        
        # Create forecast dataframe
        last_date = self.data['date'].max()
        future_dates = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecasted_sales': predictions.round(0)
        })
        
        return forecast_df
    
    def get_trend_analysis(self):
        """Analyze sales trend."""
        # Calculate growth rate
        recent_avg = self.data['sales_amount'].tail(3).mean()
        older_avg = self.data['sales_amount'].head(3).mean()
        growth_rate = ((recent_avg - older_avg) / older_avg) * 100
        
        # Get trend direction from model coefficient
        trend_coefficient = self.model.coef_[0]
        
        analysis = {
            'trend_direction': 'Upward' if trend_coefficient > 0 else 'Downward',
            'monthly_change': round(trend_coefficient, 2),
            'overall_growth_rate': round(growth_rate, 2),
            'recent_3mo_avg': round(recent_avg, 2)
        }
        
        return analysis
    
    def save_forecast(self, forecast_df, path='models/sales_forecast.json'):
        """Save forecast to file."""
        import os
        os.makedirs('models', exist_ok=True)
        
        forecast_data = {
            'forecast': forecast_df.to_dict('records'),
            'trend_analysis': self.get_trend_analysis(),
            'generated_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        
        # Convert date objects to strings
        for record in forecast_data['forecast']:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        with open(path, 'w') as f:
            json.dump(forecast_data, f, indent=2)
        
        print(f"\nForecast saved to {path}")


def main():
    """Test sales forecasting."""
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Load and prepare data
    forecaster.load_data('data/sales_data.csv')
    forecaster.prepare_features()
    
    # Train model
    forecaster.train()
    
    # Evaluate
    forecaster.evaluate()
    
    # Generate forecast
    print("\n" + "="*60)
    print("3-Month Sales Forecast:")
    print("="*60)
    forecast = forecaster.forecast(periods=3)
    print(forecast.to_string(index=False))
    
    # Trend analysis
    print("\n" + "="*60)
    print("Trend Analysis:")
    print("="*60)
    trend = forecaster.get_trend_analysis()
    for key, value in trend.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Save forecast
    forecaster.save_forecast(forecast)


if __name__ == "__main__":
    main()
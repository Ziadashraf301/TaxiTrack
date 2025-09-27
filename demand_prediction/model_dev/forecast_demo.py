import pandas as pd
from forecast import forecast_demand, TimeSeriesForecaster
from pathlib import Path
import sys
import logging 
from config import PATH_CONFIG 


# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(PATH_CONFIG["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'forecast_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# Demo function to test the forecasting
def demo_forecast_with_encoders():
    """Demo function to test forecasting with proper encoder loading"""

    print("🚀 Testing Forecast with Encoder Loading")
    
    
    # Create forecaster instance
    forecaster = TimeSeriesForecaster()
    
    # Define groups to forecast for
    groups = [
        {'pickup_zone': 'East Harlem North', 'pickup_borough': 'Manhattan', 'service_type': 'yellow_trip'},
        {'pickup_zone': 'Clinton East', 'pickup_borough': 'Manhattan', 'service_type': 'green_trip'},
    ]
    
    print("📊 Making forecast...")
    try:
        results = forecaster.forecast(groups, horizon_hours=12, model_name='xgboost')
        print(f"✅ Forecast successful! Generated {len(results)} predictions")
        print("\n📋 Sample predictions:")
        print(results.head())
        results.to_csv("results/forecast_results.csv", index=False)
        # Check if encoders were loaded properly
        if forecaster.encoders:
            print(f"\n🔧 Encoders loaded:")
            print(f"   Service columns: {len(forecaster.encoders.get('service_cols', []))}")
            print(f"   Borough columns: {len(forecaster.encoders.get('borough_cols', []))}")
            print(f"   Zone frequencies: {len(forecaster.encoders.get('zone_freq', {}))}")
        
        return results
        
    except Exception as e:
        print(f"❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_single_forecast():
    """Demo: Forecast for specific groups"""
    print("🚀 Starting Demand Forecast Demo")
    
    # Define groups to forecast for
    groups_to_forecast = [
        {'pickup_zone': 'Manhattan', 'pickup_borough': 'Manhattan', 'service_type': 'Yellow'},
        {'pickup_zone': 'Brooklyn', 'pickup_borough': 'Brooklyn', 'service_type': 'Green'},
        {'pickup_zone': 'Queens', 'pickup_borough': 'Queens', 'service_type': 'Yellow'},
    ]
    
    # Forecast next 24 hours using XGBoost
    print("📊 Forecasting next 24 hours...")
    results = forecast_demand(
        groups=groups_to_forecast,
        hours_ahead=24,
        model='decision_tree'
    )
    
    print(f"✅ Forecast completed! {len(results)} predictions generated")
    print("\n📋 Sample predictions:")
    print(results.head(10))
    
    # Save results
    results.to_csv('forecast_results.csv', index=False)
    print("💾 Results saved to forecast_results.csv")
    
    return results

def demo_multiple_models():
    """Demo: Compare multiple models"""
    print("\n🔍 Comparing Multiple Models")
    
    forecaster = TimeSeriesForecaster()
    
    groups_to_forecast = [
        {'pickup_zone': 'Manhattan', 'pickup_borough': 'Manhattan', 'service_type': 'Yellow'}
    ]
    
    # Forecast with all available models
    all_predictions = forecaster.forecast_multiple_models(
        groups=groups_to_forecast,
        horizon_hours=12
    )
    
    for model_name, predictions in all_predictions.items():
        avg_pred = predictions['prediction'].mean()
        print(f"📈 {model_name}: Average forecast = {avg_pred:.2f} trips/hour")
    
    return all_predictions

def demo_advanced_usage():
    """Demo: Advanced forecasting with custom parameters"""
    print("\n🎯 Advanced Forecasting Demo")
    
    forecaster = TimeSeriesForecaster()
    
    # Get available groups from historical data
    historical_groups = forecaster.get_latest_historical_data([], lookback_hours=24)
    unique_groups = historical_groups[['pickup_zone', 'pickup_borough', 'service_type']].drop_duplicates()
    
    print(f"📊 Found {len(unique_groups)} unique groups in recent data")
    
    # Forecast for first 5 groups
    groups_to_forecast = unique_groups.head(5).to_dict('records')
    
    results = forecaster.forecast(
        groups=groups_to_forecast,
        horizon_hours=48,
        model_name='decision_tree'
    )
    
    # Analyze results
    summary = results.groupby(['pickup_zone', 'service_type'])['prediction'].agg(['mean', 'max', 'min'])
    print("\n📈 Forecast Summary by Group:")
    print(summary)
    
    return results

if __name__ == "__main__":
    # Run demos
    results1 = demo_single_forecast()
    results2 = demo_multiple_models() 
    results3 = demo_advanced_usage()
    
    print("\n🎉 All demos completed successfully!")
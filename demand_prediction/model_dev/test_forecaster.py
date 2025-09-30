from forecast import TimeSeriesForecaster
from config import DATA_CONFIG
from datetime import datetime, timedelta

def run_forecast_pipeline():
    """Main sequence for running the forecast pipeline"""
    print("🚀 Starting Taxi Demand Forecast Pipeline...")
    
    try:
        # Step 1: Initialize forecaster
        print("1. Initializing TimeSeriesForecaster...")
        forecaster = TimeSeriesForecaster()
        # Note: _load_trained_artifacts() is now called inside forecast() method
        print("✅ Forecaster initialized successfully")
        
        # Step 2: Define forecast parameters
        print("2. Setting up forecast parameters...")
        groups = [{
            "pickup_zone": "SoHo", 
            "pickup_borough": "Manhattan", 
            "service_type": "yellow_trip"
        }]
        
        forecast_params = {
            "groups": groups,
            "horizon_hours": 12,
            "model_name": "decision_tree",  # Options: xgboost, decision_tree, random_forest, etc.
            "start_date": "2024-12-23",
            "end_date": "2024-12-30"
        }
        
        print(f"   - Groups: {len(groups)}")
        print(f"   - Horizon: {forecast_params['horizon_hours']} hours")
        print(f"   - Model: {forecast_params['model_name']}")
        print(f"   - Historical data: {forecast_params['start_date']} to {forecast_params['end_date']}")
        
        # Step 3: Run forecast
        print("3. Running forecast...")
        results = forecaster.forecast(**forecast_params)
        
        # Step 4: Validate results
        if results is None or results.empty:
            print("⚠️ No predictions generated!")
            return None
        
        # Get the correct timestamp column name from config
        timestamp_col = DATA_CONFIG["timestamp_col"]
        
        # Step 5: Display results summary
        print("4. Forecast Results:")
        print(f"   - Total predictions: {len(results)}")
        print(f"   - Date range: {results[timestamp_col].min()} to {results[timestamp_col].max()}")
        print(f"   - Average prediction: {results['prediction'].mean():.1f} trips")
        print(f"   - Min prediction: {results['prediction'].min()} trips")
        print(f"   - Max prediction: {results['prediction'].max()} trips")
        
        # Step 6: Show sample predictions
        print("\n5. Sample predictions:")
        display_cols = [timestamp_col, 'pickup_zone', 'prediction', 'prediction_lower', 'prediction_upper', 'model_used']
        print(results[display_cols].head(10).to_string(index=False))
        
        # Step 7: Show statistics by group
        print("\n6. Statistics by group:")
        group_stats = results.groupby(['pickup_zone', 'pickup_borough', 'service_type'])['prediction'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(2)
        print(group_stats)
        
        print("\n🎉 Forecast pipeline completed successfully!")
        print(f"📁 Results saved to: results/forecast_results.csv")
        
        return results
        
    except KeyError as e:
        print(f"❌ Configuration error: {e}")
        print("   Check that your config file has all required fields")
        import traceback
        traceback.print_exc()
        return None
        
    except ValueError as e:
        print(f"❌ Validation error: {e}")
        print("   Check model name and parameters")
        return None
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_multi_group_forecast():
    """Example with multiple groups"""
    print("🚀 Starting Multi-Group Forecast Pipeline...")
    
    forecaster = TimeSeriesForecaster()
    
    # Define multiple groups
    groups = [
        {"pickup_zone": "SoHo", "pickup_borough": "Manhattan", "service_type": "yellow_trip"},
        {"pickup_zone": "Times Square", "pickup_borough": "Manhattan", "service_type": "yellow_trip"},
        {"pickup_zone": "Williamsburg", "pickup_borough": "Brooklyn", "service_type": "green_trip"},
    ]
    
    try:
        results = forecaster.forecast(
            groups=groups,
            horizon_hours=24,
            model_name="decision_tree",
            start_date="2024-12-23",
            end_date="2024-12-30"
        )
        
        print(f"\n✅ Generated {len(results)} predictions for {len(groups)} groups")
        return results
        
    except Exception as e:
        print(f"❌ Multi-group forecast failed: {e}")
        return None


def run_forecast_with_validation():
    """Run forecast with artifact validation"""
    print("🚀 Starting Forecast with Validation...")
    
    try:
        forecaster = TimeSeriesForecaster()
        
        # Manually load artifacts to check what's available
        forecaster._load_trained_artifacts()
        
        available_models = list(forecaster.models.keys())
        print(f"✅ Available models: {available_models}")
        
        if not available_models:
            print("❌ No models found! Train models first.")
            return None
        
        # Use first available model
        model_to_use = available_models[0]
        print(f"📊 Using model: {model_to_use}")
        
        groups = [{
            "pickup_zone": "SoHo", 
            "pickup_borough": "Manhattan", 
            "service_type": "yellow_trip"
        }]
        
        results = forecaster.forecast(
            groups=groups,
            horizon_hours=12,
            model_name=model_to_use,
            start_date="2024-12-23",
            end_date="2024-12-30"
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Choose which function to run:
    
    # Option 1: Single group forecast
    results = run_forecast_pipeline()
    
    # Option 2: Multiple groups (uncomment to use)
    # results = run_multi_group_forecast()
    
    # Option 3: With validation (uncomment to use)
    # results = run_forecast_with_validation()
    
    if results is not None:
        print(f"\n📊 Final results shape: {results.shape}")
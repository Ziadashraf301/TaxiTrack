import logging
import sys
import os
from pathlib import Path
from venv import logger
from config import CLICKHOUSE_CONFIG, PATH_CONFIG

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import TimeSeriesForecastPipeline
from config import CLICKHOUSE_CONFIG


# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(PATH_CONFIG["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def test_data_loading():
    """Test data loading from ClickHouse"""
    print("🧪 Testing Data Loading...")
    
    try:
        pipeline = TimeSeriesForecastPipeline()
        
        # Test with a small date range first
        df = pipeline.data_loader.get_processed_data(
            table_name=CLICKHOUSE_CONFIG["default_table"],
            start_date="2019-01-01",  # Adjust based on your data
            end_date="2019-08-01",     # Just 1 week for testing
            train= False
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")
        print(f"   Unique groups: {df[['pickup_zone', 'pickup_borough', 'service_type']].drop_duplicates().shape[0]}")
        
        return df
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def test_feature_engineering(df):
    """Test feature engineering"""
    print("\n🧪 Testing Feature Engineering...")
    
    try:
        pipeline = TimeSeriesForecastPipeline()
        df_features = pipeline.feature_engineer.transform(df)
        
        print(f"✅ Feature engineering successful!")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   After feature engineering: {len(df_features.columns)}")
        print(f"   New features: {[col for col in df_features.columns if col not in df.columns][:10]}...")  # First 10 new features
        
        # Check for NaN values
        nan_counts = df_features.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"   ⚠️  Columns with NaN: {len(nan_cols)}")
        else:
            print(f"   ✅ No NaN values in features")
            
        return df_features
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None

def test_small_pipeline():
    """Test the complete pipeline with a small dataset"""
    print("\n🧪 Testing Complete Pipeline (Small Scale)...")
    
    try:
        pipeline = TimeSeriesForecastPipeline()
        
        # Run pipeline with small date range
        pipeline.run_pipeline(
            table_name=CLICKHOUSE_CONFIG["default_table"],
            start_date="2019-01-01",
            end_date="2025-01-01", 
            train= False
        )
        
        print(f"✅ Pipeline completed successfully!")
        
        # Print results summary
        if hasattr(pipeline, 'results'):
            logger.info("\n=== MODEL RESULTS SUMMARY ===")
            for model_name, results in pipeline.results.items():
                logger.info(f"{model_name.upper():<15}: "
                          f"Train RMSE: {results['train_rmse']:.2f}, "
                          f"Test RMSE: {results['test_rmse']:.2f}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return None

def check_artifacts():
    """Check if pipeline artifacts were created"""
    print("\n🔍 Checking Generated Artifacts...")
    
    from config import PATH_CONFIG
    import joblib
    
    artifacts_path = Path(PATH_CONFIG["models_dir"]) / PATH_CONFIG["pipeline_artifacts"]
    
    if artifacts_path.exists():
        artifacts = joblib.load(artifacts_path)
        print("✅ Pipeline artifacts found!")
        print(f"   Feature names: {len(artifacts.get('feature_names', []))}")
        print(f"   Models trained: {list(artifacts.get('results', {}).keys())}")
        return True
    else:
        print("❌ No artifacts found")
        return False

def main():
    """Run all tests"""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting Pipeline Tests...")
    print("=" * 50)
    
    # # Test 1: Data Loading
    # df = test_data_loading()
    # if df is None:
    #     print("❌ Stopping tests - data loading failed")
    #     return
    
    # # Test 2: Feature Engineering
    # df_features = test_feature_engineering(df)
    # if df_features is None:
    #     print("❌ Stopping tests - feature engineering failed")
    #     return
    
    # Test 3: Complete Pipeline (small scale)
    pipeline = test_small_pipeline()
    
    # Test 4: Check Artifacts
    artifacts_exist = check_artifacts()
    
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY:")
    # print(f"   Data Loading: {'✅' if df is not None else '❌'}")
    # print(f"   Feature Engineering: {'✅' if df_features is not None else '❌'}")
    print(f"   Pipeline Execution: {'✅' if pipeline is not None else '❌'}")
    print(f"   Artifacts Created: {'✅' if artifacts_exist else '❌'}")
    #df is not None, df_features is not None, pipeline is not None, artifacts_exist
    if all([pipeline is not None, artifacts_exist]):
        print("\n🎉 ALL TESTS PASSED! You can now run the full pipeline.")
        print("\nNext steps:")
        print("1. Run the full pipeline with your complete date range")
        print("2. Check the results in the 'results/' folder")
        print("3. Start building the Streamlit app")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
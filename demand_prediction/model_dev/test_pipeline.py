import logging
import sys
import os
from pathlib import Path
from config import PATH_CONFIG, TABLE_CONFIG
from pipeline import TimeSeriesForecastPipeline
from logger import setup_logger

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")


def test_data_loading():
    """Test data loading from ClickHouse"""
    logger.info("🧪 Testing Data Loading...")
    
    try:
        pipeline = TimeSeriesForecastPipeline()
        
        # Test with a small date range first
        df = pipeline.data_loader.get_processed_data(
            table_name=TABLE_CONFIG["table"],
            start_date="2024-01-01",
            end_date="2024-06-31"
        )
        
        logger.info(f"✅ Data loaded successfully: {len(df)} rows, "
                    f"{len(df.columns)} columns, "
                    f"date range {df['pickup_datetime'].min()} → {df['pickup_datetime'].max()}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}", exc_info=True)
        return None


def test_feature_engineering(df):
    """Test feature engineering"""
    logger.info("🧪 Testing Feature Engineering...")
    
    try:
        pipeline = TimeSeriesForecastPipeline()
        df_features = pipeline.feature_engineer.transform(df)
        
        logger.info(f"✅ Feature engineering successful: "
                    f"{len(df.columns)} → {len(df_features.columns)} features")
        return df_features
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}", exc_info=True)
        return None


def test_small_pipeline():
    """Test the complete pipeline with a small dataset"""
    logger.info("🧪 Testing Complete Pipeline (Small Scale)...")
    
    try:
        pipeline = TimeSeriesForecastPipeline()
        
        pipeline.run_pipeline(
            table_name=TABLE_CONFIG["table"],
            start_date="2024-07-01",
            end_date="2025-01-01",
        )
        
        logger.info("✅ Pipeline completed successfully!")
        
        if hasattr(pipeline, 'results'):
            logger.info("=== MODEL RESULTS SUMMARY ===")
            for model_name, results in pipeline.results.items():
                logger.info(f"{model_name.upper():<15}: "
                            f"Train RMSE: {results['train_rmse']:.2f}, "
                            f"Test RMSE: {results['test_rmse']:.2f}")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return None


def main():
    """Run all tests"""
    logger.info("🚀 Starting Pipeline Tests...")
    logger.info("=" * 50)
    
    # df = test_data_loading()
    # if df is None:
    #     logger.error("❌ Stopping tests - data loading failed")
    #     return
    
    # df_features = test_feature_engineering(df)
    # if df_features is None:
    #     logger.error("❌ Stopping tests - feature engineering failed")
    #     return
    
    pipeline = test_small_pipeline()
    
    logger.info("=" * 50)
    if pipeline is not None:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.warning("⚠️ Some tests failed. Check logs above.")


if __name__ == "__main__":
    main()

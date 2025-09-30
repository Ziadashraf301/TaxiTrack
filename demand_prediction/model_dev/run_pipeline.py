import sys
from pathlib import Path
from config import PATH_CONFIG, TABLE_CONFIG
from pipeline import TimeSeriesForecastPipeline
from logger import setup_logger

# Setup logger
log_dir = Path(PATH_CONFIG["logs_dir"])
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")


def run_training():
    """Run the full training pipeline in production mode"""
    logger.info("🚀 Starting Production Training Pipeline...")
    logger.info("=" * 60)

    try:
        # Initialize pipeline
        pipeline = TimeSeriesForecastPipeline()

        # Run pipeline on configured date range
        pipeline.run_pipeline(
            table_name=TABLE_CONFIG["table"],
            start_date=TABLE_CONFIG.get("start_date", "2019-01-01"),
            end_date=TABLE_CONFIG.get("end_date", "2025-08-31"),
        )

        logger.info("✅ Pipeline completed successfully!")

        # Log results summary
        if hasattr(pipeline, "results"):
            logger.info("=== MODEL RESULTS SUMMARY ===")
            for model_name, results in pipeline.results.items():
                logger.info(
                    f"{model_name.upper():<15}: "
                    f"Month={results['month']} | "
                    f"Train RMSE={results['train_rmse']:.2f} | "
                    f"Test RMSE={results['test_rmse']:.2f} | "
                    f"Test Size={results['test_size']}"
                )
                
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🎉 Training job finished successfully!")


if __name__ == "__main__":
    run_training()
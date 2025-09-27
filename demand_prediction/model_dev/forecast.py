import pandas as pd 
import numpy as np 
import joblib 
import logging 
from datetime import datetime, timedelta 
from data_loader import ClickHouseDataLoader 
from feature_engineer import TimeSeriesFeatureEngineer 
from config import DATA_CONFIG, PATH_CONFIG, CLICKHOUSE_CONFIG 
from pathlib import Path
import sys

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

class TimeSeriesForecaster: 
    def __init__(self): 
        self.config = DATA_CONFIG
        self.path_config = PATH_CONFIG 
        self.CLICKHOUSE_CONFIG = CLICKHOUSE_CONFIG
        self.data_loader = ClickHouseDataLoader() 
        self.feature_engineer = TimeSeriesFeatureEngineer() 
        self.feature_engineer.Training = False
        self.models = {} 
        self.pipeline_artifacts = None 
        self.encoders = {}  # Store loaded encoders
        # Load trained models and artifacts
        self._load_trained_artifacts() 

    def _load_trained_artifacts(self): 
        """Load trained models and pipeline artifacts""" 
        try: 
            # Load pipeline artifacts
            artifacts_path = f"airflow/{self.path_config['models_dir']}/pipeline_artifacts.pkl" 
            self.pipeline_artifacts = joblib.load(artifacts_path) 
            logger.info("✅ Pipeline artifacts loaded successfully")
            # Load individual models
            model_names = list(self.pipeline_artifacts['results'].keys()) 
            for model_name in model_names: 
                model_path = f"airflow/{self.path_config['models_dir']}/{model_name}_model.pkl" 
                self.models[model_name] = joblib.load(model_path) 
                logger.info(f"✅ {model_name} model loaded successfully") 
        except Exception as e: 
            logger.error(f"❌ Error loading artifacts: {e}") 
            raise

    # ---------------------------------------------------------
    # 1️⃣ Data Handling
    # ---------------------------------------------------------
    def get_latest_historical_data(self,groups = None, start_date= None ,end_date= None, lookback_days=7):
        """Fetch historical data (last N days) from ClickHouse"""
        loader = self.data_loader

        # end_date = datetime(2021, 1, 1).date()  # fixed end of training
        # start_date = end_date - timedelta(days=lookback_days)

        df = loader.get_processed_data(
            start_date=start_date,
            end_date=end_date,
            table_name=self.CLICKHOUSE_CONFIG["default_table"],
            groups = groups,
            calculate_zones_freq = False,
            train = False
        )
        return df


    # ---------------------------------------------------------
    # 3️⃣ Prepare feature matrix for prediction
    # ---------------------------------------------------------
    def prepare_feature_matrix(self, future_df):
        """Ensure features match training format"""
        # Get expected features from training
        expected_features = self.pipeline_artifacts.get("feature_names", [])
        for feat in expected_features:
            if feat not in future_df.columns:
                logger.warning(f"⚠️ Missing feature in future data: {feat}")
                future_df[feat] = 0

        X_pred = future_df[expected_features].copy()

        # Ensure numeric
        for col in X_pred.columns:
            if not pd.api.types.is_numeric_dtype(X_pred[col]):
                logger.warning(f"⚠️ Non-numeric feature converted to numeric: {col}")
                X_pred[col] = pd.to_numeric(X_pred[col], errors="coerce")

        X_pred = X_pred.backfill().ffill()
        return X_pred

    # ---------------------------------------------------------
    # 4️⃣ Forecasting
    # ---------------------------------------------------------
    def forecast(self, groups= None, horizon_hours=24, model_name="decision_tree", start_date=None, end_date=None):
        """Generate forecasts using consistent training transformations"""
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} not found. Available: {list(self.models.keys())}"
            )

        logger.info("📊 Fetching historical data...")
        historical_df = self.get_latest_historical_data(groups = groups, start_date = start_date, end_date = end_date)

        if historical_df.empty:
            logger.warning("⚠️ No historical data found, forecasts may be unreliable.")
            last_timestamp = datetime(2020, 4, 30, 23, 0, 0)
        else:
            last_timestamp = historical_df[self.config["timestamp_col"]].max()

        # 1️⃣ Build future timestamps
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=horizon_hours,
            freq="h"
        )

        # 2️⃣ Create future rows
        future_data = []
        for ts in future_timestamps:
            for g in groups:
                future_data.append({
                    self.config["timestamp_col"]: ts,
                    "pickup_zone": g["pickup_zone"],
                    "pickup_borough": g["pickup_borough"],
                    "service_type": g["service_type"],
                    self.config["target_col"]: np.nan
                })
        future_df = pd.DataFrame(future_data)

        # 3️⃣ Combine historical + future
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)

        # 4️⃣ Apply feature engineering (same as training)
        transformed = self.feature_engineer.transform(combined_df)

        # 6️⃣ Keep only future rows
        mask_future = transformed[self.config["timestamp_col"]] > last_timestamp
        X_pred = self.prepare_feature_matrix(transformed[mask_future])

        # 7️⃣ Predictions
        model = self.models[model_name]
        preds = model.predict(X_pred)
        # Round to nearest integer and ensure non-negative
        preds = np.rint(preds).astype(int)
        preds = np.maximum(preds, 0)  # avoid negative trips

        # 8️⃣ Assemble results
        results = transformed.loc[mask_future, [
            self.config["timestamp_col"],
            "pickup_zone",
            "pickup_borough",
            "service_type"
        ]].copy()
        results["prediction"] = preds
        results["model_used"] = model_name
        results["prediction_timestamp"] = datetime.now()
        results["prediction_lower"] = results["prediction"] * 0.8
        results["prediction_upper"] = results["prediction"] * 1.2

        results.to_csv("results/forecast_results.csv", index=False)

        logger.info(f"✅ Forecast done for {len(results)} rows")
        return results

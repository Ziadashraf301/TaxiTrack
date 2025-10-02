import os
import pandas as pd 
import numpy as np 
import joblib 
from datetime import datetime, timedelta 
from data_loader import ClickHouseDataLoader 
from feature_engineer import TimeSeriesFeatureEngineer 
from encoder import TimeSeriesEncoder
from config import DATA_CONFIG, PATH_CONFIG, CLICKHOUSE_CONFIG, TABLE_CONFIG
from pathlib import Path
from logger import setup_logger

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/forecast_pipeline.log")

class TimeSeriesForecaster: 
    def __init__(self): 
        self.config = DATA_CONFIG
        self.path_config = PATH_CONFIG 
        self.CLICKHOUSE_CONFIG = CLICKHOUSE_CONFIG
        self.data_loader = ClickHouseDataLoader()
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.encoder = TimeSeriesEncoder()
        self.data_loader.Training = False
        self.feature_engineer.Training = False 
        self.feature_engineer.Split = False
        self.encoder.Training = False
        self.models = {} 
        self.pipeline_artifacts = None 
        self.encoders = {}
        self._is_loaded = False
        
    def _load_trained_artifacts(self): 
        """Load trained models and pipeline artifacts - only once""" 
        if self._is_loaded:
            logger.info("✅ Artifacts already loaded")
            return
            
        try: 
            artifacts_path = f"{self.path_config['models_dir']}/pipeline_artifacts.pkl" 
            self.pipeline_artifacts = joblib.load(artifacts_path) 
            logger.info("✅ Pipeline artifacts loaded successfully")
            
            model_names = list(self.pipeline_artifacts['results'].keys()) 
            for model_name in model_names: 
                model_path = f"{self.path_config['models_dir']}/{model_name}_model.pkl" 
                self.models[model_name] = joblib.load(model_path) 
                logger.info(f"✅ {model_name} model loaded successfully") 
                
            self._is_loaded = True
            
        except Exception as e: 
            logger.error(f"❌ Error loading artifacts: {e}") 
            raise

    def get_latest_historical_data(self, groups=None, start_date=None, end_date=None):
        """Fetch historical data from ClickHouse"""
        return self.data_loader.get_processed_data(
            start_date=start_date,
            end_date=end_date,
            table_name=TABLE_CONFIG["table"],
            groups=groups
        )

    def prepare_feature_matrix(self, future_df):
        """Ensure features match training format"""
        expected_features = self.pipeline_artifacts.get("feature_names", [])
        
        # Add missing features with appropriate defaults
        for feat in expected_features:
            if feat not in future_df.columns:
                logger.warning(f"⚠️ Missing feature: {feat}, adding with default value")
                if 'lag' in feat or 'rolling' in feat:
                    # For temporal features, use the mean from historical data
                    future_df[feat] = future_df.get(self.config["target_col"], pd.Series([0])).mean()
                else:
                    future_df[feat] = 0

        X_pred = future_df[expected_features].copy()

        return X_pred

    def _generate_predictions(self, model, feature_matrix):
        """Generate predictions from model"""
        try:
            preds = model.predict(feature_matrix)
            # Round to nearest integer and ensure non-negative
            preds = np.rint(preds).astype(int)
            return np.maximum(preds, 0)
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

    def _create_future_dataframe_with_features(self, groups, last_timestamp, horizon_hours):
        """Create future timestamps with proper time features pre-computed"""
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=horizon_hours,
            freq="h"
        )

        future_data = []
        for ts in future_timestamps:
            for g in groups:
                # Pre-compute basic time features that the feature engineer expects
                future_data.append({
                    self.config["timestamp_col"]: ts,
                    "pickup_zone": g["pickup_zone"],
                    "pickup_borough": g["pickup_borough"],
                    "service_type": g["service_type"],
                    self.config["target_col"]: np.nan,
                    "pickup_date": ts.date(),  # Add date
                    "pickup_hour": ts.hour,    # Add hour
                })
        
        future_df = pd.DataFrame(future_data)
    
        # Ensure proper datetime type
        future_df[self.config["timestamp_col"]] = pd.to_datetime(future_df[self.config["timestamp_col"]])
        
        return future_df

    def _prepare_single_result(self, prediction_row, prediction, model_name, step):
        """Prepare result for a single prediction step"""
        result_row = prediction_row[[
            self.config["timestamp_col"],
            "pickup_zone",
            "pickup_borough", 
            "service_type"
        ]].copy()
        
        result_row["prediction"] = prediction
        result_row["model_used"] = model_name
        result_row["prediction_timestamp"] = datetime.now()
        result_row["prediction_lower"] = int(prediction * 0.8)
        result_row["prediction_upper"] = int(prediction * 1.2)
        result_row["forecast_step"] = step + 1
        
        return result_row

    def _get_last_timestamp(self, historical_df):
        """Get the last timestamp from historical data"""
        if historical_df.empty:
            logger.warning("⚠️ No historical data found, using default timestamp")
            return datetime(2020, 4, 30, 23, 0, 0)
        return historical_df[self.config["timestamp_col"]].max()

    def _save_results(self, results):
        """Save forecast results"""
        os.makedirs("results", exist_ok=True)
        results.to_csv("results/forecast_results.csv", index=False)
        logger.info("💾 Results saved to results/forecast_results.csv")

    def _forecast_single_group(self, group_dict, group_id, historical_group_df, horizon_hours, model):
        """Forecast for a single group iteratively"""
        group_predictions = []
        
        # Get last timestamp for this group
        if historical_group_df.empty:
            last_timestamp = datetime(2025, 8, 31, 23, 0, 0)
        else:
            last_timestamp = historical_group_df[self.config["timestamp_col"]].max()
        
        # Create working copy of historical data
        working_df = historical_group_df.copy()
        
        # Generate future timestamps
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=horizon_hours,
            freq="h"
        )
        
        for step, ts in enumerate(future_timestamps):
            # Create next timestep row
            next_row = pd.DataFrame([{
                self.config["timestamp_col"]: ts,
                self.config["group_col"]: group_id,
                self.config["target_col"]: np.nan
            }])
            
            # Combine with historical data for feature engineering
            temp_df = pd.concat([working_df, next_row], ignore_index=True)
            
            # Apply feature engineering (only transform, no fit)
            try:
                X, y, timestamps = self.feature_engineer.transform(temp_df)

                transformed = self.encoder.transform(X)

            except Exception as e:
                logger.error(f"❌ Feature engineering failed for {group_dict} at step {step}: {e}")
                # Use fallback prediction
                prediction = int(working_df[self.config["target_col"]].tail(24).mean()) if len(working_df) > 0 else 0
            else:
                # Get the last row (which is our prediction row)
                prediction_row = transformed.iloc[[-1]].copy()
                
                # Prepare features
                X_pred = self.prepare_feature_matrix(prediction_row)
                
                # Generate prediction
                prediction = self._generate_predictions(model, X_pred)[0]
            
            # Store prediction
            result_data = {
                self.config["timestamp_col"]: ts,
                "pickup_zone": group_dict["pickup_zone"],
                "pickup_borough": group_dict["pickup_borough"],
                "service_type": group_dict["service_type"],
                "prediction": prediction,
                "model_used": model.__class__.__name__,
                "prediction_timestamp": datetime.now(),
                "prediction_lower": int(prediction * 0.8),
                "prediction_upper": int(prediction * 1.2),
                "forecast_step": step + 1
            }
            group_predictions.append(result_data)
            
            # Update working dataframe with prediction
            next_row[self.config["target_col"]] = prediction
            working_df = pd.concat([working_df, next_row], ignore_index=True)
            
            logger.debug(f"  Step {step + 1}: Predicted {prediction} trips at {ts}")
        
        return pd.DataFrame(group_predictions)

    def forecast(self, groups=None, horizon_hours=24, model_name="decision_tree", start_date=None, end_date=None):
        """Main forecast method with iterative prediction per group"""
        
        # Load artifacts if not already loaded
        self._load_trained_artifacts()
        
        # Validation
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")

        logger.info(f"📊 Starting iterative forecast pipeline for {len(groups)} groups...")
        
        # Get historical data
        historical_df = self.get_latest_historical_data(groups=groups, start_date=start_date, end_date=end_date)
        
        if historical_df.empty:
            logger.warning("⚠️ No historical data found!")
        
        model = self.models[model_name]
        all_predictions = []
        
        # Forecast each group independently
        for i, group_dict in enumerate(groups):
            logger.info(f"🔮 Forecasting group {i+1}/{len(groups)}: {group_dict}")
            
            group_col = DATA_CONFIG["group_col"]

            # Build group_id from dict (zone_borough_service)
            group_id = f"{group_dict['pickup_zone']}_{group_dict['pickup_borough']}_{group_dict['service_type']}".strip()
            group_id = group_id.replace(" ", "_").replace("/", "_")

            # Filter historical data by group_id
            historical_group_df = historical_df[historical_df[group_col] == group_id].copy()
            
            # Forecast for this group
            group_predictions = self._forecast_single_group(
                group_dict, 
                group_id,
                historical_group_df, 
                horizon_hours, 
                model
            )
            
            all_predictions.append(group_predictions)
            logger.info(f"✅ Group {i+1} completed: {len(group_predictions)} predictions")
        
        # Combine all predictions
        results = pd.concat(all_predictions, ignore_index=True)
        
        # Save and return
        self._save_results(results)
        logger.info(f"✅ Iterative forecast completed: {len(results)} total predictions")
        
        return results
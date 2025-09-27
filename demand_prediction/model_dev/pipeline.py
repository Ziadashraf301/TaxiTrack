import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.metrics import mean_squared_error
from data_loader import ClickHouseDataLoader
from feature_engineer import TimeSeriesFeatureEngineer
from config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG
from pathlib import Path
import sys
import os
from xgboost import XGBRegressor


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

setup_logging()
logger = logging.getLogger(__name__)

class TimeSeriesForecastPipeline:
    def __init__(self):
        self.config = DATA_CONFIG
        self.model_config = MODEL_CONFIG
        self.path_config = PATH_CONFIG
        
        self.data_loader = ClickHouseDataLoader()
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.models = {}
        self.feature_names = []
        
    def run_pipeline(self, table_name, start_date=None, end_date=None, train= False):
        """Run complete pipeline with simple train-test split"""
        logger.info("Starting time series forecasting pipeline...")
        
        # 1. Load and preprocess data
        df = self.data_loader.get_processed_data(table_name, start_date, end_date, train = train)
        logger.info(f"Data loaded: {len(df)} rows")
        logger.info(f"Date range: {df[self.config['timestamp_col']].min()} to {df[self.config['timestamp_col']].max()}")
        
        # 2. Feature engineering
        df_features = self.feature_engineer.transform(df)
        logger.info(f"Feature engineering completed. Shape: {df_features.shape}")
        
        # 3. Prepare data for modeling
        X, y, timestamps, groups = self._prepare_features_target(df_features)
        
        # 4. Time-based split ensuring equal time periods for all groups
        X_train, X_test, y_train, y_test, train_times, test_times = self._group_aware_time_split(X, y, timestamps, groups)
        
        # 5. Train models
        self._train_models_simple(X_train, X_test, y_train, y_test, train_times, test_times)
        
        # 6. Save pipeline artifacts
        self._save_pipeline_artifacts(X_train.columns.tolist())

        # 7. Save feature_importance
        self.save_feature_importance()
        
        logger.info("Pipeline completed successfully!")
        
    def _prepare_features_target(self, df):
        """Prepare features and target variable - NO NaN HANDLING"""
        timestamp_col = self.config["timestamp_col"]
        target_col = self.config["target_col"]
        group_cols = self.config["group_cols"]
        
        # Remove rows where target is NaN (let it fail if there are NaNs)
        df_clean = df.dropna(subset=[target_col])
        
        # Feature columns - exclude timestamp, target, and group columns
        exclude_cols = [timestamp_col, target_col, "pickup_date", "pickup_hour"] + group_cols
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        # Get features, target, timestamps, and groups
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        timestamps = df_clean[timestamp_col]
        groups = df_clean[group_cols].astype(str).agg('_'.join, axis=1)
        
        # Ensure all arrays have the same length
        assert len(X) == len(y) == len(timestamps) == len(groups), \
            f"Length mismatch: X={len(X)}, y={len(y)}, timestamps={len(timestamps)}, groups={len(groups)}"
        
        logger.info(f"Final dataset - X: {X.shape}, y: {len(y)}, timestamps: {len(timestamps)}, groups: {groups.nunique()}")
        
        return X, y, timestamps, groups
    
    def _group_aware_time_split(self, X, y, timestamps, groups):
        """Split each month: train on all but last 3 months, test on last 3 months"""
        # Ensure timestamps are datetime
        timestamps = pd.to_datetime(timestamps)

        # Get month boundaries
        month_end = timestamps.max().normalize()

        # Cutoff = last 3 months
        cutoff_date = month_end - pd.DateOffset(months=6)
        # Train = everything before or equal cutoff
        train_mask = timestamps <= cutoff_date
        test_mask = timestamps > cutoff_date

        # Apply masks
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_times, test_times = timestamps[train_mask], timestamps[test_mask]
        train_groups, test_groups = groups[train_mask], groups[test_mask]

        # Logging
        logger.info(f"Train set: {len(X_train)} samples ({train_times.min()} → {train_times.max()})")
        logger.info(f"Test set: {len(X_test)} samples ({test_times.min()} → {test_times.max()})")
        logger.info(f"Train groups: {train_groups.nunique()}, Test groups: {test_groups.nunique()}")

        return X_train, X_test, y_train, y_test, train_times, test_times

    
    def _train_models_simple(self, X_train, X_test, y_train, y_test, train_times=None, test_times=None):
        """Train models with simple train-test split, evaluate, 
        then retrain incrementally on full month data."""

        self.models = {
            'xgboost': XGBRegressor(**self.model_config["models"]["xgboost"])
        }

        results = {}

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            try:
                model_path = f"{self.path_config['models_dir']}/{model_name}_model.pkl"

                # --- 1) Train on train split ---
                model.fit(X_train, y_train)

                # --- 2) Evaluate on test split ---
                y_pred = model.predict(X_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # --- 2.5) Record time period ---
                train_start = str(train_times.min().date()) if train_times is not None else None
                train_end   = str(train_times.max().date()) if train_times is not None else None
                test_start  = str(test_times.min().date()) if test_times is not None else None
                test_end    = str(test_times.max().date()) if test_times is not None else None
                month_id    = pd.to_datetime(train_times.min()).strftime("%Y-%m") if train_times is not None else None

                results[model_name] = {
                    'month': month_id,
                    'train_period': f"{train_start} → {train_end}",
                    'test_period': f"{test_start} → {test_end}",
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_size': len(y_test)
                }
                logger.info(f"{model_name} [{month_id}] Train RMSE: {train_rmse:.4f} - Test RMSE: {test_rmse:.4f}")

                # --- 3) Retrain on full month (train + test) ---
                X_full = pd.concat([X_train, X_test])
                y_full = pd.concat([y_train, y_test])

                logger.info(f"{model_name} [{month_id}] Train Full Date")
                model.fit(X_full, y_full)

                # --- 4) Save updated model for next month ---
                joblib.dump(model, model_path)
                logger.info(f"💾 Saved {model_name} model to {model_path}")

            except Exception as e:
                logger.error(f"{model_name} training failed: {e}")
                raise

        self.results = results
        return results

    
    def _save_pipeline_artifacts(self, feature_names):
        """Save pipeline artifacts and results"""
        # Create directories if they don't exist
        os.makedirs(self.path_config["models_dir"], exist_ok=True)
        os.makedirs(self.path_config["results_dir"], exist_ok=True)

        # Save pipeline artifacts
        pipeline_artifacts = {
            'feature_names': feature_names,
            'config': self.config,
            'model_config': self.model_config,
            'results': self.results if hasattr(self, 'results') else {}
        }
        joblib.dump(pipeline_artifacts, f"{self.path_config['models_dir']}/pipeline_artifacts.pkl")

        # Save results to CSV (append mode with header only if file doesn't exist)
        if hasattr(self, 'results') and self.results:
            results_df = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
            results_df.rename(columns={'index': 'model_name'}, inplace=True)

            results_path = f"{self.path_config['results_dir']}/model_results.csv"
            file_exists = os.path.isfile(results_path)

            results_df.to_csv(
                results_path,
                mode='a' if file_exists else 'w',   # append if file exists
                header=not file_exists,             # only write header if new file
                index=False
            )

            logger.info(f"📊 Results appended to {results_path}")

        logger.info("✅ Pipeline artifacts saved successfully")




    def save_feature_importance(self, model_name="xgboost"):
        """
        Save feature importances of the trained XGBoost model to CSV,
        overwrite monthly, and also log the results.
        """
        try:
            # Build model path
            model_path = os.path.join(self.path_config["models_dir"], f"{model_name}_model.pkl")

            # Load model
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Model file not found at {model_path}")
                return None

            model = joblib.load(model_path)

            # Ensure it's XGBoost with importances
            if not hasattr(model, "feature_importances_"):
                logger.warning(f"⚠️ Model {model_name} does not have feature_importances_ attribute.")
                return None

            # Extract importances
            importances = model.feature_importances_
            feature_names = (
                model.get_booster().feature_names
                if hasattr(model, "get_booster") and model.get_booster().feature_names is not None
                else [f"feature_{i}" for i in range(len(importances))]
            )

            # Create DataFrame
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            # Ensure results dir exists
            results_dir = os.path.join(self.path_config["results_dir"], "feature_importances")
            os.makedirs(results_dir, exist_ok=True)

            # Use current month as filename → overrides each month
            csv_path = os.path.join(results_dir, f"feature_importances.csv")

            # Save CSV (overwrite each month)
            importance_df.to_csv(csv_path, index=False)

            logger.info(f"✅ Feature importances saved to {csv_path}")
            return importance_df

        except Exception as e:
            logger.error(f"❌ Failed to save feature importances: {e}", exc_info=True)
            return None

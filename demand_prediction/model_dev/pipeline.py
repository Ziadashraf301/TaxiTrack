import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from data_loader import ClickHouseDataLoader
from feature_engineer import TimeSeriesFeatureEngineer
from encoder import TimeSeriesEncoder
from config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG
from pathlib import Path
import os
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import lightgbm as lgb
import gc
from logger import setup_logger
from datetime import datetime

log_dir = Path(PATH_CONFIG["logs_dir"])
logger = setup_logger(__name__, f"{log_dir}/pipeline.log")

class TimeSeriesForecastPipeline:
    def __init__(self):
        self.config = DATA_CONFIG
        self.model_config = MODEL_CONFIG
        self.path_config = PATH_CONFIG
        
        self.data_loader = ClickHouseDataLoader()
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.encoder = TimeSeriesEncoder()
        self.feature_engineer.Training = True
        self.feature_engineer.Split = True
        self.encoder.Training = True

        self.models = {}
        self.feature_names = []
        
    def run_pipeline(self, table_name, start_date=None, end_date=None):
        """Run complete pipeline with simple train-test split"""
        logger.info("🚀 Starting time series forecasting pipeline...")
        
        try:
            # 1. Load and preprocess data
            df = self.data_loader.get_processed_data(table_name, start_date, end_date)
        except Exception as e:
            logger.error(f"❌ Data loading failed for {table_name}: {e}", exc_info=True)
            raise

        if df.empty:
            logger.error("❌ Loaded dataframe is empty, aborting pipeline!")
            return

        logger.info(f"✅ Data loaded: {len(df)} rows")
        logger.info(f"Date range: {df[self.config['timestamp_col']].min()} to {df[self.config['timestamp_col']].max()}")
        
        # 2. Feature engineering
        X_train, X_test, y_train, y_test, train_times, test_times = self.feature_engineer.transform(df)
        
        del df
        gc.collect()

        # 3. Encoding
        X_train_scaled = self.encoder.fit_transform(X_train)
        X_test_scaled = self.encoder.transform(X_test)

        del X_train, X_test
        gc.collect()

        # 4. Train models
        self._train_models_simple(X_train_scaled, X_test_scaled, y_train, y_test, train_times, test_times)

        del X_test_scaled, y_train, y_test, train_times, test_times
        gc.collect()

        logger.debug("🧹 GC: Deleted train/test sets after training")

        # 5. Save pipeline artifacts
        self._save_pipeline_artifacts(X_train_scaled.columns.tolist())

        del X_train_scaled

        # 6. Save feature_importance
        self.save_feature_importance()
        
        logger.info("🎉 Pipeline completed successfully!")
        
    
    def _train_models_simple(self, X_train, X_test, y_train, y_test, train_times=None, test_times=None):
        """Train models with simple train-test split, evaluate, 
        then retrain incrementally on full month data."""

        self.models = {
             'LIGHTGBM' : LGBMRegressor(**self.model_config["models"]["LIGHTGBM"])

        }

        results = {}

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            try:

                model_path = f"{self.path_config['models_dir']}/{model_name}_model.pkl"

                # --- 1) Train on train split ---
                if model_name == 'LIGHTGBM':
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=50),
                                                                                        lgb.log_evaluation(period=100)])
                else:
                    model.fit(X_train, y_train)

                # --- 2) Evaluate on test split ---
                # After training, LightGBM stores evaluation results

                y_train_pred= model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
            

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
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'test_size': len(y_test)
                }
                logger.info(f"{model_name} [{month_id}] train_mae: {train_mae:.4f} - test_mae: {test_mae:.4f} ")

                # --- 3) Retrain on full month (train + test) ---
                X_full = pd.concat([X_train, X_test])
                y_full = pd.concat([y_train, y_test])

                logger.info(f"{model_name} [{month_id}] Train Full Date")
                
                if model_name == 'LIGHTGBM':
                    best_round = getattr(model, "best_iteration_", None)
                    if best_round is not None:
                        # keep only the useful number of trees
                        model.set_params(n_estimators=best_round + 1)
                    model.set_params(callbacks=None)

                # Train on full data without early stopping
                model.fit(X_full, y_full)
                
                # --- 4) Save updated model for next month ---
                joblib.dump(model, model_path)
                logger.info(f"💾 Saved {model_name} model to {model_path}")

                # Free memory
                del X_full, y_full
                gc.collect()

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


    def save_feature_importance(self, model_name=None):
        """
        Save feature importances or coefficients for all trained models in models_dir.
        - If model_name is provided, only that model is processed.
        - Works with tree-based models (XGB, LightGBM, RF, GBDT) and linear models (Ridge, LogisticRegression).
        - Adds timestamp for when importances were saved.
        """
        try:
            model_files = []
            if model_name:
                model_files = [f"{model_name}_model.pkl"]
            else:
                model_files = [f for f in os.listdir(self.path_config["models_dir"]) if f.endswith("_model.pkl")]

            if not model_files:
                logger.warning("⚠️ No model files found to extract feature importances.")
                return None

            all_importances = []
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for mf in model_files:
                model_path = os.path.join(self.path_config["models_dir"], mf)
                name = mf.replace("_model.pkl", "")

                if not os.path.exists(model_path):
                    logger.warning(f"⚠️ Model file not found at {model_path}")
                    continue

                model = joblib.load(model_path)

                # --- Tree-based models ---
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_names = getattr(model, "feature_names_in_", None)
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(len(importances))]

                # --- Linear models (Ridge, LogisticRegression, etc.) ---
                elif hasattr(model, "coef_"):
                    importances = model.coef_.ravel() if hasattr(model.coef_, "ravel") else model.coef_
                    feature_names = getattr(model, "feature_names_in_", None)
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(len(importances))]

                else:
                    logger.warning(f"⚠️ Model {name} has no importances or coefficients.")
                    continue

                # Build DataFrame (with timestamp)
                importance_df = pd.DataFrame({
                    "Model": name,
                    "Feature": feature_names,
                    "Importance": importances,
                    "Saved_At": now_str
                }).sort_values(by="Importance", ascending=False)

                all_importances.append(importance_df)

            if not all_importances:
                logger.warning("⚠️ No importances extracted from any model.")
                return None

            final_df = pd.concat(all_importances, ignore_index=True)

            # Save results
            results_dir = os.path.join(self.path_config["results_dir"], "feature_importances")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, "feature_importances.csv")
            final_df.to_csv(csv_path, index=False)

            logger.info(f"✅ Feature importances saved to {csv_path}")
            return final_df

        except Exception as e:
            logger.error(f"❌ Failed to save feature importances: {e}", exc_info=True)
            return None

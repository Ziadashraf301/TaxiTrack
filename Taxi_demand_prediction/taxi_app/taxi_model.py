import os
import joblib
import pandas as pd
import logging

class TaxiPredictionModel:
    def __init__(self, model_filename='xgboost_pipeline.pkl'):
        self.model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '..',
                'model_training',
                model_filename
            )
        )
        self.model = self.load_model()

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}: {e}")
            return None

    def make_prediction(self, features):
        if self.model is None:
            return None, None

        try:
            df = pd.DataFrame([features])
            prediction = self.model.predict(df)
            if prediction < 0:
                prediction = 1
            df['prediction'] = round(prediction[0], 0)
            return df, round(prediction[0], 0)
        except Exception as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            return None, None

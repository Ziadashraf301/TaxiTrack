import psycopg2
import logging
import streamlit as st
from datetime import datetime

class DatabaseHandler:
    def __init__(self, host='localhost', dbname='ingest_db', user='ingest_user', password='ingest_password', port=5433):
        self.conn_params = {
            'host': host,
            'database': dbname,
            'user': user,
            'password': password,
            'port': port
        }

    def get_connection(self):
        try:
            conn = psycopg2.connect(**self.conn_params)
            logging.info("Database connection established.")
            return conn
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return None

    def insert_prediction(self, df):
        conn = self.get_connection()
        if conn is None:
            st.error("Could not connect to the database.")
            return

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_results (
                        trip_month INTEGER,
                        trip_day INTEGER,
                        trip_weekday INTEGER,
                        trip_hour INTEGER,
                        day_type TEXT,
                        vendor_label TEXT,
                        payment_type_label TEXT,
                        trip_type TEXT,
                        avg_distance FLOAT,
                        avg_passenger_count INTEGER,
                        unique_pickup_locations INTEGER,
                        prediction FLOAT,
                        prediction_time TIMESTAMP
                    )
                """)
                logging.info("Ensured prediction_results table exists.")

                # Add timestamp column to DataFrame
                df['prediction_time'] = datetime.now()

                for _, row in df.iterrows():
                    cols = ', '.join(df.columns)
                    placeholders = ', '.join(['%s'] * len(df.columns))
                    sql = f"INSERT INTO prediction_results ({cols}) VALUES ({placeholders})"
                    cursor.execute(sql, tuple(row))

                conn.commit()
                logging.info("Prediction inserted into the database.")
        except Exception as e:
            logging.error(f"Failed to insert data: {e}")
            st.error(f"Failed to insert into the database: {e}")
        finally:
            conn.close()

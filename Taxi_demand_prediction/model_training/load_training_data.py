import psycopg2
import pandas as pd
import logging
from tqdm import tqdm
from io import StringIO

# Set up logging configuration
logging.basicConfig(
    filename='load_training_data.log',  # Log file name
    level=logging.INFO,  # Log level (INFO will log everything above and including INFO level)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define connection parameters
connection_params = {
    "dbname": "ingest_db",
    "user": "ingest_user",
    "password": "ingest_password",
    "host": "localhost",
    "port": "5433"
}

# Define chunk size for fetching rows
chunk_size = 100000

def connect_to_db(connection_params):
    """Establish and return a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()
        logging.info('Successfully connected to the database.')
        return connection, cursor
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        raise

def fetch_data_to_memory(cursor, query):
    """Fetch data from the database and return it as a StringIO object."""
    try:
        output = StringIO()
        cursor.copy_expert(query, output)
        logging.info("Data successfully copied to memory.")
        output.seek(0)  # Reset StringIO pointer to start
        return output
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

def load_data_in_chunks(output, chunk_size):
    """Load data from the StringIO object into a DataFrame in chunks."""
    try:
        df = pd.DataFrame()
        for chunk in pd.read_csv(output, chunksize=chunk_size):
            df = pd.concat([df, chunk], ignore_index=True)
        logging.info(f"Data loaded into DataFrame with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading data into DataFrame: {e}")
        raise

def close_db_connection(cursor, connection):
    """Close the database connection and cursor."""
    try:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        logging.info('Connection closed.')
    except Exception as e:
        logging.error(f"Error closing connection: {e}")

def get_data(table_name):
    """Main function to orchestrate the data fetching and loading process."""
    try:
        # Log the start of the process
        logging.info('Starting connection to the database...')
        
        # Establish database connection
        connection, cursor = connect_to_db(connection_params)

        # Define the query to export data from the database
        query = f'COPY (SELECT * FROM "{table_name}") TO STDOUT WITH CSV HEADER DELIMITER \',\';'

        # Fetch the data into memory
        output = fetch_data_to_memory(cursor, query)

        # Load data in chunks and create a DataFrame
        df = load_data_in_chunks(output, chunk_size)

        # Print and log the number of rows
        print(f"Data fetched into DataFrame with {len(df)} rows.")
        logging.info(f"Data fetched into DataFrame with {len(df)} rows.")
        
        close_db_connection(cursor, connection)
        return df
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")







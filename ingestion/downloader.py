import os
import logging

def download_file(url, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure 'data/' folder exists

    if os.path.exists(output_file):
        logging.info(f"File already exists, skipping download: {output_file}")
        return
    try:
        os.system(f"curl -L {url} -o {output_file}")
        logging.info(f"File downloaded successfully: {output_file}")
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        raise

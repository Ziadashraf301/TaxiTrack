import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Ingest CSV data to Postgres')

    parser.add_argument('--user', required=True, help='Postgres username')
    parser.add_argument('--password', required=True, help='Postgres password')
    parser.add_argument('--host', required=True, help='Postgres host')
    parser.add_argument('--port', required=True, help='Postgres port')
    parser.add_argument('--db', required=True, help='Database name')
    parser.add_argument('--file_name', required=True, help='Text file with URLs')

    return parser.parse_args()

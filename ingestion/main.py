from config import parse_args
from logger import setup_logging
from ingestor import ingest_data_for_file

def main(params):
    setup_logging()
    with open(params.file_name, "r") as f:
        params.base_url = f.readline().strip()
        params.processed_urls = set()
        for file_line in f:
            ingest_data_for_file(params, file_line)

if __name__ == '__main__':
    args = parse_args()
    main(args)

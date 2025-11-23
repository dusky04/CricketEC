from utils import setup_and_download_dataset
from pathlib import Path


if __name__ == "__main__":
    DATASET_NAME = "CricketEC"
    CRICKET_EC_URL = "https://drive.google.com/file/d/1QRM360a5HvRKvPF3k7vOT1PS0suxioJd/view?usp=sharing"

    setup_and_download_dataset(
        DATASET_NAME, url=CRICKET_EC_URL, download_dir=Path("zipped_data")
    )

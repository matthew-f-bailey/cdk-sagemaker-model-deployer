""" Module for all constants and overall project settings """
import logging
import os
from pathlib import Path

logging.basicConfig(
    format="%(name)s|%(asctime)s|%(levelname)s: %(message)s",
    level=getattr(logging, os.environ.get("LOGGING_LEVEL", "DEBUG")),
)

INPUT_DIR = Path(Path(__file__).parent, "data")
OUTPUT_DIR = Path(Path(__file__).parent, "data")
os.makedirs(str(OUTPUT_DIR.resolve()), exist_ok=True)


# Data related stuff
def get_data_filepath(file: str) -> Path:
    return Path(INPUT_DIR, file)


def get_db_filepath(db_name: str):
    pth = get_data_filepath(f"{db_name}/1/")
    files = [x for x in pth.glob("*")]
    for file in files:
        if ".parquet" in str(file):
            return file

    raise FileNotFoundError(f"Could not find DB file of {pth}\nFiles: {files}")


def get_output_filepath(file: str) -> Path:
    return Path(OUTPUT_DIR, file)

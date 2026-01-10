import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOVELS_DIR = os.path.join(DATA_DIR, "novels")


def load_train_test():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def load_novel(novel_name):
    """
    novel_name comes from CSV (e.g. 'In Search of the Castaways')
    Actual file is 'In search of the castaways.txt'
    This function matches safely.
    """

    # normalize name
    novel_name = novel_name.lower().strip()

    for filename in os.listdir(NOVELS_DIR):
        if filename.lower().startswith(novel_name):
            novel_path = os.path.join(NOVELS_DIR, filename)
            with open(novel_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    raise FileNotFoundError(f"No novel found for name: {novel_name}")

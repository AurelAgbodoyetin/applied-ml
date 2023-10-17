from typing import Dict


class Dataset:
    def __init__(self, path: str, sep: str, name: str, skip_rows: int = 0):
        self.path: str = path
        self.sep: str = sep
        self.name = name
        self.skip_rows: int = skip_rows


datasets: Dict[str, Dataset] = {
    '100k': Dataset(path="data/100k/u.data", sep="\t", name="100k"),
    '1m': Dataset(path="data/1m/ratings.dat", sep="::", name="1m"),
    '10m': Dataset(path="../datasets/10M/ratings.dat", sep="::", name="10m"),
    '20m': Dataset(path="../datasets/20M/ratings.csv", sep=",", skip_rows=1, name="20m"),
    '25m': Dataset(path="../datasets/25M/ratings.csv", sep=",", skip_rows=1, name="25m"),
}



from typing import Dict


class Dataset:
    def __init__(self, path: str, items_path: str, sep: str, name: str, items_sep: str, features_sep: str = "|",
                 skip_rows: int = 0):
        self.path: str = path
        self.items_path = items_path
        self.sep: str = sep
        self.items_sep: str = items_sep
        self.features_sep: str = features_sep
        self.name = name
        self.skip_rows: int = skip_rows


datasets: Dict[str, Dataset] = {
    '100k': Dataset(path="data/100k/u.data", sep="\t", name="100k", items_path="data/100k/u.item", items_sep="|"),
    '100k_csv': Dataset(path="data/100k_csv/ratings.csv", sep=",", name="100k_csv",
                        items_path="data/100k_csv/movies.csv", items_sep=",", skip_rows=1),
    '1m': Dataset(path="data/1m/ratings.dat", sep="::", name="1m", items_path="data/1m/movies.dat", items_sep="::"),
    '10m': Dataset(path="../datasets/10M/ratings.dat", sep="::", name="10m", items_path="data/10m/movies.dat",
                   items_sep=","),
    '25m': Dataset(path="../datasets/25M/ratings.csv", sep=",", skip_rows=1, name="25m",
                   items_path="../datasets/25M/movies.csv", items_sep="::"),
}

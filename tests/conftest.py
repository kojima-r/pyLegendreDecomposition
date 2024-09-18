from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from PIL import Image


@pytest.fixture
def random_X_array() -> NDArray[np.float64]:
    return np.random.uniform(0, 0.1, size=(10, 10, 10, 10))


@pytest.fixture
def coil_100_array() -> NDArray[np.float64]:
    X = []
    for k in range(1, 5):
        file_name = Path("tests") / "resources" / "coil-100" / f"obj{k}__0.png"
        assert file_name.exists()
        img = Image.open(file_name)
        img = img.resize((32, 32))
        array_obj = np.asarray(img)
        X.append(array_obj)
    X = np.array(X)
    X = (X + 1) / 256.0
    return X


@pytest.fixture
def movie_lens_array() -> NDArray[np.float64]:
    movies_df = pd.read_csv(Path("tests") / "resources" / "ml-latest-small" / "movies.csv")
    ratings_df = pd.read_csv(Path("tests") / "resources" / "ml-latest-small" / "ratings.csv")

    genres_set = set()
    for i, r in movies_df.iterrows():
        for el in r["genres"].split("|"):
            genres_set.add(el)
    genres_list = list(genres_set)
    edges_mg_dict = {}
    mid_dict = {}
    for i, r in movies_df.iterrows():
        for el in r["genres"].split("|"):
            genres_set.add(el)
            j = genres_list.index(el)
            m = int(r["movieId"])
            if m not in mid_dict:
                mid_dict[m] = len(mid_dict)
            if mid_dict[m] not in edges_mg_dict:
                edges_mg_dict[mid_dict[m]] = []
            edges_mg_dict[mid_dict[m]].append(j)

    edges = []
    for i, r in ratings_df.iterrows():
        m = int(r["movieId"])
        if m in mid_dict:
            for g in edges_mg_dict[mid_dict[m]]:
                e = (int(r["userId"]) - 1, mid_dict[m], g, r["rating"])
                edges.append(e)
    u_max = max([e[0] for e in edges])
    m_max = max([e[1] for e in edges])
    g_max = max([e[2] for e in edges])
    M = np.zeros((u_max + 1, m_max + 1, g_max + 1))
    for u, m, g, r in edges:
        M[u, m, g] = r
    return M[:10, :10, :]

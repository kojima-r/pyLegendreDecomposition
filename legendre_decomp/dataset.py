import requests
import zipfile
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import Counter
import pandas as pd


def build_coil(N=5, W=32, H=32, redownload=False):
    url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
    zip_filename = "coil-100.zip"
    extract_dir = "datasets"

    # Check if the directory already exists
    if os.path.exists(extract_dir + "/coil-100"):
        print(
            f"Directory {extract_dir+'/coil-100'} already exists. Skipping download and extraction."
        )
    else:
        # Download the zip file
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # Unzip the file
        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")

    # Optional: Clean up the zip file
    # os.remove(zip_filename)

    X = []
    for k in range(1, N + 1):
        # ファイル名、パス
        file_name = "datasets/coil-100/obj{}__0.png".format(k)
        img = Image.open(file_name)
        img = img.resize((W, H))
        array_obj = np.asarray(img)
        # print(array_obj.shape)
        # 画像の表示
        # plt.imshow(array_obj)
        # plt.show()
        X.append(array_obj)
    X = np.array(X)
    X = (X + 1) / 256.0
    return X


def draw_coil(Q, scaleX):
    # 前処理の逆変換
    X_recons = (X * 256).astype(np.int32)
    for i in range(len(X_recons)):
        plt.imshow(X_recons[i])
        plt.show()


def draw_coil_recons(Q, scaleX):
    # 前処理の逆変換
    X_recons = (Q * scaleX * 256).astype(np.int32)

    for i in range(len(X_recons)):
        plt.imshow(X_recons[i])
        plt.show()


def build_random(N, order, low=0.0, high=0.1):
    X = np.random.uniform(low, high, size=tuple([N] * order))
    return X


def build_random_diff_size(order, low=0.0, high=0.1):
    shape = tuple([m + 2 for m in range(order)])
    X = np.random.uniform(low, high, size=shape)
    return X


def build_movielens(movie_thresh=100, user_thresh=20):
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_filename = "ml-latest-small.zip"
    extract_dir = "datasets"

    # Check if the directory already exists
    if os.path.exists(extract_dir + "/ml-latest-small"):
        print(
            f"Directory {extract_dir+'/ml-latest-small'} already exists. Skipping download and extraction."
        )
    else:
        # Download the zip file
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # Unzip the file
        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")

    # Optional: Clean up the zip file
    # os.remove(zip_filename)

    # user-movie-genre graph with rating weight
    df = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    # count movies and users to decide target users/movies
    movie_count = Counter()
    for i, r in df.iterrows():
        # Get the movie ID and convert it to integer
        movie_count[m] += 1

    user_count = Counter()
    for i, r in df.iterrows():
        m = int(r["movieId"])
        if movie_count[m] >= movie_thresh:
            u = int(r["userId"])
            user_count[u] += 1
    mid_list = []
    for k, v in movie_count.items():
        if v >= movie_thresh:
            mid_list.append(k)
    uid_list = []
    for k, v in user_count.items():
        if v >= user_thresh:
            uid_list.append(k)
    df_m = pd.read_csv("datasets/ml-latest-small/movies.csv")
    genre_set = set()
    for i, r in df_m.iterrows():
        # "genres"列の値を"|"で分割し、各ジャンルをセットに追加
        for el in r["genres"].split("|"):
            genre_set.add(el)
    # セットをリストに変換
    genre_list = list(genre_set)

    # movie index => genre index
    edges_mg_dict = {}
    for i, r in df_m.iterrows():
        # Split the genres string by '|' and add each genre to the set
        for el in r["genres"].split("|"):
            j = genre_list.index(el)
            m = int(r["movieId"])
            if m in mid_list:
                k = mid_list.index(m)
                # If the movie index is not in the edges_mg_dict, create an empty list for it
                if k not in edges_mg_dict:
                    edges_mg_dict[k] = []
                edges_mg_dict[k].append(j)
    edges_mg_dict

    # user-movie-genre graph with rating weight
    edges = []
    for i, r in df.iterrows():
        # Get the movie ID and convert it to integer
        m = int(r["movieId"])
        u = int(r["userId"])
        if m in mid_list and u in uid_list:
            # Create an edge tuple (userId, movieIndex, genreIndex, rating)
            for g in edges_mg_dict[mid_list.index(m)]:
                # Create an edge tuple (userId, movieIndex, genreIndex, rating)
                e = (uid_list.index(u), mid_list.index(m), g, r["rating"])
                # Append the edge to the edges list
                edges.append(e)

    # Find the maximum values for userId, movieIndex, and genreIndex
    u_max = max([e[0] for e in edges])
    m_max = max([e[1] for e in edges])
    g_max = max([e[2] for e in edges])

    # Print the maximum values
    print("tensor:", u_max, m_max, g_max)
    import numpy as np

    # Create a 3D NumPy array (tensor) with dimensions based on max indices
    M = np.zeros((u_max + 1, m_max + 1, g_max + 1))
    # Populate the tensor with ratings from the edges
    for u, m, g, r in edges:
        M[u, m, g] = r

    return M, {"user_list": uid_list, "movie_list": mid_list, "genre_list": genre_list}


def build_matrix_train_factor(low=0.0, high=0.1, M=3, order=3, independent=None):
    l = []
    X_list = []
    I_list = []
    N = order - 1
    for i in range(N):
        if independent is not None and i in independent:
            if i == 0:
                X = np.random.uniform(low, high, size=(M,))
                l.append(X)
                X_list.append(X)
                l.append((i,))
                I_list.append((i,))
            elif i == N - 1:
                X = np.random.uniform(low, high, size=(M,))
                l.append(X)
                X_list.append(X)
                l.append((i + 1,))
                I_list.append((i + 1,))
        else:
            X = np.random.uniform(low, high, size=(M, M))
            l.append(X)
            X_list.append(X)
            l.append((i, i + 1))
            I_list.append((i, i + 1))
    X = np.einsum(*l, [i for i in range(N + 1)])
    return X, {"x_list": X_list, "I": I_list}


def build_matrix_train_factor_diff_size(low=0.0, high=0.1, order=3, independent=None):
    l = []
    X_list = []
    N = order - 1
    I_list = []
    for i in range(N):
        if independent is not None and i in independent:
            if i == 0:
                X = np.random.uniform(low, high, size=(i + 2,))
                l.append(X)
                X_list.append(X)
                l.append((i,))
                I_list.append((i,))
            elif i == N - 1:
                X = np.random.uniform(low, high, size=(i + 3,))
                l.append(X)
                X_list.append(X)
                l.append((i + 1,))
                I_list.append((i + 1,))
        else:
            X = np.random.uniform(low, high, size=(i + 2, i + 3))
            l.append(X)
            X_list.append(X)
            l.append((i, i + 1))
            I_list.append((i, i + 1))
    X = np.einsum(*l, [i for i in range(N + 1)])
    return X, {"x_list": X_list, "I": I_list}


def build_matrix_ring_factor(low=0.0, high=0.1, M=3, order=3):
    l = []
    X_list = []
    N = order
    I_list = []
    for i in range(N - 1):
        X = np.random.uniform(0, 0.1, size=(M, M))
        l.append(X)
        X_list.append(X)
        l.append((i, i + 1))
        I_list.append((i, i + 1))
    X = np.random.uniform(0, 0.1, size=(M, M))
    l.append(X)
    X_list.append(X)
    l.append((0, N - 1))
    I_list.append((0, N - 1))
    #
    X = np.einsum(*l, [i for i in range(N)])
    return X, {"x_list": X_list, "I": I_list}


def build_matrix_ring_factor_diff_size(low=0.0, high=0.1, order=3):
    l = []
    X_list = []
    N = order
    I_list = []
    for i in range(N - 1):
        X = np.random.uniform(0, 0.1, size=(i + 2, i + 3))
        l.append(X)
        X_list.append(X)
        l.append((i, i + 1))
        I_list.append((i, i + 1))
    X = np.random.uniform(0, 0.1, size=(2, N - 1 + 2))
    l.append(X)
    X_list.append(X)
    l.append((0, N - 1))
    I_list.append((0, N - 1))
    #
    X = np.einsum(*l, [i for i in range(N)])
    return X, {"x_list": X_list, "I": I_list}


def build_bmatrix_train_factor(prob=0.5, M=3, order=3, independent=None):
    l = []
    X_list = []
    I_list = []
    N = order - 1
    for i in range(N):
        if independent is not None and i in independent:
            if i == 0:
                X = np.random.binomial(1, prob, size=(M,))
                l.append(X)
                X_list.append(X)
                l.append((i,))
                I_list.append((i,))
            elif i == N - 1:
                X = np.random.binomial(1, prob, size=(M,))
                l.append(X)
                X_list.append(X)
                l.append((i + 1,))
                I_list.append((i + 1,))
        else:
            X = np.random.binomial(1, prob, size=(M, M))
            l.append(X)
            X_list.append(X)
            l.append((i, i + 1))
            I_list.append((i, i + 1))
    X = np.einsum(*l, [i for i in range(N + 1)])
    return X, {"x_list": X_list, "I": I_list}


def build_bmatrix_train_factor_diff_size(prob=0.5, order=3, independent=None):
    l = []
    X_list = []
    N = order - 1
    I_list = []
    for i in range(N):
        if independent is not None and i in independent:
            if i == 0:
                X = np.random.binomial(1, prob, size=(i + 2,))
                l.append(X)
                X_list.append(X)
                l.append((i,))
                I_list.append((i,))
            elif i == N - 1:
                X = np.random.binomial(1, prob, size=(i + 3,))
                l.append(X)
                X_list.append(X)
                l.append((i + 1,))
                I_list.append((i + 1,))
        else:
            X = np.random.binomial(1, prob, size=(i + 2, i + 3))
            l.append(X)
            X_list.append(X)
            l.append((i, i + 1))
            I_list.append((i, i + 1))
    X = np.einsum(*l, [i for i in range(N + 1)])
    return X, {"x_list": X_list, "I": I_list}


def build_bmatrix_ring_factor(prob=0.5, M=3, order=3):
    l = []
    X_list = []
    N = order
    I_list = []
    for i in range(N - 1):
        X = np.random.binomial(1, prob, size=(M, M))
        l.append(X)
        X_list.append(X)
        l.append((i, i + 1))
        I_list.append((i, i + 1))
    X = np.random.binomial(1, prob, size=(M, M))
    l.append(X)
    X_list.append(X)
    l.append((0, N - 1))
    I_list.append((0, N - 1))
    #
    X = np.einsum(*l, [i for i in range(N)])
    return X, {"x_list": X_list, "I": I_list}


def build_bmatrix_ring_factor_diff_size(prob=0.5, order=3):
    l = []
    X_list = []
    N = order
    I_list = []
    for i in range(N - 1):
        X = np.random.binomial(1, prob, size=(i + 2, i + 3))
        l.append(X)
        X_list.append(X)
        l.append((i, i + 1))
        I_list.append((i, i + 1))
    X = np.random.binomial(1, prob, size=(2, N - 1 + 2))
    l.append(X)
    X_list.append(X)
    l.append((0, N - 1))
    I_list.append((0, N - 1))
    #
    X = np.einsum(*l, [i for i in range(N)])
    return X, {"x_list": X_list, "I": I_list}

import pandas as pd; import numpy as np; import tqdm; import os
from multiprocessing import Pool
import argparse

src_folder = "stage2_embeddings_b5_sz512_fold0"
dst_folder = "series_embeddings_seq_b5_sz512_fold0"
src_folder_t = "stage2_test_embeddings_b5_sz512_fold0"
dst_folder_t = "series_test_embeddings_seq_b5_sz512_fold0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", action="store_true",
                            help="inference procedure")
    parser.add_argument("-p", action="store_true",
                            help="dummy printing")
    args = parser.parse_args()

    return args

def func(series):
    print(series)
    seq_len = 1024
    series_embedding = []
    df = df_train[df_train["SeriesInstanceUID"] == series]
    assert len(df)
    series_len = len(df)
    if series_len < seq_len:
        padsize = seq_len - series_len
        before_pad = int(padsize / 2)
        after_pad = int(padsize / 2)
        if padsize > (before_pad+after_pad):
            after_pad += 1
        assert before_pad + after_pad + len(df) == seq_len
        for _ in range(before_pad):
            series_embedding.append(np.zeros((32,)))
        for instance in df["SOPInstanceUID"].values:
            series_embedding.append(np.load(f"embeddings_data/{src_folder}/{instance}.npy"))
        for _ in range(after_pad):
            series_embedding.append(np.zeros((32,)))
    elif series_len > seq_len:
        truncate = series_len - seq_len
        before_truncate = int(truncate / 2)
        after_truncate = int(truncate / 2)
        if truncate > (before_truncate+after_truncate):
            after_truncate += 1
        assert - before_truncate - after_truncate + len(df) == seq_len
        df = df.iloc[before_truncate:-(after_truncate)]
        for instance in df["SOPInstanceUID"].values:
            series_embedding.append(np.load(f"embeddings_data/{src_folder}/{instance}.npy"))
    else:
        for instance in df["SOPInstanceUID"].values:
            series_embedding.append(np.load(f"embeddings_data/{src_folder}/{instance}.npy"))
    series_embedding = np.vstack(series_embedding)
    np.save(f"embeddings_data/{dst_folder}/{series}.npy", series_embedding)

def func_t(series):
    print(series)
    seq_len = 1024
    series_embedding = []
    df = df_train[df_train["StudyInstanceUID"] == series]
    assert len(df)
    series_len = len(df)
    if series_len < seq_len:
        padsize = seq_len - series_len
        before_pad = int(padsize / 2)
        after_pad = int(padsize / 2)
        if padsize > (before_pad+after_pad):
            after_pad += 1
        assert before_pad + after_pad + len(df) == seq_len
        for _ in range(before_pad):
            series_embedding.append(np.zeros((32,)))
        for instance in df["SOPInstanceUID"].values:
            series_embedding.append(np.load(f"embeddings_data/{src_folder_t}/{instance}.npy"))
        for _ in range(after_pad):
            series_embedding.append(np.zeros((32,)))
    elif series_len > seq_len:
        truncate = series_len - seq_len
        before_truncate = int(truncate / 2)
        after_truncate = int(truncate / 2)
        if truncate > (before_truncate+after_truncate):
            after_truncate += 1
        assert - before_truncate - after_truncate + len(df) == seq_len
        df = df.iloc[before_truncate:-(after_truncate)]
        for instance in df["SOPInstanceUID"].values:
            series_embedding.append(np.load(f"embeddings_data/{src_folder_t}/{instance}.npy"))
    else:
        for instance in df["SOPInstanceUID"].values:
            series_embedding.append(np.load(f"embeddings_data/{src_folder_t}/{instance}.npy"))
    series_embedding = np.vstack(series_embedding)
    np.save(f"embeddings_data/{dst_folder_t}/{series}.npy", series_embedding)
    
if __name__ == "__main__":
    args = parse_args()
    if args.p:
        print("Dummy test passed.")
    elif args.t:
        df_train = pd.read_csv("data_csv/rsna-str-test-index.csv")
        series_list = df_train["StudyInstanceUID"].unique()
        if not os.path.exists(f"embeddings_data/{dst_folder_t}"):
            os.makedirs(f"embeddings_data/{dst_folder_t}")
        p = Pool()
        p.map(func_t,list(series_list))
    else:
        df_train = pd.read_csv("data_csv/rsna-str-train-index.csv")
        series_list = df_train["SeriesInstanceUID"].unique()
        if not os.path.exists(f"embeddings_data/{dst_folder}"):
            os.makedirs(f"embeddings_data/{dst_folder}")
        p = Pool()
        p.map(func,list(series_list))

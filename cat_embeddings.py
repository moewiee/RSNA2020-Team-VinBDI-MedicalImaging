import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool

df = pd.read_csv("data_csv/rsna-str-train-index.csv")

def cat_embeddings(series):
    print(f"Processing {series}...")
    df_t = df[df["SeriesInstanceUID"] == series]
    series_len = len(df_t)
    sequence_len = 31
    for i in range(series_len):
        slices = []
        for ii in range(1, int((sequence_len + 1) / 2)):
            slices.append(np.load(os.path.join('/home/datnt/Code/rsna-str/embeddings_data/embeddings_b5_sz512_fold0/', df_t[df_t["sub_index"] == max(i-ii, 0)]["SOPInstanceUID"].values[0] + '.npy')))
        slices = slices[::-1]
        slices.append(np.load(os.path.join('/home/datnt/Code/rsna-str/embeddings_data/embeddings_b5_sz512_fold0/', df_t[df_t["sub_index"] == i]["SOPInstanceUID"].values[0] + '.npy')))
        for ii in range(1, int((sequence_len + 1) / 2)):
            slices.append(np.load(os.path.join('/home/datnt/Code/rsna-str/embeddings_data/embeddings_b5_sz512_fold0/', df_t[df_t["sub_index"] == min(i+ii, series_len-1)]["SOPInstanceUID"].values[0] + '.npy')))
        slices = np.vstack(slices)
        np.save(os.path.join('/home/datnt/Code/rsna-str/embeddings_data/embeddings_cat_b5_sz512_fold0/', df_t[df_t["sub_index"] == i]["SOPInstanceUID"].values[0] + '.npy'), slices)

series_list = df["SeriesInstanceUID"].unique()

p = Pool()
p.map(cat_embeddings, series_list)

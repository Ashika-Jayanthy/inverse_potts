
import pandas as pd
import pickle
import numpy as np
import time

dir = "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/combined_counts"

split_files = ["xaa","xab","xac","xad","xae","xaf","xag","xah","xai","xaj","xak","xal","xam","xan","xao","xap","xaq","xar","xas","xat","xau","xav","xaw","xax","xay","xaz","xba"]
num_files = len(split_files)

npb_indices = np.loadtxt(f"{dir}/npb_indices.txt")
npb_indices = [int(a)-1 for a in npb_indices] # in the dataframe, the first column is set to the index column and the new column numbers are off by 1
t1_indices = np.loadtxt(f"{dir}/timeseries1_indices.txt")
t1_indices = [int(a)-1 for a in t1_indices]
t2_indices = np.loadtxt(f"{dir}/timeseries2_indices.txt")
t2_indices = [int(a)-1 for a in t2_indices]
indices = [npb_indices,t1_indices,t2_indices]
index_names = ["npb","timeseries1","timeseries2"]

for i,split_file in enumerate(split_files):
    start = time.time()
    df = pd.read_csv(f"{dir}/{split_file}", sep="\t", header=None,index_col=0)
    print(df.shape)
    for j,index in enumerate(indices):
        df_subset = df.iloc[:,index]
        name = index_names[j]
        with open(f"{dir}/df_{name}_{i}.pkl",'wb') as handle:
            pickle.dump(df_subset,handle)
    end = time.time()
    print(f"Finished {i} of {num_files}, time {end-start}")

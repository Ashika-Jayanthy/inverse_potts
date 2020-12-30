import pandas as pd
import pickle
# Timeseries2 too large, was skipped
dir = "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/combined_counts"

split_files = ["xaa","xab","xac","xad","xae","xaf","xag","xah","xai","xaj","xak","xal","xam","xan","xao","xap","xaq","xar","xas","xat","xau","xav","xaw","xax","xay","xaz","xba"]
num_files = len(split_files)

index_names = ["npb","timeseries1","timeseries2"]

for name in index_names:
    df_parts = []
    for i in range(num_files):
        with open(f"{dir}/df_{name}_{i}.pkl",'rb') as handle:
            df_parts.append(pickle.load(handle))
        print(f"Finished {i} of {num_files}")
    df = pd.concat(df_parts, axis=0, join='outer', ignore_index=False, keys=None,levels=None, names=None, verify_integrity=False, copy=True)
    print(df.shape)
    df.to_csv(f"{dir}/{name}.tsv",sep="\t")
    print(f"Finished {name}")

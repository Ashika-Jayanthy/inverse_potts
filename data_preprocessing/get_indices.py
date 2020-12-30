import numpy as np

dir = "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/combined_counts"

with open(f"{dir}/GSE113074_Corrected_combined.annotated_counts.tsv",'r') as ff:
    xt = ff.readlines()

replicates = np.array(xt[1].split())

npb_indices = np.where(replicates == 'NPB_dissection')
np.savetxt(f"{dir}/npb_indices.txt",npb_indices,fmt='%u')

t1_indices = np.where(replicates == 'Timeseries_1')
np.savetxt(f"{dir}/timeseries1_indices.txt",t1_indices,fmt='%u')

t2_indices = np.where(replicates == 'Timeseries_2')
np.savetxt(f"{dir}/timeseries2_indices.txt",t2_indices,fmt='%u')

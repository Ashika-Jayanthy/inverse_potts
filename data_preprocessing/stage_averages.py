import numpy as np


dir = "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/combined_counts"

with open(f"{dir}/timeseries1.tsv", 'r') as ff:
    t1 = ff.readlines()
outfile = open(f"{dir}/timeseries1_stage_averages.tsv", 'w')

stage_names = ['Stage_8', 'Stage_10', 'Stage_12', 'Stage_14',
               'Stage_16', 'Stage_18', 'Stage_20', 'Stage_22']
stage_indices = []
stage_line = np.array([s.strip() for s in t1[7].split()])
outfile.write("gene_name")
outfile.write("\t")
for name in stage_names:
    outfile.write(name)
    outfile.write("\t")
outfile.write("\n")

for stage in stage_names:
    indices = np.where(stage_line == stage)
    indices = [i - 1 for i in indices[0]]
    stage_indices.append(indices)

for linenum in range(10, len(t1)):
    linesplit = np.array(t1[linenum].split()[1:], dtype=np.float64)
    gene_name = t1[linenum].split()[0].strip()
    outfile.write(gene_name)
    outfile.write("\t")
    for idx in stage_indices:
        outfile.write(str(np.mean(linesplit[idx])))
        outfile.write("\t")
    outfile.write("\n")

outfile.close()

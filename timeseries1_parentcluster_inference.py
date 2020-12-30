import pandas as pd
import numpy as np
from inversepotts_utils import *
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr
import copy

indir = "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/combined_counts"
outdir = "/Users/ashika/inverse_potts/Inferences/by_parent_cluster"
periodic_boundary_conditions = True


timeseries1_parentcluster_mean = pd.read_csv(
    f"{indir}/timeseries1_parent_cluster_averages.tsv", sep="\t", header=0, index_col=0)
gene_names = np.array(list(timeseries1_parentcluster_mean.index))
parent_cluster_names = list(timeseries1_parentcluster_mean.columns.values)

species_transcripts = name_to_sequence(
    gene_names, "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/fasta_files/Xtropicalisv9.0.Named.primaryTrs.fa")
# stages in rows, gene counts corresponding to gene_names in columns
counts_matrix = timeseries1_parentcluster_mean.to_numpy().T


max_length = max([len(seq) for seq in species_transcripts])

if periodic_boundary_conditions:
    transcript_spins = np.zeros((len(species_transcripts), max_length))
    transcript_pairs = np.zeros((len(species_transcripts), max_length))
    for n, seq in enumerate(species_transcripts):
        spins = sequence_to_spins(seq)
        transcript_spins[n, 0:len(seq)] = spins
        transcript_pairs[n, 0:len(seq)] = pair_corr(
            spins, periodic_boundary_conditions=True)
else:
    transcript_spins = np.zeros((len(species_transcripts), max_length))
    transcript_pairs = np.zeros((len(species_transcripts), max_length - 1))
    for n, seq in enumerate(species_transcripts):
        spins = sequence_to_spins(seq)
        transcript_spins[n, 0:len(seq)] = spins
        transcript_pairs[n, 0:len(
            seq) - 1] = pair_corr(spins, periodic_boundary_conditions=False)

possible_spins = [-2, -1, 0, 1, 2]
masked_spins = []
for i in [-2, -1, 0, 1, 2]:
    masked_spins.append(np.ma.masked_equal(transcript_spins, i).mask)
masked_spins = np.array(masked_spins)
possible_pairs = [-4, -2, -1, 0, 1, 2, 4]
masked_pairs = []
for i in [-4, -2, -1, 0, 1, 2, 4]:
    masked_pairs.append(np.ma.masked_equal(transcript_pairs, i).mask)
masked_pairs = np.array(masked_pairs)

masked_values = (masked_spins, masked_pairs)


nseqs, max_length = transcript_spins.shape
Es = []

hinit = np.ones((nseqs, max_length))
if periodic_boundary_conditions:
    Jinit = np.ones((nseqs, max_length))
else:
    Jinit = np.ones((nseqs, max_length - 1))

for parentclusternum in range(len(parent_cluster_names)):
    cluster_counts = counts_matrix[parentclusternum]
    relative_counts = cluster_counts / np.sum(cluster_counts)
    initial_h = copy.deepcopy(hinit)
    initial_j = copy.deepcopy(Jinit)

    cluster = InferHJ(transcriptome_spins=transcript_spins, transcriptome_pairs=transcript_pairs, masked_arrays=masked_values, counts=cluster_counts, h_init=initial_h, J_init=initial_j,
                      k=1e-7, e1=[1e-1, 2e-1, 0, -2e-1, -1e-1], e2=[5e-2, 1e-1, 2e-1, 0, -2e-1, -1e-1, 5e-2], max_runs=30, periodic_boundary_conditions=periodic_boundary_conditions)
    h, J, E, P = cluster.run_inference()
    corr = pearsonr(relative_counts, P)

    h_spins = []
    for j in range(h.shape[1]):
        h_spins.append(np.mean(np.unique(h[:, j])))
    J_spins = []
    for j in range(J.shape[1]):
        J_spins.append(np.mean(np.unique(J[:, i])))

    print(f"{parent_cluster_names[parentclusternum]}:", corr)
    print("E:", np.mean(E), np.var(E))

    filename = (parent_cluster_names[parentclusternum]).replace("/", "|")
    np.savetxt(f"{outdir}/timeseries1_E_{filename}.txt", E)
    np.savetxt(f"{outdir}/timeseries1_h_{filename}.txt",
               np.array(h_spins, dtype=np.float64))
    np.savetxt(f"{outdir}/timeseries1_J_{filename}.txt",
               np.array(J_spins, dtype=np.float64))
    np.savetxt(f"{outdir}/timeseries1_P_{filename}.txt", P)

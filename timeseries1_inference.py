from scipy.stats import pearsonr
from inversepotts_utils import *
from Bio import SeqIO
import pandas as pd
import numpy as np
import copy

periodic_boundary_conditions = True
indir = "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/combined_counts"
outdir = "/Users/ashika/inverse_potts/Inferences/by_stage"


timeseries1_mean = pd.read_csv(
    f"{indir}/timeseries1_stage_averages_edited.tsv", sep="\t", header=0, index_col=0)
gene_names = np.array(list(timeseries1_mean.index))
species_transcripts = name_to_sequence(
    gene_names, "/Users/ashika/inverse_potts/Data/earlyembryo_xtropicalis/fasta_files/Xtropicalisv9.0.Named.primaryTrs.fa")
# stages in rows, gene counts corresponding to gene_names in columns
counts_matrix = timeseries1_mean.to_numpy().T


stage_names = ['Stage_8', 'Stage_10', 'Stage_12', 'Stage_14',
               'Stage_16', 'Stage_18', 'Stage_20', 'Stage_22']
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


masked_spins = []
for i in [-2, -1, 0, 1, 2]:
    masked_spins.append(np.ma.masked_equal(transcript_spins, i).mask)
masked_spins = np.array(masked_spins)
print(masked_spins.shape)
masked_pairs = []
for i in [-4, -2, -1, 0, 1, 2, 4]:
    masked_pairs.append(np.ma.masked_equal(transcript_pairs, i).mask)
masked_pairs = np.array(masked_pairs)
print(masked_pairs.shape)
masked_values = (masked_spins, masked_pairs)


nseqs, max_length = transcript_spins.shape
Es = []
hinit = np.ones((nseqs, max_length))
if periodic_boundary_conditions:
    Jinit = np.ones((nseqs, max_length))
else:
    Jinit = np.ones((nseqs, max_length - 1))


for stagenum in range(8):
    stage_counts = counts_matrix[stagenum]
    relative_counts = stage_counts / np.sum(stage_counts)
    initial_h = copy.deepcopy(hinit)
    initial_j = copy.deepcopy(Jinit)

    stage = InferHJ(transcriptome_spins=transcript_spins, transcriptome_pairs=transcript_pairs, masked_arrays=masked_values, counts=stage_counts, h_init=initial_h, J_init=initial_j,
                    k=1e-7, e1=[1e-1, 2e-1, 0, -2e-1, -1e-1], e2=[5e-2, 1e-1, 2e-1, 0, -2e-1, -1e-1, -5e-2], max_runs=30, periodic_boundary_conditions=periodic_boundary_conditions)
    h, J, E, P = stage.run_inference()
    corr = pearsonr(relative_counts, P)

    print(f"{stagenum}:", corr)
    print("E:", np.mean(E), np.var(E))

    filename = stage_names[stagenum]
    np.savetxt(f"{outdir}/timeseries1_E_{filename}.txt", E)
    np.savetxt(f"{outdir}/timeseries1_h_{filename}.txt",
               np.array(h, dtype=np.float64))
    np.savetxt(f"{outdir}/timeseries1_J_{filename}.txt",
               np.array(J, dtype=np.float64))
    np.savetxt(f"{outdir}/timeseries1_P_{filename}.txt", P)


from scipy.stats import pearsonr
from inversepotts_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
sns.set_style("darkgrid")

periodic_boundary_conditions = True
nseqs, min_length, max_length = 250, 500, 700
num_runs = 100
correlations = []

hinit = np.ones((nseqs, max_length))
if periodic_boundary_conditions:
    Jinit = np.ones((nseqs, max_length))
else:
    Jinit = np.ones((nseqs, max_length - 1))


for nr in range(num_runs):
    print(nr)
    random_transcriptome = [DNA(l) for l in np.random.choice(
        np.arange(min_length, max_length), size=nseqs, replace=True)]
    counts = np.random.choice(np.arange(0, 900), size=nseqs, replace=True)
    true_probabilities = counts / np.sum(counts)

    if periodic_boundary_conditions:
        transcriptome_spins = np.zeros((len(random_transcriptome), max_length))
        transcriptome_pairs = np.zeros((len(random_transcriptome), max_length))
        for n, seq in enumerate(random_transcriptome):
            spins = sequence_to_spins(seq)
            transcriptome_spins[n, 0:len(seq)] = spins
            transcriptome_pairs[n, 0:len(seq)] = pair_corr(
                spins, periodic_boundary_conditions=True)
    else:
        transcriptome_spins = np.zeros((len(random_transcriptome), max_length))
        transcriptome_pairs = np.zeros(
            (len(random_transcriptome), max_length - 1))
        for n, seq in enumerate(random_transcriptome):
            spins = sequence_to_spins(seq)
            transcriptome_spins[n, 0:len(seq)] = spins
            transcriptome_pairs[n, 0:len(
                seq) - 1] = pair_corr(spins, periodic_boundary_conditions=False)

    masked_spins = []
    for i in [-2, -1, 0, 1, 2]:
        masked_spins.append(np.ma.masked_equal(transcriptome_spins, i).mask)
    masked_spins = np.array(masked_spins)

    masked_pairs = []
    for i in [-4, -2, -1, 0, 1, 2, 4]:
        masked_pairs.append(np.ma.masked_equal(transcriptome_pairs, i).mask)
    masked_pairs = np.array(masked_pairs)

    masked_values = (masked_spins, masked_pairs)
    random_cell = InferHJ(transcriptome_spins=transcriptome_spins, transcriptome_pairs=transcriptome_pairs, masked_arrays=masked_values,
                          counts=counts, h_init=hinit, J_init=Jinit, k=1e-7, e1=[1e2, 2e2, 0, -2e2, -1e2], e2=[5e1, 1e2, 2e2, 0, -2e2, -1e2, 5e1], max_runs=100)
    h, J, E, P = random_cell.run_inference()
    corr = pearsonr(true_probabilities, P)
    correlations.append(corr[0])

    print("Pearsons:", corr)


sns.distplot(correlations)
plt.title("Correlations between true and inferred probabilities")
plt.xlabel("Pearsons correlation")
plt.savefig("RandomSeqs_Correlations.pdf")
plt.show()

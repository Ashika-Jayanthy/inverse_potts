#!/usr/bin/python3
# @Author: Ashika Jayanthy
# @Title: Core functions and classes for Inverse Potts formulation of Cell States

from scipy.special import logsumexp
import matplotlib.pyplot as plt
from Bio import SeqIO
from numba import jit
import numpy as np
import random
import time

letter_dict = {'A': 1, 'T': -1, 'G': 2, 'C': -2, 'N': 0}


def DNA(length):
    """
    Return a random DNA sequence of length 'length'
    """
    return ''.join(random.choice('CGTA') for _ in range(length))


def sequence_to_spins(string_sequence):
    """
    Convert string sequence to numerical sequence
    """
    return np.array([letter_dict[i] if i in letter_dict.keys() else 0 for i in string_sequence.upper()])


def name_to_sequence(gene_names, gene_fasta):
    gene_ids = SeqIO.index(gene_fasta, "fasta")  # Dict with seq
    gene_sequences = []
    for i, gene in enumerate(gene_names):
        seq = gene_ids[gene].seq.upper()
        gene_sequences.append(seq)
    return gene_sequences


@jit
def logZ(energies):
    return logsumexp(-energies)


@jit
def pair_corr(vector, periodic_boundary_conditions=True):
    L = len(vector)
    if periodic_boundary_conditions:
        return [vector[i] * vector[(i + 1) % (L)] for i in range(L)]
    else:
        return [vector[i] * vector[i + 1] for i in range(L - 1)]


class InferHJ():
    def __init__(self, transcriptome_spins, transcriptome_pairs, masked_arrays, counts, h_init=None, J_init=None, k=1e-5, e1=1e12, e2=1e12, max_runs=100, periodic_boundary_conditions=True):

        self.spin_matrix = transcriptome_spins
        self.nseqs, self.max_length = self.spin_matrix.shape
        self.pair_matrix = transcriptome_pairs

        self.k = k

        self.masked_spins = masked_arrays[0]
        self.masked_pairs = masked_arrays[1]

        self.max_runs = max_runs
        self.e1 = e1
        self.e2 = e2

        if counts is None:
            self.counts = np.ones(self.nseqs) / self.nseqs
        else:
            self.counts = counts / np.sum(counts)

        self.possible_spins = [-2, -1, 0, 1, 2]
        self.possible_pairs = [-4, -2, -1, 0, 1, 2, 4]

        if h_init is None:
            self.h = np.ones((nseqs, max_length))
        else:
            self.h = h_init

        if (J_init is None) and (periodic_boundary_conditions == True):
            self.J = np.ones((nseqs, max_length))
        elif (J_init is None) and (periodic_boundary_conditions == False):
            self.J = np.ones((nseqs, max_length - 1))
        else:
            self.J = J_init

    @jit
    def calc_P_obs(self):
        self.P1_obs = np.array([(self.masked_spins[i].T * self.counts).sum(1)
                                for i in range(len(self.possible_spins))])
        self.P2_obs = np.array([(self.masked_pairs[i].T * self.counts).sum(1)
                                for i in range(len(self.possible_pairs))])
        return

    @jit
    def calc_E(self):
        self.E = (np.sum(self.spin_matrix * self.h, axis=1) +
                  np.sum(self.pair_matrix * self.J, axis=1))
        return

    @jit
    def calc_P_model(self):
        self.P_model = np.exp(-self.E * self.k) / \
            np.sum(np.exp(-self.E * self.k))
        self.P1_model = np.array([(self.masked_spins[i].T * self.P_model).sum(1)
                                  for i in range(len(self.possible_spins))])
        self.P1_model[self.P1_model == 0] = 1
        self.P2_model = np.array([(self.masked_pairs[i].T * self.P_model).sum(1)
                                  for i in range(len(self.possible_pairs))])
        self.P2_model[self.P2_model == 0] = 1
        return

    @jit
    def calc_costs(self):
        self.cost1 = np.log(self.P1_obs / self.P1_model)
        self.cost1[self.cost1 == -np.inf] = 0
        self.cost2 = np.log(self.P2_obs / self.P2_model)
        self.cost2[self.cost2 == -np.inf] = 0
        self.cost = np.round(np.mean(self.counts / self.P_model), 15)
        return

    @jit
    def update_hJ(self):
        for i in range(len(self.possible_spins)):
            h_update = self.masked_spins[i] * self.cost1[i]
            self.h += self.e1[i] * (h_update)
        for i in range(len(self.possible_pairs)):
            J_update = self.masked_pairs[i] * self.cost2[i]
            self.J += self.e2[i] * (J_update)
        return

    def run_inference(self):
        self.calc_P_obs()
        self.calc_E()
        self.calc_P_model()
        self.calc_costs()

        if np.abs(1 - self.cost) > 1e-15:
            old_cost = np.inf
            nruns = 0

            while np.abs(1 - old_cost) > np.abs(1 - self.cost):
                if nruns > self.max_runs:
                    break
                start = time.time()
                old_cost = self.cost
                self.update_hJ()
                self.calc_E()
                self.calc_P_model()
                self.calc_costs()
                nruns += 1
                end = time.time()
                print(end - start)
        return self.h, self.J, self.E, np.nan_to_num(self.P_model)

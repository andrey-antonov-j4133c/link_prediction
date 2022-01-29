import numpy as np
import os
import networkx as nx

from data.synthetic.acMark.acMark import acmark_model
from data.synthetic.helpers import write_features, write_network
from fixed_params import *

from settings import *


def generate_network(alpha, phi_c, beta, sigma, omega):
    S, X, Label = acmark_model.acmark(n, m, k, d,max_deg, M, D, alpha, phi_c, beta, sigma, omega)
    return nx.convert_matrix.from_scipy_sparse_matrix(S), X


def main():
    alpha = [1]
    phi_c = [1]
    beta = [0.1]
    sigma = [0]
    omega = [0.2]

    for a in alpha:
        for p in phi_c:
            for b in beta:
                for s in sigma:
                    for o in omega:
                        G, X = generate_network(a, p, b, s, o)
                        write_network(G, SYNTHETIC_PATH + f'acMark/acMark-a={a};p={p};b={b};s={s};o={o}/')
                        write_features(X, SYNTHETIC_PATH + f'acMark/acMark-a={a};p={p};b={b};s={s};o={o}/')


main()

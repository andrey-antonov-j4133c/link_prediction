import networkx as nx

from data.synthetic.acMark.acMark import acmark_model
from data.synthetic.helpers import write_features, write_network
from params import *

from settings import *


def generate_network(a, b, s, o):
    S, X, Label = acmark_model.acmark(n, m, k, d, max_deg, M, D, a, phi_c, b, s, o)
    return nx.convert_matrix.from_scipy_sparse_matrix(S), X


def main():
    for a in alpha:
        for b in beta:
            for s in sigma:
                for o in omega:
                    for r in RUN:
                        G, X = generate_network(a, b, s, o)
                        write_network(G, SYNTHETIC_PATH + f'acMark/acMark-a={a};b={b};s={s};o={o}_run{r}/')
                        write_features(X, SYNTHETIC_PATH + f'acMark/acMark-a={a};b={b};s={s};o={o}_run{r}/')


main()

import random
from networkx.generators.community import LFR_benchmark_graph

from settings import *

from data.synthetic.LFR.params import *
from data.synthetic.helpers import write_network


def generate_network(tau1, tau2, mu, average_degree, seed):
    return LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=average_degree, min_community=25, seed=random.randint(0,100))


def main():
    seed = random.randint(0, 1000)

    for t1 in tau1:
        for t2 in tau2:
            for m in mu:
                for a in average_degree:
                    for r in run:
                        G = generate_network(t1, t2, m, a, seed)
                        write_network(G, SYNTHETIC_PATH + f'LFR/LFR-t1={t1};t2={t2};mu={m};average_degree={a};run{r}/')


main()

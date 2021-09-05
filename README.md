# Link prediction quality classes

This work is an addition to the paper "Link predictability classes in complex networks" by Stavinova et al.

## Abstract

In this paper, we study how the observed quality of a network link prediction model applied to a part of a network can be further used for the analysis of the whole network. Namely, we first show that it can be determined for a part of the network which topological features of node pairs lead to a certain level of link prediction quality. Based on it, we construct a link predictability (prediction quality) classifier for the network links. This is further used in the other part of the network for controlling the link prediction quality typical for the model on the par- ticular network. The actual prediction model is not used then already. Experiments with synthetic and real-world networks show a good per- formance of the proposed pipeline.


## Experiments

<p align="left">
  <img width="500"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/pipeline.png">
</p>


We run experiments on synthetic and real networks to establish the quality of predicting a spesific edge in a graph, using topological features (such as Jaccard’s coefficient, Adamic-Adar index and others) as input.

We split the data in the following way:

<p align="left">
  <img width="500"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/data_spit.png">
</p>

Then we run our first model, to reconstruct links between nodes, based on the features. Based on the absolute prediction error we then train the next model to give us prediction quality (1-good, 0-bad).

We also calculate feature importance for each classifier and network, and look at the distribution of each feature:

<p align="left">
  <img width="500"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/feature_imp.png">
</p>

<p align="left">
  <img width="500"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/feature_dist.png">
</p>


## Instation

```
pip install requirements.txt
```
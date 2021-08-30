# Link prediction quality classes

In this work we attempt to learn quality classes of link prediction model based on topological features. We use different link prediciton models to predict edges&
Then we use edge-based topological features of like Jacard coefficent to predict witch edges will be restored by the model with good quality, and witch will not. 

We currently use the following networks and features:
- LFR benchmark graph and couple of simple features from NetworkX (LFR_Forest-pred_Logi-class.ipynp)
- Real networks from Optomal Link prediction (https://github.com/Aghasemian/OptimalLinkPrediction) and their features

Example of feature generation:
<p align="left">
  <img width="500"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/features.png">
</p>

Then run a link prediction model and classify the link-predictions.
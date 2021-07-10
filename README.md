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

Then run a link prediction model and classify the the link-predictions. We divide the error value into 4 labels (great: [0; 0.25], good: [0.25; 0.5], ok: [0.5, 0.75], poor: [0.75, 1]).
Here is a distribution of ABS-error of one of one of the models:
<p align="left">
  <img width="800"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/error_distribution.png">
</p>

Finaly we learn on the quality classes and topological features and later try to predict the quality class based on the features:
<p align="left">
  <img width="800"src="https://raw.githubusercontent.com/andrey-antonov-j4133c/link_prediction/master/images/conf_matrix.png">
</p>

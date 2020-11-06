# MOGWO feature selection
Irrelevant or partially relevant features can negatively impact model performance. in this project a multi-objective grey wolf optimzer is implemented to select the features which contribute most to the output in which we are interested in.

## Fitness function
The fitness function is computed as follows:
```
F = w * accuracy + (1-w) * 1/(number_of_selected_features)
```
This fitness function allow the algorithm to drop as many features as possible while keeping the accuary high.

## Experimental results
The [mobile price dataset](https://www.kaggle.com/iabhishekofficial/mobile-price-classification) was employed to test the algorithm.
### without feature selection
number of features: 20
accuracy: 90.4%
### with feature selection
number of features: 5
accuracy: 94.6%
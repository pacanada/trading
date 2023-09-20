This module attempts to train a transformer with univariate open feature for all cryptos.

Some comments from the notebooks exploration (transformer_model_categorical_univariate.ipynb)

## Comments

The idea of this notebooks is to explore the feasability of training a transformer model for classification

- Lets assume that we use only 700 last record and there is no data leakage when performing. For xgboost (no sequence) it was enough to have the ewm, rsi column value for the most recent interval, no need to consider sequences (and possibly null values if we are not fetching enough samples (less than the intervals used for the domain markers)) as in the transformer approach 
- Lets agree for now in target 20 min
- Doesnt make sense to normalize the target since we are classifying
- neg target means it increase the price: class 0 means it will increase, 6 it will decrease
- predictions look okeish (more exhaustive check must be made)

TODO:
- give wieght to unbalance classes-> doesnt seem to help, on the contrary, the model is predicting all the classes with the same frequency
- include linear layer before positional embedding
- check if just open feature and a linear layer is better (not necessarily but for simplicity, let's start with that)
- Be careful with the normalization!!! we were sampling randomly and the sequences were not in order
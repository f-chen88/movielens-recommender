# DataLabAssignement1

## generate.py
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use and it saves the complete matrix as *output.npy*.
DO NOT CHANGE THE LINES TO LOAD AND SAVE THE TABLE. Between those to you are free to use any method for matrix completion. 
Example:
  > python3 generate.py --name ratings_train.npy

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt

## Our method: Bayesian SVD++ *flipped*, averaged over 8 MCMC chains

Variant « flipped » of Rendle, Zhang & Koren (2019), *On the Difficulty of Evaluating Baselines: A Study on Recommender Systems* (arXiv:1905.01395), without the time term (the data has no timestamps).

- **Model:** matrix factorization with biases. Each user is also described by the movies they rated, and each movie by the users who rated it (implicit feedback on both sides, normalised by |N|^-1/2).
- **Learning:** Bayesian, by Gibbs sampling (MCMC), with the [myFM](https://myfm.readthedocs.io/) library. Regularisation strengths are sampled from the data, so there is no λ to tune. The predictions of 8 independent chains (seeds 0 to 7) are averaged.
- **Hyperparameters:** rank 16 and 145 MCMC iterations per chain, chosen by 5-fold cross-validation on the train set with Optuna (30 trials). The number of chains (8) was chosen by grid search over {1, 2, 4, 8}.
- **Cost:** about 95 s on `ratings_train.npy` and about 135 s on train + test (2 CPU cores), under 1 GB of memory.
- **Tested** from an empty environment with Python 3.11 (myfm 0.4.0) and 3.12 (myfm 0.5.0). Both give the same output.
- **Fallback:** if `myfm` cannot be imported, `generate.py` prints a warning and uses a bias model, so `output.npy` is always produced.

### Local results

Same protocol for every method:
- 5-fold cross-validation on the train set (same folds for all methods), 30 Optuna trials per method;
- then refit on the full train set and a single measurement on the test set.

| Method | RMSE validation (5 folds) | RMSE test |
|---|---|---|
| Neighbourhood (course) | 0.994 ± 0.009 | 1.006 |
| Matrix factorization, gradient descent (course) | 0.905 ± 0.009 | 0.912 |
| SVD++ (Koren 2008) | 0.871 ± 0.011 | 0.877 |
| Bayesian matrix factorization | 0.864 ± 0.011 | 0.867 |
| Bayesian SVD++ (paper) | 0.864 ± 0.010 | 0.867 |
| Bayesian SVD++ flipped | 0.853 ± 0.011 | 0.856 |
| Neighbourhood on bias residuals (Koren 2010) | 0.870 ± 0.009 | 0.875 |
| **Bayesian SVD++ flipped, 8 chains (submitted)** | **0.848 ± 0.011** | **0.850** |
| Blend 0.84 × previous + 0.16 × neighbourhood on residuals | 0.848 ± 0.011 | 0.849 |

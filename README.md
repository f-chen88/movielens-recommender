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

## Our method: Bayesian SVD++ *flipped* with movie side information, averaged over 8 MCMC chains

Variant « flipped » of Rendle, Zhang & Koren (2019), *On the Difficulty of Evaluating Baselines: A Study on Recommender Systems* (arXiv:1905.01395). It is a factorization machine (Rendle 2010) learned by Gibbs sampling with [myFM](https://myfm.readthedocs.io/):

- **user** = [user id | movies rated by the user, normalised by |N(u)|^-0.51]
- **movie** = [movie id | users who rated the movie, normalised by |N(i)|^-0.50 | genres (one-hot, norm 1) | release period (5-year bins)]

The prediction is the sum of all linear terms and all pairwise interactions between these features, averaged over the posterior samples of 8 independent chains (seeds 0 to 7). Genres and release period mostly help movies with few ratings: the median movie has 3 ratings in the train set.

- **Data:** genres and titles come from `namesngenre2.npy`, the corrected metadata file. `generate.py` looks for it in the working directory, next to `generate.py`, and next to the ratings file. If it is not found, it prints a warning and runs the same model without side information (rank 16, tuned priors).
- **Hyperparameters:** rank 42, 150 iterations, myFM hyperpriors `alpha_0=0.25, beta_0=0.016, gamma_0=0.416, reg_0=0.461`.
  - **Platform-like protocol:** the platform trains on more data than our local train, so we tuned on `ratings_train` + one half of `ratings_test` (1.5x the data), measured on the other half, then swapped the halves.
  - **Search:** Optuna, each trial = mean of 2 chains.
  - **Validation:** the selected configuration was validated with 4 averaged chains on seeds not used during the search.
- **Cost:** about 225 s on `ratings_train.npy` and about 290 s on train + test (2 CPU cores), about 1.3 GB of memory.
- **Tested** with Python 3.11 (myfm 0.4.0) and 3.12 (myfm 0.5.0).

### Results

| Method | Local test (train → test) | Platform-like protocol (4 chains) | Platform |
|---|---|---|---|
| Bayesian SVD++ flipped, default priors | 0.8501 | 0.8303 | 0.809 |
| Stacking (flipped + ordered probit + kNN), default priors | 0.8473 | 0.8280 | 0.807 |
| Stacking, tuned priors | 0.8472 | 0.8269 | 0.805 |
| + genres, rank 16 | | 0.8157 | |
| + genres + release period, rank 16 | | 0.8091 | |
| **+ genres + release period (5-year bins), rank 42 (submitted)** | **0.8242** | **0.8076** | |

With the side information, stacking with the ordered-probit model and the neighbourhood model no longer helps much (0.8076 → 0.8072), so we went back to a single model.

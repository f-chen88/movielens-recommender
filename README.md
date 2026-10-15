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

## Our method: FWLS stacking of Bayesian SVD++ *flipped* with two complementary models

Final prediction = feature-weighted linear stacking (FWLS, Sill et al. 2009, *Feature-Weighted Linear Stacking*, arXiv:0911.0460) of three models:

1. **Bayesian SVD++ flipped** (Rendle, Zhang & Koren 2019, arXiv:1905.01395), Gaussian likelihood, 8 MCMC chains (rank 16, 145 iterations). This was our previous submission (platform RMSE 0.809).
2. **The same model with an ordered-probit likelihood** (myFM `MyFMOrderedProbit`, 10 classes for 0.5 to 5 stars), 8 chains. It treats ratings as ordered categories instead of real numbers with uniform noise. The expected rating is computed from the posterior of the latent score and of the cutpoints (Gaussian approximation).
3. **Item-based neighbourhood on the residuals of a bias model** (Koren 2010): shrunk Pearson similarity, k = 21. It captures local item-to-item effects that a rank-16 factorization smooths out.

**Hyperparameters at platform scale.** The platform trains on more data than our local train (0.809 on the platform vs 0.850 locally for the same model). We therefore re-tuned on a platform-like protocol: train on `ratings_train` + one half of `ratings_test` (1.5x the data), measure on the other half, then swap the halves. Every candidate was validated with 4 averaged chains on seeds not used during the search.
- Rank, number of iterations, burn-in and initialisation: the previous values (rank 16, about 150 iterations, no burn-in, init 0.1) are already on the plateau. Wider ranges (rank 4–48, 100–500 iterations) give nothing better.
- myFM hyperpriors (`alpha_0, beta_0, gamma_0, reg_0`), left at 1 until now, and the exponent of the implicit normalisation |N|^-ex (Koren uses 1/2): `alpha_0=0.885, beta_0=0.15, gamma_0=0.366, reg_0=7.806`, `ex_u=0.373, ex_i=0.417`. Gaussian model: 0.8303 → 0.8273; ordered probit: 0.8323 → 0.8305. The priors give about 2/3 of the gain, the exponents 1/3.
- Neighbourhood: k = 21, shrinkage 88 kept, because the better kNN found (alone 0.8627 → 0.8595) does not improve the stack.

**Stacker.** The weight of each model depends on the cell, through 7 meta-features: constant, log number of ratings of the movie, log number of ratings of the user, user mean, user standard deviation, fraction of half-star ratings of the user, movie standard deviation. That gives 7 × 3 = 21 weights. They were learned by ridge regression on the predictions of the platform-like protocol (base models trained on train + one half of the test, predictions on the other half; ridge penalty chosen by 5-fold cross-validation), and are hard-coded in `generate.py`.

- **Cost:** about 290 s on `ratings_train.npy` and about 410 s on train + test (2 CPU cores), under 1 GB of memory.
- **Lighter variant:** set `STACK = "reg"` at the top of `generate.py` to keep only the Bayesian SVD++ flipped (8 chains). It is about 2.5 times faster (about 130 s).
- **Fallback:** if `myfm` cannot be imported, `generate.py` uses a bias model, so `output.npy` is always produced.

### Local results (train → test)

| Method | Local test (train → test) | Platform-like protocol (4 chains) | Platform |
|---|---|---|---|
| Bayesian SVD++ flipped, default priors | 0.8501 | 0.8303 | 0.809 |
| Stacking, default priors | 0.8473 | 0.8280 | 0.807 |
| Bayesian SVD++ flipped, tuned priors (`STACK = "reg"`) | 0.8492 | 0.8273 | |
| **Stacking, tuned priors (submitted)** | **0.8472** | **0.8269** | |

All methods use the same protocol: 5-fold cross-validation on the train set with the same folds for every method, then a single measurement on the test set.

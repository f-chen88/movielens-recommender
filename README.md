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

**Stacker.** The weight of each model depends on the cell, through 7 meta-features: constant, log number of ratings of the movie, log number of ratings of the user, user mean, user standard deviation, fraction of half-star ratings of the user, movie standard deviation. That gives 7 × 3 = 21 weights. They were learned by ridge regression on out-of-fold predictions (5 folds of the train set, ridge penalty chosen by nested cross-validation) and are hard-coded in `generate.py`. No information from the test set was used.

- **Cost:** about 290 s on `ratings_train.npy` and about 410 s on train + test (2 CPU cores), under 1 GB of memory.
- **Lighter variant:** set `STACK = "reg+knn"` at the top of `generate.py` to drop the probit model. It is about twice as fast, with a test RMSE of 0.8482.
- **Fallback:** if `myfm` cannot be imported, `generate.py` uses a bias model, so `output.npy` is always produced.

### Local results (train → test)

| Method | RMSE test |
|---|---|
| Bayesian SVD++ flipped, 8 chains (previous submission) | 0.8501 |
| Stacking reg + neighbourhood | 0.8482 |
| **Stacking reg + probit + neighbourhood (submitted)** | **0.8473** |

All methods use the same protocol: 5-fold cross-validation on the train set with the same folds for every method, then a single measurement on the test set.


import os
os.environ.setdefault("TQDM_DISABLE", "1")   # must be set before tqdm is imported: no myFM progress bars
import numpy as np
from tqdm import tqdm, trange
import argparse
import time
import warnings

import scipy.sparse as sp

# Bayesian SVD++ flipped (Rendle, Zhang & Koren, 2019), averaged over 8 independent MCMC chains.
# Hyperparameters chosen by 5-fold cross-validation on the train set (Optuna): rank 16, 145 iterations, 8 chains.
RANK, N_ITER, N_CHAINS = 16, 145, 8

warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")        # myfm <= 0.4 on Python <= 3.11


def bayesian_svdpp_flipped(R, seed):
    """One MCMC chain. R: movies x users (NaN = missing). Returns the full movies x users prediction matrix.

    rating(i, u) = w0 + w_u + w_i + < v_u + |N(u)|^-1/2 sum_{j in N(u)} y_j , v_i + |N(i)|^-1/2 sum_{w in N(i)} x_w >
    N(u) = movies rated by u, N(i) = users who rated i (in the input matrix only).
    """
    import myfm
    m, n = R.shape
    M = ~np.isnan(R)
    Nu = sp.csr_matrix(M.T.astype(float))                                       # user  -> movies rated
    Nu = sp.diags(1 / np.sqrt(np.maximum(np.asarray(Nu.sum(1)).ravel(), 1))) @ Nu
    Ni = sp.csr_matrix(M.astype(float))                                         # movie -> users who rated it
    Ni = sp.diags(1 / np.sqrt(np.maximum(np.asarray(Ni.sum(1)).ravel(), 1))) @ Ni
    Xu = sp.hstack([sp.identity(n, format="csr"), Nu]).tocsr()                  # [ user id  | N(u) ]
    Xi = sp.hstack([sp.identity(m, format="csr"), Ni]).tocsr()                  # [ movie id | N(i) ]

    ii, uu = np.where(M)
    fm = myfm.MyFMRegressor(rank=RANK, init_stdev=0.1, random_seed=seed)
    fm.fit(sp.csr_matrix((len(ii), 0)), R[ii, uu],
           X_rel=[myfm.RelationBlock(uu, Xu), myfm.RelationBlock(ii, Xi)],
           n_iter=N_ITER, group_shapes=[n, m, m, n])                           # groups: u, N(u), i, N(i)

    # Full matrix computed from the MCMC samples (much faster than predicting millions of cells one by one).
    pu, P = Xu.shape[1], np.zeros((m, n))
    Xu2, Xi2 = Xu.multiply(Xu).tocsr(), Xi.multiply(Xi).tocsr()
    for w0, w, V in zip(fm.w0_samples, fm.w_samples, fm.V_samples):
        Au, Bi = Xu @ V[:pu], Xi @ V[pu:]
        hu = 0.5 * ((Au ** 2).sum(1) - (Xu2 @ (V[:pu] ** 2)).sum(1))           # interactions inside the user block
        hi = 0.5 * ((Bi ** 2).sum(1) - (Xi2 @ (V[pu:] ** 2)).sum(1))           # interactions inside the movie block
        P += w0 + (Xu @ w[:pu] + hu)[None, :] + (Xi @ w[pu:] + hi)[:, None] + Bi @ Au.T
    return P / len(fm.w0_samples)


def bias_baseline(R, beta=3.0):
    """Fallback, only used if myFM cannot be imported: global mean + regularised movie and user biases."""
    M = ~np.isnan(R); Z = np.where(M, R, 0.0); mu = Z.sum() / M.sum()
    b_i = ((Z - mu) * M).sum(1) / (beta + M.sum(1))
    b_u = ((Z - mu - b_i[:, None]) * M).sum(0) / (beta + M.sum(0))
    return mu + b_i[:, None] + b_u[None, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    t0 = time.time()
    R = table.T.astype(float)                                                   # movies x users, NaN = missing
    try:
        P = 0
        for seed in range(N_CHAINS):
            P = P + bayesian_svdpp_flipped(R, seed) / N_CHAINS
            print(f"  chain {seed + 1}/{N_CHAINS} done ({time.time() - t0:.0f} s)")
        print(f"Bayesian SVD++ flipped, {N_CHAINS} chains (rank {RANK}, {N_ITER} iterations): {time.time() - t0:.0f} s")
    except ImportError as e:
        print(f"WARNING: myFM unavailable ({e}); falling back to the bias baseline.")
        P = bias_baseline(R)
    table = np.clip(P, 0.5, 5).T                                                # users x movies, every cell filled
    assert np.isfinite(table).all()

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        

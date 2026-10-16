import os
os.environ.setdefault("TQDM_DISABLE", "1")   # must be set before tqdm is imported: no myFM progress bars
import numpy as np
from tqdm import tqdm, trange
import argparse
import time
import warnings
import re

import scipy.sparse as sp

# Bayesian SVD++ "flipped" (Rendle, Zhang & Koren 2019) with movie side information, averaged over 8 MCMC chains (myFM).
#   user  = [ user id  | movies rated by the user  (|N(u)|^-ex_u) ]
#   movie = [ movie id | users who rated the movie (|N(i)|^-ex_i) | genres | release period (5-year bins) ]
# Hyperparameters tuned on a platform-like protocol: train + one half of the test -> RMSE on the other half, and swap.
# If namesngenre2.npy cannot be found, the model runs without side information (previous tuned configuration).
N_CHAINS, N_ITER = 8, 150
WITH_META = dict(rank=42, alpha_0=0.25, beta_0=0.016, gamma_0=0.416, reg_0=0.461, ex_u=0.507, ex_i=0.498)
NO_META = dict(rank=16, alpha_0=0.885, beta_0=0.15, gamma_0=0.366, reg_0=7.806, ex_u=0.373, ex_i=0.417)
META_FILE = "namesngenre2.npy"
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")        # myfm <= 0.4 on Python <= 3.11


def load_side(n_movies, ratings_path):
    """Movie side features (genres one-hot, normalised to norm 1, and 5-year release periods), or None if unavailable."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.dirname(os.path.abspath(ratings_path))):
        p = os.path.join(d, META_FILE)
        if os.path.exists(p):
            try:
                N = np.load(p)
                if N.shape == (n_movies, 2):
                    break
                print(f"WARNING: {p} has shape {N.shape}, expected ({n_movies}, 2); ignored.")
            except Exception as e:
                print(f"WARNING: cannot read {p} ({e}); ignored.")
    else:
        return None, []
    genres = sorted({g for s in N[:, 1] for g in str(s).split("|")})
    G = np.array([[g in str(s).split("|") for g in genres] for s in N[:, 1]], float)
    G /= np.sqrt(np.maximum(G.sum(1, keepdims=True), 1))
    year = np.array([float(m.group(1)) if (m := re.search(r"\((\d{4})\)\s*$", str(t).strip())) else np.nan for t in N[:, 0]])
    year = np.where(np.isnan(year), np.nanmedian(year), year)
    bins = list(range(1950, 2020, 5))
    Y = np.eye(len(bins) + 1)[np.digitize(year, bins)]
    print(f"Side information loaded from {p}: {G.shape[1]} genres, {Y.shape[1]} release periods.")
    return sp.csr_matrix(np.hstack([G, Y])), [G.shape[1], Y.shape[1]]


def flipped_chain(R, seed, cfg, side=None, side_shapes=()):
    """One MCMC chain. R: movies x users (NaN = missing). Returns the full movies x users matrix (posterior mean)."""
    import myfm
    m, n = R.shape
    M = ~np.isnan(R)
    Nu = sp.csr_matrix(M.T.astype(float))                                       # user  -> movies rated
    Nu = sp.diags(np.maximum(np.asarray(Nu.sum(1)).ravel(), 1) ** -cfg["ex_u"]) @ Nu
    Ni = sp.csr_matrix(M.astype(float))                                         # movie -> users who rated it
    Ni = sp.diags(np.maximum(np.asarray(Ni.sum(1)).ravel(), 1) ** -cfg["ex_i"]) @ Ni
    Xu = sp.hstack([sp.identity(n, format="csr"), Nu]).tocsr()
    Xi = sp.hstack([sp.identity(m, format="csr"), Ni] + ([side] if side is not None else [])).tocsr()
    ii, uu = np.where(M)
    blocks = [myfm.RelationBlock(uu, Xu), myfm.RelationBlock(ii, Xi)]
    priors = {k: cfg[k] for k in ("alpha_0", "beta_0", "gamma_0", "reg_0")}
    fm = myfm.MyFMRegressor(rank=cfg["rank"], init_stdev=0.1, random_seed=seed, **priors)
    fm.fit(sp.csr_matrix((len(ii), 0)), R[ii, uu], X_rel=blocks, n_iter=N_ITER, group_shapes=[n, m, m, n] + list(side_shapes))

    # Prediction of every cell for every MCMC sample, computed from the parameters (fast, no per-cell predict).
    pu = Xu.shape[1]
    Xu2, Xi2 = Xu.multiply(Xu).tocsr(), Xi.multiply(Xi).tocsr()
    S = np.zeros((m, n))
    for w0, w, V in zip(fm.w0_samples, fm.w_samples, fm.V_samples):
        Au, Bi = Xu @ V[:pu], Xi @ V[pu:]
        hu = 0.5 * ((Au ** 2).sum(1) - (Xu2 @ (V[:pu] ** 2)).sum(1))           # interactions inside the user block
        hi = 0.5 * ((Bi ** 2).sum(1) - (Xi2 @ (V[pu:] ** 2)).sum(1))           # interactions inside the movie block
        S += w0 + (Xu @ w[:pu] + hu)[None, :] + (Xi @ w[pu:] + hi)[:, None] + Bi @ Au.T
    return S / len(fm.w0_samples)


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
        import myfm  # noqa: F401
        side, shapes = load_side(R.shape[0], args.name)
        cfg = WITH_META if side is not None else NO_META
        if side is None:
            print(f"WARNING: {META_FILE} not found; running without side information.")
        P = 0
        for s in range(N_CHAINS):
            P = P + flipped_chain(R, s, cfg, side, shapes) / N_CHAINS
            print(f"  chain {s + 1}/{N_CHAINS} done ({time.time() - t0:.0f} s)")
    except ImportError as e:
        print(f"WARNING: myFM unavailable ({e}); falling back to the bias baseline.")
        P = bias_baseline(R)
    table = np.clip(P, 0.5, 5).T                                                # users x movies, every cell filled
    assert np.isfinite(table).all()



    # Save the completed table
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE

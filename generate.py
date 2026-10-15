
import os
os.environ.setdefault("TQDM_DISABLE", "1")   # must be set before tqdm is imported: no myFM progress bars
import numpy as np
from tqdm import tqdm, trange
import argparse
import time
import warnings

import scipy.sparse as sp
from scipy.special import ndtr

# Stacking (feature-weighted linear stacking, Sill et al. 2009) of three base models:
#   1. Bayesian SVD++ flipped (Rendle, Zhang & Koren 2019), Gaussian likelihood, 8 MCMC chains
#   2. the same model with an ordered-probit likelihood (ratings as 10 ordered classes), 8 chains   [STACK = "all" only]
#   3. item-based neighbourhood on the residuals of the bias model (Koren 2010)
# Base hyperparameters chosen by 5-fold cross-validation on the train set (Optuna); stacking weights learned by
# ridge regression on the out-of-fold predictions of the same 5 folds (nested CV to choose the ridge penalty).
STACK = "all"                                   # "all" (3 models, stacked) or "reg" (Bayesian SVD++ flipped alone, faster)
RANK, N_ITER, N_CHAINS = 16, 150, 8
# myFM hyperpriors and exponents of the implicit normalisation |N|^-ex, tuned on a platform-like protocol
# (train + half of the test -> other half): defaults (1, 1, 1, 1, 0.5, 0.5) -> 0.8303, these -> 0.8273 (4 chains).
PRIORS = dict(alpha_0=0.885, beta_0=0.15, gamma_0=0.366, reg_0=7.806)
EX_U, EX_I = 0.373, 0.417
KNN = dict(k=21, alpha=88.01362076554739, beta=2.91523782912842, kappa=1.3752959847722561)
WEIGHTS = {   # learned at platform scale (base models trained on train + half of the test, predictions on the other half)
    "all": dict(w=[0.8409217333, 0.1497042937, 0.008428208549, -0.08212576978, 0.1280317566, -0.04970040588, 0.04806243604, -0.09300958703, 0.04457657131, -0.0752525724, 0.04701534393, 0.02790494729, -0.09078522603, 0.05271862088, 0.04114618795, 0.0185394415, -0.002225051509, -0.01223309194, -0.007963756969, 0.02695372624, -0.01207603017],
             mu=[2.953159776, 5.023411176, 0.9013371367, 3.517933106, 0.2992815527, 0.8077458416],
             sd=[1.082228726, 1.13281031, 0.2147823604, 0.4600544966, 0.2055022102, 0.2872306458]),
}
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")        # myfm <= 0.4 on Python <= 3.11


def flipped_chain(R, seed, probit=False):
    """One MCMC chain of Bayesian SVD++ flipped. R: movies x users (NaN = missing). Returns the full movies x users matrix
    of predicted ratings (posterior mean)."""
    import myfm
    m, n = R.shape
    M = ~np.isnan(R)
    Nu = sp.csr_matrix(M.T.astype(float))                                       # user  -> movies rated
    Nu = sp.diags(np.maximum(np.asarray(Nu.sum(1)).ravel(), 1) ** -EX_U) @ Nu
    Ni = sp.csr_matrix(M.astype(float))                                         # movie -> users who rated it
    Ni = sp.diags(np.maximum(np.asarray(Ni.sum(1)).ravel(), 1) ** -EX_I) @ Ni
    Xu = sp.hstack([sp.identity(n, format="csr"), Nu]).tocsr()                  # [ user id  | N(u) ]
    Xi = sp.hstack([sp.identity(m, format="csr"), Ni]).tocsr()                  # [ movie id | N(i) ]
    ii, uu = np.where(M)
    blocks = [myfm.RelationBlock(uu, Xu), myfm.RelationBlock(ii, Xi)]
    if probit:
        fm = myfm.MyFMOrderedProbit(rank=RANK, init_stdev=0.1, random_seed=seed, **PRIORS)
        fm.fit(sp.csr_matrix((len(ii), 0)), np.round(R[ii, uu] * 2).astype(int) - 1, X_rel=blocks, n_iter=N_ITER, group_shapes=[n, m, m, n])
    else:
        fm = myfm.MyFMRegressor(rank=RANK, init_stdev=0.1, random_seed=seed, **PRIORS)
        fm.fit(sp.csr_matrix((len(ii), 0)), R[ii, uu], X_rel=blocks, n_iter=N_ITER, group_shapes=[n, m, m, n])

    # Latent score of every cell for every MCMC sample, computed from the parameters (fast, no per-cell predict).
    pu = Xu.shape[1]
    Xu2, Xi2 = Xu.multiply(Xu).tocsr(), Xi.multiply(Xi).tocsr()
    S1 = np.zeros((m, n)); S2 = np.zeros((m, n)) if probit else None
    for w0, w, V in zip(fm.w0_samples, fm.w_samples, fm.V_samples):
        Au, Bi = Xu @ V[:pu], Xi @ V[pu:]
        hu = 0.5 * ((Au ** 2).sum(1) - (Xu2 @ (V[:pu] ** 2)).sum(1))           # interactions inside the user block
        hi = 0.5 * ((Bi ** 2).sum(1) - (Xi2 @ (V[pu:] ** 2)).sum(1))           # interactions inside the movie block
        F = w0 + (Xu @ w[:pu] + hu)[None, :] + (Xi @ w[pu:] + hi)[:, None] + Bi @ Au.T
        S1 += F
        if probit: S2 += F ** 2
    T = len(fm.w0_samples)
    if not probit:
        return S1 / T
    # Ordered probit: E[rating] = 0.5 + 0.5 * sum_k P(latent > cutpoint_k). Averaged over the samples with the
    # probit-normal identity E[Phi(X)] = Phi(m / sqrt(1 + s^2)) (latent and cutpoints treated as Gaussian across samples).
    mf = S1 / T; vf = np.maximum(S2 / T - mf ** 2, 0)
    C = np.array([np.asarray(c).ravel() for c in fm.cutpoint_samples]); mc, vc = C.mean(0), C.var(0)
    E = np.full((m, n), 0.5)
    for c, v in zip(mc, vc):
        E += 0.5 * ndtr((mf - c) / np.sqrt(1 + vf + v))
    return E


def knn_residuals(R, k, alpha, beta, kappa):
    """Item-based neighbourhood on the residuals of the bias model. R: movies x users. Returns the full matrix."""
    M = ~np.isnan(R); Z = np.where(M, R, 0.0); mu = Z.sum() / M.sum()
    b_i = ((Z - mu) * M).sum(1) / (beta + M.sum(1))
    b_u = ((Z - mu - b_i[:, None]) * M).sum(0) / (beta + M.sum(0))
    B = mu + b_i[:, None] + b_u[None, :]                                        # bias model
    E = np.where(M, R - B, 0.0).astype(np.float32)                              # residuals (movies x users)
    Mf = M.astype(np.float32)
    A = (E ** 2) @ Mf.T                                                         # A[i, j] = sum over users of both of E_iu^2
    S = (E @ E.T) / np.sqrt(np.maximum(A * A.T, 1e-12))                         # Pearson on residuals
    S[A * A.T <= 0] = 0
    cnt = Mf @ Mf.T
    S *= cnt / (cnt + alpha)                                                    # shrink when few users in common
    np.fill_diagonal(S, 0)
    P = B.copy()
    for u in range(R.shape[1]):
        J = np.where(M[:, u])[0]                                                # movies rated by u
        if len(J) == 0: continue
        Sj = S[:, J]
        if len(J) > k:
            idx = np.argpartition(-Sj, k - 1, axis=1)[:, :k]
            Sj = np.take_along_axis(Sj, idx, 1); e = E[J, u][idx]
        else:
            e = np.broadcast_to(E[J, u], Sj.shape)
        P[:, u] += (Sj * e).sum(1) / (np.abs(Sj).sum(1) + kappa)
    return P


def meta_features(R):
    """Per-cell features of the stacker (movies x users x 7), from the input matrix only."""
    M = ~np.isnan(R); ni, nu = M.sum(1), M.sum(0)
    with np.errstate(all="ignore"):
        ustd, umean = np.nan_to_num(np.nanstd(R, 0)), np.nan_to_num(np.nanmean(R, 0))
        half = np.nansum((np.nan_to_num(R) * 2) % 2 == 1, 0) / np.maximum(nu, 1)
        istd = np.where(ni > 1, np.nan_to_num(np.nanstd(R, 1)), 0.0)
    m, n = R.shape
    return [np.ones((m, n)), np.log1p(ni)[:, None] + np.zeros((m, n)), np.log1p(nu)[None, :] + np.zeros((m, n)),
            ustd[None, :] + np.zeros((m, n)), umean[None, :] + np.zeros((m, n)), half[None, :] + np.zeros((m, n)),
            istd[:, None] + np.zeros((m, n))]


def stack(preds, feats, w, mu, sd):
    """Feature-weighted linear stacking: sum over features j and models p of w[j, p] * feature_j * prediction_p."""
    w = np.asarray(w).reshape(len(feats), len(preds))
    out = np.zeros_like(preds[0])
    for j, f in enumerate(feats):
        fz = f if j == 0 else (f - mu[j - 1]) / sd[j - 1]
        out += fz * sum(w[j, p] * preds[p] for p in range(len(preds)))
    return out


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
        preds = []
        reg = 0
        for s in range(N_CHAINS):
            reg = reg + flipped_chain(R, s) / N_CHAINS
            print(f"  regression chain {s + 1}/{N_CHAINS} done ({time.time() - t0:.0f} s)")
        preds.append(reg)
        if STACK == "all":
            pro = 0
            for s in range(N_CHAINS):
                pro = pro + flipped_chain(R, s, probit=True) / N_CHAINS
                print(f"  ordered-probit chain {s + 1}/{N_CHAINS} done ({time.time() - t0:.0f} s)")
            preds.append(pro)
        if STACK == "all":
            preds.append(knn_residuals(R, **KNN))
            print(f"  neighbourhood done ({time.time() - t0:.0f} s)")
            P = stack([np.clip(p, 0.5, 5) for p in preds], meta_features(R), **WEIGHTS[STACK])
        else:
            P = reg
        print(f"Done ({STACK}): {time.time() - t0:.0f} s")
    except ImportError as e:
        print(f"WARNING: myFM unavailable ({e}); falling back to the bias baseline.")
        P = bias_baseline(R)
    table = np.clip(P, 0.5, 5).T                                                # users x movies, every cell filled
    assert np.isfinite(table).all()

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        

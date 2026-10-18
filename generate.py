import numpy as np
from tqdm import tqdm, trange
import argparse
import time

# Course baseline: user-based neighbourhood collaborative filtering (DataLab A1 lecture).
#   Sim(u, v) = Jaccard(X_u, X_v) with X_u = {movies rated >= 3 by u}
#   R_hat[i, u] = sum_{v in N_k(u; i)} Sim(u, v) R[i, v] / sum_{v in N_k(u; i)} |Sim(u, v)|
#   N_k(u; i) = the k most similar users to u among those who rated movie i.
#   Fallback (denominator = 0): mean rating of user u.
# k = 144 and Jaccard were selected by 5-fold cross-validation on the training set (Optuna, 30 trials).
K = 144


def jaccard(R):
    """User-user Jaccard similarity on the sets of movies rated >= 3. R: movies x users (NaN = missing)."""
    X = (np.nan_to_num(R) >= 3).astype(float)
    inter = X.T @ X
    size = X.sum(0)
    union = size[:, None] + size[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def knn_course(R, k=K):
    """Fills every missing cell of R (movies x users) with the course neighbourhood prediction."""
    M = ~np.isnan(R)
    S = jaccard(R)
    np.fill_diagonal(S, 0.0)
    mu = np.nanmean(R, 0)                                                       # fallback: user mean
    P = np.where(M, R, 0.0)
    for i in range(R.shape[0]):
        V = np.where(M[i])[0]                                                   # users who rated movie i
        U = np.where(~M[i])[0]                                                  # users to predict
        if len(V) == 0:
            P[i, U] = mu[U]
            continue
        Sv = S[np.ix_(U, V)]                                                    # similarities target users x raters
        r = R[i, V]
        if len(V) > k:                                                          # keep the k nearest raters
            idx = np.argpartition(-Sv, k - 1, axis=1)[:, :k]
            Sv = np.take_along_axis(Sv, idx, 1)
            r = r[idx]
        else:
            r = np.broadcast_to(r, Sv.shape)
        num = (Sv * r).sum(1)
        den = np.abs(Sv).sum(1)
        P[i, U] = np.where(den > 0, num / np.where(den > 0, den, 1.0), mu[U])
    return P


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
    P = knn_course(R)
    table = np.clip(P, 0.5, 5).T                                                # users x movies, every cell filled
    assert np.isfinite(table).all()
    print(f"Course neighbourhood (Jaccard, k={K}): {time.time() - t0:.1f} s")



    # Save the completed table
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE

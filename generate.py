import numpy as np
from tqdm import tqdm, trange
import argparse
import time

import scipy.sparse as sp

# Course model: matrix factorization R ~ I U^T learned by full-batch gradient descent on the observed entries S
# (DataLab A1 lecture, "missing values" slide):
#   C(I, U) = sum_{(i,u) in S} (R_iu - <I_i, U_u>)^2 + lam ||I||_F^2 + mu ||U||_F^2
#   dC/dI = -2 E U + 2 lam I,   dC/dU = -2 E^T I + 2 mu U,   E = residuals on S (0 elsewhere)
#   I <- I - eta dC/dI,   U <- U - eta dC/dU        (simultaneous update, fixed step)
# Hyperparameters selected by 5-fold cross-validation on the training set (Optuna, 30 trials).
K, LAM, MU, ETA, N_ITER, SEED = 1, 0.014684803764875042, 0.9492861281053446, 0.0002730449359463328, 694, 0


def mf_course(R, k=K, lam=LAM, mu=MU, eta=ETA, n_iter=N_ITER, seed=SEED):
    """R: movies x users (NaN = missing). Returns the full matrix I @ U.T."""
    rng = np.random.default_rng(seed)
    m, n = R.shape
    ii, uu = np.where(~np.isnan(R))
    r = R[ii, uu]
    I = 0.1 * rng.standard_normal((m, k))
    U = 0.1 * rng.standard_normal((n, k))
    for _ in range(n_iter):
        E = sp.csr_matrix((r - np.einsum("ij,ij->i", I[ii], U[uu]), (ii, uu)), shape=(m, n))   # residuals on S
        gI = -2 * (E @ U) + 2 * lam * I
        gU = -2 * (E.T @ I) + 2 * mu * U
        I, U = I - eta * gI, U - eta * gU
    return I @ U.T


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
    P = mf_course(R)
    table = np.clip(P, 0.5, 5).T                                                # users x movies, every cell filled
    assert np.isfinite(table).all()
    print(f"Course matrix factorization (gradient descent, rank {K}): {time.time() - t0:.1f} s")



    # Save the completed table
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE

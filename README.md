# DataLab Assignment 1 — Group 16

Omar Bargach · Franck Chen · Nourcène Brahem

**Final model: flipped Bayesian SVD++ with movie metadata — platform RMSE 0.793.**

## Repository content

| File | Content |
|---|---|
| `generate.py` | Final model, the file run by the platform |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |
| `namesngenre2.npy` | Movie titles and genres (row j = column j of the ratings matrices), with `MovieLens-README.txt` |
| `report.pdf` | Report |
| `slides.pdf` | Presentation |
| `notebooks/seven_methods_comparison.ipynb` | Comparison of seven methods under Protocol 1 (Table 1 of the report) |
| `notebooks/metadata_experiments.ipynb` | Ablation A → E under Protocol 2 and final local test (Table 2 of the report) |

The ratings matrices are not included. To run the notebooks, place `ratings_train.npy` and `ratings_test.npy` in `notebooks/` for `seven_methods_comparison.ipynb`, and at the repository root (or in the folder given by the `A1_DATA` environment variable) for `metadata_experiments.ipynb`.

## Usage

```
pip install -r requirements.txt
python3 generate.py --name ratings_train.npy
```

`generate.py` completes the ratings table given by `--name` and saves it as `output.npy`. The lines that load and save the table are unchanged.

## Model

Flipped Bayesian SVD++ (Rendle, Zhang & Koren, 2019, *On the Difficulty of Evaluating Baselines*, arXiv:1905.01395), written as a factorization machine (Rendle, 2010) and learned by Gibbs sampling with [myFM](https://myfm.readthedocs.io/). Each rating (i, u) becomes one sparse row of features:

- **user block** = [user id | movies rated by the user, normalized by |N(u)|^-0.507]
- **movie block** = [movie id | users who rated the movie, normalized by |N(i)|^-0.498 | genres (normalized to norm 1) | release period (5-year bins, year parsed from the title)]

For one parameter set θ, the prediction is

ŷ_θ(x_iu) = w0 + b_u + b_i + ⟨P_u, Q_i⟩ + h_u + h_i,

where P_u and Q_i sum the embeddings of the active columns of each block (the SVD++ flipped terms, plus the genre and period embeddings in Q_i). The final prediction averages ŷ over 8 independent chains × 150 draws (seeds 0 to 7) and is clipped to [0.5, 5]. Genres and release period mostly help movies with few ratings: the median movie has 3 ratings in the train set.

- **Hyperparameters:** rank 42, 150 iterations, no burn-in, myFM hyperpriors `alpha_0=0.25, beta_0=0.016, gamma_0=0.416, reg_0=0.461`.
- **Cost:** about 300 s on the platform (train + test as input), about 1.3 GB of memory.
- **Metadata file:** `generate.py` looks for `namesngenre2.npy` next to itself, in the working directory and next to the ratings file (and their parent or `data/` folders). If the file cannot be found, it stops with a `FileNotFoundError` listing the folders searched.
- **Tested** with Python 3.11 (myfm 0.4.0) and 3.12 (myfm 0.5.0).

## Evaluation protocols

- **Protocol 1 (comparing methods):** 5 folds of the train set shared by all methods (one rating per movie and per user always kept for training); Optuna, 30 trials per method; the best setting is refitted on the full train set and evaluated once on the test set.
- **Protocol 2 (tuning at platform scale):** the platform trains on train + test, twice our training set. The test set is split into two halves H1, H2; each setting is trained on train + H1 and scored on H2, then the reverse. Optuna trials average 2 chains (seeds 0–1); selected settings are verified with 4 new chains (seeds 2–5).
- **Platform:** hidden eval set, never used for tuning.

## Results

| Configuration | Protocol 2 RMSE | Platform RMSE |
|---|---|---|
| Neighborhood (course) | — | 0.951 |
| Matrix factorization (course) | 0.8943 | 0.872 |
| A. Bayesian SVD++ flipped, 8 chains, default priors | 0.8308 | 0.809 |
| B. + priors tuned at platform scale | 0.8274 | 0.806 |
| C. + genres | 0.8158 | 0.799 |
| D. + release period | 0.8092 | — |
| **E. 5-year bins, re-tuned priors, rank 42 (submitted)** | **0.8077** | **0.793** |

Under Protocol 1 (local test), configuration E reaches 0.8242, against 0.9123 for the course matrix factorization and 0.8501 for configuration A.

## Data and license

`namesngenre2.npy` is included so that `generate.py` finds it at execution time. It is derived from **MovieLens Latest Small** (GroupLens, September 2018) and redistributed under the same license conditions; see `MovieLens-README.txt`. It contains no ratings.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

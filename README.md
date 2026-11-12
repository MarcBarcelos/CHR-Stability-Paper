# CHR Stability Paper

Code and results for *Do Fandoms Have a Voice? Tracing Stylistic Conventions Across Fan Communities and Genre Norms* (`paper.pdf`). The study looks at stylistic structure in fan-authored fiction on Archive of Our Own (AO3): which linguistic, semantic, and affective patterns co-vary across texts, and how the resulting stylistic dimensions differ by fandom, Maturity Rating, and publication time. Built on the [`Expressive Profile Pipeline`](https://anonymous.4open.science/r/Expressive-Profile-Pipeline/README.md) feature-extraction and EFA toolkit.

## Data

The corpus has six groups of around 3,000 works each:

- five fandom samples, published 2000–2024: Harry Potter (HP), Lord of the Rings (LOTR), The Magnus Archives (MAG), Percy Jackson (PJ), and Real Person Fiction (RPF)
- a random AO3 sample, published 2022–2026, used as a control

After deduplication, English-language filtering, and removing texts in the bottom 5th percentile of length (< 619 words), the working corpus is **16,966 texts**.

Raw AO3 texts and the per-text metric scores are **not shared**: they are treated as sensitive personal data under GDPR. The only data file tracked in this repository is:

- `dat/anonymized_data.csv` has one row per text (16,966 rows). Columns: `id`, `anon_author` (anonymized author ID), `fandom_label`, `rating`, `published`, and the 10 factor scores `F1`–`F10`.

## Pipeline

Run the steps in this order.

| Step | File | What it does | Needs private data? |
|---|---|---|---|
| 1 | `scripts/run_AO3_scoring.py` | Combines the two source corpora and scores all ~110 linguistic, semantic, affective, and structural metrics via `ep_pipeline` → `data/AO3metrics_full.csv` | Yes |
| 2 | `nbs/check_pre_efa_assumptions.ipynb` | Checks sample size, prunes for multicollinearity, applies a Yeo–Johnson normality transform, winsorizes outliers, runs KMO/Bartlett, and prunes variables iteratively by per-variable MSA → `data/efa_state.joblib` (74 variables retained) | Yes |
| 3 | `nbs/fit_efa_AO3.ipynb` | Runs parallel analysis for the factor count and fits an oblimin-rotated EFA (minres). Checks communalities and loadings, drops poor-fitting variables, refits (71 variables retained), and names the factors → `data/efa_factor_scores.csv` | Yes |
| 4 | `nbs/explore_factor_scores.ipynb` | Descriptive stats, distributions, factor intercorrelations, profile tests (PERMANOVA and dispersion, including length-adjusted versions), and trajectories of the 10 factors over time, by rating, and by fandom | Yes (uses word counts) |
| 5 | `scripts/classification.py` | 5-fold cross-validated multinomial logistic regression predicting fandom and Maturity Rating from the 10 factor scores. Rating uses `class_weight="balanced"`; fandom is unweighted because its groups are roughly balanced. 'Not Rated' texts are excluded from the rating model. Saves per-class metrics, coefficients, confusion matrices, and coefficient plots | No: runs on `dat/anonymized_data.csv` |

Steps 2–4 read from `data/`, which is not tracked in git. Without the private scoring output they will not run. Their saved outputs and figures are in `outputs/`. Step 5 can be reproduced from this repository alone:

```bash
python scripts/classification.py
```

Overall EFA fit: KMO ≈ 0.85, Bartlett p < .001. Ten factors were retained: parallel analysis supported up to 16, and solution quality plateaued at k ≈ 10–11. The 10 factors explain 53.8% of the variance.

## The 10 factors

| Factor | Name | High score means more... |
|---|---|---|
| F1 | Syntactic Complexity | subordination/coordination, deeper parse trees, sentence-length variability |
| F2 | Lexical Sophistication | longer/rarer words, higher AoA, harder readability |
| F3 | Negative Affect | anger/fear/sadness/disgust, lower valence |
| F4 | Lexical Richness | unique vocabulary, stronger topic focus, less repetition |
| F5 | Repetitiveness | duplicated n-grams/lines, OOV/unusual characters |
| F6 | Concreteness | concrete, noun-heavy, sensory language |
| F7 | Conversationality | dialogue markers: questions, quotes, exclamations |
| F8 | Positive Affect | joy/trust/anticipation, higher valence |
| F9 | Narrative Drift | semantic dispersion/movement across the text |
| F10 | Structural Variability | variable sentence/dependency/parse structure |

Full loadings are in the paper's appendix and in `outputs/results/efa_assumptions/efa_full_loadings_table.csv`. Interpretation notes are in the markdown cell at the top of `nbs/fit_efa_AO3.ipynb` and `nbs/explore_factor_scores.ipynb`.

## Outputs

- `outputs/results/classification_weighted/` contains the current classification results written by `scripts/classification.py`: overall and per-fold metrics (JSON), per-class metrics, and coefficients for fandom and rating.
- `outputs/results/classification/` holds an earlier run without class weights, kept for comparison.
- `outputs/results/efa_assumptions/` holds the EFA diagnostics: multicollinearity pairs and drop list, normality after transform, the MSA drop log, the factor summary, and the full loadings table.
- `outputs/visualizations/` has the diagnostic and classification plots: correlation heatmap, scree plot, communalities, MSA, word-length distribution, confusion matrices, and coefficient plots.
- `outputs/visualizations/factor_explore/` has the exploratory plots: factor distributions, radar profiles, and trajectories by fandom and rating.

## Setup

Steps 1–4 require `ep_pipeline` to be installed, the model weights and lexicons in `../models/`, and the private source corpora in `data/`. None of these are tracked in git. Step 5 needs only `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

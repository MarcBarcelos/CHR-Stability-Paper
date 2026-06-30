# CHR Stability Paper

A study of stylistic structure in fan-authored fiction on Archive of Our Own (AO3): which linguistic, semantic, and affective patterns co-vary across texts, and how the resulting stylistic dimensions differ by content rating, fandom, and publication time. Built on the [`Expressive Profile Pipeline`](https://github.com/MarcBarcelos/Expressive-Profile-Pipeline) feature-extraction and EFA toolkit.

## Data

The corpus contains a general AO3 sample, published 2022–2026 as well as a fixed subset spanning five specific fandoms (HP, LOTR, MAG, PJ, RPF), published 2000–2024. All six groups have arround 3,000 fanfictions included.

After deduplication, English-language filtering, and outliers in the bottomn 5th percentile in length, the working corpus is **16,966 texts**.

## Pipeline

Run in this order; each notebook persists its state so later stages can be re-run independently once the upstream cells have executed at least once.

| Step | File | What it does |
|---|---|---|
| 1 | `run_AO3_scoring.py` | Combines the two source corpora and scores all ~110 linguistic/semantic/affective/structural metrics via `ep_pipeline` → `data/AO3metrics_full.csv` |
| 2 | `check_pre_efa_assumptions.ipynb` | Sample size, multicollinearity pruning, normality (Yeo–Johnson transform), outlier winsorization, KMO/Bartlett, iterative per-variable MSA pruning → `data/efa_state.joblib` (74 variables retained) |
| 3 | `fit_efa_AO3.ipynb` | Parallel analysis for factor count, fits an oblimin-rotated EFA (minres), communality/loading diagnostics, drops poor-fitting variables and refits, names the factors → `data/efa_factor_scores.csv` |
| 4 | `explore_factor_scores.ipynb` | Descriptive stats, distributions, factor intercorrelations, and trajectories of the 10 factors over time, by content rating, and by fandom (point plots, radar charts, heatmaps) |
| — | `explore_chr.ipynb` | Scratch notebook for inspecting the scored metrics table; not part of the main pipeline |

Overall fit: KMO ≈ 0.85, Bartlett p < .001, 10 factors retained (parallel analysis supported up to 16; solution quality plateaued by k≈10–11).

## The 10 factors

| Factor | Name | High score means more... |
|---|---|---|
| F1 | Syntactic Complexity | subordination/coordination, deeper parse trees, sentence-length variability |
| F2 | Lexical Sophistication | longer/rarer words, higher AoA, harder readability |
| F3 | Negative Affect | anger/fear/sadness/disgust, lower valence |
| F4 | Lexical Richness | unique vocabulary, stronger topic focus, less repetition |
| F5 | Repetitiveness | duplicated n-grams/lines, OOV/unusual characters |
| F6 | Concreteness | concrete, noun-heavy, sensory language |
| F7 | Conversationality | dialogue markers — questions, quotes, exclamations |
| F8 | Positive Affect | joy/trust/anticipation, higher valence |
| F9 | Narrative Drift | semantic dispersion/movement across the text |
| F10 | Structural Variability | variable sentence/dependency/parse structure |

Full loading-level interpretation notes are in the markdown cell at the top of `fit_efa_AO3.ipynb` and `explore_factor_scores.ipynb`.

## Outputs

- `data/efa_state.joblib` — cleaned, scaled variable set + correlation matrix (input to step 3)
- `data/efa_factor_scores.csv` — one row per text: metadata + 10 factor scores (input to step 4)
- `outputs/visualizations/` — diagnostic and exploratory plots (correlation heatmap, scree plot, communalities, MSA, factor distributions, radar profiles, trajectories)
- `outputs/results/` — scoring checkpoints and tabular diagnostics (multicollinearity pairs, normality, MSA drop log, etc.)

## Setup

Requires `ep_pipeline` installed plus the model weights/lexicons in `../models/`. `data/` and `../models/` are not tracked in git.

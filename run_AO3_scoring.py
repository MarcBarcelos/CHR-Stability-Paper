from pathlib import Path
import pandas as pd

from ep_pipeline.config import MetricsConfig
from ep_pipeline.io import load_jsonl, load_csv, read_checkpoint, write_table
from ep_pipeline.models import pick_device, load_embedder, load_spacy_model
from ep_pipeline.scoring.get_td_linguistic import tokenize, load_td_nlp, get_td_metrics_batch
from ep_pipeline.scoring.get_other_linguistic import (mattr, windowed_unigram_entropy, trigram_entropy)
from ep_pipeline.scoring.get_semantic import (
    _empty_metrics, chunk_by_words, l2_normalize, adj_cos, shannon_entropy,
    compute_semantic_metrics)
from ep_pipeline.scoring.get_lexicon import (
    vad_metrics, emotion_metrics, score_lexicon,
    load_vad_lexicon, load_emotion_lexicon, load_norm_lexicon, load_norm_by_name)
from ep_pipeline.scoring.get_syntax_punct import (
    punctuation_metrics, syntax_complexity_metrics)
from ep_pipeline.assemble_corpus import extract_passage, build_text_table
from ep_pipeline.scoring.runner import map_with_checkpoints, map_with_checkpoints_batched, _key


print("Starting AO3 Scoring")

# PATHS FOR THIS PROJECT
PROJECT_ROOT  = Path("/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/chr_stability_paper")
MODEL_ROOT = Path("/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/models")
FULL_FP       = PROJECT_ROOT / "data" / "human_texts_2023_2025.csv"
# PROMPTS_FP    = PROJECT_ROOT / "outputs" / "ai_continuation" / "prompts.jsonl"    not needed for this project
# OUTPUT_FP     = PROJECT_ROOT / "outputs" / "ai_continuation" / "outputs.jsonl"
# EXCERPTS_FP   = PROJECT_ROOT / "outputs" / "ai_continuation" / "excerpts.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "results"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
TD_FP, SEM_FP, LING_FP, AFF_FP, SYN_FP = OUT_DIR / "checkpoints" / "td_checkpoint.csv", OUT_DIR / "checkpoints" / "sem_checkpoint.csv", OUT_DIR / "checkpoints" / "ling_checkpoint.csv", OUT_DIR / "checkpoints" / "affect_checkpoint.csv", OUT_DIR / "checkpoints" / "synpun_checkpoint.csv"
OUT_FP = PROJECT_ROOT / "data" / "AO3metrics_full.csv"

cfg = MetricsConfig(affect_mode="modern", spacy_model="en_core_web_md")

#MODELS AND LEXICONS FOR SCORING
E5_MODEL_PATH = MODEL_ROOT / "e5-small"

LEX_DIR = MODEL_ROOT / "lexicons"
CONC_LEX = load_norm_by_name(LEX_DIR / "concretness" / "Concreteness_ratings_Brysbaert_et_al_BRM.xlsx", term_field="Word", score_field="Conc.M")
if cfg.affect_mode == "modern":
    VAD_LEX  = load_vad_lexicon(LEX_DIR / "NRC-VAD-Lexicon-v2.1" / "NRC-VAD-Lexicon-v2.1.txt")
    EMO_LEX  = load_emotion_lexicon(LEX_DIR / "NRC-Emotion-Intensity-Lexicon" / "NRC-Emotion-Intensity-Lexicon-v1.txt")
    AOA_LEX  = load_norm_by_name(LEX_DIR / "AoA" / "AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx", term_field="Word", score_field="Rating.Mean")
    PREV_LEX = load_norm_by_name(LEX_DIR / "word_prevelance" / "English_Word_Prevalences.xlsx", term_field="Word", score_field="Prevalence")

# LOADING EXPENSIVE OBJECTS ONCE
embedder = load_embedder(E5_MODEL_PATH)
nlp_tok = load_spacy_model(cfg.spacy_model, for_tokenizing=True)
nlp_parse = load_spacy_model(cfg.spacy_model, for_tokenizing=False)
nlp_td = load_td_nlp(cfg.spacy_model, cfg.td_metrics)

# PREPING DATA
raw = (
    pd.read_csv(FULL_FP)
    .dropna(subset=["text"])
    .rename(columns={"work_id": "id"})
    .assign(text=lambda df: df["text"].str.replace(r"^Work Text:\s*", "", regex=True))
)
chunks_df = raw.assign(source="human")
records = chunks_df.to_dict("records")

#SCORING
def score_linguistic(r):
    toks = tokenize(r["text"], nlp_tok)
    H_mean, H_std, PPL = windowed_unigram_entropy(toks, entropy_window=cfg.entropy_window)
    H3, PPL3 = trigram_entropy(toks, cfg.trigram_test_frac, cfg.trigram_alpha, cfg.seed)
    mattr_score = mattr(toks, cfg.mattr_window)
    return {**r, "id": r["id"], "source": r["source"], "mattr": mattr_score, "H_unigram_win_mean_nats": H_mean, "H_unigram_win_std_nats": H_std, "PPL_unigram_win_mean": PPL, "H_3gram_self_nats": H3, "PPL_3gram_self_nats": PPL3}


def score_semantic(r):
    semantic_metrics = compute_semantic_metrics(r["text"], embedder, chunk_size=cfg.embed_chunk_size, overlap=cfg.embed_overlap, batch_size=cfg.batch_size, prefix=cfg.e5_prefix, seed=cfg.seed)
    return {**r, "id": r["id"], "source": r["source"], **semantic_metrics}

def score_affect(r):
    toks = tokenize(r["text"], nlp_tok)
    out = {**r, "id": r["id"], "source": r["source"]}
    # Concreteness is era-stable, so it runs in every mode.
    out.update(score_lexicon(toks, CONC_LEX, prefix="concreteness"))
    # VAD / emotion / register depend on modern annotator associations and
    # don't transfer cleanly to pre-20th-century text, so they're modern-only.
    if cfg.affect_mode == "modern":
        out.update(vad_metrics(toks, VAD_LEX))
        out.update(emotion_metrics(toks, EMO_LEX))
        out.update(score_lexicon(toks, AOA_LEX,  prefix="aoa"))
        out.update(score_lexicon(toks, PREV_LEX, prefix="prevalence", agg=("mean",)))
    return out

def score_syntax_punct(r):
    text = r["text"][:cfg.max_text_chars]
    if len(text) > nlp_parse.max_length:
        nlp_parse.max_length = len(text) + 1
    doc = nlp_parse(text)
    out = {**r, "id": r["id"], "source": r["source"]}
    out.update(punctuation_metrics(text))
    out.update(syntax_complexity_metrics(doc))
    return out

ling = map_with_checkpoints(records, score_linguistic, LING_FP, ["id", "source"], checkpoint_every=cfg.checkpoint_every)
td = map_with_checkpoints_batched(records, lambda batch: get_td_metrics_batch(batch, nlp_td, cfg.max_text_chars), TD_FP, ["id", "source"], batch_size=cfg.checkpoint_every)
sem = map_with_checkpoints(records, score_semantic, SEM_FP, ["id", "source"], checkpoint_every=cfg.checkpoint_every)
affect = map_with_checkpoints(records, score_affect, AFF_FP / "affect_checkpoint.csv", ["id","source"], checkpoint_every=cfg.checkpoint_every)
synpun = map_with_checkpoints(records, score_syntax_punct, SYN_FP / "synpun_checkpoint.csv", ["id","source"], checkpoint_every=cfg.checkpoint_every)

# MERGE AND WRITE FINAL SCORES
key_cols = ["id", "source"]
original_cols = set(chunks_df.columns)

for df in (ling, td, sem, affect, synpun):
    df["id"], df["source"] = df["id"].astype(str), df["source"].astype(str)

all_metrics = ling[[c for c in ling.columns if c not in original_cols or c in key_cols]]
for df in (td, sem, affect, synpun):
    cols = [c for c in df.columns if c not in original_cols or c in key_cols]
    all_metrics = all_metrics.merge(df[cols], on=key_cols, how="left")

meta_cols = [c for c in chunks_df.columns if c != "text"]
meta_df = chunks_df[meta_cols].copy()
meta_df["id"] = meta_df["id"].astype(str)
meta_df["source"] = meta_df["source"].astype(str)
all_metrics = meta_df.merge(all_metrics, on=key_cols, how="left")
write_table(all_metrics, OUT_FP)
print(f"Saved: {all_metrics.shape} to {OUT_FP}")
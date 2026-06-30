from pathlib import Path
from multiprocessing import Process
import pandas as pd

from ep_pipeline.config import MetricsConfig
from ep_pipeline.io import read_checkpoint, write_table
from ep_pipeline.scoring.pipeline_workers import worker_lingaff_synpun, worker_td, worker_sem


print("Starting AO3 Scoring")

# PATHS
PROJECT_ROOT = Path("/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/chr_stability_paper")
MODEL_ROOT   = Path("/Users/au728638/Library/CloudStorage/OneDrive-Aarhusuniversitet/Desktop/3. PhD Project/3. Code/models")
FULL_FP      = PROJECT_ROOT / "data" / "human_texts_2023_2025.csv"
CHR27_FP     = PROJECT_ROOT / "data" / "data_subset_chr27.csv"
OUT_DIR      = PROJECT_ROOT / "outputs" / "results"
LEX_DIR      = MODEL_ROOT / "lexicons"
E5_PATH      = MODEL_ROOT / "e5-small"
TD_FP        = OUT_DIR / "checkpoints" / "td_checkpoint.csv"
SEM_FP       = OUT_DIR / "checkpoints" / "sem_checkpoint.csv"
LINGAFF_FP   = OUT_DIR / "checkpoints" / "lingaff_checkpoint.csv"
SYN_FP       = OUT_DIR / "checkpoints" / "synpun_checkpoint.csv"
RECORDS_TMP  = OUT_DIR / "checkpoints" / "_records_tmp.parquet"
OUT_FP       = PROJECT_ROOT / "data" / "AO3metrics_full.csv"

cfg = MetricsConfig(affect_mode="modern", spacy_model="en_core_web_md")


if __name__ == "__main__":

    # COMBINE BOTH DATASETS INTO ONE AND SAVE TO TEMP PARQUET FOR WORKERS
    chr27 = pd.read_csv(CHR27_FP).rename(columns={
        "work_title": "title",
        "work_author": "author",
        "fandoms": "fandom",
        "date": "published",
    }).drop(columns=["work_url"], errors="ignore")

    raw = pd.concat([
        pd.read_csv(FULL_FP),
        chr27,
    ], ignore_index=True).dropna(subset=["text"]).rename(
        columns={"work_id": "id"}
    ).assign(text=lambda df: df["text"].str.replace(r"^Work Text:\s*", "", regex=True))

    chunks_df = raw.assign(source="human")
    print(f"Total records: {len(chunks_df)}")

    RECORDS_TMP.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_parquet(RECORDS_TMP, index=False)

    # SPAWN 3 PARALLEL WORKERS
    # A: lingaff + synpun  (sequential inside, sharing one nlp_parse)
    # B: td                (batched nlp.pipe, own nlp_td)
    # C: sem               (per-record, own embedder)
    p_a = Process(
        target=worker_lingaff_synpun,
        args=(RECORDS_TMP, LEX_DIR, LINGAFF_FP, SYN_FP, cfg),
        name="lingaff-synpun")
    p_b = Process(
        target=worker_td,
        args=(RECORDS_TMP, TD_FP, cfg),
        name="td")
    p_c = Process(
        target=worker_sem,
        args=(RECORDS_TMP, E5_PATH, SEM_FP, cfg),
        name="sem")

    print("Launching 3 parallel scoring workers...")
    for p in (p_a, p_b, p_c):
        p.start()
    for p in (p_a, p_b, p_c):
        p.join()

    failed = [p.name for p in (p_a, p_b, p_c) if p.exitcode != 0]
    if failed:
        raise RuntimeError(
            f"Workers failed: {failed}. "
            "Checkpoints are preserved — fix the issue and rerun to resume.")

    RECORDS_TMP.unlink(missing_ok=True)
    print("All workers finished. Merging results...")

    # READ CHECKPOINT RESULTS
    lingaff = read_checkpoint(LINGAFF_FP)
    td      = read_checkpoint(TD_FP)
    sem     = read_checkpoint(SEM_FP)
    synpun  = read_checkpoint(SYN_FP)

    missing = [name for name, df in
               [("lingaff", lingaff), ("td", td), ("sem", sem), ("synpun", synpun)]
               if df is None]
    if missing:
        raise RuntimeError(f"Checkpoint files missing after workers completed: {missing}")

    # MERGE AND WRITE FINAL SCORES
    key_cols      = ["id", "source"]
    original_cols = set(chunks_df.columns)

    for df in (lingaff, td, sem, synpun):
        if df["id"].dtype != object:
            df["id"] = df["id"].astype(str)
        if df["source"].dtype != object:
            df["source"] = df["source"].astype(str)

    all_metrics = lingaff[[c for c in lingaff.columns if c not in original_cols or c in key_cols]]
    for df in (td, sem, synpun):
        cols = [c for c in df.columns if c not in original_cols or c in key_cols]
        all_metrics = all_metrics.merge(df[cols], on=key_cols, how="left")

    meta_cols = [c for c in chunks_df.columns if c != "text"]
    meta_df   = chunks_df[meta_cols].copy()
    meta_df["id"]     = meta_df["id"].astype(str)
    meta_df["source"] = meta_df["source"].astype(str)
    all_metrics = meta_df.merge(all_metrics, on=key_cols, how="left")

    write_table(all_metrics, OUT_FP)
    print(f"Saved: {all_metrics.shape} to {OUT_FP}")

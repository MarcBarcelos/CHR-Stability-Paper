# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

sns.set_theme(style='whitegrid', font_scale=1.0)

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ is one level under chr_stability_paper/
DATA_PATH = PROJECT_ROOT / "data" / "final_full_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "classification"
FIG_DIR = PROJECT_ROOT / "outputs" / "visualizations"

# EFA factor -> interpreted construct name
FACTOR_TO_CATEGORY = {
    "F1":  "Syntactic Complexity",
    "F2":  "Lexical Sophistication",
    "F3":  "Negative Affect",
    "F4":  "Lexical Richness",
    "F5":  "Repetitiveness",
    "F6":  "Concreteness",
    "F7":  "Conversationality",
    "F8":  "Positive Affect",
    "F9":  "Narrative Drift",
    "F10": "Structural Variability",
}

TARGETS = {
    "fandom": {
        "column": "fandom_label",
        "class_labels": None,  # sorted(y.unique()) at runtime
        "colors": ["#F49F0A", "#4B9531", "#89023E", "#26547C", "#5AB3B3", "#F24236"],
        "grid": (2, 3),
    },
    "rating": {
        "column": "rating",
        "class_labels": ["General Audiences", "Teen And Up Audiences", "Mature", "Explicit"],
        "colors": {
            "General Audiences":     "#88D498",
            "Teen And Up Audiences": "#508991",
            "Mature":                "#FFBC42",
            "Explicit":              "#D36135",
        },
        "grid": (2, 2),
    },
}


def load_target(df, spec):
    df = df.copy()
    if spec["class_labels"] is not None:
        df[spec["column"]] = df[spec["column"]].str.strip("[]'")
        df = df[df[spec["column"]].isin(spec["class_labels"])]

    y = df[spec["column"]]
    X = df[[col for col in df.columns if col.startswith("F") and col[1:].isdigit()]]
    class_labels = spec["class_labels"] or sorted(y.unique())
    return X, y, class_labels


def run_classification(X, y, class_labels, random_state=1999, n_splits=5):
    predictor_columns = X.columns
    model = make_pipeline(
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_state),
    )
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = {
        k: []
        for k in ["fold", "accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
    }
    per_class_scores = {
        f"{c}_{m}": [] for c in class_labels for m in ["precision", "recall", "f1"]
    }
    all_coefs, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        roc_auc = roc_auc_score(
            label_binarize(y_test, classes=class_labels), y_proba, multi_class="ovr"
        )
        report = classification_report(y_test, y_pred, output_dict=True)

        for c in class_labels:
            key = str(c)
            per_class_scores[f"{c}_precision"].append(report[key]["precision"])
            per_class_scores[f"{c}_recall"].append(report[key]["recall"])
            per_class_scores[f"{c}_f1"].append(report[key]["f1-score"])

        scores["fold"].append(fold)
        scores["accuracy"].append(model.score(X_test, y_test))
        scores["macro_precision"].append(report["macro avg"]["precision"])
        scores["macro_recall"].append(report["macro avg"]["recall"])
        scores["macro_f1"].append(report["macro avg"]["f1-score"])
        scores["roc_auc_ovr"].append(roc_auc)
        all_coefs.append(model.named_steps["logisticregression"].coef_)

    coef_array = np.array(all_coefs)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)
    return scores, per_class_scores, coef_array, cm, predictor_columns


def print_summary(scores, per_class_scores, class_labels):
    print("\n--- Average Performance Across Folds ---")
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]:
        print(f"{metric}: {np.mean(scores[metric]):.4f} ± {np.std(scores[metric]):.4f}")

    print("\n--- Per-Class Performance (mean ± std) ---")
    for c in class_labels:
        print(f"\nClass {c}:")
        for metric in ["precision", "recall", "f1"]:
            key = f"{c}_{metric}"
            print(
                f"  {metric}: {np.mean(per_class_scores[key]):.4f} ± {np.std(per_class_scores[key]):.4f}"
            )


def plot_confusion_matrix(cm, class_labels, out_path):
    n_classes = len(class_labels)
    fig_size = max(3, n_classes * 0.9)
    plt.figure(figsize=(fig_size, fig_size), dpi=500)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_labels, yticklabels=class_labels,
    )
    plt.xlabel(r"$\bf{Predicted}$")
    plt.ylabel(r"$\bf{True}$")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_json_results(class_labels, scores, per_class_scores, coef_array, cm, predictor_columns):
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr"]
    return {
        "n_classes": len(class_labels),
        "class_labels": list(class_labels),
        "n_folds": len(scores["fold"]),
        "overall_metrics": {
            m: {"mean": float(np.mean(scores[m])), "std": float(np.std(scores[m]))}
            for m in metrics
        },
        "per_fold_metrics": {m: [float(v) for v in scores[m]] for m in metrics},
        "per_class_metrics": {
            name: {
                m: {
                    "mean": float(np.mean(per_class_scores[f"{name}_{m}"])),
                    "std": float(np.std(per_class_scores[f"{name}_{m}"])),
                }
                for m in ["precision", "recall", "f1"]
            }
            for name in class_labels
        },
        "coefficients": {
            name: {
                feat: {
                    "mean": float(coef_array[:, idx, fi].mean()),
                    "std": float(coef_array[:, idx, fi].std()),
                }
                for fi, feat in enumerate(predictor_columns)
            }
            for idx, name in enumerate(class_labels)
        },
        "confusion_matrix": cm.tolist(),
    }


def plot_coefficients(coef_array, class_labels, predictor_columns, colors, grid, out_path):
    n_classes = len(class_labels)
    n_rows, n_cols = grid
    predictors = [FACTOR_TO_CATEGORY[col] for col in predictor_columns]
    palette = colors if isinstance(colors, dict) else dict(zip(class_labels, colors))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 4 * n_rows), sharey=True, dpi=300)
    axes = np.atleast_2d(axes)

    for flat_idx, name in enumerate(class_labels):
        color = palette[name]
        ax = axes[flat_idx // n_cols, flat_idx % n_cols]
        means = coef_array[:, flat_idx, :].mean(axis=0)
        stds = coef_array[:, flat_idx, :].std(axis=0)
        y_display = np.arange(len(predictors))[::-1]

        reliable = np.abs(means) >= 3 * stds

        ax.scatter(means[reliable], y_display[reliable],
                   color=color, edgecolor="black", linewidth=1, s=80, alpha=1, zorder=2)
        ax.scatter(means[~reliable], y_display[~reliable],
                   color="white", edgecolor="black", linewidth=1, s=80, zorder=2)
        ax.errorbar(means, y_display, xerr=3 * stds,
                    fmt="none", ecolor="black", elinewidth=.5, capsize=15, capthick=.5, zorder=1)

        ax.axvline(0, color="0.1", linestyle="--", zorder=0.5)
        ax.set_title("Random" if name == "random" else name, fontweight="bold")

        # x-axis limits tailored to this class's own coefficient range, symmetric about 0
        class_lim = np.abs(np.concatenate([means - stds, means + stds])).max()
        class_lim *= 1.15
        ax.set_xlim(-class_lim, class_lim)

        ax.grid(True, color="0.4", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        if flat_idx % n_cols != 0:
            ax.tick_params(labelleft=False)

    for flat_idx in range(n_classes, n_rows * n_cols):
        axes[flat_idx // n_cols, flat_idx % n_cols].set_visible(False)

    for row in range(n_rows):
        axes[row, 0].set_yticks(range(len(predictors)))
        axes[row, 0].set_yticklabels([s.title() for s in predictors[::-1]], fontsize=12, fontweight="bold")
    fig.supxlabel("Feature Coefficient (log-odds)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved coefficient plot to {out_path}")


def run_target(df, target_name, spec):
    print(f"\n{'=' * 70}\nRunning classification for target: {target_name}\n{'=' * 70}")

    X, y, class_labels = load_target(df, spec)
    scores, per_class_scores, coef_array, cm, predictor_columns = run_classification(
        X, y, class_labels
    )

    print_summary(scores, per_class_scores, class_labels)

    plot_confusion_matrix(cm, class_labels, FIG_DIR / f"{target_name}_confusion_matrix.png")

    results = build_json_results(class_labels, scores, per_class_scores, coef_array, cm, predictor_columns)
    results_path = RESULTS_DIR / f"{target_name}_classification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Saved structured results to {results_path}")

    per_class_df = pd.DataFrame(
        [
            {
                "class_label": c,
                "metric": m,
                "mean": np.mean(per_class_scores[f"{c}_{m}"]),
                "std": np.std(per_class_scores[f"{c}_{m}"]),
            }
            for c in class_labels
            for m in ["precision", "recall", "f1"]
        ]
    )
    per_class_df.to_csv(RESULTS_DIR / f"{target_name}_per_class_metrics.csv", index=False)

    coef_df = pd.DataFrame(
        [
            {
                "class_name": name,
                "coef_mean": coef_array[:, idx, fi].mean(),
                "coef_std": coef_array[:, idx, fi].std(),
            }
            for idx, name in enumerate(class_labels)
            for fi in range(len(predictor_columns))
        ]
    )
    coef_df.to_csv(RESULTS_DIR / f"{target_name}_coefficients.csv", index=False)
    print(f"Saved per-class metrics and coefficients CSVs to {RESULTS_DIR}")

    plot_coefficients(
        coef_array, class_labels, predictor_columns, spec["colors"], spec["grid"],
        FIG_DIR / f"{target_name}_coef_plot.png",
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    df = pd.read_csv(DATA_PATH)

    for target_name, spec in TARGETS.items():
        run_target(df, target_name, spec)


if __name__ == "__main__":
    main()

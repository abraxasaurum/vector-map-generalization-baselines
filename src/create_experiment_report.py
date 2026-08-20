from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

CSV_FILE = REPORTS_DIR / "model_comparison.csv"
MARKDOWN_FILE = REPORTS_DIR / "model_comparison.md"
BAR_FIGURE = REPORTS_DIR / "model_comparison.png"
CURVE_FIGURE = REPORTS_DIR / "validation_learning_curves.png"

CNN_HISTORY = PROJECT_ROOT / "outputs" / "cnn_training_history.csv"
TRANSFORMER_HISTORY = PROJECT_ROOT / "outputs" / "transformer_training_history.csv"

results = [
    {
        "model": "Residual MLP",
        "dataset": "Single area (Charlottenburg)",
        "n_buildings": 166,
        "n_train": 132,
        "n_parameters": 33088,
        "identity_test_mse": 0.001203,
        "model_test_mse": 0.002451,
        "improvement_percent": -103.7,
        "interpretation": "Strong overfitting",
    },
    {
        "model": "Residual circular 1D-CNN",
        "dataset": "Eight Berlin areas",
        "n_buildings": 3282,
        "n_train": 2231,
        "n_parameters": 1634,
        "identity_test_mse": 0.007481,
        "model_test_mse": 0.007366,
        "improvement_percent": 1.5,
        "interpretation": "Best test baseline",
    },
    {
        "model": "Residual Transformer",
        "dataset": "Eight Berlin areas",
        "n_buildings": 3282,
        "n_train": 2231,
        "n_parameters": 69314,
        "identity_test_mse": 0.007481,
        "model_test_mse": 0.007403,
        "improvement_percent": 1.0,
        "interpretation": "Small gain; no CNN advantage",
    },
]

comparison = pd.DataFrame(results)
comparison.to_csv(CSV_FILE, index=False)

markdown_table = comparison.to_markdown(index=False, floatfmt=".6f")

markdown_text = f"""# Model comparison

## Evaluation principle

Each model is compared with an identity baseline: the input building geometry
is returned unchanged. A positive improvement means the learned model achieved
a lower test MSE than that baseline.

## Results

{markdown_table}

## Interpretation

- The MLP strongly overfit the small single-area dataset.
- Expanding the data from 166 to 3,282 buildings substantially improved the
  experimental setup.
- The circular 1D-CNN achieved the best test result, with a 1.5% improvement
  over the identity baseline.
- The Transformer reduced validation loss strongly but achieved only a 1.0%
  test improvement. With the current Douglas-Peucker targets, global attention
  did not clearly outperform local convolution.
- The MLP experiment uses a different, much smaller dataset. Its absolute MSE
  must therefore not be compared directly with CNN and Transformer MSE values.

## Limitation

The target geometry is produced by Douglas-Peucker with a fixed 2 m tolerance.
It does not incorporate map scale, neighbouring buildings, object semantics,
or topological conflicts. A context-aware extension would require targets that
depend on such contextual information.
"""

MARKDOWN_FILE.write_text(markdown_text, encoding="utf-8")

plot_data = comparison.copy()
colors = [
    "#E45756" if value < 0 else "#54A24B"
    for value in plot_data["improvement_percent"]
]

labels = [
    "MLP\n166 buildings",
    "Circular 1D-CNN\n3,282 buildings",
    "Transformer\n3,282 buildings",
]

fig, axis = plt.subplots(figsize=(9, 5))
bars = axis.bar(
    labels,
    plot_data["improvement_percent"],
    color=colors,
    edgecolor="white",
    linewidth=1,
)

axis.axhline(0, color="black", linewidth=0.8)
axis.set_ylabel("Test-MSE improvement over identity baseline [%]")
axis.set_title("Generalization performance relative to identity baseline")
axis.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, plot_data["improvement_percent"]):
    vertical_alignment = "bottom" if value >= 0 else "top"
    offset = 1.5 if value >= 0 else -4
    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.1f}%",
        ha="center",
        va=vertical_alignment,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(BAR_FIGURE, dpi=180, bbox_inches="tight")
plt.close()

cnn_history = pd.read_csv(CNN_HISTORY)
transformer_history = pd.read_csv(TRANSFORMER_HISTORY)

fig, axis = plt.subplots(figsize=(9, 5))

axis.plot(
    cnn_history["epoch"],
    cnn_history["validation_mse"],
    color="#4C78A8",
    linewidth=2,
    label="Circular 1D-CNN",
)

axis.plot(
    transformer_history["epoch"],
    transformer_history["validation_mse"],
    color="#F58518",
    linewidth=2,
    label="Transformer",
)

axis.set_xlabel("Epoch")
axis.set_ylabel("Validation MSE")
axis.set_title("Validation learning curves on the multi-area dataset")
axis.grid(alpha=0.25)
axis.legend()

plt.tight_layout()
plt.savefig(CURVE_FIGURE, dpi=180, bbox_inches="tight")
plt.close()

print(f"CSV: {CSV_FILE}")
print(f"Markdown report: {MARKDOWN_FILE}")
print(f"Comparison figure: {BAR_FIGURE}")
print(f"Learning curves: {CURVE_FIGURE}")

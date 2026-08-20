from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

CSV_FILE = REPORTS_DIR / "final_experiment_summary.csv"
MARKDOWN_FILE = REPORTS_DIR / "final_experiment_summary.md"
FIGURE_FILE = REPORTS_DIR / "final_experiment_summary.png"

results = [
    {
        "experiment": "MLP baseline",
        "task": "Fixed 2 m Douglas-Peucker target",
        "evaluation": "Random split, single Berlin area",
        "n_buildings": 166,
        "n_test": 34,
        "n_parameters": 33088,
        "identity_test_mse": 0.001203,
        "model_test_mse": 0.002451,
        "improvement_percent": -103.7,
        "conclusion": "Strong overfitting on small dataset",
    },
    {
        "experiment": "Circular 1D-CNN",
        "task": "Fixed 2 m Douglas-Peucker target",
        "evaluation": "Spatial holdout: Wedding + Lichtenberg",
        "n_buildings": 3282,
        "n_test": 856,
        "n_parameters": 1634,
        "identity_test_mse": 0.007128,
        "model_test_mse": 0.007012,
        "improvement_percent": 1.6,
        "conclusion": "Best fixed-target spatial result",
    },
    {
        "experiment": "Transformer",
        "task": "Fixed 2 m Douglas-Peucker target",
        "evaluation": "Random split across eight Berlin areas",
        "n_buildings": 3282,
        "n_test": 657,
        "n_parameters": 69314,
        "identity_test_mse": 0.007481,
        "model_test_mse": 0.007403,
        "improvement_percent": 1.0,
        "conclusion": "Small gain; no CNN advantage",
    },
    {
        "experiment": "Geometry + context CNN",
        "task": "Context-adaptive 1/2/3 m Douglas-Peucker target",
        "evaluation": "Spatial holdout: Wedding + Lichtenberg",
        "n_buildings": 3259,
        "n_test": 848,
        "n_parameters": 2130,
        "identity_test_mse": 0.005368,
        "model_test_mse": 0.005309,
        "improvement_percent": 1.1,
        "conclusion": "Context-aware proof of concept",
    },
]

summary = pd.DataFrame(results)
summary.to_csv(CSV_FILE, index=False)

markdown_table = summary.to_markdown(index=False, floatfmt=".6f")

markdown_text = f"""# Final experiment summary

## Important comparison note

Absolute MSE values must only be compared within the same target task and
evaluation split. The fixed 2 m target and the context-adaptive 1/2/3 m target
represent different prediction tasks.

## Results

{markdown_table}

## Main findings

- The MLP strongly overfit the small single-area dataset.
- The circular 1D-CNN improved over an identity baseline on a spatially
  separated test set for fixed 2 m Douglas-Peucker targets.
- The Transformer reduced validation loss but did not outperform the compact
  CNN on the fixed-target task.
- The context-aware CNN produced a positive improvement on a spatial holdout
  for the synthetic context-adaptive target task.
- Results are intentionally interpreted as proof-of-concept outcomes, not as
  evidence of a superior cartographic generalization method.

## Limitations

- Targets are generated synthetically with Douglas-Peucker simplification.
- Context tolerance is a transparent didactic proxy, not a cartographic rule.
- The context model uses spatial but no semantic features such as building use,
  height, or road context.
- More diverse cities, map scales, and expert-labelled data would be needed
  for a substantive research benchmark.
"""

MARKDOWN_FILE.write_text(markdown_text, encoding="utf-8")

plot_data = summary.copy()
colors = [
    "#E45756" if value < 0 else "#54A24B"
    for value in plot_data["improvement_percent"]
]

labels = [
    "MLP\nsmall dataset",
    "1D-CNN\nfixed target,\nspatial holdout",
    "Transformer\nfixed target,\nrandom split",
    "CNN + context\nadaptive target,\nspatial holdout",
]

fig, axis = plt.subplots(figsize=(11, 6))

bars = axis.bar(
    labels,
    plot_data["improvement_percent"],
    color=colors,
    edgecolor="white",
    linewidth=1,
)

axis.axhline(0, color="black", linewidth=0.8)
axis.set_ylabel("Test-MSE improvement over identity baseline [%]")
axis.set_title("Model experiments: improvement relative to identity baseline")
axis.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, plot_data["improvement_percent"]):
    vertical_alignment = "bottom" if value >= 0 else "top"
    offset = 1.0 if value >= 0 else -4.0

    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.1f}%",
        ha="center",
        va=vertical_alignment,
        fontweight="bold",
    )

axis.text(
    0.02,
    0.03,
    "Do not compare absolute MSE across different target definitions.",
    transform=axis.transAxes,
    fontsize=9,
    color="#555555",
)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"CSV: {CSV_FILE}")
print(f"Markdown report: {MARKDOWN_FILE}")
print(f"Figure: {FIGURE_FILE}")

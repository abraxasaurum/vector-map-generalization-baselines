from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXED_SEQUENCE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "building_sequences_32points.npz"
)
FIXED_METADATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sequence_metadata.csv"
)
FIXED_MODEL_FILE = (
    PROJECT_ROOT / "outputs" / "cnn_spatial_holdout.pt"
)

CONTEXT_SEQUENCE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "context_adaptive_sequences.npz"
)
CONTEXT_METADATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "context_adaptive_metadata.csv"
)
CONTEXT_MODEL_FILE = (
    PROJECT_ROOT / "outputs" / "context_cnn_spatial_holdout.pt"
)

SUMMARY_FILE = PROJECT_ROOT / "reports" / "geometry_quality_summary.csv"
DETAIL_FILE = PROJECT_ROOT / "reports" / "geometry_quality_per_building.csv"
FIGURE_FILE = PROJECT_ROOT / "reports" / "geometry_quality_comparison.png"

TEST_AREAS = {"wedding", "lichtenberg"}


class ResidualCircularCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=5, padding=2, padding_mode="circular"),
        )

    def forward(self, x):
        features = self.network(x.transpose(1, 2)).transpose(1, 2)
        return x + features


class ContextResidualCircularCNN(nn.Module):
    def __init__(self, context_size=3):
        super().__init__()

        self.geometry_network = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
        )

        self.context_network = nn.Sequential(
            nn.Linear(context_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.output_layer = nn.Conv1d(
            32,
            2,
            kernel_size=5,
            padding=2,
            padding_mode="circular",
        )

    def forward(self, geometry, context_features):
        geometry_features = self.geometry_network(geometry.transpose(1, 2))

        context_features = self.context_network(context_features)
        context_features = context_features.unsqueeze(-1).expand(
            -1,
            -1,
            geometry_features.shape[-1],
        )

        combined_features = torch.cat(
            [geometry_features, context_features],
            dim=1,
        )

        correction = self.output_layer(combined_features).transpose(1, 2)
        return geometry + correction


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def restore_coordinates(sequence, metadata_row):
    center = np.array(
        [metadata_row["center_x"], metadata_row["center_y"]],
        dtype=np.float64,
    )

    return sequence.astype(np.float64) * metadata_row["scale_m"] + center


def make_polygon(coordinates):
    closed_coordinates = np.vstack([coordinates, coordinates[0]])
    return Polygon(closed_coordinates)


def compute_metrics(
    task,
    model_name,
    metadata,
    source_sequences,
    target_sequences,
    predicted_sequences,
):
    rows = []

    for sequence_index, metadata_row in metadata.iterrows():
        source_coordinates = restore_coordinates(
            source_sequences[sequence_index],
            metadata_row,
        )

        target_coordinates = restore_coordinates(
            target_sequences[sequence_index],
            metadata_row,
        )

        predicted_coordinates = restore_coordinates(
            predicted_sequences[sequence_index],
            metadata_row,
        )

        source_polygon = make_polygon(source_coordinates)
        target_polygon = make_polygon(target_coordinates)
        predicted_polygon = make_polygon(predicted_coordinates)

        paired_point_rmse_m = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (predicted_coordinates - target_coordinates) ** 2,
                        axis=1,
                    )
                )
            )
        )

        try:
            hausdorff_distance_m = predicted_polygon.hausdorff_distance(
                target_polygon
            )
        except Exception:
            hausdorff_distance_m = np.nan

        target_area_m2 = target_polygon.area
        predicted_area_m2 = predicted_polygon.area

        relative_area_error_pct = (
            abs(predicted_area_m2 - target_area_m2)
            / max(target_area_m2, 1e-9)
            * 100
        )

        rows.append(
            {
                "task": task,
                "model": model_name,
                "building_id": metadata_row["building_id"],
                "source_area": metadata_row["source_area"],
                "context_tolerance_m": metadata_row.get(
                    "context_tolerance_m",
                    np.nan,
                ),
                "paired_point_rmse_m": paired_point_rmse_m,
                "hausdorff_distance_m": hausdorff_distance_m,
                "relative_area_error_pct": relative_area_error_pct,
                "source_is_valid": source_polygon.is_valid,
                "target_is_valid": target_polygon.is_valid,
                "prediction_is_valid": predicted_polygon.is_valid,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metrics):
    return (
        metrics.groupby(["task", "model"])
        .agg(
            n_buildings=("building_id", "count"),
            mean_paired_point_rmse_m=("paired_point_rmse_m", "mean"),
            median_paired_point_rmse_m=("paired_point_rmse_m", "median"),
            mean_hausdorff_distance_m=("hausdorff_distance_m", "mean"),
            median_hausdorff_distance_m=("hausdorff_distance_m", "median"),
            mean_relative_area_error_pct=("relative_area_error_pct", "mean"),
            median_relative_area_error_pct=("relative_area_error_pct", "median"),
            valid_prediction_rate_pct=(
                "prediction_is_valid",
                lambda value: value.mean() * 100,
            ),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------
# Fixed 2-m target task: Identity versus spatial-holdout CNN
# ---------------------------------------------------------------------
fixed_data = np.load(FIXED_SEQUENCE_FILE)

X_fixed = fixed_data["X"].astype(np.float32)
y_fixed = fixed_data["y"].astype(np.float32)

fixed_metadata = (
    pd.read_csv(FIXED_METADATA_FILE)
    .sort_values("building_id")
    .reset_index(drop=True)
)

fixed_test_indices = np.where(
    fixed_metadata["source_area"].isin(TEST_AREAS)
)[0]

fixed_model = load_checkpoint(
    ResidualCircularCNN(),
    FIXED_MODEL_FILE,
)

with torch.no_grad():
    fixed_predictions = fixed_model(
        torch.from_numpy(X_fixed[fixed_test_indices])
    ).numpy()

fixed_identity_metrics = compute_metrics(
    task="Fixed 2 m target",
    model_name="Identity baseline",
    metadata=fixed_metadata.iloc[fixed_test_indices].reset_index(drop=True),
    source_sequences=X_fixed[fixed_test_indices],
    target_sequences=y_fixed[fixed_test_indices],
    predicted_sequences=X_fixed[fixed_test_indices],
)

fixed_cnn_metrics = compute_metrics(
    task="Fixed 2 m target",
    model_name="Circular 1D-CNN",
    metadata=fixed_metadata.iloc[fixed_test_indices].reset_index(drop=True),
    source_sequences=X_fixed[fixed_test_indices],
    target_sequences=y_fixed[fixed_test_indices],
    predicted_sequences=fixed_predictions,
)

# ---------------------------------------------------------------------
# Context-adaptive target task: Identity versus Geometry + Context CNN
# ---------------------------------------------------------------------
context_data = np.load(CONTEXT_SEQUENCE_FILE)

X_context = context_data["X"].astype(np.float32)
y_context = context_data["y"].astype(np.float32)
context_features = context_data["context"].astype(np.float32)

context_metadata = (
    pd.read_csv(CONTEXT_METADATA_FILE)
    .sort_values("building_id")
    .reset_index(drop=True)
)

context_test_indices = np.where(
    context_metadata["source_area"].isin(TEST_AREAS)
)[0]

context_model = load_checkpoint(
    ContextResidualCircularCNN(),
    CONTEXT_MODEL_FILE,
)

with torch.no_grad():
    context_predictions = context_model(
        torch.from_numpy(X_context[context_test_indices]),
        torch.from_numpy(context_features[context_test_indices]),
    ).numpy()

context_identity_metrics = compute_metrics(
    task="Context-adaptive 1/2/3 m target",
    model_name="Identity baseline",
    metadata=context_metadata.iloc[context_test_indices].reset_index(drop=True),
    source_sequences=X_context[context_test_indices],
    target_sequences=y_context[context_test_indices],
    predicted_sequences=X_context[context_test_indices],
)

context_cnn_metrics = compute_metrics(
    task="Context-adaptive 1/2/3 m target",
    model_name="Geometry + context CNN",
    metadata=context_metadata.iloc[context_test_indices].reset_index(drop=True),
    source_sequences=X_context[context_test_indices],
    target_sequences=y_context[context_test_indices],
    predicted_sequences=context_predictions,
)

all_metrics = pd.concat(
    [
        fixed_identity_metrics,
        fixed_cnn_metrics,
        context_identity_metrics,
        context_cnn_metrics,
    ],
    ignore_index=True,
)

summary = summarize_metrics(all_metrics)

DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
all_metrics.to_csv(DETAIL_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)

print("\nGeometry-quality summary:")
print(summary.round(4).to_string(index=False))

plot_labels = [
    "Fixed:\nIdentity",
    "Fixed:\nCNN",
    "Context:\nIdentity",
    "Context:\nCNN + context",
]

plot_order = [
    ("Fixed 2 m target", "Identity baseline"),
    ("Fixed 2 m target", "Circular 1D-CNN"),
    ("Context-adaptive 1/2/3 m target", "Identity baseline"),
    ("Context-adaptive 1/2/3 m target", "Geometry + context CNN"),
]

plot_rows = pd.DataFrame(
    [
        summary[
            (summary["task"] == task)
            & (summary["model"] == model)
        ].iloc[0]
        for task, model in plot_order
    ]
)

colors = ["#9E9E9E", "#4C78A8", "#9E9E9E", "#54A24B"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(
    plot_labels,
    plot_rows["median_paired_point_rmse_m"],
    color=colors,
)
axes[0].set_title("Median paired-point RMSE")
axes[0].set_ylabel("Metres")
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(
    plot_labels,
    plot_rows["median_hausdorff_distance_m"],
    color=colors,
)
axes[1].set_title("Median Hausdorff distance")
axes[1].set_ylabel("Metres")
axes[1].grid(axis="y", alpha=0.25)

axes[2].bar(
    plot_labels,
    plot_rows["valid_prediction_rate_pct"],
    color=colors,
)
axes[2].set_title("Valid predicted polygons")
axes[2].set_ylabel("Valid polygons [%]")
axes[2].set_ylim(0, 105)
axes[2].grid(axis="y", alpha=0.25)

for axis in axes:
    axis.tick_params(axis="x", labelsize=9)

fig.suptitle(
    "Geometry quality on spatial-holdout test areas\n"
    "Compare models only within the same target task.",
    y=1.03,
)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
plt.close()

print(f"\nDetailed metrics: {DETAIL_FILE}")
print(f"Summary: {SUMMARY_FILE}")
print(f"Figure: {FIGURE_FILE}")

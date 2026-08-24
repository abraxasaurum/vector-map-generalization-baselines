# Vector Map Generalization Baselines

A compact GeoAI research prototype for learning cartographic building-footprint
generalization from OpenStreetMap vector data.

The project was developed as a hands-on preparation exercise for research on
generative Transformer models for cartographic generalization of vector data.
It demonstrates a reproducible workflow from OpenStreetMap retrieval and
geometric preprocessing to PyTorch baselines, spatial holdout evaluation and a
small context-aware extension.

> **Scope:** This is a didactic proof of concept. Douglas-Peucker-generated
> geometries are synthetic learning targets, not expert-labelled cartographic
> reference data.

## Research questions

1. Can neural models predict simplified building-boundary sequences from
   detailed OpenStreetMap building polygons?
2. Do compact sequence models outperform an identity baseline?
3. Do results remain positive when testing on spatially separated urban areas?
4. How can local building context be represented for context-adaptive
   vector-generalization experiments?

## Workflow

```text
OpenStreetMap building footprints
            |
            v
GeoPandas / Shapely quality filtering and EPSG:25833 projection
            |
            v
Douglas-Peucker reference targets
            |
            v
Fixed-length polygon boundary sequences: 32 x/y coordinate pairs
            |
            +------------------------------+
            |                              |
            v                              v
Geometry-only baselines            Spatial context features
MLP / circular 1D-CNN /            nearest neighbour distance,
Transformer                        neighbour count, local density
            |                              |
            +---------------+--------------+
                            |
                            v
Spatial holdout evaluation
```

## Dataset

Building footprints were retrieved from OpenStreetMap using OSMnx and the
Overpass API. The multi-area dataset contains 5,328 initial building polygons
from eight spatially separated Berlin areas:

- Charlottenburg
- Moabit
- Schöneberg
- Kreuzberg
- Friedrichshain
- Prenzlauer Berg
- Wedding
- Lichtenberg

After geometric quality filtering and removal of examples without an effective
simplification target, the fixed-tolerance experiment used 3,282 buildings.
The context-adaptive experiment used 3,259 buildings.

All geometric operations use `EPSG:25833` (ETRS89 / UTM zone 33N), ensuring
that distances, areas and simplification tolerances are measured in metres.

### Data attribution

This project uses OpenStreetMap data:

> © OpenStreetMap contributors, available under the Open Data Commons Open
> Database License (ODbL).

Raw and processed OSM data are deliberately not included in this repository.
They can be reproduced with the scripts in `src/`.

## Target generation

### Fixed target baseline

The initial task uses topology-preserving Douglas-Peucker simplification:

```text
Input:  detailed OSM building polygon
Target: simplify(tolerance=2 m, preserve_topology=True)
```

This retained 3,282 examples. The median outer-boundary complexity decreased
from 13 to 6 vertices, corresponding to a mean vertex reduction of 53.7%.

### Context-adaptive target extension

A second, explicitly synthetic task uses local context to define the
simplification tolerance:

| Local context | Rule | Tolerance |
|---|---|---:|
| Isolated | 0 neighbours within 25 m | 1 m |
| Normal | 1-2 neighbours within 25 m | 2 m |
| Dense | 3 or more neighbours within 25 m | 3 m |

This is a transparent didactic proxy, not a claim of a cartographic standard.
It creates a controlled experiment where spatial context influences the target
geometry.

## Models

All models predict coordinate corrections rather than an entirely new polygon:

```text
prediction = source_sequence + learned_coordinate_correction
```

- **Residual MLP:** Fully connected baseline operating on flattened
  `32 x 2` coordinate sequences.
- **Residual circular 1D-CNN:** Lightweight local sequence model with circular
  padding, so the first and final polygon points are treated as neighbours.
- **Residual Transformer:** PyTorch `TransformerEncoder` over 32 point tokens,
  with positional embeddings and multi-head self-attention.
- **Geometry + context CNN:** Circular CNN augmented with nearest-neighbour
  distance, local neighbour count and local building density.

## Evaluation

### Identity baseline

Every learned model is compared against an identity baseline:

```text
prediction = input geometry
```

A positive improvement indicates a lower test MSE than returning the original
building boundary unchanged.

### Spatial holdout

To avoid an overly optimistic random building split, the main CNN experiment
uses spatially separated areas:

| Partition | Areas | Buildings |
|---|---|---:|
| Training | Charlottenburg, Moabit, Schöneberg, Kreuzberg, Friedrichshain | 1,806 |
| Validation | Prenzlauer Berg | 620 |
| Test | Wedding, Lichtenberg | 856 |

## Results

| Experiment | Task | Evaluation | Test improvement over identity |
|---|---|---|---:|
| Residual MLP | Fixed 2 m target | Random split, single area | -103.7% |
| Circular 1D-CNN | Fixed 2 m target | Spatial holdout | **+1.6%** |
| Residual Transformer | Fixed 2 m target | Random split, multi-area | +1.0% |
| Geometry + context CNN | Context-adaptive 1/2/3 m target | Spatial holdout | +1.1% |

The MLP strongly overfit the small single-area dataset. The compact circular
CNN produced the strongest fixed-target result on a spatially separated test
set. The Transformer substantially reduced validation loss but did not
outperform the CNN on the synthetic, primarily local Douglas-Peucker task.

Absolute MSE values must not be compared across the fixed and context-adaptive
tasks, because the target definitions differ.

## Key figures

Generated figures are not committed automatically because they are reproducible
artifacts. After running the pipeline locally, see:

```text
reports/final_experiment_summary.png
reports/validation_learning_curves.png
reports/spatial_context_overview.png
reports/context_adaptive_targets.png
outputs/cnn_spatial_holdout_predictions.png
outputs/context_cnn_spatial_holdout_predictions.png
```

## Installation

Create and activate the Conda environment:

```bash
conda create -n cart2former-prep python=3.11 -y
conda activate cart2former-prep
```

Install dependencies:

```bash
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Reproducing the pipeline

Run commands from the repository root.

```bash
# 1. Retrieve building footprints from eight Berlin areas
python src/download_buildings_multiarea.py

# 2. Fixed 2 m Douglas-Peucker targets and sequence representation
python src/create_baseline.py
python src/prepare_sequences.py

# 3. Geometry-only model baselines
python src/train_mlp.py
python src/train_cnn.py
python src/train_transformer.py

# 4. Spatially separated CNN evaluation
python src/train_cnn_spatial.py

# 5. Spatial context and context-adaptive targets
python src/analyse_spatial_context.py
python src/create_context_adaptive_targets.py
python src/prepare_context_sequences.py
python src/train_context_cnn_spatial.py

# 6. Geometric evaluation on spatial-holdout test areas
python src/evaluate_geometry_metrics.py

# 7. Reports
python src/create_experiment_report.py
python src/create_final_report.py
```

## Geometric quality checks

In addition to normalized coordinate MSE, the spatial-holdout predictions are
evaluated after reconstruction in `EPSG:25833`. This enables geometrically
interpretable quality checks in metres.

The evaluation script reports the following metrics separately for the fixed
2 m and context-adaptive target tasks:

- **Paired-point RMSE [m]:** Root mean squared distance between corresponding
  resampled boundary points.
- **Hausdorff distance [m]:** Maximum geometric discrepancy between predicted
  and target polygon boundaries.
- **Relative area error [%]:** Absolute predicted-versus-target area deviation,
  normalized by target area.
- **Valid prediction rate [%]:** Share of raw predicted closed polygons that
  satisfy Shapely geometry-validity checks.

Raw predictions are evaluated without automatic topology repair. This avoids
hiding potential self-intersections or geometric artifacts through
post-processing.

Results are stored in:

```text
reports/geometry_quality_summary.csv
reports/geometry_quality_per_building.csv
reports/geometry_quality_comparison.png
```

Geometry metrics are compared only within the same target definition:

- Fixed 2 m target: Identity baseline versus circular 1D-CNN.
- Context-adaptive 1/2/3 m target: Identity baseline versus geometry-plus-context CNN.

## Limitations and next steps

- Targets are generated synthetically with Douglas-Peucker simplification.
- The project uses building footprints only; it does not include roads,
  water features or multiple cartographic object classes.
- Context is spatial but not semantic: building type, height, land use and
  road-network context are not yet included.
- A stronger research benchmark would use expert-labelled multi-scale data,
  multiple cities and map scales, topology-aware losses, neighbourhood graphs,
  and vector-object context tokens.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # Ignored: downloaded OSM data
│   └── processed/           # Ignored: derived GeoJSON, NumPy arrays
├── outputs/                 # Ignored: models and intermediate figures
├── reports/                 # CSV, Markdown summaries and selected figures
├── src/                     # Reproducible data, model and report scripts
├── .gitignore
├── requirements.txt
└── README.md
```

## License

Code in this repository is provided under the MIT License. OpenStreetMap data
remain subject to the ODbL and associated attribution requirements.

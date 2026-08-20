# Final experiment summary

## Important comparison note

Absolute MSE values must only be compared within the same target task and
evaluation split. The fixed 2 m target and the context-adaptive 1/2/3 m target
represent different prediction tasks.

## Results

| experiment             | task                                            | evaluation                             |   n_buildings |   n_test |   n_parameters |   identity_test_mse |   model_test_mse |   improvement_percent | conclusion                          |
|:-----------------------|:------------------------------------------------|:---------------------------------------|--------------:|---------:|---------------:|--------------------:|-----------------:|----------------------:|:------------------------------------|
| MLP baseline           | Fixed 2 m Douglas-Peucker target                | Random split, single Berlin area       |           166 |       34 |          33088 |            0.001203 |         0.002451 |           -103.700000 | Strong overfitting on small dataset |
| Circular 1D-CNN        | Fixed 2 m Douglas-Peucker target                | Spatial holdout: Wedding + Lichtenberg |          3282 |      856 |           1634 |            0.007128 |         0.007012 |              1.600000 | Best fixed-target spatial result    |
| Transformer            | Fixed 2 m Douglas-Peucker target                | Random split across eight Berlin areas |          3282 |      657 |          69314 |            0.007481 |         0.007403 |              1.000000 | Small gain; no CNN advantage        |
| Geometry + context CNN | Context-adaptive 1/2/3 m Douglas-Peucker target | Spatial holdout: Wedding + Lichtenberg |          3259 |      848 |           2130 |            0.005368 |         0.005309 |              1.100000 | Context-aware proof of concept      |

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

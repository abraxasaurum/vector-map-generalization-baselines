# Model comparison

## Evaluation principle

Each model is compared with an identity baseline: the input building geometry
is returned unchanged. A positive improvement means the learned model achieved
a lower test MSE than that baseline.

## Results

| model                    | dataset                      |   n_buildings |   n_train |   n_parameters |   identity_test_mse |   model_test_mse |   improvement_percent | interpretation               |
|:-------------------------|:-----------------------------|--------------:|----------:|---------------:|--------------------:|-----------------:|----------------------:|:-----------------------------|
| Residual MLP             | Single area (Charlottenburg) |           166 |       132 |          33088 |            0.001203 |         0.002451 |           -103.700000 | Strong overfitting           |
| Residual circular 1D-CNN | Eight Berlin areas           |          3282 |      2231 |           1634 |            0.007481 |         0.007366 |              1.500000 | Best test baseline           |
| Residual Transformer     | Eight Berlin areas           |          3282 |      2231 |          69314 |            0.007481 |         0.007403 |              1.000000 | Small gain; no CNN advantage |

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

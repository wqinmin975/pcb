# solderbench

**An open-source synthetic benchmark for PCB solder-joint defect inspection algorithm development.**

`solderbench` provides a configurable, pure-NumPy point-cloud simulator of
surface-mount solder joints, a seven-feature geometric descriptor extractor,
six baseline classifiers, and reproducible comparison/ablation experiment
runners. It is designed for developing and fairly evaluating 3D solder-joint
defect inspection algorithms when real annotated 3D scans are scarce, and it
exposes a clean integration point for swapping in real data.

## Defect classes

| Class         | Description                                             | Simulated mechanism                                        |
|---------------|---------------------------------------------------------|------------------------------------------------------------|
| `good`        | Nominal joint                                           | Half-ellipsoid with natural size variation                 |
| `insufficient`| Too little solder (30–50% volume shrink)                | Uniformly shrunk ellipsoid                                 |
| `excessive`   | Too much solder, irregular shape                        | Inflated ellipsoid + random bump                           |
| `bridging`    | Solder bridge to the neighbouring joint                 | Abnormally small centre distance + low bridge point strip  |
| `collapse`    | Collapsed/void joint with a dented top                  | Gaussian dent at a random offset                           |

Class distributions deliberately overlap (e.g. the upper tail of
`insufficient` overlaps the lower tail of `good`), so **no classifier can
reach a perfect score** — a key methodological property of the benchmark.
Every cloud is resampled to a fixed point count so that voxel-based volume
estimates are not biased by sampling density.

## Features

| Feature            | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| `vol`              | Volume by voxel counting (voxel = 0.01)                            |
| `max_h`            | Maximum height (z)                                                 |
| `h_var`            | Variance of point heights                                          |
| `area`             | Convex-hull surface area (approximates the joint surface)          |
| `vol_area_ratio`   | `vol / area`                                                       |
| `convex_vol_ratio` | `vol / ConvexHull.volume` (shape irregularity)                     |
| `min_dist`         | Centre-to-centre distance to the nearest neighbouring joint        |

## Installation

```bash
git clone https://github.com/wqinmin975/pcb.git
cd pcb
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
(Python ≥ 3.8).

## Quickstart

```python
from solderbench import SolderJointGenerator, extract_features, get_baselines, run_comparison, run_ablation

gen = SolderJointGenerator(n_points=1500, noise=0.02, seed=42)
clouds, labels = gen.generate(n_per_class=250)   # point count / noise / defect strength are all configurable
X, y = gen.to_features(clouds, labels)           # 7-feature matrix + integer labels

results = run_comparison(X, y)                   # DataFrame: 6 methods x Acc/P/R/F1 + 5-fold CV
print(results[["no", "name", "Accuracy", "F1"]])

abl = run_ablation(X, y)                         # leave-one-out + single-feature ΔF1
print(abl.head(8))
```

## Reproducing the benchmark

```bash
python examples/run_benchmark.py
```

writes `results/tables/table_baseline.{csv,tex}`,
`results/tables/table_ablation.{csv,tex}` and
`results/figures/confusion_matrix.png` (250 samples × 5 classes, seed 42,
stratified 7:3 split, 5-fold CV on the training set).

### Benchmark snapshot (this repository, seed = 42)

| No. | Method                  | Accuracy | Precision | Recall | F1     | CV Acc. (5-fold) |
|-----|-------------------------|----------|-----------|--------|--------|------------------|
| B1  | S3DA (volume threshold) | 0.8960   | 0.8990    | 0.8960 | 0.8964 | 0.9234 ± 0.0138  |
| B2  | Logistic Regression     | 0.9387   | 0.9416    | 0.9387 | 0.9386 | 0.9657 ± 0.0198  |
| B3  | SVM (RBF)               | 0.9387   | 0.9390    | 0.9387 | 0.9385 | 0.9771 ± 0.0177  |
| B4  | KNN                     | 0.8933   | 0.8944    | 0.8933 | 0.8921 | 0.9234 ± 0.0168  |
| B5  | GradientBoosting        | 0.9413   | 0.9417    | 0.9413 | 0.9412 | 0.9577 ± 0.0138  |
| B6  | **RandomForest (Ours)** | **0.9440** | **0.9452** | **0.9440** | **0.9439** | **0.9680 ± 0.0160** |

Ablation highlights (RandomForest, macro F1; full model = 0.9439): removing
`h_var` costs the most (ΔF1 = −0.0459), followed by `vol_area_ratio`
(−0.0105); the best single feature is `convex_vol_ratio` (F1 = 0.6412) and
the weakest is `min_dist` (F1 = 0.3616).

## Repository structure

```
solderbench/
  __init__.py      # public API, __version__
  generator.py     # SolderJointGenerator: configurable point-cloud simulator
  features.py      # extract_features: the 7 geometric features
  baselines.py     # get_baselines: the 6 baseline classifiers
  evaluate.py      # run_comparison / run_ablation + CSV/LaTeX/figure export
  data.py          # load_dataset(): unified entry (real-data integration point)
tests/             # unittest suite (generator / features / baselines)
examples/
  run_benchmark.py # one-command reproduction of the full benchmark
pyproject.toml     # setuptools build, dependencies
codemeta.json      # software metadata (SoftwareX/Zenodo friendly)
LICENSE            # MIT
```

## Using real data

Replace the body of `solderbench.data.load_dataset` (or pass your own
`X, y` directly): read the real 3D-camera scans, segment individual solder
joints, and call `solderbench.extract_features` to compute the same seven
features. All downstream experiments (`run_comparison`, `run_ablation`)
work unchanged on the returned arrays.

## Testing

```bash
python -m unittest discover tests -v
```

The suite checks dataset shapes and class balance, exact same-seed
reproducibility, feature validity (finite values, positive volume,
`convex_vol_ratio ∈ (0, 1]`), a statistical assertion that
insufficient-solder joints have significantly smaller volume than good
joints, and that all six baselines fit/predict with RandomForest accuracy
in the expected range.

## Citation

If you use `solderbench` in your research, please cite:

```bibtex
@software{solderbench,
  title   = {solderbench: An open-source synthetic benchmark for PCB
             solder-joint defect inspection algorithm development},
  author  = {Qinmin Wang, Zixin Wang, Hongyuan Wang, and Zongzhong Wang},
  year    = {2025},
  version = {0.1.1},
  doi     = {10.5281/zenodo.22959193},
  url     = {https://github.com/wqinmin975/pcb}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Authors

- **Qinmin Wang** — School of Computer Science and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan, China (wangqinmin@gdust.edu.cn)
- **Zixin Wang** — School of Electronic and Communication Engineering, Tianjin Normal University, Tianjin, China (396514839@qq.com)
- **Hongyuan Wang** — School of Computer Science and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan, China
- **Zongzhong Wang** — School of Electromechanical Engineering, Guangdong University of Technology, Guangzhou, China (wangzz@gdut.edu.cn, corresponding author)

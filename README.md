# solderbench

**An open-source synthetic benchmark for PCB solder-joint defect inspection algorithm development.**

[![Version](https://img.shields.io/badge/version-v0.1.2-blue)](https://github.com/wqinmin975/pcb/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22959192.svg)](https://doi.org/10.5281/zenodo.22959192)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.13-blue.svg)](requirements.txt)

`solderbench` provides a configurable, pure-NumPy point-cloud simulator of
surface-mount solder joints, a seven-feature geometric descriptor extractor,
six baseline classifiers, and reproducible comparison/ablation experiment
runners. It is designed for developing and fairly evaluating 3D solder-joint
defect inspection algorithms when real annotated 3D scans are scarce, and it
exposes a clean integration point for swapping in real data.

> **Scope note.** Solderbench is a *parametric geometric model*, not a physics
> simulator. Metrics obtained on it are a controlled, reproducible proxy for
> algorithm development; they must not be read as predictions of industrial
> AOI performance. Final validation requires real production scans.

---

## Table of contents

1. [Defect classes](#defect-classes)
2. [Generator parameters](#generator-parameters)
3. [Features](#features)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Reproducing the paper's results](#reproducing-the-papers-results)
7. [Benchmark results](#benchmark-results)
8. [Robustness ablations](#robustness-ablations)
9. [Repository structure](#repository-structure)
10. [Using real data](#using-real-data)
11. [Testing](#testing)
12. [Reproduction environment](#reproduction-environment)
13. [Citation](#citation)
14. [License and authors](#license)

---

## Defect classes

| Class | Description | Simulated mechanism |
|---|---|---|
| `good` | Nominal joint | Half-ellipsoid with natural size variation |
| `insufficient` | Too little solder | Uniformly shrunk ellipsoid (0.55–0.85 of nominal) |
| `excessive` | Too much solder, irregular shape | Inflated ellipsoid (mean 1.30×) + random bump |
| `bridging` | Solder bridge to the neighbour | Abnormally small centre distance + low bridge strip |
| `collapse` | Collapsed / voided joint | Gaussian dent on the top surface |

Class distributions deliberately overlap (the upper tail of `insufficient`
overlaps the lower tail of `good`), so **no classifier can reach a perfect
score** — a key methodological property of the benchmark. Every cloud is
resampled to a fixed point count so voxel-based volume estimates are not
biased by sampling density.

## Generator parameters

All fifteen parameters of `SolderJointGenerator` (defaults are those used for
every number reported in the paper; all spatial quantities are in **mm**):

| Parameter | Default | Physical meaning |
|---|---|---|
| `n_points` | 1500 | fixed point count per cloud (simulates scanner resolution) |
| `noise` | 0.005 | Gaussian sensor-noise standard deviation σ (mm) |
| `base_radius` | 0.12 | nominal joint horizontal radius *a* (mm) |
| `base_height` | 0.10 | nominal joint height *c* (mm) |
| `good_scale_std` | 0.09 | relative size variation of good joints |
| `insufficient_scale_range` | (0.55, 0.85) | shrink factor of insufficient joints (overlaps good lower tail) |
| `excess_scale_mean` | 1.30 | mean inflation factor of excessive joints |
| `excess_scale_std` | 0.12 | inflation variation (overlaps good upper tail) |
| `excess_bump_range` | (150, 350) | number of extra points forming the bump |
| `normal_dist_mean` | 0.65 | mean centre-to-centre spacing, non-bridging (mm) |
| `normal_dist_std` | 0.07 | spacing variation (mm) |
| `bridge_dist_mean` | 0.34 | mean spacing for bridging joints (abnormally small, mm) |
| `bridge_dist_std` | 0.06 | bridging spacing variation (mm) |
| `bridge_points_range` | (80, 180) | number of points forming the bridge |
| `collapse_depth_range` | (0.28, 0.50) | dent depth as a fraction of joint height |

Narrowing or widening the overlap ranges makes the task easier or harder
(see [Robustness ablations](#robustness-ablations)).

## Features

`extract_features` returns seven descriptors. All coordinates are in **mm**;
the voxel edge is δ = 0.01 mm, so `vol` = *N*<sub>vox</sub> · δ³ mm³.

| Feature | Definition | Physical meaning | Units / computation |
|---|---|---|---|
| `vol` | *N*<sub>vox</sub> δ³ | total solder volume | mm³, unique occupied voxels at δ = 0.01 mm |
| `max_h` | max<sub>*i*</sub> *z<sub>i</sub>* | joint height | mm |
| `h_var` | Var(*z*) | height-distribution spread (dent / bump) | mm² |
| `area` | Surf(Conv(*P*)) | joint surface extent | mm², convex-hull area |
| `vol_area_ratio` | `vol / area` | compactness | dimensionless |
| `convex_vol_ratio` | `vol / Vol(Conv(*P*))` | shape irregularity (voids / dents) | dimensionless, ≤ 1 |
| `min_dist` | min<sub>*j*≠*i*</sub> ‖*c<sub>i</sub>* − *c<sub>j</sub>*‖ | bridging exposure | mm; synthetic: from the spacing distribution; real: distance between segmented joint centroids |

## Installation

### From source (recommended)

```bash
git clone https://github.com/wqinmin975/pcb.git
cd pcb
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # pinned versions (bit-for-bit reproduction)
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
(Python ≥ 3.8; every number in the paper was produced on **Python 3.13** —
see [Reproduction environment](#reproduction-environment)).

### With Docker

```bash
docker build -t solderbench:0.1.2 .
docker run --rm solderbench:0.1.2            # print the exact pinned environment
docker run --rm -v "$PWD/results:/app/results" solderbench:0.1.2 \
       python examples/run_benchmark.py      # regenerate every table and figure
docker run --rm solderbench:0.1.2 \
       python -m unittest discover tests -v  # run the test suite
```

## Quickstart

```python
from solderbench import (SolderJointGenerator, extract_features,
                         get_baselines, run_comparison, run_ablation)

gen = SolderJointGenerator(n_points=1500, noise=0.005, seed=42)
clouds, labels = gen.generate(n_per_class=250)   # 5 classes x 250 = 1250 joints
X, y = gen.to_features(clouds, labels)           # 7-feature matrix + integer labels

results = run_comparison(X, y)                   # 6 methods x Acc/P/R/F1 + 5-fold CV
print(results[["no", "name", "Accuracy", "F1"]])

abl = run_ablation(X, y)                         # leave-one-out + single-feature ΔF1
print(abl.head(8))
```

> **API stability.** The snippet above matches **v0.1.2**; later releases may
> change it. To freeze the exact behaviour, check out the corresponding tag
> (`git checkout v0.1.2`) or install the archived Zenodo release.

## Reproducing the paper's results

| Script | What it produces | Runtime |
|---|---|---|
| `python examples/run_benchmark.py` | the main benchmark snapshot (seed 42) → `results/tables/table_baseline.{csv,tex}`, `results/tables/table_ablation.{csv,tex}`, `results/figures/confusion_matrix.png` | ≈ 1 min |
| `python examples/run_revision.py` | all supplementary revision experiments: 5-seed mean ± std, sensor-noise ablation, defect-strength ablation, and the three point-cloud figures → `results/revision/` and `results/figures/` | ≈ 15 min |

Both scripts are deterministic: re-running them on the pinned dependencies
reproduces the reported numbers bit-for-bit.

## Benchmark results

### Main snapshot (seed 42, 1250 samples, stratified 7:3 split)

| No. | Method | Accuracy | Precision | Recall | F1 | CV Acc. (5-fold) |
|---|---|---|---|---|---|---|
| B1 | S3DA (volume threshold) | 0.8960 | 0.8990 | 0.8960 | 0.8964 | 0.9234 ± 0.0138 |
| B2 | Logistic Regression | 0.9387 | 0.9416 | 0.9387 | 0.9386 | 0.9657 ± 0.0198 |
| B3 | SVM (RBF) | 0.9387 | 0.9390 | 0.9387 | 0.9385 | 0.9771 ± 0.0177 |
| B4 | KNN | 0.8933 | 0.8944 | 0.8933 | 0.8921 | 0.9234 ± 0.0168 |
| B5 | GradientBoosting | 0.9413 | 0.9417 | 0.9413 | 0.9412 | 0.9577 ± 0.0138 |
| B6 | **RandomForest (Ours)** | **0.9440** | **0.9452** | **0.9440** | **0.9439** | **0.9680 ± 0.0160** |

### Five-seed reliability (mean ± std over seeds 42, 123, 456, 789, 1011)

| No. | Method | Accuracy | Precision | Recall | macro-F1 | CV Acc. |
|---|---|---|---|---|---|---|
| B1 | S3DA (vol. rule) | 0.9131 ± 0.0156 | 0.9164 ± 0.0153 | 0.9131 ± 0.0156 | 0.9136 ± 0.0153 | 0.9184 ± 0.0149 |
| B2 | Logistic Regression | 0.9579 ± 0.0114 | 0.9588 ± 0.0105 | 0.9579 ± 0.0114 | 0.9578 ± 0.0114 | 0.9625 ± 0.0157 |
| B3 | SVM (RBF) | 0.9611 ± 0.0119 | 0.9616 ± 0.0119 | 0.9611 ± 0.0119 | 0.9610 ± 0.0119 | 0.9696 ± 0.0115 |
| B4 | KNN | 0.9253 ± 0.0178 | 0.9272 ± 0.0184 | 0.9253 ± 0.0178 | 0.9249 ± 0.0184 | 0.9239 ± 0.0176 |
| B5 | GradientBoosting | 0.9477 ± 0.0078 | 0.9491 ± 0.0079 | 0.9477 ± 0.0078 | 0.9476 ± 0.0079 | 0.9557 ± 0.0140 |
| B6 | **RandomForest (Ours)** | **0.9531 ± 0.0115** | **0.9538 ± 0.0116** | **0.9531 ± 0.0115** | **0.9530 ± 0.0115** | 0.9650 ± 0.0127 |

Reading of the five-seed table — stated conservatively:

* The industrial volume rule S3DA (B1) is already a strong baseline
  (0.9131 ± 0.0156), confirming that solder volume carries most of the
  class signal.
* **Most** learning-based methods improve on the rule; kNN (0.9253) is
  within one standard deviation of S3DA and is therefore **statistically
  tied** with it.
* RandomForest attains the best *mean test* accuracy (0.9531), but it leads
  GradientBoosting by only 0.0053 while the inter-seed std of that gap is
  0.0065 — the gap is smaller than its own variability, so **we do not claim
  RF strictly outperforms GB**.
* The RBF SVM has the highest *cross-validation* accuracy (0.9696) but a
  lower held-out test accuracy (0.9611); report the independent test set,
  not cross-validation alone.

Ablation highlights (RandomForest, macro F1; full model = 0.9439): removing
`h_var` costs the most (ΔF1 = −0.0459), followed by `vol_area_ratio`
(−0.0105); the best single feature is `convex_vol_ratio` (F1 = 0.6412) and
the weakest is `min_dist` (F1 = 0.3616, informative for bridging only).

## Robustness ablations

Reproduce with `python examples/run_revision.py`.

### Sensor noise σ (mm), seed 42

| σ | RF Acc. | S3DA Acc. | GB Acc. | Accuracy span over all 6 methods |
|---|---|---|---|---|
| 0.005 | 0.9440 | 0.8960 | 0.9413 | [0.8933, 0.9440] |
| 0.010 | 0.9360 | 0.9013 | 0.9307 | [0.8960, 0.9387] |
| 0.020 | 0.9173 | 0.8667 | 0.9120 | [0.8667, 0.9307] |
| 0.050 | 0.8640 | 0.8293 | 0.8560 | [0.8293, 0.8827] |
| 0.100 | 0.7467 | 0.7200 | 0.7307 | [0.6933, 0.7547] |

Accuracy degrades monotonically and gracefully, and the relative ordering of
methods is preserved.

### Defect strength / class overlap, seed 42

| Configuration | RF Acc. | S3DA Acc. | Worst-method Acc. |
|---|---|---|---|
| narrow (easier) | 0.9653 | 0.9040 | 0.9040 |
| default | 0.9440 | 0.8960 | 0.8933 |
| wide (harder) | 0.9440 | 0.8960 | 0.8960 |

Difficulty is controllable in the easy direction; widening the overlap does
not further degrade the seven-feature representation, which stays
discriminative under aggressive overlap.

## Repository structure

```
solderbench/
  __init__.py       # public API, __version__
  generator.py      # SolderJointGenerator: configurable point-cloud simulator
  features.py       # extract_features: the 7 geometric features
  baselines.py      # get_baselines: the 6 baseline classifiers (incl. S3DA)
  evaluate.py       # run_comparison / run_ablation + CSV/LaTeX/figure export
  data.py           # load_dataset(): unified entry (real-data integration point)
tests/              # 15-test unittest suite (generator / features / baselines)
examples/
  run_benchmark.py  # one-command reproduction of the main benchmark
  run_revision.py   # supplementary experiments (5 seeds, noise/defect ablations, figures)
paper/              # SoftwareX manuscript source (solderbench_softwarex_draft.tex)
requirements.txt    # pinned dependency set for bit-for-bit reproduction
Dockerfile          # reproducible container image
pyproject.toml      # setuptools build, dependencies
codemeta.json       # software metadata (SoftwareX / Zenodo friendly)
LICENSE             # MIT
```

## Using real data

Replace the body of `solderbench.data.load_dataset` (or pass your own `X, y`
directly): read the real 3D-camera scans, segment individual solder joints,
and call `solderbench.extract_features` to compute the same seven features.
All downstream experiments (`run_comparison`, `run_ablation`) work unchanged.
The raw point clouds can also be handed directly to external deep models
(e.g. PointNet / PointNet++).

## Testing

```bash
python -m unittest discover tests -v
```

The suite checks dataset shapes and class balance, exact same-seed
reproducibility, feature validity (finite values, positive volume,
`convex_vol_ratio ∈ (0, 1]`), a Welch *t*-test that insufficient-solder
joints have significantly smaller volume than good joints, and that all six
baselines fit/predict with RandomForest accuracy in the expected range.

## Reproduction environment

Every number in the paper was produced with the exact stack below
(`requirements.txt` pins all five libraries):

| Component | Version |
|---|---|
| Python | 3.13 |
| NumPy | 2.5.2 |
| SciPy | 1.18.0 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.1 |

All randomness derives from a single `RandomState` created inside
`SolderJointGenerator` (never the global NumPy seed), so a given seed yields
point-wise identical clouds, features, and tables. There is no compiled code,
no GPU, and no network access at runtime; the package is OS-independent
(tested on Windows and Linux via the Docker image).

## Citation

If you use `solderbench` in your research, please cite the archived software.
The DOI below is the Zenodo **concept DOI**, which always resolves to the
latest release:

```bibtex
@software{solderbench,
  title   = {solderbench: An open-source synthetic benchmark for PCB
             solder-joint defect inspection algorithm development},
  author  = {Wang, Qinmin and Wang, Zixin and Wang, Hongyuan and Wang, Zongzhong},
  year    = {2026},
  version = {0.1.2},
  doi     = {10.5281/zenodo.22959192},
  url     = {https://github.com/wqinmin975/pcb}
}
```

* Concept DOI (all versions, always latest): `10.5281/zenodo.22959192`
* v0.1.1 version DOI (frozen): `10.5281/zenodo.22959193`

## License

MIT, see [LICENSE](LICENSE).

## Authors

- **Qinmin Wang** — School of Computer Science and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan, China (wangqinmin@gdust.edu.cn)
- **Zixin Wang** — School of Electronic and Communication Engineering, Tianjin Normal University, Tianjin, China (396514839@qq.com)
- **Hongyuan Wang** — School of Computer Science and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan, China
- **Zongzhong Wang** — School of Electromechanical Engineering, Guangdong University of Technology, Guangzhou, China (wangzz@gdut.edu.cn, corresponding author)

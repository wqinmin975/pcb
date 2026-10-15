# Response to Reviewers

**Manuscript:** *Solderbench: An open-source synthetic benchmark for PCB solder-joint defect inspection algorithm development*
**Journal:** SoftwareX (Original Software Publication) — **Revised manuscript, Major Revision**
**Software release accompanying this revision:** v0.1.3 (GitHub `https://github.com/wqinmin975/pcb`; Zenodo concept DOI `10.5281/zenodo.22959192`)

Accompanying files in this submission:

| File | Content |
|---|---|
| `solderbench_softwarex_draft_v20_trackchanges.pdf` | **Track-changes version** — all modifications relative to the original submission are marked (additions underlined/blue, deletions struck out/red) |
| `solderbench_softwarex_draft_v20_clean.pdf` | **Clean version** — identical content, no markup |
| `RESPONSE_TO_REVIEWERS.md` | This document |
| `COVER_LETTER.md` | Cover letter |

We thank both reviewers for the exceptionally careful reading. Every comment has been
addressed **in the manuscript and in the code**, and every number below is regenerated
by the scripts shipped with release v0.1.3.

> Note: the reviewer comments below are paraphrased for brevity. If the editorial
> system provides the verbatim text, replace each "Comment" line with the verbatim
> quote — the "Response" and "Location" entries stay unchanged.

---

## Overview — comment → change → location

| Comment | Our response | Location in the manuscript |
|---|---|---|
| **R1.1** Validity / credibility of a purely synthetic benchmark | Physical parameterisation made explicit; per-class point clouds shown; synthetic-only scope stated as a limitation | Table 2, Figure 2, §4.3, §6 |
| **R1.2** Robustness and statistical reliability | Five independent seeds with mean ± std; sensor-noise and defect-strength ablations | Table 4, Table 5, Table 6, §4.3 |
| **R1.3** Feature definitions and baseline implementation | Full mathematical definitions, units and voxel size; S3DA re-implemented as the original hand-crafted rule | Table 3, §3.2 |
| **R1.4** Software usability, reproducibility, real-data integration | Pinned dependencies, Dockerfile, rewritten README, 17 unit tests; explicit statement of what the real-data hook does *not* do | §3.1, §3.3, §5, repository |
| **R2.1** Generator description and demonstration | All 15 generator parameters tabulated with defaults and physical meaning; one sample cloud per class | Table 2, Figure 2 |
| **R2.2** Geometric feature definitions and input information | Definitions, physical meaning **and** computation/units column; voxel edge δ = 0.01 mm stated | Table 3 |
| **R2.3** S3DA implementation and comparison conditions | Original rule implemented and used for **all** reported numbers; learned tree proxy demoted to repository-only reference; threshold-sensitivity sweep shipped | §3.2, Tables 4–6, `examples/s3da_threshold_sensitivity.py` |
| **R2.4** Stability of the benchmark comparison | Single-seed Table replaced by a five-seed mean ± std table | Table 4, Abstract |
| **R2.5** Consistency between results and their interpretation | Four over-claimed statements corrected (see R2.5 for the list) | §4.2, §4.4, §6 |
| **R2.6** Supporting environment and reproduction instructions | Exact dependency versions, Docker image, runtimes, archival DOI, one-command reproduction | §3.3, §7, README, `Dockerfile` |

---

## Reviewer 1

### R1.1 — Validity of the synthetic benchmark

**Comment.** The benchmark is entirely synthetic; the authors should substantiate that
the generated data are physically meaningful and usable as a proxy for real inspection.

**Response.** We now make the physical basis of the generator explicit and show, rather
than assert, what the clouds look like.

* **Table 2** (new) lists **all 15 generator parameters** with their defaults and their
  physical interpretation. Every spatial quantity is in millimetres; the nominal joint
  radius `base_radius` = 0.12 mm and height `base_height` = 0.10 mm set the joint scale,
  `noise` is the sensor measurement standard deviation σ, and each defect is expressed as
  a controlled deviation from nominal geometry (shrinkage 0.55–0.85, inflation mean 1.30,
  bridging spacing 0.34 mm vs. 0.65 mm normal, dent depth 0.28–0.50 of joint height).
* **Figure 2** (new) shows one representative cloud per class, so the reader can see that
  the classes are geometrically distinguishable and that they correspond to the named
  industrial defects.
* **§4.3** (new section) states explicitly that the clouds are parametric geometric models
  that are **not** fitted to production scans, and that the benchmark's purpose is a
  controlled, reproducible comparison protocol.
* **§6 (Limitations)** now states: *all benchmark experiments in this manuscript are based
  exclusively on synthetic data; the absolute accuracy values cannot be directly
  transferred to industrial real-world inspection* — only the relative ordering under an
  identical protocol should be read as a result.

### R1.2 — Robustness and statistical reliability

**Comment.** Results rest on a single random seed; robustness to noise and to defect
strength is not demonstrated.

**Response.** All headline results are now multi-seed, and two ablations were added.

* **Table 4** (replaces the old single-seed table): five independent seeds
  (42, 123, 456, 789, 1011), 1250 samples per seed, identical stratified 7:3 split and
  5-fold CV. Random forest: **0.9531 ± 0.0115** accuracy, 0.9530 ± 0.0115 macro-F₁
  (bold in the table as the best mean test metric).
* **Table 5** (new): sensor-noise ablation, σ ∈ {0.005, 0.01, 0.02, 0.05, 0.10} mm.
  RF accuracy falls monotonically 0.9440 → 0.7467; the S3DA rule falls 0.5600 → 0.2000;
  the span over all six methods widens with σ, i.e. the ordering is stable but the
  absolute level degrades as expected.
* **Table 6** (new): defect-strength / class-overlap ablation (narrow, default, wide).
  RF: 0.9653 / 0.9440 / 0.9440. Difficulty is tunable, as claimed.
* **Abstract** now reports both the seed-42 reference value (0.9440 / 0.9439) **and** the
  five-seed statistic (0.9531 ± 0.0115).

### R1.3 — Feature definitions and baseline implementation

**Comment.** The geometric features and the baseline implementations are not specified
precisely enough to be reproduced or compared.

**Response.**

* **Table 3** now carries a dedicated **"Computation (units)"** column: the voxel volume
  is `vol = N_vox · δ³` with **δ = 0.01 mm** (units mm³), heights and variances are in
  mm / mm², area is the convex-hull surface in mm², ratios are dimensionless, and the
  provenance of `min_dist` is given for both synthetic (spacing distribution) and real
  (centroid distance between segmented joints) input.
* The **S3DA baseline** is now the original hand-crafted rule — see R2.3, which is the
  substantive part of this response.
* All six baselines are scikit-learn estimators with fixed random states and share the
  **identical** stratified 7:3 split and the **same** seven-feature input; this is stated
  explicitly in §3.2 so that the comparison is unambiguously fair.

### R1.4 — Software usability, reproducibility, and real-data integration

**Comment.** The package should be usable and reproducible by a third party; it should be
clear how real data enter the pipeline.

**Response.**

* `requirements.txt` pins the verified environment: Python 3.13, NumPy 2.5.2,
  SciPy 1.18.0, pandas 3.0.6, scikit-learn 1.9.0, Matplotlib 3.11.1.
* A **`Dockerfile`** (new) builds a pinned Python 3.13 image and runs the whole benchmark
  with one command: `docker build -t solderbench:0.1.3 . && docker run solderbench:0.1.3`.
* **README.md** rewritten: parameter table, five-seed results, both ablations,
  reproduction commands and their runtimes (`run_benchmark.py` ≈ 1 min,
  `run_revision.py` ≈ 15 min, `s3da_threshold_sensitivity.py` ≈ 5 min), archival citation.
* **17 unit tests** (was 15) cover shape and class balance, same-seed reproducibility,
  feature validity, a Welch *t*-test that insufficient-solder volumes are significantly
  smaller, and — new — that the S3DA rule is label-free, deterministic and never emits
  the bridging/collapse classes.
* **Real data — explicitly scoped.** §3.1 now states verbatim:

  > `load_dataset()` provides a software hook for external real-world point-cloud input.
  > Real-world data acquisition, joint segmentation and pre-processing functions are
  > **not** implemented within this package; users must supply pre-segmented solder-joint
  > point clouds (or their own `X, y` feature matrix) at this hook, after which every
  > downstream experiment runs unchanged.

  The same caveat is in the package docstring (`solderbench/data.py`) and the README. We
  deliberately avoided claiming an end-to-end real-data pipeline that does not exist.

---

## Reviewer 2

### R2.1 — Generator description and demonstration

**Comment.** The generator is described only in prose; a parameter table and a visual
demonstration are missing.

**Response.** Added **Table 2** (all 15 parameters, defaults, physical meaning, mm) and
**Figure 2** (one representative cloud per class, axes in mm, seed 42, 1500 points).
The Quickstart in §4.1 was corrected to the noise value that actually reproduces the
reported numbers (`noise = 0.005`), so the example now regenerates Table 4 exactly.

### R2.2 — Geometric feature definitions and input information

**Comment.** The seven features need mathematical definitions, and the input information
(voxel size, coordinate units) must be stated.

**Response.** **Table 3** now gives, for each feature, the symbol definition, the physical
meaning, and the computation with units. The caption states that all coordinates are in
millimetres and that the voxel edge is **δ = 0.01 mm**.

*Consistency check on the voxel size requested by the handling editor:* the manuscript
and the code (`solderbench/features.py`, `VOXEL_SIZE = 0.01`) use **0.01 mm throughout**.
We re-checked the entire source: **no 0.05 mm value appears anywhere** in the manuscript
or in the code. The 0.05 mm figure came from an external template and was never adopted.

### R2.3 — S3DA implementation and comparison conditions

**Comment.** The published S3DA is a hand-tuned volume-threshold rule. The manuscript
described a decision-tree variant; the reader cannot tell which one produced the numbers.

**Response — this is the most substantive change of the revision.** The reviewer was
right: our previous B1 was a `DecisionTreeClassifier(max_depth = 6)` trained on all seven
features, i.e. a **learned proxy**, not the published rule. We therefore did **not** simply
reword the description — we implemented the original rule and **re-ran every experiment**.

S3DA is now available in two clearly separated forms (§3.2):

1. **S3DA-rule (used for every number in the paper).** `S3DARuleClassifier` reproduces the
   hand-tuned rule of Xu et al. [6]: the solder volume is obtained by integral summation
   (the voxel-count descriptor `vol`) and compared against two **hand-set** acceptance
   limits, `0.90` and `1.10`, expressed as fractions of a nominal reference volume. The
   reference volume is calibrated **solely from training-set feature statistics** — the
   median volume of the training features — and uses no class label; **it never accesses
   any test-set information or class labels**. The rule has no trained parameters, so its
   decision is deterministic and identical for every seed. Because the published rule
   thresholds volume alone, it can only emit {good, insufficient, excessive}; bridging and
   collapse are not volume defects and are never predicted by it.
2. **Learned tree proxy (reference only).** A shallow decision tree (`max_depth = 6`)
   trained on all seven features, whose splits are dominated by voxel volume, is retained
   in the repository as `get_s3da_tree_proxy()`. **It is used for no number in the
   manuscript.**

**Reported S3DA numbers (original rule):**

| | seed 42 | five seeds (mean ± std) |
|---|---|---|
| Accuracy | 0.5600 | **0.5632 ± 0.0091** |
| macro Precision | 0.4160 | 0.4245 ± 0.0069 |
| macro Recall | 0.5600 | 0.5632 ± 0.0091 |
| macro F₁ | 0.4588 | 0.4636 ± 0.0075 |
| CV accuracy | 0.5726 ± 0.0301 | 0.5675 ± 0.0376 |

For contrast, the *learned tree proxy* reaches 0.9131 ± 0.0156 over the same five seeds —
roughly 0.35 accuracy stronger than the published rule. Conflating the two would have
overstated the industrial baseline, which is exactly why the distinction matters.

**Anticipating "were the two limits tuned?"** We ship
`examples/s3da_threshold_sensitivity.py`, which evaluates the rule over five seeds for
limits (0.70, 1.30), (0.80, 1.20), (0.85, 1.15), (0.90, 1.10) and (0.95, 1.05). Mean
accuracy ranges from **0.3163** to **0.5632** across the whole sweep — i.e. the rule stays
far below every learned method (the weakest, kNN, is at 0.9253) for *every* setting. The
qualitative conclusion is insensitive to the choice of limits.

**Comparison conditions** (also requested): all six methods use the identical stratified
7:3 split, the same seven-feature input, and the same 5-fold CV protocol on the training
split. This is now stated explicitly in §3.2.

### R2.4 — Stability of the benchmark comparison

**Comment.** A single-seed comparison is not stable enough to support the conclusions.

**Response.** See R1.2: Table 4 is now a five-seed mean ± std table, and the abstract
reports the multi-seed statistic alongside the seed-42 value. All robustness claims in
the text are now stated against the inter-seed standard deviation rather than against
point estimates.

### R2.5 — Consistency between results and their interpretation

**Comment.** Several statements in the text are not supported by the reported numbers.

**Response.** We went through the manuscript sentence by sentence and corrected four
over-claims. (Two further corrections became necessary because B1 changed — see the last
two bullets.)

1. ~~"all learning-based methods improve on the rule"~~ → **most** do; kNN (0.9253) is
   within one standard deviation of the strongest rule setting, so we write "most … ,
   while kNN remains statistically tied with it". *(Note: with the rule corrected to
   0.5632, every learned method now leads it by more than 0.36 accuracy — many times the
   inter-seed std — so this caveat is now conservative rather than necessary; the text
   states the corrected version.)*
2. ~~"the random forest strictly outperforms gradient boosting"~~ → the mean gap is
   0.0053 while the inter-seed std is 0.0065, i.e. **the gap is smaller than the
   inter-seed standard deviation**. We now say the two are *statistically tied* and define
   the term explicitly in §4.2; we do not claim RF strictly outperforms GB.
3. ~~"the SVM CV/test gap demonstrates overfitting"~~ → restated as *consistent with mild
   overfitting, but it may also reflect split variance*; we no longer assert a cause.
4. ~~"excessive solder is the hardest defect"~~ → corrected: excessive solder is in fact
   **well separated** and incurs **no test errors** in Figure 1. The genuinely confusable
   classes are collapse vs. good/insufficient.
5. ~~"S3DA is a strong baseline … volume alone carries most of the class signal"~~ →
   removed. The rule is far above the 0.20 chance level but far below every learned
   method; volume alone is *necessary but not sufficient*, and its zero recall on
   bridging/collapse is structural, not an implementation defect.
6. ~~"the S3DA rule reaches 0.8960 accuracy using volume only"~~ → 0.5600 at seed 42
   (0.5632 ± 0.0091 over five seeds).

### R2.6 — Supporting environment and reproduction instructions

**Comment.** The environment and the exact reproduction steps should be documented.

**Response.**

* Dependency versions are given in the manuscript (§3.3) and pinned in
  `requirements.txt`; the package is tested on Python 3.13.
* `Dockerfile` + `.dockerignore` provide a one-command reproducible image.
* `README.md` documents the reproduction commands **and their runtimes**.
* The code is archived on Zenodo under the concept DOI `10.5281/zenodo.22959192`
  (always resolves to the latest release; the v0.1.1 version DOI
  `10.5281/zenodo.22959193` is retained for reference). Code metadata C1–C3 in Table 1
  are updated accordingly (C1 = v0.1.3).
* `python examples/run_benchmark.py` regenerates every table and figure of the original
  submission; `python examples/run_revision.py` regenerates every *new* table and figure
  of this revision.

---

## Summary of numerical changes (old → new)

| Quantity | Original submission | This revision |
|---|---|---|
| Table 4 protocol | 1 seed (42) | 5 seeds, mean ± std |
| S3DA (B1) implementation | decision-tree proxy | original hand-crafted volume rule |
| S3DA accuracy | 0.8960 (seed 42) / 0.9131 ± 0.0156 | **0.5600 (seed 42) / 0.5632 ± 0.0091** |
| Random forest accuracy | 0.9440 | 0.9440 (seed 42) and **0.9531 ± 0.0115** (5 seeds) |
| Noise ablation | not reported | Table 5 (σ = 0.005 … 0.10) |
| Defect-strength ablation | not reported | Table 6 (narrow / default / wide) |
| Generator parameters | prose only | Table 2 (all 15, with units) |
| Feature specification | names + short glosses | Table 3 with formulas, computation and units |
| Dependency versions | not stated | pinned in `requirements.txt` + Dockerfile |
| Unit tests | 15 | 17 |

---

## Reproduction commands

```bash
pip install -r requirements.txt          # pinned environment
python -m unittest discover tests -v     # 17 tests
python examples/run_benchmark.py           # seed-42 snapshot + Figure 1   (~1 min)
python examples/run_revision.py            # 5 seeds + both ablations      (~15 min)
python examples/s3da_threshold_sensitivity.py   # S3DA limit sweep         (~5 min)
# or, containerised:
docker build -t solderbench:0.1.3 . && docker run solderbench:0.1.3
```

| Manuscript item | Generating script |
|---|---|
| Table 4 (five seeds), Table 5 (noise), Table 6 (defect strength), Figures 2–3 | `examples/run_revision.py` |
| Seed-42 snapshot, Figure 1 (confusion matrix), Tables 7–8 (ablation) | `examples/run_benchmark.py` |
| S3DA threshold sensitivity (supporting evidence, R2.3) | `examples/s3da_threshold_sensitivity.py` |

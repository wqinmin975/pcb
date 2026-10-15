# Cover Letter

**To:** The Editor-in-Chief / Handling Editor, *SoftwareX*
**Date:** 15 October 2026
**Article type:** Original Software Publication
**Manuscript title:** *Solderbench: An open-source synthetic benchmark for PCB solder-joint defect inspection algorithm development*
**Manuscript number:** [insert manuscript number]
**Revision:** Major Revision — revised manuscript resubmitted

Dear Editor,

Thank you for the opportunity to revise our manuscript, and please pass our thanks to
both reviewers. Their comments exposed one genuinely serious problem in the original
submission, and addressing it has materially improved the paper. We have implemented
every requested change **in the manuscript and in the released software**, and we
resubmit a revised manuscript together with a point-by-point response.

## What we changed — summary

1. **The S3DA baseline is now the original hand-crafted rule (the substantive fix).**
   In the original submission our S3DA baseline was a shallow decision tree trained on
   all seven features — a learned proxy for, rather than a reproduction of, the
   hand-tuned volume-threshold rule of Xu et al. [6]. We have now implemented the
   published rule itself (`S3DARuleClassifier`: integral-summation volume compared
   against hand-set acceptance limits of 0.90 and 1.10 times a nominal reference volume,
   the reference being calibrated from training-set statistics only, with no access to
   test data or class labels) and **re-run every experiment**. The reported S3DA accuracy
   changes accordingly, from 0.9131 ± 0.0156 (proxy) to **0.5632 ± 0.0091** (rule) over
   five seeds. All S3DA numbers in the paper now come from the rule; the tree proxy is
   retained in the repository for reference only and is used for no reported number. We
   additionally ship a threshold-sensitivity sweep showing that the conclusion holds for
   every setting of the two limits.
2. **All headline results are now multi-seed.** The single-seed comparison table is
   replaced by a five-seed (42/123/456/789/1011) mean ± std table, and the abstract now
   reports both the seed-42 reference value (0.9440 accuracy, 0.9439 macro-F₁) and the
   five-seed statistic (**0.9531 ± 0.0115** for the random forest).
3. **Two new robustness studies.** A sensor-noise ablation (σ from 0.005 to 0.10 mm) and
   a defect-strength / class-overlap ablation (narrow / default / wide) are reported as
   new tables, with a figure showing the effect of noise on a nominal joint.
4. **The generator and the features are now fully specified.** A new table lists all 15
   generator parameters with defaults and physical meaning (mm); a new figure shows one
   representative point cloud per defect class; and the feature table now gives
   mathematical definitions together with a computation/units column (voxel volume
   `vol = N_vox·δ³`, δ = 0.01 mm; mm and mm² units throughout). We confirm that
   **δ = 0.01 mm is used consistently** in the manuscript and in the code — no 0.05 mm
   value appears anywhere.
5. **Four over-claimed statements corrected.** (i) "all learning-based methods improve on
   the rule" now reads "most"; (ii) we no longer claim the random forest strictly
   outperforms gradient boosting — the mean gap (0.0053) is smaller than the inter-seed
   standard deviation (0.0065), so the two are reported as statistically tied, with the
   term defined; (iii) the SVM cross-validation/test gap is described as *consistent
   with* mild overfitting rather than as proof of it; (iv) excessive solder, previously
   called the hardest defect, is corrected — it is in fact well separated and incurs no
   test errors.
6. **Reproducibility and real-data scope.** Exact dependency versions are pinned
   (Python 3.13; NumPy 2.5.2, SciPy 1.18.0, pandas 3.0.6, scikit-learn 1.9.0,
   Matplotlib 3.11.1), a `Dockerfile` builds a one-command reproducible image, the README
   has been rewritten with parameters, results and reproduction commands with runtimes,
   and the test suite now has 17 tests. Following Reviewer 1's comment 4, the manuscript
   now states explicitly that `load_dataset()` is a **software hook only**: real-world
   data acquisition, joint segmentation and pre-processing are **not** implemented, and
   users must supply pre-segmented solder-joint point clouds.
7. **A synthetic-only scope statement** has been added to the Limitations section: the
   absolute accuracy values cannot be transferred directly to industrial inspection;
   only the relative ordering under a controlled, identical protocol should be read as a
   result.

## Files enclosed

| File | Content |
|---|---|
| `solderbench_softwarex_draft_v20_trackchanges.pdf` | Revised manuscript **with track changes** (additions underlined/blue, deletions struck out/red) |
| `solderbench_softwarex_draft_v20_clean.pdf` | Revised manuscript, **clean copy** (identical content) |
| `RESPONSE_TO_REVIEWERS.md` | Point-by-point response to all ten reviewer comments |

## Software and data availability

The software accompanying this revision is release **v0.1.3**, available at
`https://github.com/wqinmin975/pcb` and archived on Zenodo under the concept DOI
`10.5281/zenodo.22959192` (which always resolves to the latest release). The benchmark
dataset is not distributed as a fixed file: it is generated deterministically at run
time, and `python examples/run_benchmark.py` regenerates every table and figure in the
paper.

## Declarations

* **Funding:** this research did not receive any specific grant from funding agencies in
  the public, commercial, or not-for-profit sectors.
* **Competing interests:** the authors declare none.
* **CRediT authorship contribution statement:** included in the manuscript.
* **Authors:** Qinmin Wang, Zixin Wang, Hongyuan Wang, Zongzhong Wang (corresponding).
* **Corresponding author:** Zongzhong Wang, School of Electromechanical Engineering,
  Guangdong University of Technology, Guangzhou, China — wangzz@gdut.edu.cn.

We believe the revised manuscript and the accompanying software release fully address
the reviewers' concerns, and we thank you for your time and consideration.

Sincerely,

**Zongzhong Wang** (on behalf of all authors)
School of Electromechanical Engineering, Guangdong University of Technology
wangzz@gdut.edu.cn

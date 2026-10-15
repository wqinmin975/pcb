"""solderbench -- an open-source synthetic benchmark for PCB solder-joint
defect inspection algorithm development.

Public API
----------
- :class:`SolderJointGenerator` -- configurable point-cloud simulator
  (five classes: good / insufficient / excessive / bridging / collapse).
- :func:`extract_features` -- seven geometric features per joint cloud.
- :func:`get_baselines` -- the six baseline classifiers.
- :func:`run_comparison`, :func:`run_ablation` -- experiment runners.
- :func:`load_dataset` -- one-call dataset builder (real-data entry point).
"""

__version__ = "0.1.2"

from .baselines import get_baselines
from .data import load_dataset
from .evaluate import (
    confusion_matrix_ours,
    evaluate_classifier,
    plot_confusion_matrix,
    run_ablation,
    run_comparison,
    write_ablation_tex,
    write_baseline_tex,
)
from .features import FEATURES, VOXEL_SIZE, extract_features, features_to_array
from .generator import (
    CLASSES,
    CLASS_NAMES_EN,
    CLASS_TO_IDX,
    SolderJointGenerator,
    SolderSample,
)

__all__ = [
    "__version__",
    "SolderJointGenerator",
    "SolderSample",
    "CLASSES",
    "CLASS_NAMES_EN",
    "CLASS_TO_IDX",
    "FEATURES",
    "VOXEL_SIZE",
    "extract_features",
    "features_to_array",
    "get_baselines",
    "run_comparison",
    "run_ablation",
    "evaluate_classifier",
    "confusion_matrix_ours",
    "plot_confusion_matrix",
    "write_baseline_tex",
    "write_ablation_tex",
    "load_dataset",
]

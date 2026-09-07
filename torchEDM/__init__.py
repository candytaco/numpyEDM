"""Python tools for EDM with a pyTorch backend"""

import torch
torch.set_grad_enabled(False)

# array-in, array-out predictors
from .EDM.Predictors import SimplexPredict, SimplexGenerate, SMapPredict, SMapGenerate
from .EDM.Multiview import MultiviewPredict
from .EDM.ConvergentCrossMap import ConvergentCrossMap
from .EDM.MDE import MDE
# parameter sweeps
from .Hyperparameters import (FindOptimalEmbeddingDimensionality, FindOptimalPredictionHorizon,
                              FindSMapNeighborhood, FindSelfPredictionEmbeddingDimension)
# parameter-holding wrappers with Fit(X_train, Y_train, X_test, Y_test)
from . import Fitters
from .Utils import SurrogateData
from .FitterExamples import FitterExamples

from .EDM.Results import (
    SimplexResult,
    SMapResult,
    MultiviewResult,
    MDEResult,
    BatchedCCMResult,
    ResultsIO,
)
from .Visualization import (
    plot_prediction,
    plot_smap_coefficients,
    plot_ccm,
    plot_multiview,
    plot_embed_dimension,
    plot_predict_interval,
    plot_predict_nonlinear
)

__version__     = "4"
__versionDate__ = "2026-09-07"

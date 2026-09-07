from typing import Union

import numpy
import torch

from ..EDM.MDE import MDE
from .EDMFitter import EDMFitter


class MDEFitter(EDMFitter):
	"""Parameter holder for MDE."""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = False,
				 Convergent: Union[str, bool] = 'pre',
				 Metric: str = "correlation",
				 BatchSize: int = 1000,
				 dtype: torch.dtype = torch.float32,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5,),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 MinCandidatePerformance: float = 0.5,
				 IterativeDimensionSearch: bool = False,
				 TimeDelay: int = 0,
				 progressBar: bool = True,
				 device = None):
		"""Parameters as in MDE."""
		super().__init__(progressBar)
		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.dtype = dtype
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose
		self.UseSMap = UseSMap
		self.Theta = Theta
		self.stdThreshold = stdThreshold
		self.CCMLibraryPercentiles = CCMLibraryPercentiles
		self.CCMNumSamples = CCMNumSamples
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.CCMSeed = CCMSeed
		self.CCMMaxEmbeddingDimensions = CCMMaxEmbeddingDimensions
		self.MinPredictionThreshold = MinPredictionThreshold
		self.MinCandidatePerformance = MinCandidatePerformance
		self.IterativeDimensionSearch = IterativeDimensionSearch
		self.TimeDelay = TimeDelay
		self.device = device
		self.MDE = None

	def MDEKeywords(self) -> dict:
		return dict(maxD = self.MaxD, include_target = self.IncludeTarget, convergent = self.Convergent,
					metric = self.Metric, batch_size = self.BatchSize, dtype = self.dtype,
					embedDimensions = self.EmbedDimensions, predictionHorizon = self.PredictionHorizon,
					knn = self.KNN, step = self.Step, exclusionRadius = self.ExclusionRadius, verbose = self.Verbose,
					useSMap = self.UseSMap, theta = self.Theta, stdThreshold = self.stdThreshold,
					CCMLibraryPercentiles = self.CCMLibraryPercentiles, CCMNumSamples = self.CCMNumSamples,
					CCMConvergenceThreshold = self.CCMConvergenceThreshold, CCMSeed = self.CCMSeed,
					CCMMaxEmbeddingDimensions = self.CCMMaxEmbeddingDimensions,
					MinPredictionThreshold = self.MinPredictionThreshold,
					MinCandidatePerformance = self.MinCandidatePerformance,
					IterativeDimensionSearch = self.IterativeDimensionSearch, TimeDelay = self.TimeDelay,
					device = self.device)

	def Fit(self, X_train, Y_train, X_test = None, Y_test = None):
		self.MDE = MDE(X_train, Y_train, X_test, Y_test, **self.MDEKeywords())
		self.Result = self.MDE.Run()
		return self.Result

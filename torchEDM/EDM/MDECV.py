"""
Cross-validated greedy variable selection over contiguous folds of one series.
"""
from typing import List, Optional

import numpy
import torch
from tqdm import tqdm as ProgressBar

from .MDE import MDE
from .Predictors import SimplexPredict, SMapPredict
from .Results import MDECVResult, MDEResult
from .Setup import AsRuns
from ..Scoring import Correlation


class MDECV:
	"""
	MDE runs once per fold; the final variable set is taken from the best fold, or from the
	most frequent selections, and then predicts a separate test set.
	"""

	def __init__(self,
				 X_train: numpy.ndarray,
				 Y_train: numpy.ndarray,
				 maxD: int = 5,
				 include_target: bool = True,
				 convergent = 'post',
				 metric: str = "correlation",
				 batch_size: int = 1000,
				 dtype: torch.dtype = torch.float32,
				 folds: int = 5,
				 final_feature_mode: str = "best_fold",
				 columns = None,
				 embedDimensions: int = 0,
				 predictionHorizon: int = 1,
				 knn: int = 0,
				 step: int = -1,
				 exclusionRadius: int = 0,
				 verbose: bool = False,
				 useSMap: bool = False,
				 theta: float = 0.0,
				 stdThreshold: float = 1e-3,
				 CCMLibraryPercentiles = numpy.linspace(10, 90, 5, ),
				 CCMNumSamples: int = 10,
				 CCMConvergenceThreshold: float = 0.01,
				 CCMSeed = None,
				 CCMMaxEmbeddingDimensions: int = 15,
				 MinPredictionThreshold: float = 0.0,
				 MinCandidatePerformance: float = 0.5,
				 IterativeDimensionSearch: bool = False,
				 TimeDelay: int = 0,
				 device = None):
		"""
		:param X_train:		[nSamples, nFeatures] one series of candidate columns
		:param Y_train:		[nSamples, nTargets]
		:param folds:		contiguous folds; each fold's rows are held out and the rows before and after it train
		:param final_feature_mode:	'best_fold' takes the best fold's selection, 'frequency' the most frequent columns
		Other parameters as in MDE.
		"""
		self.X = AsRuns(X_train)[0]
		self.Y = AsRuns(Y_train)[0]
		if self.X.shape[0] != self.Y.shape[0]:
			raise ValueError(f'X_train has {self.X.shape[0]} rows but Y_train has {self.Y.shape[0]}')
		self.maxD = maxD
		self.include_target = include_target
		self.convergent = convergent
		self.metric = metric
		self.batch_size = batch_size
		self.dtype = dtype
		self.folds = folds
		self.final_feature_mode = final_feature_mode
		self.columns = columns
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.verbose = verbose
		self.useSMap = useSMap
		self.theta = theta
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

		self.fold_results: List[MDEResult] = []
		self.test_accuracy = []
		self.bestFold = None
		self.best_fold_features = None
		self.best_fold_accuracy = None

	def _MDEKeywords(self):
		return dict(maxD = self.maxD, include_target = self.include_target, convergent = self.convergent,
					metric = self.metric, batch_size = self.batch_size, dtype = self.dtype, columns = self.columns,
					embedDimensions = self.embedDimensions, predictionHorizon = self.predictionHorizon, knn = self.knn,
					step = self.step, exclusionRadius = self.exclusionRadius, verbose = self.verbose,
					useSMap = self.useSMap, theta = self.theta, stdThreshold = self.stdThreshold,
					CCMLibraryPercentiles = self.CCMLibraryPercentiles, CCMNumSamples = self.CCMNumSamples,
					CCMConvergenceThreshold = self.CCMConvergenceThreshold, CCMSeed = self.CCMSeed,
					CCMMaxEmbeddingDimensions = self.CCMMaxEmbeddingDimensions,
					MinPredictionThreshold = self.MinPredictionThreshold,
					MinCandidatePerformance = self.MinCandidatePerformance,
					IterativeDimensionSearch = self.IterativeDimensionSearch, TimeDelay = self.TimeDelay,
					device = self.device)

	def FoldBounds(self):
		"""(start, stop) of each contiguous held-out block."""
		nRows = self.X.shape[0]
		edges = numpy.linspace(0, nRows, self.folds + 1).astype(int)
		return [(int(edges[i]), int(edges[i + 1])) for i in range(self.folds)]

	def fit(self, scoring_function = Correlation) -> None:
		"""Run MDE on every fold; the rows before and after the held-out block are separate training runs."""
		self.fold_results = []
		progressBar = ProgressBar(total = self.folds, desc = 'MDE CV Fold', leave = False, disable = not self.verbose)
		for start, stop in self.FoldBounds():
			trainX = [run for run in (self.X[:start], self.X[stop:]) if run.shape[0] > 0]
			trainY = [run for run in (self.Y[:start], self.Y[stop:]) if run.shape[0] > 0]
			result = self.fitSingleFold(trainX, trainY, self.X[start:stop], self.Y[start:stop], scoring_function = scoring_function)
			self.fold_results.append(result)
			progressBar.update(1)
		progressBar.close()

		self.test_accuracy = [float(r.score[0]) for r in self.fold_results]
		self.bestFold = int(numpy.argmax(self.test_accuracy))
		self.best_fold_accuracy = self.test_accuracy[self.bestFold]
		self.best_fold_features = self.fold_results[self.bestFold].selected_variables

	def fitSingleFold(self, X_train, Y_train, X_test, Y_test, return_predictions: bool = True,
					  scoring_function = Correlation) -> MDEResult:
		mde = MDE(X_train, Y_train, X_test, Y_test, **self._MDEKeywords())
		return mde.Run(return_predictions = return_predictions, scoring_function = scoring_function)

	def SelectedVariables(self) -> numpy.ndarray:
		"""Final [nTargets, maxD] selection padded with -1, per final_feature_mode."""
		if self.final_feature_mode == 'frequency':
			return self._get_frequency_features()
		return self.best_fold_features

	def predict(self, X_test: numpy.ndarray, Y_test: Optional[numpy.ndarray] = None,
				scoring_function = Correlation) -> MDECVResult:
		"""
		Predict a separate test set from the whole training series with the final selection.

		:param X_test:	[nTest, nFeatures] with the same columns as X_train
		:param Y_test:	[nTest, nTargets]; when given the result carries a score per target
		"""
		if not self.fold_results:
			raise RuntimeError('Call fit() first')
		selected = self.SelectedVariables()
		nTargets = self.Y.shape[1]
		X_test = AsRuns(X_test)[0]
		Y_test = None if Y_test is None else AsRuns(Y_test)[0]
		allTrain = numpy.column_stack([self.X, self.Y])
		allTest = numpy.column_stack([X_test, Y_test if Y_test is not None else numpy.full((X_test.shape[0], nTargets), numpy.nan)])
		Y_pred = numpy.full((X_test.shape[0], nTargets), numpy.nan)
		scores = numpy.full(nTargets, numpy.nan)
		for j in range(nTargets):
			variables = [int(v) for v in selected[j] if v >= 0]
			if len(variables) == 0:
				continue
			common = dict(X_train = allTrain[:, variables], Y_train = self.Y[:, j], X_test = allTest[:, variables],
						  Y_test = None if Y_test is None else Y_test[:, j], embedDimensions = 1, step = self.step,
						  predictionHorizon = self.predictionHorizon, scoringFunction = scoring_function,
						  device = self.device, dtype = self.dtype)
			if self.useSMap:
				result = SMapPredict(knn = self.knn, theta = self.theta, **common)
			else:
				result = SimplexPredict(knn = len(variables) + 1, **common)
			Y_pred[:, j] = result.Y_pred
			if result.score is not None:
				scores[j] = result.score[0]

		return MDECVResult(
			Y_pred = Y_pred,
			selected_variables = selected,
			fold_results = self.fold_results,
			fold_performances = numpy.array(self.test_accuracy),
			best_fold = numpy.array(self.bestFold),
			score = scores if Y_test is not None else None)

	def _get_frequency_features(self) -> numpy.ndarray:
		"""Per target, the maxD columns selected in the most folds, [nTargets, maxD] padded with -1."""
		nTargets = self.Y.shape[1]
		selected = numpy.full((nTargets, self.maxD), -1, dtype = int)
		for j in range(nTargets):
			counts = {}
			for result in self.fold_results:
				for column in result.selected_variables[j]:
					if column >= 0:
						counts[int(column)] = counts.get(int(column), 0) + 1
			ranked = sorted(counts.items(), key = lambda item: item[1], reverse = True)[:self.maxD]
			for k, (column, _) in enumerate(ranked):
				selected[j, k] = column
		return selected

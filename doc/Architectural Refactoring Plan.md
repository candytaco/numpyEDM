# pyEDM Architectural Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to address architectural issues in the pyEDM library, specifically focusing on the API layer in src/pyEDM/API.py and its interaction with the class hierarchy.

## Current Architecture Analysis

### Module Structure

```
src/pyEDM/
├── EDM.py           # Base class with FindNeighbors, EmbedData, etc.
├── Simplex.py       # Inherits from EDM
├── SMap.py          # Inherits from EDM
├── CCM.py           # Composition: contains two Simplex instances
├── Multiview.py     # Composition: creates Simplex instances
├── API.py           # Function wrappers around classes
├── PoolFunc.py      # Worker functions for multiprocessing
├── Utils.py         # ComputeError, PlotObsPred, PlotCoeff
├── Embed.py         # Time delay embedding
└── LoadData.py      # Sample data loading
```

### Class Hierarchy

```
EDM (base)
├── Simplex (inherits EDM)
└── SMap (inherits EDM)

CCM (composition)
├── FwdMap: Simplex instance
└── RevMap: Simplex instance

Multiview (composition)
└── Creates Simplex instances dynamically
```

### Current API Pattern

Each API function follows this pattern:
1. Validate input parameters (sometimes)
2. Instantiate class with all parameters
3. Call Run() or Generate() based on flags
4. Optionally call plotting functions
5. Return either:
   - The full object (if returnObject=True)
   - A subset of results (default)
   - A dictionary of results (SMap, CCM, Multiview)

## Identified Problems

### 1. Parameter Duplication

All four core methods (Simplex, SMap, CCM, Multiview) share 15+ parameters:
- embedDimensions vs D in Multiview: D is the state-space dimension (number of variables to combine), while embedDimensions is the time-delay embedding dimension for each variable
- All parameters repeated in PoolFunc.py worker functions

Note: trainLib in Multiview is a distinct boolean flag that controls whether to use in-sample evaluation for ranking (trainLib=True) vs out-of-sample evaluation (trainLib=False). This is not a naming inconsistency with the train parameter.

### 2. Mixed Responsibilities

API.py functions handle:
- Input validation (solver checking at API.py:122-135)
- Object instantiation
- Execution control (Generate vs Run)
- Side effects (plotting)
- Return value formatting

### 3. Conditional Return Types

- Simplex: array or object (API.py:81-84)
- SMap: dict or object (API.py:169-175)
- CCM: array or dict or object (API.py:261-269)
- Multiview: dict or object (API.py:369-372)

### 4. Plotting Integration

Plotting functions from Utils.py are called directly within API functions (API.py:79, 165-167), mixing computation with presentation.

### 5. Execution Model Opacity

The choice between Generate() and Run() is buried in conditionals (API.py:73-76) rather than being explicit in the API.

### 6. Multiprocessing Complexity Leak

Low-level multiprocessing parameters (numProcess, mpMethod, chunksize) appear in high-level API (API.py:290-292, 391-393).

### 7. Parameter Validation Inconsistency

Some validation happens in API functions (solver in SMap), some in class constructors (__init__), some in Validate() methods.

## Proposed Solutions

### Phase 1: Parameter Configuration System

Create dataclasses for shared parameter groups:

```python
# src/pyEDM/Parameters.py

from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class EDMParameters:
    """Common parameters for all EDM methods"""
    data: np.ndarray
    columns: Optional[List[int]] = None
    target: Optional[int] = None
    embedDimensions: int = 0
    predictionHorizon: int = 1
    knn: int = 0
    step: int = -1
    exclusionRadius: int = 0
    embedded: bool = False
    validLib: List = None
    noTime: bool = False
    ignoreNan: bool = True
    verbose: bool = False

    def __post_init__(self):
        if self.validLib is None:
            self.validLib = []

@dataclass
class DataSplit:
    """Train/test split configuration"""
    train: Optional[Tuple[int, int]] = None
    test: Optional[Tuple[int, int]] = None

@dataclass
class GenerationParameters:
    """Parameters for iterative generation"""
    generateSteps: int = 0
    generateConcat: bool = False

@dataclass
class MultiprocessingConfig:
    """Multiprocessing execution configuration"""
    numProcess: int = 1
    mpMethod: Optional[str] = None
    chunksize: int = 1
    sequential: bool = False

@dataclass
class SMapParameters:
    """S-Map specific parameters"""
    theta: float = 0.0
    solver: Optional[object] = None

@dataclass
class CCMParameters:
    """CCM specific parameters"""
    trainSizes: List[int] = None
    sample: int = 0
    seed: Optional[int] = None
    includeData: bool = False

    def __post_init__(self):
        if self.trainSizes is None:
            self.trainSizes = []

@dataclass
class MultiviewParameters:
    """Multiview specific parameters"""
    D: int = 0  # State-space dimension (number of variables to combine)
    multiview: int = 0  # Number of top-ranked ensembles (k in paper)
    trainLib: bool = True  # Use in-sample evaluation for ranking
    excludeTarget: bool = False  # Exclude target from combinations

    def __post_init__(self):
        """
        trainLib behavior:
        - True (default): Use in-sample evaluation for ranking top combinations
                         (sets test=train during ranking phase). This is faster
                         but may overfit to arbitrary non-constant vectors.
        - False: Use proper out-of-sample evaluation with specified train/test
                splits for ranking. Requires explicit train and test parameters.
        """
        pass
```

Benefits:
- Single source of truth for parameter definitions
- Type hints and validation in one place
- Easy to extend without changing function signatures
- Can add validation in __post_init__

### Phase 2: Result Objects

Replace conditional returns with structured result classes:

```python
# src/pyEDM/Results.py

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SimplexResult:
    """Results from Simplex prediction"""
    projection: np.ndarray  # [Time, Observations, Predictions]
    embedDimensions: int
    predictionHorizon: int

    @property
    def time(self):
        return self.projection[:, 0]

    @property
    def observations(self):
        return self.projection[:, 1]

    @property
    def predictions(self):
        return self.projection[:, 2]

    def compute_error(self):
        from .Utils import ComputeError
        return ComputeError(self.observations, self.predictions)

@dataclass(frozen=True)
class SMapResult:
    """Results from S-Map prediction"""
    projection: np.ndarray
    coefficients: np.ndarray
    singularValues: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    theta: float

    @property
    def predictions(self):
        return SimplexResult(
            self.projection,
            self.embedDimensions,
            self.predictionHorizon
        )

    def compute_error(self):
        from .Utils import ComputeError
        return ComputeError(
            self.projection[:, 1],
            self.projection[:, 2]
        )

@dataclass(frozen=True)
class CCMResult:
    """Results from Convergent Cross Mapping"""
    libMeans: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    predictStats1: Optional[np.ndarray] = None
    predictStats2: Optional[np.ndarray] = None

@dataclass(frozen=True)
class MultiviewResult:
    """Results from Multiview prediction"""
    projection: np.ndarray  # Averaged prediction
    view: List  # Rankings of column combinations
    topRankProjections: dict
    topRankStats: dict
    D: int
    embedDimensions: int
    predictionHorizon: int
```

Benefits:
- Consistent return types
- Self-documenting (user knows what fields are available)
- Can add convenience methods
- No more returnObject flag needed

### Phase 3: Visualization Module

Separate plotting from computation:

```python
# src/pyEDM/Visualization.py

import matplotlib.pyplot as plt
from matplotlib.pyplot import show, axhline

def plot_prediction(result, title=""):
    """Plot observations vs predictions from any result object"""
    from .Utils import PlotObsPred

    if hasattr(result, 'projection'):
        PlotObsPred(
            result.projection,
            title,
            result.embedDimensions,
            result.predictionHorizon
        )
    else:
        raise ValueError("Result object has no projection to plot")

def plot_smap_coefficients(result, title=""):
    """Plot S-Map coefficients"""
    from .Utils import PlotCoeff

    if not isinstance(result, SMapResult):
        raise ValueError("plot_smap_coefficients requires SMapResult")

    PlotCoeff(
        result.coefficients,
        title,
        result.embedDimensions,
        result.predictionHorizon
    )

def plot_ccm(result, title=""):
    """Plot CCM convergence"""
    if not isinstance(result, CCMResult):
        raise ValueError("plot_ccm requires CCMResult")

    fig, ax = plt.subplots()
    title_str = f'E = {result.embedDimensions}'

    if result.libMeans.shape[1] == 3:
        ax.plot(result.libMeans[:, 0], result.libMeans[:, 1],
                linewidth=3, label='Col 1')
        ax.plot(result.libMeans[:, 0], result.libMeans[:, 2],
                linewidth=3, label='Col 2')
        ax.legend()
    elif result.libMeans.shape[1] == 2:
        ax.plot(result.libMeans[:, 0], result.libMeans[:, 1],
                linewidth=3)

    ax.set(xlabel="Library Size", ylabel="CCM correlation", title=title_str)
    axhline(y=0, linewidth=1)
    show()
```

API changes:
- Remove showPlot parameter from all API functions
- Users explicitly call plotting functions on results
- Clean separation of concerns

### Phase 4: Improved Class Execution Interface

Make execution methods explicit and improve class API:

```python
# In Simplex, SMap classes

class Simplex(EDMClass):
    def run(self):
        """Execute standard prediction"""
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighbors()
        self.Project()
        self.FormatProjection()
        return SimplexResult(
            projection=self.Projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    def generate(self):
        """Execute iterative generation"""
        self.Generate()
        return SimplexResult(
            projection=self.Projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )
```

Benefits:
- Explicit method calls
- Classes return result objects directly
- No conditional logic needed in API wrappers
- Classes become more usable on their own

### Phase 5: Execution Strategy Pattern

Abstract multiprocessing configuration with an enumeration-based interface:

```python
# src/pyEDM/Execution.py

from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import get_context
from typing import Callable, Iterable, Optional
import os

class ExecutionMode(Enum):
    """Enumeration of execution strategies"""
    SEQUENTIAL = "sequential"
    MULTIPROCESS = "multiprocess"
    SPAWN = "spawn"
    FORK = "fork"
    FORKSERVER = "forkserver"

class ExecutionStrategy(ABC):
    @abstractmethod
    def map(self, func: Callable, iterable: Iterable):
        """Execute function over iterable"""
        pass

class SequentialExecution(ExecutionStrategy):
    """Sequential execution (no parallelism)"""
    def map(self, func, iterable):
        return [func(*args) for args in iterable]

class MultiprocessExecution(ExecutionStrategy):
    """Multiprocessing execution with configurable context"""
    def __init__(self, numProcess: int = None, mpMethod: str = None, chunksize: int = 1):
        self.numProcess = numProcess or os.cpu_count()
        self.mpMethod = mpMethod
        self.chunksize = chunksize

    def map(self, func, iterable):
        mpContext = get_context(self.mpMethod)
        with mpContext.Pool(processes=self.numProcess) as pool:
            return pool.starmap(func, iterable, chunksize=self.chunksize)

def create_executor(
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    numProcess: Optional[int] = None,
    chunksize: int = 1
) -> ExecutionStrategy:
    """Factory function to create execution strategy from enum"""
    if mode == ExecutionMode.SEQUENTIAL:
        return SequentialExecution()
    elif mode == ExecutionMode.MULTIPROCESS:
        return MultiprocessExecution(numProcess=numProcess, chunksize=chunksize)
    elif mode in (ExecutionMode.SPAWN, ExecutionMode.FORK, ExecutionMode.FORKSERVER):
        return MultiprocessExecution(
            numProcess=numProcess,
            mpMethod=mode.value,
            chunksize=chunksize
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")

# Usage in API functions
def EmbedDimension(
    ...,
    execution: ExecutionMode = ExecutionMode.MULTIPROCESS,
    numProcess: int = 4,
    chunksize: int = 1
):
    """
    Parameters:
    -----------
    execution : ExecutionMode
        Execution strategy (SEQUENTIAL, MULTIPROCESS, SPAWN, FORK, FORKSERVER)
    numProcess : int
        Number of processes for parallel execution (ignored for SEQUENTIAL)
    chunksize : int
        Chunk size for parallel execution (ignored for SEQUENTIAL)
    """
    executor = create_executor(execution, numProcess, chunksize)

    # ... setup ...
    correlationList = executor.map(PoolFunc.EmbedDimSimplexFunc, poolArgs)
    # ...

# Alternative: Direct ExecutionStrategy parameter for advanced users
def EmbedDimension_Advanced(
    ...,
    executor: ExecutionStrategy = None
):
    """Advanced API allowing custom execution strategies"""
    if executor is None:
        executor = SequentialExecution()

    # ... setup ...
    correlationList = executor.map(PoolFunc.EmbedDimSimplexFunc, poolArgs)
    # ...
```

Benefits:
- Clean, user-friendly enumeration interface
- Type-safe execution mode selection
- Hides multiprocessing complexity from most users
- Easy to add new execution strategies (e.g., Dask, Ray, ThreadPool)
- Advanced users can still provide custom ExecutionStrategy instances
- Self-documenting API (IDE autocomplete shows available modes)
- No need to remember string values for mpMethod

### Phase 6: Refactored API Architecture

Eliminate wrapper functions and namespace conflicts by improving class interfaces directly and organizing into clean module structure.

#### Improved Class Constructors

Refactor classes to accept configuration objects instead of individual parameters:

```python
# src/pyEDM/models/simplex.py

from ..config import EDMParameters, DataSplit, GenerationParameters
from ..results import SimplexResult
from .edm import EDM

class Simplex(EDM):
    """Simplex prediction model"""

    def __init__(self,
                 params: EDMParameters,
                 split: DataSplit = None,
                 generation: GenerationParameters = None):
        """Initialize Simplex predictor

        Parameters
        ----------
        params : EDMParameters
            Core EDM parameters (data, embedDimensions, target, etc.)
        split : DataSplit, optional
            Train/test split configuration. If None, uses full dataset.
        generation : GenerationParameters, optional
            Iterative generation settings. If None, uses standard prediction.
        """
        super().__init__(params.data, isEmbedded=False, name='Simplex')

        if split is None:
            split = DataSplit()
        if generation is None:
            generation = GenerationParameters()

        # Unpack configuration objects
        self.columns = params.columns
        self.target = params.target
        self.train = split.train
        self.test = split.test
        self.embedDimensions = params.embedDimensions
        self.predictionHorizon = params.predictionHorizon
        self.knn = params.knn
        self.step = params.step
        self.exclusionRadius = params.exclusionRadius
        self.embedded = params.embedded
        self.validLib = params.validLib
        self.noTime = params.noTime
        self.ignoreNan = params.ignoreNan
        self.verbose = params.verbose
        self.generateSteps = generation.generateSteps
        self.generateConcat = generation.generateConcat

        # Map API parameter names to EDM base class names
        self.predictionHorizon = params.predictionHorizon
        self.embedStep = params.step
        self.isEmbedded = params.embedded

        self.Validate()
        self.CreateIndices()

        self.targetVec = self.Data[:, [self.target[0]]]
        if self.noTime:
            timeIndex = [i for i in range(1, self.Data.shape[0] + 1)]
            self.time = array(timeIndex, dtype=int)
        else:
            self.time = self.Data[:, 0]

    def run(self) -> SimplexResult:
        """Execute standard prediction

        Returns
        -------
        SimplexResult
            Prediction results with projection array and metadata
        """
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighbors()
        self.Project()
        self.FormatProjection()

        return SimplexResult(
            projection=self.Projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    def generate(self) -> SimplexResult:
        """Execute iterative generation

        Returns
        -------
        SimplexResult
            Generation results with projection array and metadata
        """
        self.Generate()

        return SimplexResult(
            projection=self.Projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )
```

#### Module Organization

```python
# src/pyEDM/models/__init__.py
"""EDM prediction models"""
from .simplex import Simplex
from .smap import SMap
from .ccm import CCM
from .multiview import Multiview

__all__ = ['Simplex', 'SMap', 'CCM', 'Multiview']

# src/pyEDM/__init__.py
"""pyEDM - Empirical Dynamic Modeling"""

# Import models
from .models import Simplex, SMap, CCM, Multiview

# Import configuration objects
from .config import (
    EDMParameters,
    DataSplit,
    GenerationParameters,
    SMapParameters,
    CCMParameters,
    MultiviewParameters
)

# Import result objects
from .results import (
    SimplexResult,
    SMapResult,
    CCMResult,
    MultiviewResult
)

# Import visualization functions
from .visualization import (
    plot_prediction,
    plot_smap_coefficients,
    plot_ccm
)

# Import execution configuration
from .execution import ExecutionMode

# Import utilities (maintain backward compatibility)
from .utils import ComputeError, SurrogateData
from .embed import Embed
from .loaddata import sampleData

__version__ = "3.0.0"
__all__ = [
    # Models
    'Simplex', 'SMap', 'CCM', 'Multiview',
    # Configuration
    'EDMParameters', 'DataSplit', 'GenerationParameters',
    'SMapParameters', 'CCMParameters', 'MultiviewParameters',
    # Results
    'SimplexResult', 'SMapResult', 'CCMResult', 'MultiviewResult',
    # Visualization
    'plot_prediction', 'plot_smap_coefficients', 'plot_ccm',
    # Execution
    'ExecutionMode',
    # Utilities
    'ComputeError', 'SurrogateData', 'Embed', 'sampleData'
]
```

#### User Interface

Clean, intuitive API with no namespace conflicts:

```python
# Simple usage
from pyEDM import FitSimplex, EDMParameters, DataSplit

params = EDMParameters(
   data = my_data,
   embedDimensions = 3,
   target = 1,
   predictionHorizon = 1
)

split = DataSplit(train = [1, 100], test = [101, 200])

# Create and run model
model = FitSimplex(params, split)
result = model.run()

# Access results
print(f"Correlation: {result.compute_error()['correlation']}")

# Plot if desired
from pyEDM import plot_prediction

plot_prediction(result)

# ---- More compact usage ----
result = FitSimplex(params, split).run()

# ---- SMap example ----
from pyEDM import FitSMap, SMapParameters

smap_params = SMapParameters(theta = 2.0)
smap_result = FitSMap(params, split, smap_params).run()

# Access SMap-specific results
coefficients = smap_result.coefficients
plot_smap_coefficients(smap_result)

# ---- CCM example ----
from pyEDM import FitCCM, CCMParameters

ccm_params = CCMParameters(trainSizes = [10, 50, 10], sample = 100)
ccm_model = FitCCM(params, ccm_params)
ccm_result = ccm_model.run()
plot_ccm(ccm_result)

# ---- Multiview example ----
from pyEDM import FitMultiview, MultiviewParameters

mv_params = MultiviewParameters(D = 3, trainLib = False)
mv_result = FitMultiview(params, split, mv_params).run()
print(mv_result.view)  # Rankings

# ---- With execution control ----
from pyEDM import ExecutionMode

params_with_execution = EDMParameters(
   data = my_data,
   embedDimensions = 3,
   execution = ExecutionMode.MULTIPROCESS,
   numProcess = 8
)

result = FitSimplex(params_with_execution, split).run()
```

#### Benefits

- **No namespace conflicts**: Classes use their natural names (Simplex, not SimplexClass)
- **No wrapper functions needed**: Classes are the API
- **Clean imports**: Clear module organization
- **Type-safe**: Configuration objects provide type hints and validation
- **Discoverable**: IDE autocomplete works naturally
- **Flexible**: Users can reuse configuration objects across multiple runs
- **Testable**: Each component can be tested independently
- **Maintainable**: Single responsibility for each class

## Migration Strategy

### Documentation Updates

- Update all examples to use new interface
- Provide migration guide showing old vs new
- Document the rationale for changes

## Implementation Order

1. Create Config.py with parameter dataclasses
2. Create Results.py with result objects
3. Create Visualization.py and move plotting
4. Update class run()/generate() to return result objects
5. Create Execution.py with strategy pattern
6. Refactor API.py to use new components
7. Update PoolFunc.py to work with new structure
9. Update tests
10. Update documentation and examples

## Benefits Summary

After refactoring:

- Single source of truth for parameter definitions
- Consistent, predictable return types
- Clear separation of computation and visualization
- Explicit execution model
- Hidden multiprocessing complexity
- More maintainable codebase
- Easier to test individual components
- Better user experience (both simple and advanced use cases)
- Cleaner API surface

## Open Questions

1. Should we keep API wrapper functions or expose classes directly?
2. How aggressive should we be with breaking changes?
3. Do we want to support method chaining on result objects?
4. Should we provide builder pattern for parameter objects?

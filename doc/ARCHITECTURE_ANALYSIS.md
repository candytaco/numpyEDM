# numpyEDM Software Architecture Analysis

## Overview
numpyEDM is a Python package for Empirical Dynamic Modeling (EDM) that implements various time series analysis techniques including Simplex projection, S-Map, Convergent Cross Mapping (CCM), and Multiview embedding.

## Package Structure

```
src/pyEDM/
├── __init__.py                # Package initialization
├── API.py                     # Main API functions
├── AuxFunc.py                 # Auxiliary functions
├── CCM.py                     # Convergent Cross Mapping
├── EDM.py                     # Base EDM class
├── Formatting.py              # Data formatting functions
├── LoadData.py                # Sample data loading
├── Multiview.py               # Multiview embedding
├── Neighbors.py               # Neighbor finding algorithms
├── PoolFunc.py                # Parallel processing functions
├── Simplex.py                 # Simplex projection
├── SMap.py                    # S-Map implementation
└── data/                      # Sample datasets
    ├── TentMap.csv            # Tent map time series
    ├── circle.csv             # Circle map data
    └── ... (other sample data)
```

## Input Formats

### 1. Data Input Format
- **Primary Format**: CSV files with time series data
- **Structure**: Pandas DataFrame with columns:
  - First column: Time/index (can be numeric, datetime, or string)
  - Subsequent columns: Time series variables
- **Example** (from TentMap.csv):
  ```
  Time,TentMap
  1,-0.0992
  2,-0.6013
  3,0.7998
  ...
  ```

### 2. API Function Parameters
All main functions accept these core parameters:
- `dataFrame`: Pandas DataFrame containing the time series data
- `columns`: String or list of column names to use as predictors
- `target`: String or list of column names to predict
- `lib`: Library indices (start:stop pairs)
- `pred`: Prediction indices (start:stop pairs)
- `E`: Embedding dimension
- `Tp`: Prediction interval (time steps ahead)
- `tau`: Time delay for embedding
- `knn`: Number of nearest neighbors
- `embedded`: Boolean flag for pre-embedded data
- `verbose`: Boolean for debug output
- `returnObject`: Boolean to return full object or just results

## Internal Representations

### 1. Core Data Structures
- **DataFrame**: Primary container for input/output data (Pandas DataFrame)
- **Embedding**: Time-delay embedded data with shifted columns
  - Example: For E=3, tau=-1, columns [x], embedding creates [x(t), x(t-1), x(t-2)]
- **Indices**:
  - `lib_i`: Array of library indices
  - `pred_i`: Array of prediction indices
  - `pred_i_all`: Array including NaN positions
  - `predList`: List of disjoint prediction segments

### 2. Neighbor Search
- **KDTree**: SciPy KDTree for efficient nearest neighbor search
- **knn_neighbors**: Array of neighbor indices for each prediction point
- **knn_distances**: Array of distances to neighbors

### 3. Projection Results
- **projection**: Array of predicted values
- **variance**: Array of prediction variance estimates
- **coefficients**: S-Map regression coefficients (for SMap)
- **singularValues**: SVD singular values (for SMap)

### 4. Time Handling
- **time**: Numpy array of time values (can be numeric, datetime, or string)
- Time conversion utilities handle various formats including ISO 8601 datetime strings

## Output Formats

### 1. Simplex Output
Returns a DataFrame with columns:
- `Time`: Time values aligned with predictions
- `Observations`: Actual observed values
- `Predictions`: Predicted values
- `Pred_Variance`: Prediction variance

### 2. S-Map Output
Returns a dictionary with:
- `predictions`: DataFrame with Time, Observations, Predictions, Pred_Variance
- `coefficients`: DataFrame with regression coefficients
- `singularValues`: DataFrame with SVD singular values

### 3. CCM Output
Returns a DataFrame with:
- `LibSize`: Library sizes tested
- `<column>:<target>`: Correlation values for forward mapping
- `<target>:<column>`: Correlation values for reverse mapping

### 4. Multiview Output
Returns a dictionary with:
- `Predictions`: DataFrame with multiview-averaged predictions
- `View`: DataFrame showing performance of different column combinations

## Core Classes and Inheritance

```
EDM (Base Class)
├── Simplex (inherits from EDM)
│   ├── CCM (contains two Simplex instances)
│   └── Multiview (uses Simplex)
└── SMap (inherits from EDM)
```

### EDM Class (src/pyEDM/EDM.py)
- Base class containing common functionality
- Manages data, indices, neighbors, and projections
- Key methods:
  - `Validate()`: Parameter validation
  - `CreateIndices()`: Generate library/prediction indices
  - `EmbedData()`: Create time-delay embeddings
  - `RemoveNan()`: Handle NaN values
  - `FindNeighbors()`: Find nearest neighbors (from Neighbors.py)
  - `FormatProjection()`: Format results into DataFrame (from Formatting.py)

### Simplex Class (src/pyEDM/Simplex.py)
- Implements Simplex projection algorithm
- Key methods:
  - `Project()`: Compute weighted average predictions
  - `Generate()`: Forecast future values iteratively

### SMap Class (src/pyEDM/SMap.py)
- Implements Sequential Locally Weighted Global Linear Maps
- Key methods:
  - `Project()`: Compute linear regression with localization
  - `Solver()`: Handle different regression solvers
  - `Generate()`: Forecast future values iteratively

### CCM Class (src/pyEDM/CCM.py)
- Implements Convergent Cross Mapping
- Contains two Simplex instances (forward and reverse mapping)
- Key methods:
  - `Project()`: Run cross-mapping in both directions
  - `CrossMap()`: Perform cross-mapping for a given direction

### Multiview Class (src/pyEDM/Multiview.py)
- Implements Multiview embedding for high-dimensional data
- Uses Simplex for predictions
- Key methods:
  - `Rank()`: Rank column combinations by prediction skill
  - `Project()`: Compute multiview-averaged predictions

## Key Algorithms

### 1. Time-Delay Embedding
- Creates lagged copies of time series: X(t), X(t-τ), X(t-2τ), ...
- Implemented in `API.Embed()` function
- Handles both positive and negative tau values

### 2. Nearest Neighbor Search
- Uses SciPy KDTree for efficient search
- Implemented in `Neighbors.FindNeighbors()`
- Returns sorted neighbor indices and distances

### 3. Simplex Projection
- Weighted average of neighbors' future values
- Weights: exp(-distance/min_distance)
- Implemented in `Simplex.Project()`

### 4. S-Map Localization
- Local linear regression with theta parameter
- Controls localization strength (theta=0: global, theta>0: local)
- Implemented in `SMap.Project()`

### 5. Convergent Cross Mapping
- Forward: X → Y mapping
- Reverse: Y → X mapping
- Correlation increases with library size if causal relationship exists
- Implemented in `CCM.CrossMap()`

## Data Flow

1. **Input**: Pandas DataFrame with time series data
2. **Validation**: Check parameters and data structure
3. **Index Creation**: Generate library and prediction indices
4. **Embedding**: Create time-delay embedded vectors (if not pre-embedded)
5. **Neighbor Search**: Find nearest neighbors for each prediction point
6. **Projection**: Compute predictions using chosen algorithm
7. **Formatting**: Format results into standardized DataFrame
8. **Output**: Return predictions and optionally coefficients/variance

## Parallel Processing

- `PoolFunc.py` contains functions for parallel execution
- Uses Python multiprocessing Pool
- Supports:
  - Embedding dimension optimization
  - Prediction interval optimization
  - S-Map theta optimization
  - CCM library size sampling

## Error Handling

- Comprehensive validation in `Validate()` methods
- NaN handling with `ignoreNan` parameter
- Boundary condition checks for time series edges
- Input format validation for time values

## Key Features

1. **Flexible Time Handling**: Supports numeric, datetime, and string time formats
2. **NaN Support**: Optional NaN removal from library and predictions
3. **Disjoint Libraries**: Support for non-contiguous library segments
4. **Multiple Solvers**: S-Map supports various regression solvers
5. **Forecasting**: Generate future predictions iteratively
6. **Visualization**: Built-in plotting capabilities
7. **Sample Data**: Included example datasets for testing

## Dependencies

- Python standard library: warnings, datetime, multiprocessing, itertools
- NumPy: Array operations, linear algebra
- Pandas: DataFrame manipulation
- SciPy: KDTree for neighbor search

## Usage Example

```python
import pyEDM
import pandas as pd

# Load data
data = pd.read_csv('data/TentMap.csv')

# Simplex prediction
result = pyEDM.API.Simplex(
	dataFrame = data,
	columns = 'TentMap',
	target = 'TentMap',
	lib = '1 100',
	pred = '101 200',
	E = 3,
	Tp = 1,
	tau = -1
)

# S-Map prediction
result = pyEDM.API.SMap(
	dataFrame = data,
	columns = 'TentMap',
	target = 'TentMap',
	lib = '1 100',
	pred = '101 200',
	E = 3,
	Tp = 1,
	tau = -1,
	theta = 0.5
)

# CCM analysis
result = pyEDM.API.CCM(
	dataFrame = data,
	columns = 'TentMap',
	target = 'TentMap',
	trainSizes = '10 100 10',
	sample = 10
)
```

## Architecture Summary

numpyEDM follows an object-oriented design with a clear inheritance hierarchy. The EDM base class provides core functionality that is extended by specialized classes (Simplex, SMap, CCM, Multiview). The architecture emphasizes:

1. **Modularity**: Separate concerns into different modules
2. **Reusability**: Common functionality in base classes
3. **Extensibility**: Easy to add new algorithms
4. **Flexibility**: Support for various input formats and parameters
5. **Robustness**: Comprehensive validation and error handling

The package is designed for time series analysis in ecological and dynamical systems research, with a focus on empirical dynamic modeling techniques.

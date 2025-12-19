# NumPy Migration Plan: Removing Pandas DataFrame Dependencies

## Current State Analysis

The codebase is already in transition from pandas-centric to numpy-centric architecture. Recent commits show [Embed.py](src/pyEDM/Embed.py) has been refactored to accept and return numpy arrays. However, pandas DataFrames are still required at API entry points and used for output formatting.

**Critical Dependencies on Pandas:**
- API functions require `dataFrame` parameter (pandas DataFrame)
- EDM base class validates `isinstance(self.Data, DataFrame)` at [EDM.py:273](src/pyEDM/EDM.py#L273)
- Output formatting in [Formatting.py](src/pyEDM/Formatting.py) constructs DataFrames for all results
- Sample data loading in [LoadData.py](src/pyEDM/LoadData.py) returns DataFrames
- DataFrame operations: `.iloc[]`, `.loc[]`, `.to_numpy()`, `.columns`, `.shape`, `concat()`, `.join()`

## Migration Strategy

Replace pandas DataFrames with pure numpy arrays using integer column indexing. Column names are removed entirely - users will reference columns by integer index only. This simplifies the codebase and eliminates all pandas dependencies.

### Core Design Decisions

**Input Format:** Accept only 2D numpy arrays
- 2D numpy array as sole data structure (n_samples, n_features)
- First column is always time (convention)
- Other columns referenced by integer index
- No column names, no metadata dictionaries

**Output Format:** Return simple numpy arrays
- 2D numpy arrays for outputs
- No metadata dictionaries for column tracking
- Users access columns by integer index

**Column Access:** Integer indexing only
- Replace `df.loc[:, 'col_name']` with `data[:, col_idx]`
- Replace `df.iloc[:, 0]` with `data[:, 0]`
- All column parameters become integer indices or lists of integers

## Implementation Plan

### Phase 1: EDM Base Class Refactoring

**Task 1.1: Modify EDM Data Storage and Validation**
File: [src/pyEDM/EDM.py](src/pyEDM/EDM.py)

Current behavior:
- Line ~60: Constructor accepts `dataFrame` parameter
- Line 273: Validates `isinstance(self.Data, DataFrame)`
- Stores `self.Data` as pandas DataFrame
- Accepts `columns` parameter as comma-separated string of column names

New behavior:
- Constructor accepts `data` parameter (2D numpy array)
- Validate `isinstance(self.Data, np.ndarray)` and `self.Data.ndim == 2`
- Store `self.Data` as 2D numpy array
- `columns` parameter becomes list of integer indices
- First column (index 0) is always time by convention

**Task 1.2: Update EDM Constructor**
File: [src/pyEDM/EDM.py](src/pyEDM/EDM.py)

Change signature and initialization:
```python
# Before
def __init__(self, dataFrame, lib, pred, E, Tp, knn, tau, exclusionRadius,
             columns, target, embedded, validLib, generateSteps, ...)

# After
def __init__(self, data, lib, pred, E, Tp, knn, tau, exclusionRadius,
             columns, target, embedded, validLib, generateSteps, ...)
```

Where:
- `data`: 2D numpy array (n_samples, n_features), first column is time
- `columns`: list of integer indices or None

**Task 1.3: Replace DataFrame Column Access Throughout EDM**
File: [src/pyEDM/EDM.py](src/pyEDM/EDM.py)

Replace all DataFrame operations:
- `self.Data.iloc[:, idx]` → `self.Data[:, idx]`
- `self.Data.loc[:, col_name]` → `self.Data[:, col_idx]`
- `self.Data[col_names]` → `self.Data[:, col_indices]`
- `self.Data.to_numpy()` → `self.Data`
- `self.Data.shape` → `self.Data.shape`
- Remove any usage of `self.Data.columns`

**Task 1.4: Update Column Parameter Parsing**
File: [src/pyEDM/EDM.py](src/pyEDM/EDM.py)

Current: Accepts comma-separated string like `"x,y,z"` and validates against DataFrame column names

New: Accept list of integers like `[1, 2, 3]` and validate against array shape
```python
if columns is None:
    columns = list(range(1, self.Data.shape[1]))  # All except time column
elif isinstance(columns, int):
    columns = [columns]
else:
    # Validate all are integers within bounds
    assert all(isinstance(c, int) for c in columns)
    assert all(0 <= c < self.Data.shape[1] for c in columns)
```

**Task 1.5: Update Target Column Handling**
File: [src/pyEDM/EDM.py](src/pyEDM/EDM.py)

Current: `target` is string column name

New: `target` is integer column index
```python
if target is None:
    target = columns[0] if columns else 1
assert isinstance(target, int) and 0 <= target < self.Data.shape[1]
```

**Task 1.6: Remove Pandas Imports from EDM**
File: [src/pyEDM/EDM.py](src/pyEDM/EDM.py)

Remove: `from pandas import DataFrame`

### Phase 2: Output Formatting Refactoring

**Task 2.1: Update FormatProjection Function**
File: [src/pyEDM/Formatting.py](src/pyEDM/Formatting.py)

Current: Returns pandas DataFrame with columns [Time, Observations, Predictions, Pred_Variance]

New: Return 2D numpy array
```python
# Shape: (n_samples, 4)
# Column 0: Time
# Column 1: Observations
# Column 2: Predictions
# Column 3: Pred_Variance
result = np.column_stack([time_array, obs_array, pred_array, var_array])
return result
```

**Task 2.2: Update SMap Coefficient Formatting**
File: [src/pyEDM/Formatting.py](src/pyEDM/Formatting.py)

Current: Creates DataFrame for coefficients with column names

New: Return 2D numpy array
- Coefficients: shape (n_predictions, n_coefficients)
- Singular values: shape (n_predictions, n_singular_values)

**Task 2.3: Update Time Handling Functions**
File: [src/pyEDM/Formatting.py](src/pyEDM/Formatting.py)

- `ConvertTime()`: Remove DataFrame logic, work with 1D numpy arrays
- `AddTime()`: Use numpy operations only (no DataFrame concat)

**Task 2.4: Remove Pandas Imports from Formatting**
File: [src/pyEDM/Formatting.py](src/pyEDM/Formatting.py)

Remove: `from pandas import DataFrame`

### Phase 3: Simplex Class Refactoring

**Task 3.1: Update Simplex Constructor**
File: [src/pyEDM/Simplex.py](src/pyEDM/Simplex.py)

Change to accept numpy array instead of DataFrame:
```python
# Before
def __init__(self, dataFrame, ...)

# After
def __init__(self, data, ...)
```

Pass to parent EDM class: `super().__init__(data, ...)`

**Task 3.2: Replace DataFrame Operations in Simplex**
File: [src/pyEDM/Simplex.py](src/pyEDM/Simplex.py)

Search for and replace all DataFrame-specific operations:
- Column access via `.loc[]` or `.iloc[]` → integer indexing
- Any DataFrame concatenation → `np.column_stack()` or `np.concatenate()`

**Task 3.3: Update Simplex Return Values**
File: [src/pyEDM/Simplex.py](src/pyEDM/Simplex.py)

Ensure all methods return numpy arrays (via updated FormatProjection).

**Task 3.4: Remove Pandas Imports from Simplex**
File: [src/pyEDM/Simplex.py](src/pyEDM/Simplex.py)

Remove: `from pandas import DataFrame, Series, concat`

### Phase 4: SMap Class Refactoring

**Task 4.1: Update SMap Constructor**
File: [src/pyEDM/SMap.py](src/pyEDM/SMap.py)

Change to accept numpy array:
```python
# Before
def __init__(self, dataFrame, ...)

# After
def __init__(self, data, ...)
```

**Task 4.2: Replace DataFrame Operations in SMap**
File: [src/pyEDM/SMap.py](src/pyEDM/SMap.py)

Replace DataFrame-specific operations with numpy operations.

**Task 4.3: Update SMap Return Values**
File: [src/pyEDM/SMap.py](src/pyEDM/SMap.py)

Return numpy arrays for predictions, coefficients, and singular values.

**Task 4.4: Remove Pandas Imports from SMap**
File: [src/pyEDM/SMap.py](src/pyEDM/SMap.py)

Remove: `from pandas import DataFrame, Series, concat`

### Phase 5: CCM Class Refactoring

**Task 5.1: Update CCM Constructor**
File: [src/pyEDM/CCM.py](src/pyEDM/CCM.py)

CCM is standalone (doesn't inherit from EDM) but creates two Simplex instances:
```python
# Before
def __init__(self, dataFrame, ...)

# After
def __init__(self, data, ...)
```

**Task 5.2: Update CCM Internal Data Handling**
File: [src/pyEDM/CCM.py](src/pyEDM/CCM.py)

- Store data as numpy array
- Pass numpy arrays to Simplex instances
- Column references use integer indices

**Task 5.3: Update CCM Output Format**
File: [src/pyEDM/CCM.py](src/pyEDM/CCM.py)

Current: Returns DataFrame with library size sweep results

New: Return 2D numpy array
```python
# Shape: (n_library_sizes, n_metrics)
# Column 0: LibSize
# Column 1: rho
# ... other metrics
```

**Task 5.4: Replace DataFrame Concatenation**
File: [src/pyEDM/CCM.py](src/pyEDM/CCM.py)

Replace `concat()` calls with `np.concatenate()` or `np.column_stack()`.

**Task 5.5: Remove Pandas Imports from CCM**
File: [src/pyEDM/CCM.py](src/pyEDM/CCM.py)

Remove: `from pandas import DataFrame, concat`

### Phase 6: Multiview Class Refactoring

**Task 6.1: Update Multiview Constructor**
File: [src/pyEDM/Multiview.py](src/pyEDM/Multiview.py)

Change to accept numpy array:
```python
# Before
def __init__(self, dataFrame, ...)

# After
def __init__(self, data, ...)
```

**Task 6.2: Update Column Combination Generation**
File: [src/pyEDM/Multiview.py](src/pyEDM/Multiview.py)

Generate combinations using integer column indices instead of column names.

**Task 6.3: Update Multiview Output Format**
File: [src/pyEDM/Multiview.py](src/pyEDM/Multiview.py)

Current: Returns DataFrame with column combination rankings

New: Return tuple of numpy arrays or single 2D array:
- Option 1: Return (combinations_array, rankings_array)
  - `combinations_array`: shape (n_combos, max_combo_size), integer indices
  - `rankings_array`: shape (n_combos, n_metrics)
- Option 2: Return single array with encoded combinations

**Task 6.4: Remove Pandas Imports from Multiview**
File: [src/pyEDM/Multiview.py](src/pyEDM/Multiview.py)

Remove any pandas imports.

### Phase 7: Helper Functions and Utilities

**Task 7.1: Verify Embed Function**
File: [src/pyEDM/Embed.py](src/pyEDM/Embed.py)

Recent commits already refactored this to numpy. Verify:
- Input: numpy array
- `columns` parameter: list of integer indices
- Output: numpy array
- No DataFrame dependencies

**Task 7.2: Update Auxiliary Functions**
File: [src/pyEDM/AuxFunc.py](src/pyEDM/AuxFunc.py)

- `ComputeError()`: Verify works with numpy arrays
- `SurrogateData()`: Verify numpy array handling
- `PlotObsPred()`: Update to accept numpy array, extract columns by index
- `PlotCoeff()`: Update to accept numpy array

**Task 7.3: Update LoadData Module**
File: [src/pyEDM/LoadData.py](src/pyEDM/LoadData.py)

Option 1: Use pandas for CSV reading, convert to numpy
```python
import pandas as pd
def load_data(name):
    df = pd.read_csv(...)
    return df.to_numpy()
```

Option 2: Use numpy directly
```python
def load_data(name):
    return np.genfromtxt(..., delimiter=',', skip_header=1)
```

Document convention: Column 0 is always time.

**Task 7.4: Update Pool Functions**
File: [src/pyEDM/PoolFunc.py](src/pyEDM/PoolFunc.py)

Update all parallel processing wrapper functions to accept numpy arrays:
- `EmbedDimSimplexFunc()`
- `PredictIntervalSimplexFunc()`
- `PredictNLSMapFunc()`
- `MultiviewSimplexRho()`
- `MultiviewSimplexPred()`

### Phase 8: API Layer Refactoring

**Task 8.1: Update API Function Signatures**
File: [src/pyEDM/API.py](src/pyEDM/API.py)

Replace `dataFrame` parameter with `data` in all functions:
```python
# Before
def Simplex(dataFrame, lib="", pred="", E=0, Tp=0, knn=0, tau=-1,
            exclusionRadius=0, columns="", target="", ...)

# After
def Simplex(data, lib="", pred="", E=0, Tp=0, knn=0, tau=-1,
            exclusionRadius=0, columns=None, target=None, ...)
```

Key changes:
- `data`: 2D numpy array, first column is time
- `columns`: list of integer indices or None (defaults to all except time)
- `target`: integer column index or None (defaults to first data column)

**Task 8.2: Update All API Functions**
File: [src/pyEDM/API.py](src/pyEDM/API.py)

Apply changes to:
- `Simplex()`
- `SMap()`
- `CCM()`
- `Multiview()`
- `EmbedDimension()`
- `PredictInterval()`
- `PredictNonlinear()`

**Task 8.3: Update Column Parameter Defaults**
File: [src/pyEDM/API.py](src/pyEDM/API.py)

```python
# Default: use all columns except time
if columns is None:
    columns = list(range(1, data.shape[1]))
```

**Task 8.4: Update Target Parameter Defaults**
File: [src/pyEDM/API.py](src/pyEDM/API.py)

```python
# Default: use column 1 (first data column after time)
if target is None:
    target = 1
```

**Task 8.5: Remove Pandas Imports from API**
File: [src/pyEDM/API.py](src/pyEDM/API.py)

Remove: `from pandas import concat`

### Phase 9: Testing and Validation

**Task 9.1: Update Test Files**
Files in [tests/](tests/)

For each test file:
- Replace DataFrame construction with numpy array construction
- Update column references from names to integer indices
- Update target references to integer indices
- Expect numpy array outputs
- Access output columns by integer index
- Validate array shapes and values

**Task 9.2: Numerical Validation**

Run tests to verify numerical results unchanged:
- Predictions should match within floating point precision
- Neighbor finding should be identical
- Error metrics should match

**Task 9.3: Performance Testing**

Benchmark before and after:
- Memory usage
- Execution time
- Large dataset handling

### Phase 10: Documentation and Cleanup

**Task 10.1: Update Docstrings**

Update all function and class docstrings:
- Change "dataFrame" to "data : numpy.ndarray"
- Update "columns" to "list of int"
- Update "target" to "int"
- Document column index conventions
- Add numpy array shape specifications
- Update examples to use numpy arrays

**Task 10.2: Update README**

Document breaking changes:
- DataFrames no longer accepted
- All inputs/outputs are numpy arrays
- Column 0 is always time (convention)
- Columns referenced by integer index
- Migration guide with examples

**Task 10.3: Create Migration Examples**

Before/after code examples:
```python
# Before (pandas)
import pandas as pd
df = pd.read_csv('data.csv')
result = Simplex(dataFrame=df, columns='x,y,z', target='x', E=3)
predictions = result['Predictions']

# After (numpy)
import numpy as np
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
# Assume: column 0=time, 1=x, 2=y, 3=z
result = Simplex(data=data, columns=[1,2,3], target=1, E=3)
predictions = result[:, 2]  # Column 2 is Predictions
```

**Task 10.4: Update Package Dependencies**

Remove pandas from:
- `requirements.txt`
- `setup.py` or `pyproject.toml`

Keep only: numpy, scipy, scikit-learn, matplotlib

## Execution Order Summary

1. **Phase 1**: EDM base class (foundation for everything)
2. **Phase 2**: Output formatting (affects all classes)
3. **Phase 3**: Simplex class (inherited by CCM, used by Multiview)
4. **Phase 4**: SMap class (parallel to Simplex)
5. **Phase 5**: CCM class (depends on Simplex)
6. **Phase 6**: Multiview class (depends on Simplex)
7. **Phase 7**: Helper functions and utilities
8. **Phase 8**: API layer (final user-facing interface)
9. **Phase 9**: Testing and validation
10. **Phase 10**: Documentation

Execute sequentially with testing after each phase.

## Expected Benefits

1. **Eliminated Dependency**: Remove pandas entirely (except possibly LoadData for CSV convenience)
2. **Simplified Codebase**: No DataFrame logic, conversions, or column name handling
3. **Improved Performance**: Single data representation, no conversion overhead
4. **Reduced Memory**: No dual representation
5. **Lower Barrier**: Users only need numpy

## Risks and Mitigation

**Risk 1: Breaking Changes for Users**
- Mitigation: Major version bump, clear migration guide

**Risk 2: Loss of Column Naming Convenience**
- Mitigation: Clear index conventions, documentation with constant examples

**Risk 3: Less Intuitive API**
- Mitigation: Comprehensive examples and documentation

## Success Criteria

- All pandas imports removed from core modules
- All functions accept numpy arrays
- All functions return numpy arrays
- All tests pass
- No numerical regression
- Performance meets or exceeds previous implementation

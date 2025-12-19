# Architecture Antipatterns in numpyEDM

## 1. TIGHT COUPLING WITH PANDAS DATAFRAMES

### Problem:
The entire codebase is tightly coupled to pandas DataFrames, creating several issues:

**Evidence:**
- `EDM.py` validates input is a DataFrame: `isinstance(self.Data, DataFrame)`
- `Embed.py` originally used DataFrame.shift() for embedding
- `Neighbors.py` converts DataFrame to numpy for KDTree but keeps DataFrame as primary storage
- `SMap.py` and `Simplex.py` use `.iloc[]` extensively for DataFrame access
- 21 instances of `.iloc[]` found across the codebase

**Antipatterns:**
- **Leaky Abstraction**: Internal representation (DataFrame) leaks to API
- **Impedance Mismatch**: Using DataFrames for numerical operations when numpy arrays would be more efficient
- **Dependency Hell**: Tight coupling to pandas makes the library harder to maintain and test

**Impact:**
- Performance overhead from DataFrame operations
- Memory usage higher than necessary for numerical computations
- Difficulty integrating with pure numpy-based workflows
- Testing complexity increased due to pandas dependency

## 2. MIXED PARADIGMS (OOP + PROCEDURAL)

### Problem:
The codebase mixes object-oriented and procedural programming styles inconsistently.

**Evidence:**
- `EDM.py` is an OOP class with methods
- `API.py` contains standalone functions that create and use EDM objects
- `Embed.py` has a standalone function rather than being a method
- Some functionality is split between classes and modules

**Antipatterns:**
- **Incomplete OOP**: Not fully embracing OOP principles
- **God Class**: EDM class has too many responsibilities (data storage, validation, embedding, neighbor search)
- **Feature Envy**: Methods in one class use data from another class

**Impact:**
- Code organization is confusing
- Hard to maintain and extend
- Violates Single Responsibility Principle

## 3. POOR TYPE HINTING

### Problem:
Lack of type hints makes the code harder to understand and maintain.

**Evidence:**
- No type hints in function signatures
- Return types not documented
- Parameter types not specified
- Only recent addition: `self.Embedding :numpy.ndarray` in EDM.py (incomplete)

**Antipatterns:**
- **Magic Strings**: Column names passed as strings without type safety
- **Implicit Interfaces**: Function signatures don't document expected types

**Impact:**
- IDE support limited
- Harder to catch type errors
- Documentation incomplete
- Developer experience poor

## 4. CIRCULAR DEPENDENCIES

### Problem:
Modules import from each other, creating dependency cycles.

**Evidence:**
- `EDM.py` imports from `Embed` and `API`
- `API.py` likely imports from `EDM`
- `Simplex.py` and `SMap.py` inherit from `EDM`
- `Neighbors.py` is imported as a method in `EDM`

**Antipatterns:**
- **Circular Dependency**: Modules depend on each other
- **Import Starvation**: Hard to determine proper import order

**Impact:**
- Code organization difficult
- Testing challenging
- Refactoring risky

## 5. INCONSISTENT NAMING CONVENTIONS

### Problem:
Inconsistent naming makes code harder to read and understand.

**Evidence:**
- `dataFrame` vs `data` parameter names
- `E` vs `embeddingDimensions`
- `tau` vs `stepSize`
- `lib_i` vs `pred_i` (inconsistent with `predList`)
- Mixed camelCase and snake_case

**Antipatterns:**
- **Inconsistent Naming**: No clear naming convention
- **Hungarian Notation**: `lib_i`, `pred_i` suggest type in name

**Impact:**
- Code readability suffers
- Learning curve increased
- Maintenance difficult

## 6. MAGIC NUMBERS AND STRINGS

### Problem:
Hard-coded values and strings throughout the code.

**Evidence:**
- Column index 0 assumed to be time: `self.Data.iloc[:, 0]`
- String literals for column names without validation
- Hard-coded shift values in embedding
- Magic numbers in validation logic

**Antipatterns:**
- **Magic Numbers**: Unnamed numeric constants
- **Stringly Typed**: Using strings for what should be structured data

**Impact:**
- Code fragile and error-prone
- Hard to maintain
- Documentation incomplete

## 7. LACK OF IMMUTABILITY

### Problem:
Objects are mutated extensively after creation.

**Evidence:**
- `self.Embedding` is set after object creation
- `self.lib_i`, `self.pred_i` modified in place
- Methods like `RemoveNan()` mutate object state
- No copy-on-write or defensive copying

**Antipatterns:**
- **Object Mutator**: Objects changed after creation
- **Hidden Side Effects**: Methods modify state unexpectedly

**Impact:**
- Hard to reason about code
- Testing difficult
- Thread safety issues

## 8. INCONSISTENT ERROR HANDLING

### Problem:
Error handling is inconsistent across the codebase.

**Evidence:**
- Some methods raise RuntimeError
- Some use warnings.warn()
- Some return None or empty results on error
- Inconsistent error messages

**Antipatterns:**
- **Inconsistent Exceptions**: Different exception types for similar errors
- **Silent Failures**: Some errors not properly reported

**Impact:**
- Error handling unpredictable
- Debugging difficult
- User experience inconsistent

## 9. LACK OF MODULARITY IN EMBEDDING

### Problem:
Embedding logic is scattered and not well-encapsulated.

**Evidence:**
- `Embed.py` has standalone function
- `EDM.EmbedData()` calls `Embed()` function
- Embedding happens in multiple places
- No clear single source of truth for embedding

**Antipatterns:**
- **Scattered Functionality**: Embedding logic not centralized
- **Duplicate Code**: Similar embedding logic in different places

**Impact:**
- Hard to maintain embedding logic
- Inconsistent behavior possible
- Testing difficult

## 10. POOR SEPARATION OF CONCERNS

### Problem:
Classes and functions handle multiple responsibilities.

**Evidence:**
- `EDM` class handles data storage, validation, embedding, neighbor search
- `API.py` mixes high-level API with implementation details
- Functions do multiple things (e.g., validation + computation)

**Antipatterns:**
- **God Object**: EDM class does too much
- **Fat Function**: Functions have multiple responsibilities

**Impact:**
- Violates Single Responsibility Principle
- Hard to test and maintain
- Code organization confusing

## RECOMMENDATIONS

1. **Decouple from pandas**: Use numpy arrays as primary data structure
2. **Add comprehensive type hints**: Improve code documentation and IDE support
3. **Refactor into smaller classes**: Break EDM into smaller, focused classes
4. **Centralize embedding logic**: Create a proper Embedding class
5. **Standardize naming conventions**: Use consistent snake_case throughout
6. **Add immutability**: Create copies when needed, avoid in-place mutations
7. **Improve error handling**: Consistent exception types and messages
8. **Remove circular dependencies**: Restructure imports
9. **Add proper documentation**: Docstrings for all public functions
10. **Create clear interfaces**: Define what each module exports

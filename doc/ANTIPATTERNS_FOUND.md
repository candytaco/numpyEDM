# Antipatterns Found in EDM Codebase

## 1. Type Checking Anti-Pattern (Primary Issue)
**Location**: EDM.py base class
**Problem**: Uses `self.name` parameter to determine class-specific behavior instead of proper method overriding
**Examples**:
- `if self.name == 'SMap':` for SMap-specific logic
- `if self.name in ['Simplex', 'SMap', 'Multiview']:` for validation
- `if self.name != 'CCM':` to skip CCM validation

**Impact**: Violates OOP principles, scatters class-specific logic, makes code hard to extend

## 2. Incomplete Inheritance Implementation
**Location**: EDM.py, Simplex.py, SMap.py
**Problem**: TODO comments indicate unfinished refactoring work for proper inheritance
**Examples**:
- `# TODO: change this to properly inherit and override` in `RemoveNan()`
- `# TODO: properly override these in inheritance` in `CreateIndices()`
- Methods that should be overridden but use conditional logic instead

**Impact**: Technical debt, inconsistent architecture, methods not properly specialized

## 3. Complex Index Management
**Location**: EDM.py `CreateIndices()` method
**Problem**: `# TODO: this is like, 4 levels of index shadowing - needs to be fixed`
**Issue**: Overly complex index management with multiple levels of indirection
**Impact**: Hard to understand, maintain, and debug; potential for index-related bugs

## 4. Inconsistent Architecture Pattern
**Location**: CCM.py, Multiview.py vs Simplex.py, SMap.py
**Problem**: Mixed use of inheritance (Simplex, SMap) and composition (CCM, Multiview)
**Issue**: CCM and Multiview contain Simplex instances rather than inheriting from EDM
**Impact**: Inconsistent design pattern, code duplication, harder to maintain

## 5. Excessive Print Statements with Flush
**Location**: EDM.py validation methods
**Problem**: Multiple `print(..., flush=True)` statements for verbose output
**Issue**: Debugging/verbose output mixed with core logic
**Impact**: Clutters code, hard to separate logging from business logic

## 6. Magic Numbers and Hardcoded Values
**Location**: Throughout EDM.py
**Problem**: Hardcoded values like `1E-6`, `20`, `5` without explanation
**Examples**:
- `minWeight = 1E-6` in projection calculations
- `leafsize = 20` in KDTree creation
- `xRadKnnFactor = 5` for exclusion radius adjustment

**Impact**: Makes code harder to understand and configure

## 7. Overly Complex Method Implementation
**Location**: EDM.py `FindNeighbors()` method
**Problem**: Single method handles multiple complex responsibilities
**Issues**:
- KDTree creation and querying
- Neighbor distance calculations
- Exclusion radius handling
- Multiple validation checks
- Complex index mapping logic

**Impact**: Violates Single Responsibility Principle, hard to test and maintain

## 8. Inconsistent Parameter Handling
**Location**: Across all classes
**Problem**: Mix of parameter objects and direct attribute assignment
**Issue**: Some classes use dataclass parameters, others use direct assignment
**Impact**: Inconsistent API, harder to understand parameter flow

## 9. Poor Separation of Concerns
**Location**: EDM.py `FormatProjection()` method
**Problem**: Handles time conversion, projection formatting, and SMap-specific logic
**Issue**: Mixes concerns that should be separate methods/classes
**Impact**: Hard to modify one aspect without affecting others

## 10. Inefficient Data Copying
**Location**: Simplex.py and SMap.py `Generate()` methods
**Problem**: Comment `# JP : for big data this is likely not efficient`
**Issue**: Data copying in generation loops may impact performance
**Impact**: Potential performance bottleneck for large datasets

## 11. Incomplete Error Handling
**Location**: Various methods
**Problem**: Some error conditions raise exceptions, others use warnings
**Issue**: Inconsistent error handling strategy
**Impact**: Unpredictable behavior for edge cases

## 12. Overuse of Instance Variables
**Location**: EDM class
**Problem**: Large number of instance variables (50+ in EDM base class)
**Issue**: Class has too much state, hard to track variable usage
**Impact**: Increased complexity, potential for bugs from state management

## Recommended Refactoring Priorities

1. **High Priority**: Fix type checking anti-pattern (name parameter issue)
2. **High Priority**: Complete proper inheritance implementation
3. **Medium Priority**: Simplify complex index management
4. **Medium Priority**: Standardize architecture (inheritance vs composition)
5. **Low Priority**: Extract constants for magic numbers
6. **Low Priority**: Split complex methods into smaller, focused methods

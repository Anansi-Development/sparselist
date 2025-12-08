# Sparselist

A memory-efficient list subclass that stores only explicitly-set values while maintaining a logical size. Unspecified positions return a default value without materializing them in memory.

- [Overview](#overview)
- [Core Characteristics](#core-characteristics)
- [Installation](#installation)
- [Constructor](#constructor)
- [Properties](#properties)
  - [`size` (read/write)](#size-readwrite)
  - [`default` (read/write)](#default-readwrite)
- [Indexing \& Slicing](#indexing--slicing)
  - [Single Index](#single-index)
  - [Slicing](#slicing)
  - [Deletion](#deletion)
- [List Methods](#list-methods)
  - [Mutating Methods](#mutating-methods)
  - [Query Methods](#query-methods)
- [Operators](#operators)
  - [Comparison](#comparison)
  - [Arithmetic](#arithmetic)
  - [Membership](#membership)
- [Iteration](#iteration)
- [Copy Support](#copy-support)
- [String Representation](#string-representation)
- [Performance Characteristics](#performance-characteristics)
- [Key Implementation Details](#key-implementation-details)
- [Use Cases](#use-cases)
- [License](#license)

## Overview

`sparselist` is perfect for scenarios where you need a large list but only a fraction of positions contain meaningful data. Instead of allocating memory for every position, it uses an internal dictionary to store only the explicitly-set values.

**Materialization** means creating actual Python objects in memory for every list position. Sparselist avoids this by only storing explicitly-set values: a size-1,000,000 list with 10 values uses O(10) memory, not O(1,000,000).

## Core Characteristics

- **Inherits from `list`**: All standard list operations work
- **Not hashable**: Raises `TypeError` on `hash()` attempts
- **Sparse storage**: Uses an internal dictionary to store only non-default values
- **Memory efficient for large list**: A list of size 1,000,000 with 10 values uses O(10) memory, not O(1,000,000). (For small lists the dictionary overhead is meaningful.)

## Installation

```python
from sparselist import sparselist
```

## Constructor

```python
sparselist(data, size=None, default=None)
```

**Parameters:**
- `data`: Initial data (list, dict, or iterable)
  - If dict: keys must be non-negative integers, values placed at those indices
  - If list/iterable: elements populate indices 0, 1, 2, etc.
- `size`: Total logical size (optional)
  - If omitted: inferred from data (len for lists, max_key+1 for dicts)
  - Must be integer ≥ 0
  - Must accommodate all data (≥ len(list) or ≥ max(dict.keys())+1)
- `default`: Value returned for unspecified indices (default: `None`)

**Errors:**
- `ValueError`: negative size, size too small for data
- `AttributeError`: non-integer size
- `TypeError`/`ValueError`: dict with non-integer or negative keys

**Examples:**

```python
# Create from dict (sparse initialization)
sl = sparselist({0: 'a', 100: 'b', 1000: 'c'}, default='')
len(sl)  # 1001
sl[50]   # '' (default)
sl[100]  # 'b'

# Create from list (dense initialization)
sl = sparselist(['a', 'b', 'c'], default=0)
len(sl)  # 3

# Create with explicit size
sl = sparselist({5: 'x'}, size=100, default=None)
len(sl)  # 100
```

## Properties

### `size` (read/write)
- Getting: returns current logical size
- Setting: resizes the list
  - Increasing: extends with default values
  - Decreasing: truncates, removes explicit values at positions ≥ new_size
  - Must be integer ≥ 0
  - Errors: `ValueError` (negative), `AttributeError`/`TypeError` (non-integer)

### `default` (read/write)
- Changing affects all unspecified positions immediately
- Mutating mutable defaults affects all default positions

**Examples:**

```python
sl = sparselist({0: 'a', 5: 'b'}, size=10, default='x')
sl.size = 20  # Extend to size 20
sl[15]        # 'x' (default)
sl.size = 3   # Truncate to size 3 (removes key 5)
sl.default = 'y'  # Change default
sl[2]         # 'y' (new default)
```

## Indexing & Slicing

### Single Index
- Supports positive/negative indices like normal lists
- Returns explicit value if set, otherwise default
- Assignment stores value as explicit (even if it equals default)
- `IndexError` for out-of-bounds

```python
sl = sparselist({0: 'a', 5: 'b'}, size=10, default='x')
sl[0]     # 'a'
sl[3]     # 'x' (default)
sl[-1]    # 'x' (last position, default)
sl[3] = 'c'  # Set explicit value
```

### Slicing
- Read: Returns new sparselist with same default, containing only relevant keys
- Write: Behaves exactly like list slice assignment
  - Step=1 or -1: can change length
  - Other steps: replacement must match slice length exactly
  - ValueError for step=0

```python
sl = sparselist({0: 'a', 5: 'b', 10: 'c'}, size=15, default='x')
sl[3:8]     # New sparselist with keys remapped
sl[2:6] = ['d', 'e', 'f', 'g']  # Replace slice
```

### Deletion
- `del sl[i]`: removes element, decrements size, shifts subsequent positions
- `del sl[start:stop:step]`: removes slice elements, behaves like list

```python
sl = sparselist({0: 'a', 5: 'b', 10: 'c'}, size=15, default='x')
del sl[5]   # Remove index 5, shift remaining keys
del sl[::2] # Remove every other element
```

## List Methods

### Mutating Methods
- `append(value)`: adds to end, increments size
- `extend(iterable)`: adds all items, increases size
- `insert(index, value)`: inserts at position, increments size, shifts subsequent positions
- `pop(index=-1)`: removes and returns, decrements size
- `remove(value)`: removes first occurrence (explicit or default), raises `ValueError` if not found
- `reverse()`: reverses in-place efficiently without materialization
- `sort(key=None, reverse=False)`: sorts in-place without materialization - sorts explicit items, computes logical positions with defaults
- `clear()`: empties list, removes all explicit values, sets size to 0

**Sparselist-specific methods (not part of standard list interface):**
- `compact()`: removes all implicit defaults, compacts list to contiguous explicit elements
- `unset(index)`: resets value at index to default by removing from explicit values, returns the old value

```python
sl = sparselist({0: 'c', 5: 'a', 10: 'b'}, size=15, default='x')
sl.sort()   # O(3 log 3), not O(15 log 15) - only sorts explicit values
sl.append('d')  # Adds at index 15
sl.compact()    # Removes gaps, becomes size 4 with values at 0,1,2,3
old_val = sl.unset(1)  # Sets index 1 back to default, returns old value
```

### Query Methods
- `count(value)`: counts occurrences (both explicit and default positions)
- `index(value, start=0, stop=None)`: finds first occurrence index, raises `ValueError` if not found
- `copy()`: returns independent sparselist with same size, default, and explicit values

```python
sl = sparselist({0: 'a', 5: 'a'}, size=100, default='x')
sl.count('a')  # 2
sl.count('x')  # 98 (default positions)
sl.index('a')  # 0
```

## Operators

### Comparison
- `==`, `!=`, `<`, `>`, `<=`, `>=`: compares as lists without materializing
- Compares explicit values at their positions and computes default value comparisons for gaps
- Works with both sparselists and regular lists

```python
sl1 = sparselist({0: 'a', 5: 'b'}, size=10, default='x')
sl2 = sparselist({0: 'a', 5: 'b'}, size=10, default='x')
sl1 == sl2  # True (optimized comparison for sparse lists)
```

### Arithmetic
- `+`: concatenation, returns sparselist with left operand's default
- `*`: repetition (integer multiplier only, TypeError otherwise)
- `+=`: in-place extend, supports any iterable
- `*=`: in-place repetition

```python
sl1 = sparselist({0: 'a'}, size=5, default='x')
sl2 = sparselist({0: 'b'}, size=5, default='y')
sl3 = sl1 + sl2  # size=10, default='x' (from sl1)
sl1 *= 3         # Repeat in place
```

### Membership
- `in`: checks both explicit and default values

```python
sl = sparselist({0: 'a', 5: 'b'}, size=100, default='x')
'a' in sl  # True
'x' in sl  # True (98 positions have default)
'z' in sl  # False
```

## Iteration

- Supports iterator and reverse iterator protocols
- Yields values in order (explicit or default)
- Iterator behavior under modification matches regular list behavior exactly

```python
sl = sparselist({0: 'a', 5: 'b'}, size=10, default='x')
for value in sl:
    print(value)  # prints 'a', then 8 'x's, then 'b'
```

## Copy Support

Supports both shallow and deep copying:

**Shallow copy** (`copy.copy()` or `.copy()` method):
- Creates new sparselist with independent structure
- Mutable elements and defaults are shared (same object references)
- Mutating shared objects affects both copies

**Deep copy** (`copy.deepcopy()`):
- Creates new sparselist with fully independent elements
- Mutable elements and defaults are recursively copied
- Mutations only affect the specific copy

```python
import copy

# Shallow copy - immutable elements
sl1 = sparselist({0: 'a', 5: 'b'}, size=10, default='x')
sl2 = copy.copy(sl1)
sl2[0] = 'c'
sl1[0]  # Still 'a' (reassignment doesn't affect original)

# Shallow copy - mutable elements are shared
sl3 = sparselist({0: [1, 2]}, size=5, default=[])
sl4 = copy.copy(sl3)
sl4[0].append(99)
sl3[0]  # [1, 2, 99] (mutation affects both!)

# Deep copy - mutable elements are independent
sl5 = sparselist({0: [1, 2]}, size=5, default=[])
sl6 = copy.deepcopy(sl5)
sl6[0].append(99)
sl5[0]  # Still [1, 2] (mutation doesn't affect original)
```

## String Representation

- `__repr__`: Format `<size/default>[index: value, ..., index: value]`
- Use `...` wherever there are gaps (unspecified elements), including at start and end
- No `...` for fully dense lists or empty lists

```python
sl = sparselist({0: 'a', 5: 'b', 10: 'c'}, size=15, default='x')
print(sl)  # <15/'x'>[0: 'a', ..., 5: 'b', ..., 10: 'c', ...]
```

## Performance Characteristics

The key advantage of `sparselist` is that most operations scale with the number of **explicit** values (E), not the total **size** (N):

- Constructor from dict: **O(E)** time, **O(E)** space
- Single index access/assignment: **O(1)**
- Insert/delete/pop: **O(E)** for key shifting
- Sort: **O(E log E)** - independent of size!
- Reverse: **O(E)**
- Comparison (best case): **O(E1 + E2)**

**Example**: A sparselist of size 1,000,000 with 10 explicit values:
- Sort: O(10 log 10) ≈ 33 operations, not O(1,000,000 log 1,000,000)
- Memory: ~10 entries, not 1,000,000

## Key Implementation Details

1. **Never materialize default values** - maintain sparsity for all operations including sort and comparisons
2. Maintain efficiency with large sparse lists (e.g., size=1,000,000 with few explicit keys)
3. When keys shift (insert/delete), update internal storage accordingly
4. For slicing, only copy relevant explicit values to new sparselist
5. Properties validate and maintain invariants (size ≥ 0, size accommodates data)
6. For sort: only sort explicit items, then compute their new positions considering default value ordering
7. For comparisons: iterate through positions, comparing explicit values where present and default values where not, without creating a full list

## Use Cases

- **Sparse matrices**: Store only non-zero values
- **Time series with gaps**: Store only actual measurements, not interpolated values
- **Configuration with defaults**: Store only overridden values
- **Large indexed collections**: When most indices are empty or have default values

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

from __future__ import annotations

from operator import index as op_index
from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable, Iterator
    from typing import Any, SupportsIndex

    from _typeshed import SupportsRichComparison

    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self

T = TypeVar("T")

# Threshold for switching strategies from sparse (iterate explicit keys) to dense (iterate positions)
# Only switch when both density exceeds this AND explicit count > _DENSE_COUNT_THRESHOLD
_DENSE_DENSITY_THRESHOLD = 0.5
_DENSE_COUNT_THRESHOLD = 10000


class sparselist(list[T]):  # noqa: N801
    """A memory-efficient list that stores only explicitly-set values."""

    _explicit: dict[int, T]
    _size: int
    _default: T | None

    def __init__(  # noqa: PLR0912
        self,
        data: dict[int, T] | Iterable[T] | None = None,
        size: SupportsIndex | None = None,
        default: T | None = None,
    ) -> None:
        """Initialize a sparselist from data.

        Args:
            data: Initial data (optional, defaults to empty)
                  - None: creates empty sparselist
                  - dict: keys must be non-negative integers, values placed at those indices
                  - iterable: elements populate indices 0, 1, 2, etc.
            size: Total logical size (optional, inferred from data if not provided)
            default: Value returned for unspecified indices (default: None)

        Raises:
            TypeError: If dict keys are not integers or size doesn't support __index__
            ValueError: If dict keys are negative or size is negative or size too small
        """
        super().__init__()
        self._explicit: dict[int, T] = {}
        self._default = default

        # Validate and convert size type before inference
        if size is not None:
            try:
                size = op_index(size)
            except TypeError:
                raise TypeError("size must support __index__") from None

        # Handle None data case (empty sparselist)
        if data is None:
            final_size = size if size is not None else 0
            if final_size < 0:
                raise ValueError("size must be non-negative")
            self._size = final_size
            return

        if isinstance(data, dict):
            # Validate dict keys
            for key in data:
                if not isinstance(key, int):
                    raise TypeError("dict keys must be integers")
                if key < 0:
                    raise ValueError("dict keys must be non-negative")

            # Infer size if not provided
            if size is None:
                size = max(data.keys()) + 1 if data else 0  # type: ignore[type-var, arg-type, unused-ignore]

            # Validate size value
            if size < 0:
                raise ValueError("size must be non-negative")
            if data and max(data.keys()) >= size:  # type: ignore[type-var, arg-type, unused-ignore]
                raise ValueError("size must accommodate all data")

            self._size = size
            # Copy dict to avoid aliasing
            self._explicit = dict(data.items())  # type: ignore[assignment, unused-ignore]
        else:
            # Store items from iterable while counting
            count = 0
            for i, value in enumerate(data):
                self._explicit[i] = value
                count = i + 1

            # Infer size if not provided
            if size is None:
                size = count

            # Validate size value (type already validated and converted above)
            if size < 0:
                raise ValueError("size must be non-negative")
            if count > size:
                raise ValueError("size must accommodate all data")

            self._size = size

    @property
    def size(self) -> int:
        """Get or set the logical size of the sparselist.

        :getter: Returns the total logical size.
        :setter: Truncates or extends the list to the new size.
            Truncation removes all values at indices >= new_size.

        Raises:
            TypeError: If value is not an integer (when setting)
            ValueError: If value is negative (when setting)
        """
        return self._size

    @size.setter
    def size(self, new_size: int) -> None:
        """Set the logical size. See getter for full documentation."""
        if not isinstance(new_size, int):
            raise TypeError("size must be an integer")
        if new_size < 0:
            raise ValueError("size must be non-negative")

        if new_size < self._size:
            # Decreasing: remove keys >= new_size
            # Choose strategy based on estimated work (without iterating)
            explicit_count = len(self._explicit)
            if explicit_count == 0:
                # No explicit keys, nothing to remove
                pass
            elif explicit_count == self._size:
                # Fully dense: we know exactly how many to remove
                removed_count = self._size - new_size
                if removed_count < explicit_count / 2:
                    # Remove specific keys
                    for k in range(new_size, self._size):
                        self._explicit.pop(k, None)
                else:
                    # Rebuild dict
                    self._explicit = {k: v for k, v in self._explicit.items() if k < new_size}
            else:
                # Sparse: estimate keys to remove based on density
                density = explicit_count / self._size
                estimated_removed = density * (self._size - new_size)

                if estimated_removed < explicit_count / 2:
                    # Likely removing less than half: use targeted deletion
                    keys_to_remove = [k for k in self._explicit if k >= new_size]
                    for k in keys_to_remove:
                        self._explicit.pop(k, None)
                else:
                    # Likely removing half or more: rebuild dict
                    self._explicit = {k: v for k, v in self._explicit.items() if k < new_size}

        # Update size (works for both increasing and decreasing)
        self._size = new_size

    @property
    def default(self) -> T | None:
        """Get or set the default value for unspecified positions.

        :getter: Returns the current default value.
        :setter: Updates the default value. Does not modify existing
            explicit values.

        Note:
            Changing the default affects all unspecified positions immediately.
            Mutating mutable defaults affects all default positions.
        """
        return self._default

    @default.setter
    def default(self, new_default: T | None) -> None:
        """Set the default value. See getter for full documentation."""
        self._default = new_default

    def __len__(self) -> int:
        """Return the logical size of the list."""
        return self._size

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[type[Self], tuple[()], dict[str, Any]]:
        """Return pickle data, preventing default list subclass behavior."""
        return (
            self.__class__,
            (),  # Create empty sparselist with no args
            self.__getstate__(),
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return state for pickling."""
        return {
            "data": self._explicit,
            "size": self._size,
            "default": self._default,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickling."""
        self._explicit = state["data"]
        self._size = state["size"]
        self._default = state["default"]

    @overload
    def __getitem__(self, key: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> sparselist[T]: ...

    def __getitem__(self, key: SupportsIndex | slice) -> T | sparselist[T]:
        """Get item(s) by index or slice.

        Args:
            key: Integer index or slice

        Returns:
            Single value for int index, new sparselist for slice

        Raises:
            TypeError: If key is not an integer or slice
            IndexError: If index is out of range
            ValueError: If slice step is zero
        """
        if isinstance(key, slice):
            # Handle slice
            start, stop, step = key.indices(self._size)

            # Calculate resulting slice length
            slice_length = len(range(start, stop, step))

            # Build new sparselist with remapped keys
            new_explicit: dict[int, T] = {}
            for i, old_idx in enumerate(range(start, stop, step)):
                if old_idx in self._explicit:
                    new_explicit[i] = self._explicit[old_idx]

            return sparselist(new_explicit, size=slice_length, default=self._default)
        # Single index access - convert SupportsIndex to int
        idx = op_index(key)

        # Normalize negative index
        if idx < 0:
            idx += self._size

        # Check bounds
        if not 0 <= idx < self._size:
            raise IndexError("list index out of range")

        # Return explicit value or default
        return self._explicit.get(idx, self._default)  # type: ignore[return-value]

    @overload
    def __setitem__(self, key: SupportsIndex, value: T) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, key: SupportsIndex | slice, value: T | Iterable[T]) -> None:
        """Set item(s) by index or slice.

        Args:
            key: Integer index or slice
            value: Single value for int index, iterable for slice

        Raises:
            TypeError: If key/value type combination is invalid
            IndexError: If index is out of range
            ValueError: If slice step is zero or extended slice length mismatch
        """
        if isinstance(key, slice):
            # Slice assignment requires iterable value
            start, stop, step = key.indices(self._size)

            indices = list(range(start, stop, step))
            slice_length = len(indices)

            # Convert value to list - this will fail with TypeError if not iterable
            try:
                values = list(value)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("can only assign an iterable") from None

            if step != 1:
                # Extended slice (including step=-1): must match exactly
                if len(values) != slice_length:
                    raise ValueError(
                        f"attempt to assign sequence of size {len(values)} to extended slice of size {slice_length}"
                    )

                # Simple assignment, no resizing
                for idx, val in zip(indices, values, strict=True):
                    self._explicit[idx] = val
            else:
                # Step is 1: can resize
                self._assign_basic_slice(start, stop, values)
        else:
            # Single index assignment
            idx = op_index(key)

            # Normalize negative index
            if idx < 0:
                idx += self._size

            # Check bounds
            if not 0 <= idx < self._size:
                raise IndexError("list assignment index out of range")

            # Store value (even if it equals default)
            self._explicit[idx] = value  # type: ignore[assignment]

    def _assign_basic_slice(self, start: int, stop: int, values: list[T]) -> None:
        """Handle contiguous slice assignment that can change size (step=1).

        Args:
            start: Slice start index (normalized)
            stop: Slice stop index (normalized)
            values: List of values to assign
        """
        # Clamp to bounds
        start = max(0, min(start, self._size))
        stop = max(start, min(stop, self._size))

        old_slice_len = stop - start
        new_slice_len = len(values)
        size_change = new_slice_len - old_slice_len

        if size_change == 0:
            # Same size: just replace
            # Remove old explicit values in this range
            for idx in range(start, stop):
                self._explicit.pop(idx, None)

            # Insert new values
            for i, val in enumerate(values):
                self._explicit[start + i] = val

        elif size_change > 0:
            # Growing: shift keys after stop
            self._shift_keys(stop, size_change)
            self._size += size_change

            # Remove old explicit values in range
            for idx in range(start, stop):
                self._explicit.pop(idx, None)

            # Insert new values
            for i, val in enumerate(values):
                self._explicit[start + i] = val

        else:  # size_change < 0
            # Shrinking: remove some positions
            # Remove old explicit values in deleted range
            for idx in range(start, stop):
                self._explicit.pop(idx, None)

            # Shift keys >= stop by size_change (negative)
            self._shift_keys(stop, size_change)
            self._size += size_change

            # Insert new values
            for i, val in enumerate(values):
                self._explicit[start + i] = val

    def _shift_keys(self, start_index: int, delta: int) -> None:
        """Shift all keys >= start_index by delta.

        Args:
            start_index: Index to start shifting from
            delta: Amount to shift (positive or negative)

        Note:
            Requires that start_index + delta >= 0 to ensure no keys shift to invalid positions.
        """
        if delta == 0:
            return

        # Validate that shift won't create negative indices
        assert delta <= 0 or start_index + delta >= 0, "shift would create negative indices"

        # Sort keys to process in order that avoids overwriting unprocessed keys
        # delta < 0: process ascending (move left, start with leftmost)
        # delta > 0: process descending (move right, start with rightmost)
        # Time: O(E log E), Space: O(E) where E = number of explicit keys to shift
        # This is acceptable since E is the sparse count, not the total size N
        keys_to_shift = sorted((k for k in self._explicit if k >= start_index), reverse=(delta > 0))

        # Shift each key
        for key in keys_to_shift:
            value = self._explicit.pop(key)
            self._explicit[key + delta] = value

    @overload  # type: ignore[override]
    def __delitem__(self, key: SupportsIndex) -> None: ...

    @overload
    def __delitem__(self, key: slice) -> None: ...

    def __delitem__(self, key: SupportsIndex | slice) -> None:  # noqa: PLR0912
        """Delete item(s) by index or slice.

        Args:
            key: Integer index or slice

        Raises:
            TypeError: If key is not an integer or slice
            IndexError: If index is out of range
            ValueError: If slice step is zero
        """
        if isinstance(key, slice):
            # Handle slice deletion
            # key.indices() normalizes negative start/stop to positive values
            start, stop, step = key.indices(self._size)

            # Calculate number of elements to delete
            num_deleted = len(range(start, stop, step))
            if num_deleted == 0:
                # Empty slice, nothing to delete
                return

            if abs(step) == 1:
                # Contiguous slice: use simple approach
                # Normalize to forward direction
                if step == -1:
                    start, stop = stop + 1, start + 1

                # Remove explicit values in range
                for idx in range(start, stop):
                    self._explicit.pop(idx, None)

                # Shift keys after deleted range
                self._shift_keys(stop, -(stop - start))

                # Update size
                self._size -= stop - start
            else:
                # Non-contiguous slice (step != 1 and step != -1)
                # Normalize to forward direction mathematically
                if step < 0:
                    # Backward slice: compute forward equivalent
                    # Last index: start + ((num_deleted - 1) * step)
                    # First index after reversing: that last index
                    norm_start = start + (num_deleted - 1) * step
                    norm_stop = start + 1
                    norm_step = -step
                else:
                    norm_start = start
                    norm_stop = stop
                    norm_step = step

                # Build new explicit dict with index mapping
                new_explicit: dict[int, T] = {}

                for old_idx, value in self._explicit.items():
                    # Check if old_idx is in the deletion range (without materializing)
                    if old_idx not in range(norm_start, norm_stop, norm_step):
                        # Calculate shift amount based on normalized slice
                        if old_idx < norm_start:
                            # Before deleted range: no shift
                            shift_amount = 0
                        elif old_idx < norm_stop:
                            # Within range: count deleted indices < old_idx
                            # Equivalent to len(range(norm_start, old_idx, norm_step))
                            shift_amount = (old_idx - norm_start + norm_step - 1) // norm_step
                        else:
                            # After deleted range: shift by total deleted count
                            shift_amount = num_deleted

                        new_idx = old_idx - shift_amount
                        new_explicit[new_idx] = value

                # Replace explicit dict and update size
                self._explicit = new_explicit
                self._size -= num_deleted
        else:
            # Single index deletion
            idx = op_index(key)

            # Normalize negative index
            if idx < 0:
                idx += self._size

            # Check bounds
            if not 0 <= idx < self._size:
                raise IndexError("list index out of range")

            # Remove value at index if explicit
            self._explicit.pop(idx, None)

            # Shift all keys > idx by -1
            self._shift_keys(idx + 1, -1)

            # Decrement size
            self._size -= 1

    def __iter__(self) -> Iterator[T | None]:  # type: ignore[override]
        """Return an iterator over the list values.

        Yields:
            Values in order (explicit or default)
        """
        i = 0
        while i < self._size:
            yield self._explicit.get(i, self._default)
            i += 1

    def __contains__(self, value: object) -> bool:
        """Check if value is in the list.

        Args:
            value: Value to search for

        Returns:
            True if value is found (explicit or default), False otherwise
        """
        # Early exit: if looking for default and there are default positions
        if value == self._default and len(self._explicit) < self._size:
            return True

        # Check explicit values
        return value in self._explicit.values()

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911,PLR0912
        """Return True if self equals other.

        Supports comparison with any iterable (list, tuple, other sparselist, etc.).

        Args:
            other: Object to compare with

        Returns:
            True if sequences are equal, False otherwise
        """
        # Identity check: if same object, they're equal
        if self is other:
            return True

        # Check if other is iterable
        if not hasattr(other, "__iter__"):
            return NotImplemented

        # Optimization: sparselist to sparselist with same default
        if isinstance(other, sparselist) and self._default == other._default:
            # Compare size first for early exit
            if self._size != other._size:
                return False

            # Only need to compare union of explicit keys
            all_keys = set(self._explicit) | set(other._explicit)
            for key in all_keys:
                if self._explicit.get(key, self._default) != other._explicit.get(key, self._default):
                    return False
            return True

        # General comparison with any iterable
        # Try to get length first for early exit
        if hasattr(other, "__len__"):
            try:
                other_len = len(other)  # type: ignore[arg-type, unused-ignore]
                if self._size != other_len:
                    return False
            except TypeError:
                pass  # Some iterables don't support len()

        # Compare element by element using iterator
        other_iter = iter(other)  # type: ignore[arg-type, unused-ignore]
        for i in range(self._size):
            try:
                other_val = next(other_iter)
            except StopIteration:
                # Other is shorter
                return False
            if self[i] != other_val:
                return False

        # Check if other has more elements
        try:
            next(other_iter)
            return False  # Other is longer
        except StopIteration:
            return True  # Lengths match and all elements equal

    def __ne__(self, other: object) -> bool:
        """Return True if self does not equal other.

        Args:
            other: Object to compare with

        Returns:
            True if sequences are not equal, False otherwise
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def _compare_sparse(self, other: sparselist[T]) -> int:
        """Compare two sparse sparselists efficiently by checking only explicit keys.

        Args:
            other: Another sparselist with the same default value

        Returns:
            -1 if self < other
            0 if self == other
            1 if self > other

        Raises:
            TypeError: If elements don't support rich comparison operators
        """
        min_size = min(self._size, other._size)
        lowest_diff_index = min_size
        lowest_diff_value = 0

        self_val: T | None = None
        other_val: T | None = None

        try:
            # Check self's explicit keys
            for key in self._explicit:
                if key < lowest_diff_index:
                    self_val = self._explicit[key]
                    other_val = other._explicit.get(key, other._default)
                    if self_val < other_val:  # type: ignore[operator]
                        lowest_diff_index = key
                        lowest_diff_value = -1
                    elif self_val > other_val:  # type: ignore[operator]
                        lowest_diff_index = key
                        lowest_diff_value = 1

            # Check other's explicit keys
            for key in other._explicit:
                if key < lowest_diff_index:
                    self_val = self._explicit.get(key, self._default)
                    other_val = other._explicit[key]
                    if self_val < other_val:  # type: ignore[operator]
                        lowest_diff_index = key
                        lowest_diff_value = -1
                    elif self_val > other_val:  # type: ignore[operator]
                        lowest_diff_index = key
                        lowest_diff_value = 1
        except TypeError:
            # Re-raise with clearer error message
            raise TypeError(
                f"'<' or '>' not supported between instances of "
                f"{type(self_val).__name__!r} and {type(other_val).__name__!r}"
            ) from None

        # Return result
        if lowest_diff_value != 0:
            return lowest_diff_value

        # No differences in min_size range, compare sizes
        if self._size < other._size:
            return -1
        if self._size > other._size:
            return 1
        return 0

    def _compare(self, other: Iterable[T]) -> int:
        """Compare self with other lexicographically.

        Args:
            other: Iterable to compare with

        Returns:
            -1 if self < other
            0 if self == other
            1 if self > other

        Raises:
            TypeError: If elements don't support rich comparison operators
        """
        # Optimization: sparselist to sparselist with same default
        # Only use this for sparse lists to avoid pessimization
        if isinstance(other, sparselist) and self._default == other._default:
            # Use optimization only if sparse enough to be beneficial
            combined_explicit = len(self._explicit) + len(other._explicit)
            combined_size = self._size + other._size

            if combined_explicit < combined_size * 0.5:
                return self._compare_sparse(other)

        # General comparison (including dense sparselists and other iterables)
        other_iter = iter(other)
        for i in range(self._size):
            try:
                other_val = next(other_iter)
            except StopIteration:
                # Other is shorter and all elements so far are equal
                # self is longer, so self > other
                return 1

            self_val = self[i]
            # Use < and > operators - will raise TypeError if not supported
            try:
                if self_val < other_val:  # type: ignore[operator]
                    return -1
                if self_val > other_val:  # type: ignore[operator]
                    return 1
            except TypeError:
                # Re-raise with clearer error message
                raise TypeError(
                    f"'<' or '>' not supported between instances of "
                    f"{type(self_val).__name__!r} and {type(other_val).__name__!r}"
                ) from None
            # If equal, continue to next element

        # All elements equal so far, check if other has more elements
        try:
            next(other_iter)
            # Other is longer, so self < other
            return -1
        except StopIteration:
            # Same length and all elements equal
            return 0

    def __lt__(self, other: Iterable[T]) -> bool:
        """Return True if self is lexicographically less than other.

        Args:
            other: Iterable to compare with

        Returns:
            True if self < other, False otherwise
        """
        result = self._compare(other)
        return result < 0

    def __le__(self, other: Iterable[T]) -> bool:
        """Return True if self is lexicographically less than or equal to other.

        Args:
            other: Iterable to compare with

        Returns:
            True if self <= other, False otherwise
        """
        result = self._compare(other)
        return result <= 0

    def __gt__(self, other: Iterable[T]) -> bool:
        """Return True if self is lexicographically greater than other.

        Args:
            other: Iterable to compare with

        Returns:
            True if self > other, False otherwise
        """
        result = self._compare(other)
        return result > 0

    def __ge__(self, other: Iterable[T]) -> bool:
        """Return True if self is lexicographically greater than or equal to other.

        Args:
            other: Iterable to compare with

        Returns:
            True if self >= other, False otherwise
        """
        result = self._compare(other)
        return result >= 0

    def __add__(self, other: Iterable[Any]) -> sparselist[T]:  # type: ignore[override]
        """Return concatenation of self and other.

        Args:
            other: Iterable to concatenate

        Returns:
            New sparselist with elements from self followed by elements from other

        Note:
            The result uses self's default value
        """
        # Create copy and extend
        result = sparselist(self._explicit.copy(), size=self._size, default=self._default)
        result.extend(other)
        return result

    def __radd__(self, other: Iterable[Any]) -> list[T] | sparselist[T]:
        """Return concatenation of other and self (reflected addition).

        Called when other + self is evaluated and other doesn't support __add__ with sparselist.

        Args:
            other: Iterable to concatenate (on the left)

        Returns:
            If other is a sparselist, returns sparselist.
            Otherwise returns result of other + list(self).
        """
        # If other is a sparselist, extend it with self
        if isinstance(other, sparselist):
            result = sparselist(other._explicit.copy(), size=other._size, default=other._default)
            result.extend(self)
            return result

        # Otherwise, return other + list(self)
        return other + list(self)  # type: ignore[operator, no-any-return]

    def __iadd__(self, other: Iterable[Any]) -> Self:  # type: ignore[override]
        """In-place concatenation (extend self with other).

        Args:
            other: Iterable to concatenate

        Returns:
            self (modified in place)
        """
        self.extend(other)
        return self

    def __mul__(self, other: SupportsIndex) -> sparselist[T]:
        """Return sparselist repeated n times.

        Args:
            other: Number of repetitions (must support __index__)

        Returns:
            New sparselist with self repeated n times
        """
        n = op_index(other)

        if n <= 0:
            # Return empty sparselist with same default
            return sparselist(size=0, default=self._default)

        # Create copy and repeat
        result = sparselist(self._explicit.copy(), size=self._size, default=self._default)
        result._repeat(n)
        return result

    def __rmul__(self, other: SupportsIndex) -> sparselist[T]:
        """Return sparselist repeated n times (reverse multiplication).

        Args:
            other: Number of repetitions (must support __index__)

        Returns:
            New sparselist with self repeated n times
        """
        return self.__mul__(other)

    def __imul__(self, other: SupportsIndex) -> Self:
        """Repeat sparselist in place.

        Args:
            other: Number of repetitions (must support __index__)

        Returns:
            self (modified in place)
        """
        n = op_index(other)

        if n <= 0:
            self.clear()
            return self

        self._repeat(n)
        return self

    def _is_dense(self) -> bool:
        """Check if list exceeds density threshold for sparse optimizations.

        Returns:
            True if list is dense (should use position iteration),
            False if list is sparse (should use explicit key iteration)
        """
        explicit_count = len(self._explicit)
        if self._size == 0:
            return False
        density = explicit_count / self._size
        return density > _DENSE_DENSITY_THRESHOLD and explicit_count > _DENSE_COUNT_THRESHOLD

    def _repeat(self, n: int) -> None:
        """Repeat the sparselist in place n times.

        Args:
            n: Number of total repetitions (must be >= 1)

        Note:
            Modifies self._explicit and self._size
        """
        if n == 1:
            return  # No change needed

        original_size = self._size

        # Choose strategy based on density
        if self._is_dense():
            # Dense: iterate over all positions
            for rep in range(1, n):
                offset = rep * original_size
                for idx in range(original_size):
                    if idx in self._explicit:
                        self._explicit[offset + idx] = self._explicit[idx]
        else:
            # Sparse: iterate only over explicit keys
            original_keys = list(self._explicit.keys())
            for rep in range(1, n):
                offset = rep * original_size
                for idx in original_keys:
                    self._explicit[offset + idx] = self._explicit[idx]

        self._size = original_size * n

    def __copy__(self) -> sparselist[T]:
        """Return a shallow copy of the sparselist.

        Returns:
            New sparselist with same size, default, and explicit values
        """
        return sparselist(self._explicit.copy(), size=self._size, default=self._default)

    def __repr__(self) -> str:
        """Return a string representation of the sparselist.

        Format: *<size/default>[idx: val, ..., idx: val]*

        - Uses ... for gaps (unspecified elements)
        - No ... for fully dense lists or empty lists
        """
        if self._size == 0:
            return f"<0/{self._default!r}>[]"

        if not self._explicit:
            # All defaults
            return f"<{self._size}/{self._default!r}>[...]"

        # Check if fully dense
        if len(self._explicit) == self._size:
            return (
                f"<{self._size}/{self._default!r}>"
                f"[{', '.join(f'{i}: {self._explicit[i]!r}' for i in range(self._size))}]"
            )

        # Choose strategy based on density and explicit count
        # For nearly-dense large lists, iterate positions to avoid sorting huge key list
        # For sparse lists, sort keys for better performance
        if self._is_dense():
            # Nearly dense with many items: iterate through positions
            def parts_gen() -> Iterator[str]:
                in_gap = False
                for i in range(self._size):
                    if i in self._explicit:
                        if in_gap:
                            yield "..."
                            in_gap = False
                        yield f"{i}: {self._explicit[i]!r}"
                    else:
                        in_gap = True

                # Add trailing ... if we ended in a gap
                if in_gap:
                    yield "..."
        else:
            # Sparse or small: sort keys and detect gaps
            def parts_gen() -> Iterator[str]:
                sorted_keys = sorted(self._explicit.keys())

                # Check if there's a gap at start
                if sorted_keys[0] > 0:
                    yield "..."

                # Add explicit items with gaps
                for i, key in enumerate(sorted_keys):
                    yield f"{key}: {self._explicit[key]!r}"

                    # Check if there's a gap before next item
                    if i < len(sorted_keys) - 1:
                        next_key = sorted_keys[i + 1]
                        if next_key > key + 1:
                            yield "..."

                # Check if there's a gap at end
                if sorted_keys[-1] < self._size - 1:
                    yield "..."

        return f"<{self._size}/{self._default!r}>[{', '.join(parts_gen())}]"

    def __hash__(self) -> None:  # type: ignore[override]
        """Raise TypeError as sparselists are not hashable.

        Raises:
            TypeError: Always raised since sparselists are mutable
        """
        raise TypeError("unhashable type: 'sparselist'")

    def append(self, value: T) -> None:
        """Add value to the end of the list.

        Args:
            value: Value to append
        """
        self._explicit[self._size] = value
        self._size += 1

    def clear(self) -> None:
        """Remove all items from the list."""
        self._explicit.clear()
        self._size = 0

    def compact(self) -> None:
        """Remove all implicit defaults and compact list to contiguous explicit elements.

        Moves all explicitly-set elements down to positions starting from 0, removing gaps.
        Reduces size to the count of explicit elements.
        """
        if not self._explicit:
            # All defaults - become empty list
            self._size = 0
            return

        # Build new explicit dict with compacted positions
        new_explicit: dict[int, T] = {}
        new_idx = 0

        # Choose strategy based on density to avoid unnecessary sorting
        if self._is_dense():
            # Dense: iterate through positions in order
            for old_idx in range(self._size):
                if old_idx in self._explicit:
                    new_explicit[new_idx] = self._explicit[old_idx]
                    new_idx += 1
        else:
            # Sparse: sort keys for correct ordering
            for old_idx in sorted(self._explicit.keys()):
                new_explicit[new_idx] = self._explicit[old_idx]
                new_idx += 1

        # Update state
        self._explicit = new_explicit
        self._size = new_idx

    def copy(self) -> sparselist[T]:
        """Return a shallow copy of the sparselist.

        Returns:
            New sparselist with same size, default, and explicit values
        """
        return self.__copy__()

    def count(self, value: object) -> int:
        """Return number of occurrences of value.

        Args:
            value: Value to count

        Returns:
            Number of times value appears in the list
        """
        # Count occurrences in explicit values
        explicit_count = sum(1 for v in self._explicit.values() if v == value)

        # If value matches default, add count of implicit default positions
        if value == self._default:
            implicit_count = self._size - len(self._explicit)
            return explicit_count + implicit_count

        return explicit_count

    def extend(self, iterable: Iterable[Any]) -> None:
        """Extend list by appending elements from iterable.

        Args:
            iterable: Iterable of values to append
        """
        # Optimization: sparselist with same default
        if isinstance(iterable, sparselist) and self._default == iterable._default:
            # Early exit: empty sparselist
            if iterable._size == 0:
                return

            # Add other's explicit values with offset
            offset = self._size
            for idx, value in iterable._explicit.items():
                self._explicit[offset + idx] = value

            self._size += iterable._size
        else:
            # Early exit: check for sized empty iterables
            if hasattr(iterable, "__len__"):
                try:
                    if len(iterable) == 0:  # type: ignore[arg-type]
                        return
                except TypeError:
                    pass  # Some iterables don't support len()

            # General case: iterate through iterable
            offset = self._size
            count = 0
            for i, value in enumerate(iterable):
                self._explicit[offset + i] = value
                count = i + 1

            self._size += count

    def index(self, value: object, start: SupportsIndex = 0, stop: SupportsIndex | None = None) -> int:
        """Return first index of value.

        Args:
            value: Value to search for
            start: Start index (default 0)
            stop: Stop index (default None, meaning end of list)

        Returns:
            Index of first occurrence of value

        Raises:
            ValueError: If value is not found in the specified range
        """
        # Convert to int
        start_int = op_index(start)
        stop_int = op_index(stop) if stop is not None else None

        # Normalize start
        start_int = max(0, self._size + start_int) if start_int < 0 else min(start_int, self._size)

        # Normalize stop
        if stop_int is None:
            stop_int = self._size
        elif stop_int < 0:
            stop_int = max(0, self._size + stop_int)
        else:
            stop_int = min(stop_int, self._size)

        # Choose strategy based on whether value matches default and density
        if value != self._default and not self._is_dense():
            # Value doesn't match default and list is sparse - check only explicit keys
            for idx in sorted(self._explicit.keys()):
                if start_int <= idx < stop_int and self._explicit[idx] == value:
                    return idx
            # Not found
            raise ValueError(f"{value!r} is not in list")

        # Dense or value matches default: scan all positions in range
        for i in range(start_int, stop_int):
            if self[i] == value:
                return i

        # Not found
        raise ValueError(f"{value!r} is not in list")

    def insert(self, index: SupportsIndex, value: T) -> None:
        """Insert value before index.

        Args:
            index: Index to insert before
            value: Value to insert
        """
        # Convert to int
        idx = op_index(index)

        # Normalize negative index
        idx = max(0, self._size + idx) if idx < 0 else min(idx, self._size)

        # Shift keys >= idx forward by 1
        self._shift_keys(idx, 1)

        # Insert new value
        self._explicit[idx] = value

        # Increment size
        self._size += 1

    def pop(self, index: SupportsIndex = -1) -> T:
        """Remove and return item at index.

        Args:
            index: Index to pop (default -1, last item)

        Returns:
            Value at the specified index

        Raises:
            IndexError: If list is empty or index is out of range
        """
        if self._size == 0:
            raise IndexError("pop from empty list")

        # Convert to int
        idx = op_index(index)

        # Normalize negative index
        if idx < 0:
            idx += self._size

        # Check bounds
        if not 0 <= idx < self._size:
            raise IndexError("pop index out of range")

        # Get value (explicit or default)
        value = self._explicit.get(idx, self._default)

        # Remove explicit value if present
        self._explicit.pop(idx, None)

        # Shift keys > idx backward by 1
        self._shift_keys(idx + 1, -1)

        # Decrement size
        self._size -= 1

        return value  # type: ignore[return-value]

    def remove(self, value: object) -> None:
        """Remove first occurrence of value.

        Args:
            value: Value to remove

        Raises:
            ValueError: If value is not found
        """
        # Choose strategy based on whether value matches default and density
        if value != self._default and not self._is_dense():
            # Value doesn't match default and list is sparse - check only explicit keys
            for idx in sorted(self._explicit.keys()):
                if self._explicit[idx] == value:
                    # Found it - remove at this index
                    self._explicit.pop(idx)
                    self._shift_keys(idx + 1, -1)
                    self._size -= 1
                    return
            # Not found
            raise ValueError("x not in list")

        # Dense or value matches default: scan all positions
        for i in range(self._size):
            if self[i] == value:
                # Found it - remove at this index
                self._explicit.pop(i, None)
                self._shift_keys(i + 1, -1)
                self._size -= 1
                return

        # Not found
        raise ValueError("x not in list")

    def reverse(self) -> None:
        """Reverse the list in place.

        Remaps all explicit keys according to: new_key = size - 1 - old_key
        """
        if self._size <= 1:
            # Empty or single element - nothing to reverse
            return

        # Choose strategy based on density
        if self._is_dense():
            # Dense: swap pairs in place up to midpoint
            for i in range(self._size // 2):
                j = self._size - 1 - i

                # Check if positions are explicit (can't use .get() since None is a valid value)
                i_explicit = i in self._explicit
                j_explicit = j in self._explicit

                # Swap (handle all 4 cases: both explicit, one explicit, neither explicit)
                if i_explicit and j_explicit:
                    # Both explicit - swap values
                    self._explicit[i], self._explicit[j] = self._explicit[j], self._explicit[i]
                elif i_explicit:
                    # Only i is explicit - move to j, delete i
                    self._explicit[j] = self._explicit[i]
                    del self._explicit[i]
                elif j_explicit:
                    # Only j is explicit - move to i, delete j
                    self._explicit[i] = self._explicit[j]
                    del self._explicit[j]
                # else: neither explicit - no change needed
        else:
            # Sparse: remap all keys
            new_explicit: dict[int, T] = {}
            for old_key, value in self._explicit.items():
                new_key = self._size - 1 - old_key
                new_explicit[new_key] = value
            self._explicit = new_explicit

    def sort(self, *, key: Callable[[T], SupportsRichComparison] | None = None, reverse: bool = False) -> None:
        """Sort the list in place. Sorts only the explicit values without materializing defaults.

        Args:
            key: Optional key function for sorting
            reverse: If True, sort in descending order
        """
        if self._size <= 1:
            # Empty or single element - nothing to sort
            return

        if not self._explicit:
            # All defaults - nothing to sort
            return

        # Sort explicit values (sorted() works with generators)
        if key is None:
            sorted_explicit = sorted(self._explicit.values(), reverse=reverse)  # type: ignore[type-var]
        else:
            sorted_explicit = sorted(self._explicit.values(), key=key, reverse=reverse)

        num_defaults = self._size - len(self._explicit)

        if num_defaults == 0:
            # Fully explicit - rebuild dict with sorted values
            self._explicit = dict(enumerate(sorted_explicit))
            return

        # Mixed case: have both explicit values and defaults
        # Find where default belongs in sorted order using binary search
        default_position = self._find_default_position(sorted_explicit, self._default, key, reverse)

        # Rebuild _explicit dict with new positions
        new_explicit: dict[int, T] = {}
        new_idx = 0

        for i, val in enumerate(sorted_explicit):
            # Insert defaults before this explicit value if needed
            if i == default_position:
                new_idx += num_defaults  # Skip default positions

            new_explicit[new_idx] = val
            new_idx += 1

        self._explicit = new_explicit

    def _find_default_position(
        self,
        sorted_values: list[T],
        default: T | None,
        key: Callable[[T], SupportsRichComparison] | None,
        reverse: bool,
    ) -> int:
        """Find insertion position for default value in sorted explicit values using binary search.

        Args:
            sorted_values: List of sorted explicit values
            default: Default value to position
            key: Key function used for sorting (or None)
            reverse: Whether sort is reversed

        Returns:
            Index where defaults should be inserted (0 to len(sorted_values))
        """
        if not sorted_values:
            return 0

        # Compute default key once
        default_key = key(default) if key is not None else default  # type: ignore[arg-type]

        # Binary search for insertion position
        # Normal sort: find leftmost position where default <= value
        # Reverse sort: find leftmost position where default >= value
        left, right = 0, len(sorted_values)

        while left < right:
            mid = (left + right) // 2
            val_key = key(sorted_values[mid]) if key is not None else sorted_values[mid]

            # Comparison depends on reverse flag
            should_go_right = (default_key < val_key) if reverse else (default_key > val_key)  # type: ignore[operator]

            if should_go_right:
                left = mid + 1
            else:
                right = mid

        return left

    def unset(self, index: SupportsIndex) -> T | None:
        """Reset value at index to default by removing it from explicit values.

        Args:
            index: Index to reset to default

        Returns:
            The value that was at the index before unsetting

        Raises:
            IndexError: If index is out of range

        Note:
            After calling unset(i), self[i] will return the default value.
            This does not change the list size or shift any elements.
        """
        # Convert to int
        idx = op_index(index)

        # Normalize negative index
        if idx < 0:
            idx += self._size

        # Check bounds
        if not 0 <= idx < self._size:
            raise IndexError("list index out of range")

        # Get the current value before unsetting
        old_value = self._explicit.get(idx, self._default)

        # Remove from explicit dict if present (no-op if already default)
        self._explicit.pop(idx, None)

        return old_value

# tests/test_sparselist.py
import copy
import pickle
from itertools import product

import pytest

from sparselist import sparselist

# Sentinel for omitting parameters
OMIT = object()

# Common test data
input_variants = {
    "empty_none": (None, 0),
    "empty_list": ([], 0),
    "empty_dict": ({}, 0),
    "nonempty_list": ([1, 2, 3], 3),
    "sparse_dict": ({0: 1, 3: 4}, 4),
    "far_sparse_dict": ({0: 1, 5: 9}, 6),
}

default_variants = {
    "implicit_default": OMIT,
    "explicit_None": None,
    "explicit_0": 0,
    "explicit_abcd": "abcd",
    "explicit_list": [],
    "explicit_dict": {},
}


# ---------------------
# Helper functions
# ---------------------
def _get_sparselist_key_count(sl):
    """Get the count of explicitly stored keys in a sparselist."""
    return len(sl._explicit)


# ---------------------
# Generator functions
# ---------------------
def generate_valid_construction_cases():
    """Generate all valid construction parameter combinations."""
    cases = []
    for (input_label, (data, natural_size)), (default_label, default_value) in product(
        input_variants.items(), default_variants.items()
    ):
        for size_case in ("implicit_size", "same_size", "larger_size"):
            if size_case == "implicit_size":
                size_arg = OMIT
                size_val = natural_size
            elif size_case == "same_size":
                size_arg = natural_size
                size_val = natural_size
            else:  # larger
                size_arg = natural_size + 2
                size_val = natural_size + 2

            expected_list = (
                list(data)
                if isinstance(data, list)
                else [default_value if default_value is not OMIT else None] * size_val
            )
            if isinstance(data, dict):
                for idx, val in data.items():
                    expected_list[idx] = val
            if size_val > len(expected_list):
                expected_list.extend(
                    [(default_value if default_value is not OMIT else None)] * (size_val - len(expected_list))
                )

            cases.append(
                (
                    data,
                    default_value,
                    size_arg,
                    expected_list,
                    (default_value if default_value is not OMIT else None),
                    f"{input_label}__{default_label}__{size_case}",
                )
            )
    return cases


def generate_invalid_non_integer_size_cases():
    """Generate test cases for non-integer size parameters."""
    # Note: None, True, False, and floats are all valid (None means infer, bool is int, float has __int__)
    bad_sizes = ["string", [1, 2], {}, object()]
    cases = []
    for (input_label, (data, _)), (default_label, default_value), bad_size in product(
        input_variants.items(), default_variants.items(), bad_sizes
    ):
        cases.append((data, default_value, bad_size, f"{input_label}__{default_label}__nonint_size_{bad_size!r}"))
    return cases


def generate_invalid_negative_size_cases():
    """Generate test cases for negative size parameters."""
    bad_sizes = [-1, -10]
    cases = []
    for (input_label, (data, _)), (default_label, default_value), bad_size in product(
        input_variants.items(), default_variants.items(), bad_sizes
    ):
        cases.append((data, default_value, bad_size, f"{input_label}__{default_label}__neg_size_{bad_size}"))
    return cases


def generate_invalid_too_small_size_cases():
    """Generate test cases where size is smaller than required by data."""
    cases = []
    for (input_label, (data, natural_size)), (default_label, default_value) in product(
        input_variants.items(), default_variants.items()
    ):
        if natural_size == 0:
            continue
        too_small = natural_size - 1
        cases.append((data, default_value, too_small, f"{input_label}__{default_label}__size_too_small"))
    return cases


# ---------------------
# Construction tests
# ---------------------
@pytest.mark.parametrize(
    "data, default_arg, size_arg, expected_list, expected_default",
    [(d, dv, sz, el, ed) for d, dv, sz, el, ed, _ in generate_valid_construction_cases()],
    ids=[case_id for _, _, _, _, _, case_id in generate_valid_construction_cases()],
)
def test_construction_basic(data, default_arg, size_arg, expected_list, expected_default):
    """Test valid construction with various data, default, and size combinations."""
    kwargs = {}
    if default_arg is not OMIT:
        kwargs["default"] = default_arg
    if size_arg is not OMIT:
        kwargs["size"] = size_arg

    sl = sparselist(data, **kwargs)
    assert list(sl) == expected_list
    assert sl.size == len(expected_list)
    assert sl.default == expected_default


@pytest.mark.parametrize(
    "data, default_arg, size_arg",
    [(d, dv, sz) for d, dv, sz, _ in generate_invalid_non_integer_size_cases()],
    ids=[case_id for _, _, _, case_id in generate_invalid_non_integer_size_cases()],
)
def test_invalid_non_integer_size(data, default_arg, size_arg):
    """Test that non-integer size values raise TypeError."""
    kwargs = {}
    if default_arg is not OMIT:
        kwargs["default"] = default_arg
    kwargs["size"] = size_arg
    with pytest.raises(TypeError):
        sparselist(data, **kwargs)


@pytest.mark.parametrize(
    "data, default_arg, size_arg",
    [(d, dv, sz) for d, dv, sz, _ in generate_invalid_negative_size_cases()],
    ids=[case_id for _, _, _, case_id in generate_invalid_negative_size_cases()],
)
def test_invalid_negative_size(data, default_arg, size_arg):
    """Test that negative size values raise ValueError."""
    kwargs = {}
    if default_arg is not OMIT:
        kwargs["default"] = default_arg
    kwargs["size"] = size_arg
    with pytest.raises(ValueError):
        sparselist(data, **kwargs)


@pytest.mark.parametrize(
    "data, default_arg, size_arg",
    [(d, dv, sz) for d, dv, sz, _ in generate_invalid_too_small_size_cases()],
    ids=[case_id for _, _, _, case_id in generate_invalid_too_small_size_cases()],
)
def test_invalid_too_small_size(data, default_arg, size_arg):
    """Test that size smaller than required by data raises ValueError."""
    kwargs = {}
    if default_arg is not OMIT:
        kwargs["default"] = default_arg
    kwargs["size"] = size_arg
    with pytest.raises(ValueError):
        sparselist(data, **kwargs)


@pytest.mark.parametrize(
    "invalid_dict, size, default, expected_exception",
    [
        ({-1: "a", 0: "b"}, 2, None, ValueError),
        ({0: "a", -5: "b"}, 2, None, ValueError),
        ({-1: 1}, 1, 0, ValueError),
        ({1.5: "a", 2: "b"}, 3, None, TypeError),
        ({0: "a", 2.7: "b"}, 3, None, TypeError),
        ({0.0: "a"}, 1, None, TypeError),
        ({"a": 1, 0: 2}, 2, None, TypeError),
        ({0: 1, "key": 2}, 2, None, TypeError),
        ({"0": "value"}, 1, None, TypeError),
        ({None: "a", 0: "b"}, 2, None, TypeError),
        ({(0,): "tuple_key"}, 1, None, TypeError),
        ({object(): "obj_key"}, 1, None, TypeError),
    ],
    ids=[
        "negative_-1",
        "negative_-5",
        "negative_only",
        "float_1.5",
        "float_2.7",
        "float_0.0",
        "string_a",
        "string_key",
        "string_0",
        "none_key",
        "tuple_key",
        "object_key",
    ],
)
def test_dict_invalid_keys(invalid_dict, size, default, expected_exception):
    """Test that dicts with invalid key types raise appropriate errors."""
    with pytest.raises(expected_exception):
        sparselist(invalid_dict, size=size, default=default)


# ---------------------
# Type checking tests
# ---------------------
@pytest.mark.parametrize(
    "initial_data, size, default",
    [
        ([1, 2, 3], 3, 0),
        ({0: 1, 5: 6}, 10, 0),
        (None, 5, None),
        (None, 0, None),
    ],
    ids=["dense", "sparse", "all_defaults", "empty"],
)
def test_type_checking(initial_data, size, default):
    """Test that sparselist is an instance of both list and sparselist, and is unhashable."""
    sl = sparselist(initial_data, size=size, default=default)

    assert isinstance(sl, list)
    assert isinstance(sl, sparselist)

    with pytest.raises(TypeError):
        hash(sl)


# ---------------------
# Property tests
# ---------------------
@pytest.mark.parametrize(
    "initial_default", [None, 0, "abc", [], {}], ids=["None", "0", "abc", "empty_list", "empty_dict"]
)
def test_default_reassignment(initial_default):
    """Test that changing the default property updates unspecified positions."""
    sl = sparselist([1, 2, 3], default=initial_default, size=5)
    expected_before = [1, 2, 3, initial_default, initial_default]
    assert list(sl) == expected_before

    new_default = "NEW"
    sl.default = new_default  # type: ignore[assign]
    expected_after = [1, 2, 3, new_default, new_default]
    assert list(sl) == expected_after


@pytest.mark.parametrize(
    "mutable_default, mutate_fn, expected_after",
    [
        ([], lambda x: x.append(42), [1, 2, 3, [42], [42]]),
        ({}, lambda x: x.update({"k": "v"}), [1, 2, 3, {"k": "v"}, {"k": "v"}]),
    ],
    ids=["list_default", "dict_default"],
)
def test_default_inplace_mutation_affects_all_defaults(mutable_default, mutate_fn, expected_after):
    """Test that mutating a mutable default affects all unspecified positions."""
    sl = sparselist([1, 2, 3], default=mutable_default, size=5)
    assert list(sl) == [1, 2, 3, mutable_default, mutable_default]
    mutate_fn(sl.default)
    assert list(sl) == expected_after


@pytest.mark.parametrize(
    "initial_data, initial_size, default, new_size, expected_list, expected_key_count",
    [
        ({0: 1}, 1, 0, 4, [1, 0, 0, 0], 1),
        ({0: "a", 2: "c"}, 3, None, 6, ["a", None, "c", None, None, None], 2),
        (None, 2, 0, 5, [0, 0, 0, 0, 0], 0),
        ({0: "a", 4: "z"}, 6, None, 3, ["a", None, None], 1),
        ({0: 1, 5: 6, 9: 10}, 10, 0, 7, [1, 0, 0, 0, 0, 6, 0], 2),
        ([1, 2, 3, 4, 5], 5, 0, 3, [1, 2, 3], 3),
        ({0: 1}, 5, 0, 0, [], 0),
        ([1, 2, 3], 3, None, 0, [], 0),
        ({0: 1, 5: 6}, 10, 0, 10, [1, 0, 0, 0, 0, 6, 0, 0, 0, 0], 2),
        (None, 5, None, 5, [None, None, None, None, None], 0),
        ({0: "x", 2: "y"}, 5, None, 1, ["x"], 1),
        (None, 5, 0, 1, [0], 0),
    ],
    ids=[
        "increase_sparse",
        "increase_with_gaps",
        "increase_all_defaults",
        "decrease_drops_last",
        "decrease_truncates_sparse",
        "decrease_dense",
        "reduce_to_zero_sparse",
        "reduce_to_zero_dense",
        "same_size_sparse",
        "same_size_defaults",
        "size_to_1_sparse",
        "size_to_1_defaults",
    ],
)
def test_size_property_modification(initial_data, initial_size, default, new_size, expected_list, expected_key_count):
    """Test modifying the size property increases or decreases the list correctly."""
    sl = sparselist(initial_data, size=initial_size, default=default)
    sl.size = new_size

    assert sl.size == new_size
    assert len(sl) == new_size
    assert list(sl) == expected_list
    assert _get_sparselist_key_count(sl) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, initial_size, default, bad_size, expected_exception",
    [
        ({0: 1}, 5, 0, -1, ValueError),
        ({0: 1}, 5, 0, -10, ValueError),
        ({0: 1}, 5, 0, "five", (TypeError, AttributeError)),
        ({0: 1}, 5, 0, [5], (TypeError, AttributeError)),
        ({0: 1}, 5, 0, {5}, (TypeError, AttributeError)),
        ({0: 1}, 5, 0, object(), (TypeError, AttributeError)),
        (None, 10, 0, -1, ValueError),
        ([1, 2, 3], 3, None, "bad", (TypeError, AttributeError)),
    ],
    ids=[
        "negative_-1",
        "negative_-10",
        "string",
        "list",
        "set",
        "object",
        "empty_negative",
        "dense_string",
    ],
)
def test_size_property_invalid(initial_data, initial_size, default, bad_size, expected_exception):
    """Test that invalid size values raise appropriate errors and don't modify the list."""
    sl = sparselist(initial_data, size=initial_size, default=default)

    before_list = list(sl)
    before_size = sl.size
    before_key_count = _get_sparselist_key_count(sl)

    with pytest.raises(expected_exception):
        sl.size = bad_size

    assert list(sl) == before_list
    assert sl.size == before_size
    assert _get_sparselist_key_count(sl) == before_key_count


# ---------------------
# Pickle tests
# ---------------------
def generate_pickle_roundtrip_cases():
    """Generate pickle test cases with readable IDs for all protocol versions."""
    base_cases = [
        # Empty sparselist
        (None, 0, None, "empty_none"),
        (None, 0, "default", "empty_default"),
        # Simple sparse data
        ({0: "a", 2: "c"}, 5, None, "sparse_str_none"),
        ({0: "a", 2: "c"}, 5, "_", "sparse_str_default"),
        # Dense data
        ({0: 1, 1: 2, 2: 3, 3: 4}, 4, None, "dense_int"),
        # Large dense list
        (list(range(10000)), 10000, None, "large_dense"),
        # Large sparse data
        ({0: "first", 1000: "middle", 999999: "last"}, 1000000, None, "large_sparse"),
        # Various default values
        ({5: 100}, 10, 0, "default_zero"),
        ({5: 100}, 10, -1, "default_negative"),
        ({1: "x"}, 3, "", "default_empty_str"),
        # Mutable defaults - lists
        ({0: "a", 5: "b"}, 10, [], "default_empty_list"),
        ({2: [99]}, 5, [1, 2, 3], "default_list"),
        # Mutable defaults - dicts
        ({1: "x"}, 4, {}, "default_empty_dict"),
        ({0: {"a": 1}}, 3, {"default": True}, "default_dict"),
        # Different types
        ({0: [1, 2], 2: [3, 4]}, 5, None, "type_list"),
        ({0: {"key": "val"}}, 3, None, "type_dict"),
    ]

    cases = []
    for data, size, default, case_label in base_cases:
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            test_id = f"{case_label}__proto_{protocol}"
            cases.append((data, size, default, protocol, test_id))

    return cases


@pytest.mark.parametrize(
    "data,size,default,protocol",
    [(d, s, df, p) for d, s, df, p, _ in generate_pickle_roundtrip_cases()],
    ids=[test_id for _, _, _, _, test_id in generate_pickle_roundtrip_cases()],
)
def test_pickle_roundtrip(data, size, default, protocol):
    """Test that sparselist can be pickled and unpickled correctly across all protocol versions."""
    original = sparselist(data, size=size, default=default)
    original_key_count = _get_sparselist_key_count(original)

    # Pickle and unpickle
    pickled = pickle.dumps(original, protocol=protocol)
    restored = pickle.loads(pickled)

    # Verify equality
    assert restored == original
    assert restored.size == original.size
    assert restored.default == original.default

    # Verify sparsity is preserved
    assert _get_sparselist_key_count(restored) == original_key_count


# ---------------------
# Iterator tests
# ---------------------
def test_iterator_protocol():
    """Test that sparselist properly implements the iterator protocol."""
    sl = sparselist({0: "a", 2: "c", 5: "f"}, size=7, default="_")

    it = iter(sl)
    assert iter(it) is it

    assert next(it) == "a"
    assert next(it) == "_"
    assert next(it) == "c"
    assert next(it) == "_"
    assert next(it) == "_"
    assert next(it) == "f"
    assert next(it) == "_"

    with pytest.raises(StopIteration):
        next(it)


@pytest.mark.parametrize(
    "initial_data, size, default, expected_sequence",
    [
        ({0: 1, 5: 6}, 10, 0, [1, 0, 0, 0, 0, 6, 0, 0, 0, 0]),
        ([1, 2, 3], 3, None, [1, 2, 3]),
        (None, 5, "x", ["x", "x", "x", "x", "x"]),
        (None, 0, None, []),
        ({0: "only"}, 1, None, ["only"]),
    ],
    ids=["sparse", "dense", "all_defaults", "empty", "single_element"],
)
def test_iterator_yields_all_elements(initial_data, size, default, expected_sequence):
    """Test that iteration yields exactly the right elements in order."""
    sl = sparselist(initial_data, size=size, default=default)

    result = []
    for item in sl:
        result.append(item)

    assert result == expected_sequence


def test_multiple_iterators_independent():
    """Test that multiple iterators over the same sparselist are independent."""
    sl = sparselist({0: "a", 2: "c"}, size=4, default="_")

    it1 = iter(sl)
    it2 = iter(sl)

    assert next(it1) == "a"
    assert next(it1) == "_"

    assert next(it2) == "a"

    assert next(it1) == "c"


@pytest.mark.parametrize(
    "modification_type",
    ["modify_value", "append", "insert_before", "insert_after", "delete_before", "delete_after"],
    ids=["modify_value", "append", "insert_before", "insert_after", "delete_before", "delete_after"],
)
def test_iterator_modification_matches_list(modification_type):
    """Test that sparselist iterator behaves exactly like list iterator when modified."""
    regular_list = [1, 2, 3, 4, 5]
    sparse_list = sparselist([1, 2, 3, 4, 5], default=0)

    list_it = iter(regular_list)
    sparse_it = iter(sparse_list)

    next(list_it)
    next(sparse_it)
    next(list_it)
    next(sparse_it)

    if modification_type == "modify_value":
        regular_list[3] = 99
        sparse_list[3] = 99
    elif modification_type == "append":
        regular_list.append(6)
        sparse_list.append(6)
    elif modification_type == "insert_before":
        regular_list.insert(1, 99)
        sparse_list.insert(1, 99)
    elif modification_type == "insert_after":
        regular_list.insert(3, 99)
        sparse_list.insert(3, 99)
    elif modification_type == "delete_before":
        del regular_list[0]
        del sparse_list[0]
    elif modification_type == "delete_after":
        del regular_list[4]
        del sparse_list[4]

    list_remaining = []
    sparse_remaining = []

    try:
        while True:
            list_remaining.append(next(list_it))
    except StopIteration:
        pass

    try:
        while True:
            sparse_remaining.append(next(sparse_it))
    except StopIteration:
        pass

    assert sparse_remaining == list_remaining


def test_iterator_size_modification():
    """Test iterator behavior when size property is modified."""
    sl = sparselist({0: 1, 4: 5}, size=5, default=0)
    regular_list = [1, 0, 0, 0, 5]

    sl_it = iter(sl)
    list_it = iter(regular_list)

    assert next(sl_it) == next(list_it)
    assert next(sl_it) == next(list_it)

    sl.size = 7
    regular_list.extend([0, 0])

    sl_remaining = list(sl_it)
    list_remaining = list(list_it)

    assert sl_remaining == list_remaining

    sl2 = sparselist({0: 1, 4: 5}, size=5, default=0)
    regular_list2 = [1, 0, 0, 0, 5]

    sl2_it = iter(sl2)
    list2_it = iter(regular_list2)

    assert next(sl2_it) == next(list2_it)
    assert next(sl2_it) == next(list2_it)

    sl2.size = 3
    del regular_list2[3:]

    sl2_remaining = list(sl2_it)
    list2_remaining = list(list2_it)

    assert sl2_remaining == list2_remaining

    sl3 = sparselist({0: 1, 2: 3, 4: 5}, size=5, default=0)
    regular_list3 = [1, 0, 3, 0, 5]

    sl3_it = iter(sl3)
    list3_it = iter(regular_list3)

    assert next(sl3_it) == next(list3_it)
    assert next(sl3_it) == next(list3_it)
    assert next(sl3_it) == next(list3_it)

    sl3.size = 2
    del regular_list3[2:]

    sl3_remaining = []
    list3_remaining = []

    try:
        while True:
            sl3_remaining.append(next(sl3_it))
    except StopIteration:
        pass

    try:
        while True:
            list3_remaining.append(next(list3_it))
    except StopIteration:
        pass

    assert sl3_remaining == list3_remaining


# ---------------------
# Deletion tests
# ---------------------
def generate_indexing_cases():
    """Generate test cases for valid indexing."""
    size = 10
    explicit_values = {0: "a", 3: "b", 9: "c"}
    default = 0
    full_list = [explicit_values.get(i, default) for i in range(size)]

    cases = []
    for i in [0, 3, 5, 9]:
        cases.append((explicit_values, size, default, i, full_list[i], f"pos_idx_{i}"))
    for i in [-1, -3, -5, -10]:
        cases.append((explicit_values, size, default, i, full_list[i], f"neg_idx_{i}"))
    return cases


def generate_invalid_indexing_cases():
    """Generate test cases for invalid indexing."""
    size = 5
    explicit_values = {0: 1, 2: 3}
    default = 0

    cases = []
    for idx in [5, 10, -6, -10]:
        cases.append((explicit_values, size, default, idx, f"idx_{idx}"))
    return cases


@pytest.mark.parametrize(
    "initial_data, size, default, index, expected_list, expected_size, expected_key_count, should_raise",
    [
        ({0: 1, 2: 3, 4: 5}, 5, 0, 2, [1, 0, 0, 5], 4, 2, False),
        ({0: "a", 4: "z"}, 5, None, 0, [None, None, None, "z"], 4, 1, False),
        ([1, 2, 3], 3, 0, 1, [1, 3], 2, 2, False),
        ({0: 1, 4: 5}, 5, 0, 4, [1, 0, 0, 0], 4, 1, False),
        ({0: 1, 4: 5}, 5, 0, 1, [1, 0, 0, 5], 4, 2, False),
        ([1, 2, 3, 4], 4, 0, -2, [1, 2, 4], 3, 3, False),
        ({0: 1, 4: 5}, 5, 0, -1, [1, 0, 0, 0], 4, 1, False),
        ({0: 1, 4: 5}, 5, 0, -5, [0, 0, 0, 5], 4, 1, False),
        ({0: "x"}, 1, None, 0, [], 0, 0, False),
        ([1], 1, 0, -1, [], 0, 0, False),
        ({0: 1, 2: 3}, 4, 0, 10, None, None, None, True),
        ({0: 1, 2: 3}, 4, 0, -10, None, None, None, True),
        ({0: 1, 2: 3}, 4, 0, 4, None, None, None, True),
        (None, 0, None, 0, None, None, None, True),
    ],
    ids=[
        "del_explicit_at_2",
        "del_explicit_at_0",
        "del_dense_middle",
        "del_explicit_at_4",
        "del_default",
        "del_negative_at_-2",
        "del_negative_last",
        "del_negative_first",
        "del_only_element",
        "del_only_element_negative",
        "del_out_of_bounds_pos",
        "del_out_of_bounds_neg",
        "del_at_size",
        "del_from_empty",
    ],
)
def test_delete_by_index(
    initial_data, size, default, index, expected_list, expected_size, expected_key_count, should_raise
):
    """Test deleting elements by index (positive and negative)."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        with pytest.raises(IndexError):
            del sl[index]

        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        del sl[index]
        assert list(sl) == expected_list
        assert sl.size == expected_size
        assert _get_sparselist_key_count(sl) == expected_key_count


def generate_delete_slice_cases():
    """Generate comprehensive slice deletion test cases."""

    def _calculate_key_count_after_slice_deletion(explicit_dict, size, start, stop, step):
        """Calculate how many explicit keys remain after deleting a slice."""
        slice_obj = slice(start, stop, step)
        indices_to_delete = set(range(size)[slice_obj])

        remaining_keys = 0
        for key in explicit_dict:
            if key not in indices_to_delete:
                remaining_keys += 1

        return remaining_keys

    configs = [
        ({0: 1, 1: 2, 3: 3, 5: 4}, 10, 0, "sparse"),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, None, "dense"),
        (None, 10, 0, "all_defaults"),
        (None, 0, None, "empty"),
    ]

    start_stop_values = [
        None,
        0,
        5,
        9,
        11,
        -1,
        -5,
        -10,
        -11,
    ]
    step_values = [None, 1, 2, 5, 9, 11, -1, -2, -5, -9, -11, 0]

    for explicit_values, size, default, config_name in configs:
        if isinstance(explicit_values, dict):
            full_list = [explicit_values.get(i, default) for i in range(size)]
            explicit_dict = explicit_values
        elif explicit_values is not None:
            full_list = list(explicit_values)
            explicit_dict = dict(enumerate(explicit_values))
        else:
            # None means empty sparselist - fill with defaults based on size
            full_list = [default] * size
            explicit_dict = {}

        for start, stop, step in product(start_stop_values, start_stop_values, step_values):
            if step == 0:
                yield (
                    explicit_values,
                    size,
                    default,
                    start,
                    stop,
                    step,
                    None,
                    None,
                    None,
                    ValueError,
                    f"{config_name}_start={start}_stop={stop}_step=0",
                )
            else:
                try:
                    test_list = full_list[:]
                    del test_list[start:stop:step]

                    if config_name in {"empty", "all_defaults"}:
                        expected_key_count = 0
                    elif config_name == "dense":
                        expected_key_count = len(test_list)
                    else:
                        expected_key_count = _calculate_key_count_after_slice_deletion(
                            explicit_dict, size, start, stop, step
                        )

                    yield (
                        explicit_values,
                        size,
                        default,
                        start,
                        stop,
                        step,
                        test_list,
                        len(test_list),
                        expected_key_count,
                        None,
                        f"{config_name}_start={start}_stop={stop}_step={step}",
                    )
                except Exception as e:
                    yield (
                        explicit_values,
                        size,
                        default,
                        start,
                        stop,
                        step,
                        None,
                        None,
                        None,
                        type(e),
                        f"{config_name}_start={start}_stop={stop}_step={step}_err",
                    )


@pytest.mark.parametrize(
    (
        "explicit_values",
        "size",
        "default",
        "start",
        "stop",
        "step",
        "expected_list",
        "expected_size",
        "expected_key_count",
        "expected_exception",
    ),
    [case[:-1] for case in generate_delete_slice_cases()],
    ids=[case[-1] for case in generate_delete_slice_cases()],
)
def test_delete_by_slice(
    explicit_values,
    size,
    default,
    start,
    stop,
    step,
    expected_list,
    expected_size,
    expected_key_count,
    expected_exception,
):
    """Test deleting elements by slice with various start/stop/step combinations."""
    sl = sparselist(explicit_values, size=size, default=default)

    if expected_exception:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        with pytest.raises(expected_exception):
            del sl[start:stop:step]

        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        del sl[start:stop:step]
        assert list(sl) == expected_list
        assert sl.size == expected_size
        if expected_key_count is not None:
            assert _get_sparselist_key_count(sl) == expected_key_count


# ---------------------
# Indexing tests
# ---------------------
@pytest.mark.parametrize(
    "explicit_values, size, default, index, expected_value",
    [(ev, sz, df, idx, exp) for ev, sz, df, idx, exp, _ in generate_indexing_cases()],
    ids=[case_id for _, _, _, _, _, case_id in generate_indexing_cases()],
)
def test_indexing_valid(explicit_values, size, default, index, expected_value):
    """Test that valid indexing returns correct values."""
    sl = sparselist(explicit_values, size=size, default=default)
    assert sl[index] == expected_value


@pytest.mark.parametrize(
    "explicit_values, size, default, index",
    [(ev, sz, df, idx) for ev, sz, df, idx, _ in generate_invalid_indexing_cases()],
    ids=[case_id for _, _, _, _, case_id in generate_invalid_indexing_cases()],
)
def test_indexing_invalid(explicit_values, size, default, index):
    """Test that out-of-bounds indexing raises IndexError."""
    sl = sparselist(explicit_values, size=size, default=default)
    with pytest.raises(IndexError):
        _ = sl[index]


@pytest.mark.parametrize(
    "initial_data, size, default, index, new_value",
    [
        ({0: "a", 2: "b"}, 4, None, 0, "A"),
        ({0: "a", 2: "b"}, 4, None, 1, "X"),
        ({0: "a", 2: "b"}, 4, None, -1, "Z"),
    ],
    ids=["assign_to_explicit", "assign_to_default", "assign_negative_index"],
)
def test_index_assignment_valid(initial_data, size, default, index, new_value):
    """Test that assigning to valid indices works correctly."""
    sl = sparselist(initial_data, size=size, default=default)
    sl[index] = new_value
    assert sl[index] == new_value


@pytest.mark.parametrize("bad_index", [4, -5], ids=["pos_oob", "neg_oob"])
def test_index_assignment_invalid(bad_index):
    """Test that assigning to out-of-bounds indices raises IndexError."""
    sl = sparselist({0: "a", 2: "b"}, size=4, default=None)
    with pytest.raises(IndexError):
        sl[bad_index] = "oops"


# ---------------------
# Slicing tests
# ---------------------
def generate_slice_read_cases():
    """Generate comprehensive slice read test cases."""
    size = 10
    explicit_values = {0: 1, 1: 2, 3: 3, 5: 4}
    default_value = 0
    full_list = [explicit_values.get(i, default_value) for i in range(size)]

    start_stop_values = [
        None,
        0,
        size // 2,
        size - 1,
        size + 1,
        -1,
        -(size // 2),
        -size,
        -(size + 1),
    ]
    step_values = [None, 1, 2, size // 2, size - 1, size + 1, -1, -2, -(size // 2), -(size - 1), -(size + 1), 0]

    for start, stop, step in product(start_stop_values, start_stop_values, step_values):
        case_id = f"start={start}_stop={stop}_step={step}"
        if step == 0:
            yield explicit_values, size, default_value, start, stop, step, [], ValueError, case_id
        else:
            try:
                expected = full_list[start:stop:step]
                yield explicit_values, size, default_value, start, stop, step, expected, None, case_id
            except Exception as e:
                yield explicit_values, size, default_value, start, stop, step, [], type(e), case_id


@pytest.mark.parametrize(
    "explicit_values, size, default_value, start, stop, step, expected_result, expected_exception",
    [case[:-1] for case in generate_slice_read_cases()],
    ids=[case[-1] for case in generate_slice_read_cases()],
)
def test_slice_read(explicit_values, size, default_value, start, stop, step, expected_result, expected_exception):
    """Test reading slices with various start/stop/step combinations."""
    sl = sparselist(explicit_values, size=size, default=default_value)

    if expected_exception:
        with pytest.raises(expected_exception):
            _ = sl[start:stop:step]
    else:
        result = sl[start:stop:step]
        assert isinstance(result, sparselist)
        assert result.default == default_value
        assert list(result) == expected_result


def generate_slice_equal_assignment_cases():
    """Generate slice assignment cases with equal-length replacements."""
    for (
        explicit_values,
        size,
        default_value,
        start,
        stop,
        step,
        expected_result,
        expected_exception,
        case_id,
    ) in generate_slice_read_cases():
        case_id = f"assign_{case_id}"  # noqa: PLW2901
        if expected_exception:
            yield (
                explicit_values,
                size,
                default_value,
                start,
                stop,
                step,
                [],
                expected_result,
                expected_exception,
                case_id,
            )
        else:
            replacement = ["X"] * len(expected_result)
            full_list = [explicit_values.get(i, default_value) for i in range(size)]
            expected_after = full_list[:]
            expected_after[start:stop:step] = replacement  # type: ignore[assign]
            yield explicit_values, size, default_value, start, stop, step, replacement, expected_after, None, case_id


@pytest.mark.parametrize(
    "explicit_values, size, default_value, start, stop, step, replacement, expected_result, expected_exception",
    [case[:-1] for case in generate_slice_equal_assignment_cases()],
    ids=[case[-1] for case in generate_slice_equal_assignment_cases()],
)
def test_slice_assignment_equal_length(
    explicit_values, size, default_value, start, stop, step, replacement, expected_result, expected_exception
):
    """Test slice assignment with equal-length replacements."""
    sl = sparselist(explicit_values, size=size, default=default_value)

    if expected_exception:
        with pytest.raises(expected_exception):
            sl[start:stop:step] = ["X"] * 3
    else:
        replacement = ["X"] * len(sl[start:stop:step])
        sl[start:stop:step] = replacement
        assert list(sl) == expected_result


def generate_slice_assignment_length_mismatch_cases():
    """Generate slice assignment cases with varying replacement lengths."""
    size = 10
    explicit_values = {0: 1, 1: 2, 3: 3, 5: 4}
    default_value = 0
    full_list = [explicit_values.get(i, default_value) for i in range(size)]

    start_stop_values = [
        None,
        0,
        size // 2,
        size - 1,
        size + 1,
        -1,
        -(size // 2),
        -size,
        -(size + 1),
    ]
    step_values = [None, 1, 2, size // 2, size - 1, size + 1, -1, -2, -(size // 2), -(size - 1), -(size + 1), 0]

    replacement_strategies = [
        ("shorter", lambda slice_len: max(0, slice_len - 2)),
        ("longer", lambda slice_len: slice_len + 2),
        ("empty", lambda _: 0),
    ]

    for start, stop, step in product(start_stop_values, start_stop_values, step_values):
        for strategy_name, replacement_len_fn in replacement_strategies:
            if step == 0:
                yield (
                    explicit_values,
                    size,
                    default_value,
                    start,
                    stop,
                    step,
                    [],
                    ValueError,
                    f"{strategy_name}_step0",
                )
            else:
                try:
                    slice_obj = slice(start, stop, step)
                    test_list = full_list[:]
                    slice_len = len(test_list[slice_obj])
                    replacement_len = replacement_len_fn(slice_len)
                    replacement = ["X"] * replacement_len

                    if step is not None and step != 1 and replacement_len != slice_len:
                        yield (
                            explicit_values,
                            size,
                            default_value,
                            start,
                            stop,
                            step,
                            replacement,
                            ValueError,
                            f"{strategy_name}_start={start}_stop={stop}_step={step}",
                        )
                        continue

                    test_list[slice_obj] = replacement
                    yield (
                        explicit_values,
                        size,
                        default_value,
                        start,
                        stop,
                        step,
                        replacement,
                        None,
                        f"{strategy_name}_start={start}_stop={stop}_step={step}",
                    )

                except Exception as e:
                    yield (
                        explicit_values,
                        size,
                        default_value,
                        start,
                        stop,
                        step,
                        [],
                        type(e),
                        f"{strategy_name}_start={start}_stop={stop}_step={step}_err",
                    )


@pytest.mark.parametrize(
    "explicit_values, size, default_value, start, stop, step, replacement, expected_exception",
    [case[:-1] for case in generate_slice_assignment_length_mismatch_cases()],
    ids=[case[-1] for case in generate_slice_assignment_length_mismatch_cases()],
)
def test_slice_assignment_length_mismatch(
    explicit_values, size, default_value, start, stop, step, replacement, expected_exception
):
    """Test slice assignment with varying replacement lengths."""
    sl = sparselist(explicit_values, size=size, default=default_value)

    if expected_exception:
        with pytest.raises(expected_exception):
            sl[start:stop:step] = replacement
    else:
        full_list = [explicit_values.get(i, default_value) for i in range(size)]
        full_list[start:stop:step] = replacement

        sl[start:stop:step] = replacement
        assert list(sl) == full_list


# ---------------------
# Comparison operator tests
# ---------------------
def generate_comparison_cases():
    """Generate comprehensive comparison test cases."""

    def op_to_name(op):
        """Convert operator to readable name."""
        return {
            "==": "eq",
            "!=": "ne",
            "<": "lt",
            ">": "gt",
            "<=": "le",
            ">=": "ge",
        }[op]

    value_pairs = [
        ([1, 2, 3], 3, 0, [1, 2, 3], 3, 0, "equal_content"),
        ({0: 1, 2: 3}, 3, 0, [1, 0, 3], 3, 0, "sparse_vs_dense_equal"),
        ([1, 2, 3], 3, 0, [1, 2, 3], 3, None, "equal_diff_defaults"),
        ([1, 2, 3], 3, 0, [1, 2, 4], 3, 0, "first_less"),
        ([1, 2], 2, 0, [1, 2, 3], 3, 0, "first_shorter"),
        ({0: 1, 2: 3}, 4, 0, {0: 1, 2: 3}, 4, 5, "first_less_defaults"),
        ([1, 2, 4], 3, 0, [1, 2, 3], 3, 0, "first_greater"),
        ([1, 2, 3], 3, 0, [1, 2], 2, 0, "first_longer"),
        ({0: 1, 2: 3}, 4, 5, {0: 1, 2: 3}, 4, 0, "first_greater_defaults"),
        ([], 0, 0, [], 0, 0, "both_empty"),
        ([], 0, 0, [1], 1, 0, "first_empty"),
        ([1], 1, 0, [], 0, 0, "second_empty"),
        # Large sparse test cases to verify non-materialization
        ({0: 1, 999: 1000}, 1000, 0, {0: 1, 999: 1000}, 1000, 0, "large_sparse_equal"),
        ({0: 1, 600: 2}, 1000, 0, {0: 1, 600: 3}, 1000, 0, "large_sparse_less"),
    ]

    expectations = {
        "equal_content": {"==": True, "!=": False, "<": False, ">": False, "<=": True, ">=": True},
        "sparse_vs_dense_equal": {"==": True, "!=": False, "<": False, ">": False, "<=": True, ">=": True},
        "equal_diff_defaults": {"==": True, "!=": False, "<": False, ">": False, "<=": True, ">=": True},
        "first_less": {"==": False, "!=": True, "<": True, ">": False, "<=": True, ">=": False},
        "first_shorter": {"==": False, "!=": True, "<": True, ">": False, "<=": True, ">=": False},
        "first_less_defaults": {"==": False, "!=": True, "<": True, ">": False, "<=": True, ">=": False},
        "first_greater": {"==": False, "!=": True, "<": False, ">": True, "<=": False, ">=": True},
        "first_longer": {"==": False, "!=": True, "<": False, ">": True, "<=": False, ">=": True},
        "first_greater_defaults": {"==": False, "!=": True, "<": False, ">": True, "<=": False, ">=": True},
        "both_empty": {"==": True, "!=": False, "<": False, ">": False, "<=": True, ">=": True},
        "first_empty": {"==": False, "!=": True, "<": True, ">": False, "<=": True, ">=": False},
        "second_empty": {"==": False, "!=": True, "<": False, ">": True, "<=": False, ">=": True},
        "large_sparse_equal": {"==": True, "!=": False, "<": False, ">": False, "<=": True, ">=": True},
        "large_sparse_less": {"==": False, "!=": True, "<": True, ">": False, "<=": True, ">=": False},
    }

    operators = ["==", "!=", "<", ">", "<=", ">="]

    type_combos = [
        (False, False, "sl_sl"),
        (False, True, "sl_list"),
        (True, False, "list_sl"),
    ]

    for data1, size1, default1, data2, size2, default2, desc in value_pairs:
        expected_ops = expectations[desc]

        for use_list1, use_list2, type_desc in type_combos:
            for op in operators:
                expected = expected_ops[op]
                case_id = f"{desc}__{type_desc}__{op_to_name(op)}"
                yield (data1, size1, default1, data2, size2, default2, use_list1, use_list2, op, expected, case_id)


@pytest.mark.parametrize(
    "data1, size1, default1, data2, size2, default2, use_list1, use_list2, op, expected",
    [
        (d1, s1, df1, d2, s2, df2, ul1, ul2, o, exp)
        for d1, s1, df1, d2, s2, df2, ul1, ul2, o, exp, _ in generate_comparison_cases()
    ],
    ids=[case_id for *_, case_id in generate_comparison_cases()],
)
def test_comparison_operators(data1, size1, default1, data2, size2, default2, use_list1, use_list2, op, expected):
    """Test all comparison operators with sparselists and lists without materializing defaults."""
    if use_list1:
        op1 = (
            data1
            if isinstance(data1, list) and len(data1) == size1
            else list(sparselist(data1, size=size1, default=default1))
        )
    else:
        op1 = sparselist(data1, size=size1, default=default1)

    if use_list2:
        op2 = (
            data2
            if isinstance(data2, list) and len(data2) == size2
            else list(sparselist(data2, size=size2, default=default2))
        )
    else:
        op2 = sparselist(data2, size=size2, default=default2)

    # Capture initial key counts for sparselists (comparisons should not materialize defaults)
    key_count1_before = _get_sparselist_key_count(op1) if isinstance(op1, sparselist) else None
    key_count2_before = _get_sparselist_key_count(op2) if isinstance(op2, sparselist) else None

    if op == "==":
        assert (op1 == op2) == expected
    elif op == "!=":
        assert (op1 != op2) == expected
    elif op == "<":
        assert (op1 < op2) == expected
    elif op == ">":
        assert (op1 > op2) == expected
    elif op == "<=":
        assert (op1 <= op2) == expected
    elif op == ">=":
        assert (op1 >= op2) == expected

    # Verify that comparison didn't materialize unspecified elements
    if isinstance(op1, sparselist):
        assert _get_sparselist_key_count(op1) == key_count1_before
    if isinstance(op2, sparselist):
        assert _get_sparselist_key_count(op2) == key_count2_before


# ---------------------
# Arithmetic operator tests
# ---------------------
@pytest.mark.parametrize(
    "initial_data, size, default, value, expected",
    [
        ({0: "a", 2: "c"}, 5, None, "a", True),
        ({0: "a", 2: "c"}, 5, None, "c", True),
        ({0: "a", 2: "c"}, 5, None, "z", False),
        ([1, 2, 3], 3, 0, 2, True),
        ([1, 2, 3], 3, 0, 5, False),
        ({0: "a", 2: "c"}, 5, 0, 0, True),
        ([1, 2, 3], 3, 0, 0, False),
        ({0: 0, 2: 0}, 5, 0, 0, True),
        (None, 5, None, None, True),
        (None, 0, 0, 0, False),
        ({0: 1, 5: 6}, 10, 0, 0, True),
        ({0: None}, 3, None, None, True),
    ],
    ids=[
        "explicit_a",
        "explicit_c",
        "not_present",
        "dense_present",
        "dense_not_present",
        "default_present",
        "default_all_explicit",
        "explicit_same_as_default",
        "all_defaults",
        "empty_list",
        "sparse_with_defaults",
        "explicit_none_is_default",
    ],
)
def test_in_operator(initial_data, size, default, value, expected):
    """Test the 'in' operator for explicit and default values."""
    sl = sparselist(initial_data, size=size, default=default)
    assert (value in sl) == expected


@pytest.mark.parametrize(
    "data1, size1, default1, data2, size2, default2, expected_list, expected_key_count",
    [
        ({0: 1, 2: 3}, 4, 0, {0: 5, 1: 6}, 3, 0, [1, 0, 3, 0, 5, 6, 0], 4),
        ({0: 1}, 3, 0, {0: 5}, 2, None, [1, 0, 0, 5, None], 3),
        ({0: 1}, 3, 0, [10, 20], None, None, [1, 0, 0, 10, 20], 3),
        ([10, 20], None, None, {0: 1}, 3, 0, [10, 20, 1, 0, 0], None),
        (None, 0, 0, {0: 1}, 2, 0, [1, 0], 1),
        ({0: 1}, 2, 0, None, 0, 0, [1, 0], 1),
        (None, 0, None, None, 0, None, [], 0),
        ([1, 2], 2, 0, [3, 4], 2, 0, [1, 2, 3, 4], 4),
        ({0: "a", 9: "z"}, 10, "_", {0: "A"}, 5, "-", ["a"] + ["_"] * 8 + ["z", "A"] + ["-"] * 4, 7),
    ],
    ids=[
        "same_default",
        "diff_default",
        "sparselist_plus_list",
        "list_plus_sparselist",
        "empty_left",
        "empty_right",
        "both_empty",
        "dense_lists",
        "large_sparse",
    ],
)
def test_add_operator(data1, size1, default1, data2, size2, default2, expected_list, expected_key_count):
    """Test addition operator with various operand combinations."""
    op1 = data1 if isinstance(data1, list) and size1 is None else sparselist(data1, size=size1, default=default1)
    op2 = data2 if isinstance(data2, list) and size2 is None else sparselist(data2, size=size2, default=default2)

    result = op1 + op2

    if isinstance(op1, list) and not isinstance(op1, sparselist):
        assert isinstance(result, list)
        assert not isinstance(result, sparselist)
        assert result == expected_list
    else:
        assert isinstance(result, sparselist)
        assert list(result) == expected_list
        assert result.default == default1
        if expected_key_count is not None:
            assert _get_sparselist_key_count(result) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, multiplier, expected_list, expected_key_count, should_raise",
    [
        ({0: 1, 2: 3}, 3, 0, 2, [1, 0, 3, 1, 0, 3], 4, False),
        ({0: 1, 2: 3}, 3, 0, 0, [], 0, False),
        ({0: 1, 2: 3}, 3, 0, 1, [1, 0, 3], 2, False),
        ({0: 1}, 2, 0, 3, [1, 0, 1, 0, 1, 0], 3, False),
        (None, 3, 0, 2, [0, 0, 0, 0, 0, 0], 0, False),
        ({0: 1, 2: 3}, 3, 0, -1, [], 0, False),
        ({0: 1, 2: 3}, 3, 0, -5, [], 0, False),
        ({0: 1, 5: 6}, 10, 0, 2, [1, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0, 0], 4, False),
        # Large dense test
        (
            {i: i for i in range(15000)},
            20000,
            0,
            2,
            list(range(15000)) + [0] * 5000 + list(range(15000)) + [0] * 5000,
            30000,
            False,
        ),
        ({0: 1, 2: 3}, 3, 0, 2.5, None, None, True),
        ({0: 1, 2: 3}, 3, 0, "3", None, None, True),
        ({0: 1, 2: 3}, 3, 0, [3], None, None, True),
        ({0: 1, 2: 3}, 3, 0, None, None, None, True),
    ],
    ids=[
        "2",
        "0",
        "1",
        "3",
        "defaults_by_2",
        "neg_1",
        "neg_5",
        "sparse",
        "dense_list",
        "float_error",
        "string_error",
        "list_error",
        "none_error",
    ],
)
def test_multiply_operator(initial_data, size, default, multiplier, expected_list, expected_key_count, should_raise):
    """Test multiplication operator (both sl * n and n * sl)."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        with pytest.raises(TypeError):
            _ = sl * multiplier
        with pytest.raises(TypeError):
            _ = multiplier * sl
    else:
        result = sl * multiplier
        assert isinstance(result, sparselist)
        assert list(result) == expected_list
        assert result.default == default
        assert _get_sparselist_key_count(result) == expected_key_count

        result_r = multiplier * sl
        assert isinstance(result_r, sparselist)
        assert list(result_r) == expected_list
        assert result_r.default == default
        assert _get_sparselist_key_count(result_r) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, operand, expected_result, expected_key_count, should_raise",
    [
        ({0: 1}, 2, 0, [10, 20], [1, 0, 10, 20], 3, False),
        ({0: 1}, 2, 0, [], [1, 0], 1, False),
        (None, 0, None, [1, 2], [1, 2], 2, False),
        ({0: 1}, 2, 0, sparselist({0: 10, 2: 30}, size=3, default=None), [1, 0, 10, None, 30], 4, False),
        ({0: 1}, 2, 0, sparselist([0, 0], default=0), [1, 0, 0, 0], 3, False),
        (None, 5, 0, sparselist({1: 10}, size=3, default=0), [0, 0, 0, 0, 0, 0, 10, 0], 1, False),
        ({0: 1}, 2, 0, (10, 20, 30), [1, 0, 10, 20, 30], 4, False),
        ({0: 1}, 2, 0, range(3, 6), [1, 0, 3, 4, 5], 4, False),
        ({0: 1}, 2, 0, (x * 2 for x in range(3)), [1, 0, 0, 2, 4], 4, False),
        ({0: 1}, 2, 0, "abc", [1, 0, "a", "b", "c"], 4, False),
        ({0: 1}, 2, 0, 123, None, None, True),
        ({0: 1}, 2, 0, None, None, None, True),
        ({0: 1}, 2, 0, True, None, None, True),
    ],
    ids=[
        "list",
        "empty_list",
        "list_to_empty",
        "sl",
        "sl_defaults",
        "sl_sparse",
        "tuple",
        "range",
        "generator",
        "string",
        "int_error",
        "none_error",
        "bool_error",
    ],
)
def test_iadd_operator(initial_data, size, default, operand, expected_result, expected_key_count, should_raise):
    """Test in-place addition operator (+=)."""
    obj = sparselist(initial_data, size=size, default=default)
    original_id = id(obj)

    if should_raise:
        before_data = list(obj)
        before_size = obj.size
        before_key_count = _get_sparselist_key_count(obj)

        with pytest.raises(TypeError):
            obj += operand

        assert id(obj) == original_id
        assert list(obj) == before_data
        assert obj.size == before_size
        assert _get_sparselist_key_count(obj) == before_key_count
    else:
        obj += operand

        assert id(obj) == original_id
        assert isinstance(obj, sparselist)
        assert list(obj) == expected_result
        assert _get_sparselist_key_count(obj) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, multiplier, expected_list, expected_key_count, should_raise",
    [
        ({0: 1, 2: 3}, 3, 0, 2, [1, 0, 3, 1, 0, 3], 4, False),
        ({0: 1, 2: 3}, 3, 0, 0, [], 0, False),
        ({0: 1, 2: 3}, 3, 0, 1, [1, 0, 3], 2, False),
        ({0: 1}, 2, 0, 3, [1, 0, 1, 0, 1, 0], 3, False),
        (None, 3, 0, 2, [0, 0, 0, 0, 0, 0], 0, False),
        ({0: 1, 2: 3}, 3, 0, -1, [], 0, False),
        ({0: 1, 2: 3}, 3, 0, -5, [], 0, False),
        ({0: 1, 5: 6}, 10, 0, 2, [1, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0, 0], 4, False),
        ({0: 1, 2: 3}, 3, 0, 2.5, None, None, True),
        ({0: 1, 2: 3}, 3, 0, "3", None, None, True),
        ({0: 1, 2: 3}, 3, 0, [3], None, None, True),
        ({0: 1, 2: 3}, 3, 0, None, None, None, True),
    ],
    ids=[
        "2",
        "0",
        "1",
        "3",
        "defaults",
        "neg_1",
        "neg_5",
        "sparse",
        "float_error",
        "string_error",
        "list_error",
        "none_error",
    ],
)
def test_imul_operator(initial_data, size, default, multiplier, expected_list, expected_key_count, should_raise):
    """Test in-place multiplication operator (*=)."""
    sl = sparselist(initial_data, size=size, default=default)
    original_id = id(sl)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        with pytest.raises(TypeError):
            sl *= multiplier

        assert id(sl) == original_id
        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        sl *= multiplier

        assert id(sl) == original_id
        assert list(sl) == expected_list
        assert _get_sparselist_key_count(sl) == expected_key_count


# ---------------------
# String representation tests
# ---------------------
@pytest.mark.parametrize(
    "initial_data, size, default, expected_repr",
    [
        ({0: 1, 5: 6}, 10, 0, "<10/0>[0: 1, ..., 5: 6, ...]"),
        ({0: 1, 9: 10}, 10, 0, "<10/0>[0: 1, ..., 9: 10]"),
        ([1, 2, 3], 3, None, "<3/None>[0: 1, 1: 2, 2: 3]"),
        ([1, 2, 3], 3, 0, "<3/0>[0: 1, 1: 2, 2: 3]"),
        (None, 0, None, "<0/None>[]"),
        (None, 5, 0, "<5/0>[...]"),
        ({0: "a"}, 1, None, "<1/None>[0: 'a']"),
        ({0: "a", 2: "b", 9: "g"}, 10, None, "<10/None>[0: 'a', ..., 2: 'b', ..., 9: 'g']"),
        ({0: "a", 2: "b"}, 5, "", "<5/''>[0: 'a', ..., 2: 'b', ...]"),
        ({1: "x"}, 5, "default", "<5/'default'>[..., 1: 'x', ...]"),
        ({0: 1, 1: 2, 2: 3, 3: 4}, 5, 0, "<5/0>[0: 1, 1: 2, 2: 3, 3: 4, ...]"),
        ({2: 10}, 5, 0, "<5/0>[..., 2: 10, ...]"),
        ({0: [1, 2], 3: [3, 4]}, 5, None, "<5/None>[0: [1, 2], ..., 3: [3, 4], ...]"),
        ({0: None}, 3, None, "<3/None>[0: None, ...]"),
        (None, 10, [], "<10/[]>[...]"),
        ({0: 1, 1: 2}, 2, 0, "<2/0>[0: 1, 1: 2]"),
    ],
    ids=[
        "sparse_basic",
        "sparse_ends_at_last",
        "dense_none_default",
        "dense_zero_default",
        "empty",
        "all_defaults",
        "single_element",
        "sparse_strings",
        "empty_string_default",
        "string_default_gap_both_ends",
        "gap_at_end",
        "gap_both_ends",
        "list_values",
        "explicit_none_gap_end",
        "list_default",
        "dense_no_gaps",
    ],
)
def test_repr(initial_data, size, default, expected_repr):
    """Test string representation of sparselist."""
    sl = sparselist(initial_data, size=size, default=default)
    assert repr(sl) == expected_repr


# ---------------------
# List method tests
# ---------------------
@pytest.mark.parametrize(
    "initial_data, size, default, value_to_append, expected_list",
    [
        (None, 0, None, 5, [5]),
        (None, 2, 0, 5, [0, 0, 5]),
        ({0: 1}, 2, 0, 99, [1, 0, 99]),
        ({0: 1, 5: 6}, 10, 0, "x", [1, 0, 0, 0, 0, 6, 0, 0, 0, 0, "x"]),
        ([1, 2, 3], 3, None, 4, [1, 2, 3, 4]),
    ],
    ids=["empty_append", "default_filled_append", "sparse_append", "large_sparse_append", "dense_append"],
)
def test_append(initial_data, size, default, value_to_append, expected_list):
    """Test append method adds elements to the end."""
    sl = sparselist(initial_data, size=size, default=default)
    result = sl.append(value_to_append)
    assert result is None
    assert list(sl) == expected_list
    assert sl.size == len(expected_list)


@pytest.mark.parametrize(
    "initial_data, size, default, iterable_to_extend, expected_list, expected_key_count",
    [
        ({0: 1}, 2, 0, [10, 20], [1, 0, 10, 20], 3),
        (None, 0, None, [1, 2, 3], [1, 2, 3], 3),
        ({0: 1}, 2, 0, [], [1, 0], 1),
        ({0: 1}, 2, 0, (x * 2 for x in range(3)), [1, 0, 0, 2, 4], 4),
        ({0: 1}, 2, 0, sparselist({0: 10, 2: 30}, size=3, default=None), [1, 0, 10, None, 30], 4),
        (None, 3, 0, [0, 0], [0, 0, 0, 0, 0], 2),
    ],
    ids=[
        "with_list",
        "empty_with_list",
        "with_empty",
        "with_generator",
        "with_sparselist",
        "with_defaults",
    ],
)
def test_extend(initial_data, size, default, iterable_to_extend, expected_list, expected_key_count):
    """Test extend method adds multiple elements."""
    sl = sparselist(initial_data, size=size, default=default)
    result = sl.extend(iterable_to_extend)
    assert result is None
    assert list(sl) == expected_list
    assert _get_sparselist_key_count(sl) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, index, value, expected_list, expected_size",
    [
        (None, 0, None, 0, 99, [99], 1),
        ({0: 1, 2: 3}, 4, 0, 0, 99, [99, 1, 0, 3, 0], 5),
        ({0: 1, 2: 3}, 4, 0, 2, 99, [1, 0, 99, 3, 0], 5),
        ({0: 1, 2: 3}, 4, 0, 4, 99, [1, 0, 3, 0, 99], 5),
        ({0: 1, 2: 3}, 4, 0, -1, 99, [1, 0, 3, 99, 0], 5),
        ({0: 1, 2: 3}, 4, 0, -5, 99, [99, 1, 0, 3, 0], 5),
        ({0: 1}, 2, 0, 10, 99, [1, 0, 99], 3),
    ],
    ids=[
        "empty",
        "at_0",
        "middle",
        "at_end",
        "neg_1",
        "neg_beyond",
        "beyond_size",
    ],
)
def test_insert(initial_data, size, default, index, value, expected_list, expected_size):
    """Test insert method adds element at specified position."""
    sl = sparselist(initial_data, size=size, default=default)
    result = sl.insert(index, value)
    assert result is None
    assert list(sl) == expected_list
    assert sl.size == expected_size


@pytest.mark.parametrize(
    (
        "initial_data",
        "size",
        "default",
        "index",
        "expected_return",
        "expected_list",
        "expected_size",
        "expected_key_count",
        "should_raise",
    ),
    [
        ({0: 1, 2: 3, 3: 4}, 5, 0, None, 0, [1, 0, 3, 4], 4, 3, False),
        ({0: 1, 2: 3, 4: 5}, 5, 0, 2, 3, [1, 0, 0, 5], 4, 2, False),
        ({0: 1, 2: 3}, 5, 0, 1, 0, [1, 3, 0, 0], 4, 2, False),
        ({0: 1, 2: 3}, 5, 0, -2, 0, [1, 0, 3, 0], 4, 2, False),
        ([1, 2, 3], 3, None, None, 3, [1, 2], 2, 2, False),
        ({0: 1}, 1, 0, 0, 1, [], 0, 0, False),
        (None, 0, None, None, None, None, None, 0, True),
        (None, 0, None, 0, None, None, None, 0, True),
        ({0: 1, 1: 2}, 3, 0, 5, None, [1, 2, 0], 3, 2, True),
        ({0: 1, 1: 2}, 3, 0, -5, None, [1, 2, 0], 3, 2, True),
    ],
    ids=[
        "last_default",
        "explicit",
        "default_middle",
        "negative",
        "dense",
        "only_element",
        "empty_no_index",
        "empty_with_index",
        "over_index",
        "under_index",
    ],
)
def test_pop(
    initial_data, size, default, index, expected_return, expected_list, expected_size, expected_key_count, should_raise
):
    """Test pop method removes and returns element."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        if index is None:
            with pytest.raises(IndexError):
                sl.pop()
        else:
            with pytest.raises(IndexError):
                sl.pop(index)

        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        result = sl.pop() if index is None else sl.pop(index)

        assert result == expected_return
        assert list(sl) == expected_list
        assert sl.size == expected_size
        assert _get_sparselist_key_count(sl) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, value, expected_list, expected_size, expected_key_count, should_raise",
    [
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, [0, 1, 0, 1, 0], 5, 2, False),
        ({0: 1, 2: 3}, 5, 0, 0, [1, 3, 0, 0], 4, 2, False),
        ([1, 2, 3], 3, None, 2, [1, 3], 2, 2, False),
        ({0: 0, 2: 1, 4: 0}, 5, 0, 0, [0, 1, 0, 0], 4, 2, False),
        (None, 0, None, 1, [], 0, 0, True),
        ({0: 1, 2: 3}, 5, 0, 99, [1, 0, 3, 0, 0], 5, 2, True),
        # Large dense test
        (
            {i: i for i in range(15000)},
            20000,
            0,
            100,
            list(range(100)) + list(range(101, 15000)) + [0] * 5000,
            19999,
            14999,
            False,
        ),
        # Large sparse test
        (
            {i * 100: i for i in range(100)},
            20000,
            0,
            50,
            [i // 100 if i % 100 == 0 else 0 for i in range(5000)]
            + [(i + 1) // 100 if (i + 1) % 100 == 0 else 0 for i in range(5000, 9999)]
            + [0] * 10000,
            19999,
            99,
            False,
        ),
    ],
    ids=[
        "first_explicit",
        "default",
        "dense",
        "explicit_same_as_default",
        "from_empty",
        "not_found",
        "large_dense_list",
        "large_sparse_list",
    ],
)
def test_remove(initial_data, size, default, value, expected_list, expected_size, expected_key_count, should_raise):
    """Test remove method deletes first occurrence of value."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        with pytest.raises(ValueError):
            sl.remove(value)

        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        result = sl.remove(value)
        assert result is None
        assert list(sl) == expected_list
        assert sl.size == expected_size
        assert _get_sparselist_key_count(sl) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, expected_list, expected_key_count",
    [
        ({0: "a", 4: "e"}, 5, None, ["e", None, None, None, "a"], 2),
        ([1, 2, 3], 3, 0, [3, 2, 1], 3),
        (None, 5, 0, [0, 0, 0, 0, 0], 0),
        (None, 0, None, [], 0),
        # Large dense test
        ({i: i for i in range(15000)}, 20000, 0, [0] * 5000 + list(range(14999, -1, -1)), 15000),
        # Large sparse test
        (
            {i * 100: i for i in range(100)},
            20000,
            0,
            [0] * 10000 + [i // 100 if i % 100 == 0 else 0 for i in range(9999, -1, -1)],
            100,
        ),
    ],
    ids=[
        "sparse",
        "dense",
        "all_defaults",
        "empty",
        "large_dense",
        "large_sparse",
    ],
)
def test_reverse(initial_data, size, default, expected_list, expected_key_count):
    """Test reverse method reverses list in place without materializing defaults."""
    sl = sparselist(initial_data, size=size, default=default)

    result = sl.reverse()
    assert result is None
    assert list(sl) == expected_list
    assert sl.size == len(expected_list)

    assert _get_sparselist_key_count(sl) == expected_key_count


@pytest.mark.parametrize(
    "initial_data, size, default, sort_kwargs, expected_list, should_raise",
    [
        ({0: 3, 2: 1, 4: 2}, 5, 0, {}, [0, 0, 1, 2, 3], False),
        ({0: 3, 2: 1, 4: 2}, 5, 0, {"reverse": True}, [3, 2, 1, 0, 0], False),
        ({0: "apple", 2: "zoo", 4: "bee"}, 5, "", {"key": len}, ["", "", "zoo", "bee", "apple"], False),
        ({0: 10, 2: -1, 4: 5}, 5, 0, {"key": abs, "reverse": True}, [10, 5, -1, 0, 0], False),
        (None, 5, 0, {}, [0, 0, 0, 0, 0], False),
        (None, 0, None, {}, [], False),
        ([1, "a", 3], 3, None, {}, None, True),
        # Additional test cases to verify non-materialization with many defaults
        ({0: 100, 999: 200}, 1000, 0, {}, [0] * 998 + [100, 200], False),
        ({500: "x", 700: "y"}, 1000, "", {}, [""] * 998 + ["x", "y"], False),
    ],
    ids=[
        "basic",
        "reverse",
        "key_len",
        "key_abs_reverse",
        "all_defaults",
        "empty",
        "mixed_types_error",
        "large_sparse",
        "large_sparse_strings",
    ],
)
def test_sort(initial_data, size, default, sort_kwargs, expected_list, should_raise):
    """Test sort method sorts list in place without materializing defaults."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        with pytest.raises(TypeError):
            sl.sort(**sort_kwargs)

        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        key_count_before = _get_sparselist_key_count(sl)

        result = sl.sort(**sort_kwargs)
        assert result is None
        assert list(sl) == expected_list

        # Verify that sort doesn't materialize unspecified elements
        # Sort only reorders, so key count should remain the same
        assert _get_sparselist_key_count(sl) == key_count_before


@pytest.mark.parametrize(
    "initial_data, size, default",
    [
        ({0: 1, 5: 6}, 10, 0),
        ([1, 2, 3], 3, None),
        (None, 5, "default"),
        (None, 0, None),
    ],
    ids=["sparse", "dense", "defaults_only", "empty"],
)
def test_clear(initial_data, size, default):
    """Test clear method removes all elements."""
    sl = sparselist(initial_data, size=size, default=default)
    result = sl.clear()
    assert result is None
    assert list(sl) == []
    assert sl.size == 0
    assert sl.default == default


@pytest.mark.parametrize(
    "initial_data, size, default, value, expected_count",
    [
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, 3),
        ({0: "a", 2: "b", 4: "a"}, 6, None, "a", 2),
        ([1, 2, 3, 2, 1], 5, 0, 2, 2),
        ({0: 1, 2: 1}, 6, 0, 0, 4),
        ({0: "x"}, 10, None, None, 9),
        (None, 5, 0, 0, 5),
        ({0: 0, 2: 1, 4: 0}, 6, 0, 0, 5),
        ({0: None, 2: "x"}, 5, None, None, 4),
        ({0: 1, 2: 3}, 5, 0, 99, 0),
        ([1, 2, 3], 3, None, "z", 0),
        (None, 0, None, 1, 0),
        (None, 0, 0, 0, 0),
        ([1, 1, 2, 1, 3], 5, 0, 1, 3),
        ([0, 0, 0], 3, None, 0, 3),
    ],
    ids=[
        "explicit_1",
        "explicit_a",
        "dense_2",
        "default_0_implicit",
        "default_none_implicit",
        "all_defaults",
        "mixed_explicit_implicit",
        "mixed_none",
        "not_found",
        "not_found_dense",
        "empty_1",
        "empty_0",
        "dense_1",
        "dense_0",
    ],
)
def test_count(initial_data, size, default, value, expected_count):
    """Test count method returns number of occurrences."""
    sl = sparselist(initial_data, size=size, default=default)
    assert sl.count(value) == expected_count


@pytest.mark.parametrize(
    "initial_data, size, default, value, start, stop, expected_index, should_raise",
    [
        ({2: "x", 5: "y"}, 7, 0, "x", None, None, 2, False),
        ({2: "x", 5: "y"}, 7, 0, "y", None, None, 5, False),
        ([1, 2, 3], 3, None, 2, None, None, 1, False),
        ({2: "x"}, 5, 0, 0, None, None, 0, False),
        ({0: 1, 2: 3}, 5, 0, 0, None, None, 1, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, 1, None, 2, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, 3, None, 4, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, -2, None, 4, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, -10, None, 0, False),
        ({0: 1, 2: 3}, 5, 0, 0, 2, None, 3, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, 0, 3, 0, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, 3, 6, 4, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, -2, 6, 4, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, -10, 6, 0, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, -2, -1, 4, False),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, -10, -1, 0, False),
        ({0: 1, 2: 3}, 5, 0, 0, 1, 4, 1, False),
        ({0: 1, 2: 3}, 5, 0, 3, 1, 100, 2, False),
        ({0: 1, 2: 3}, 5, 0, 99, None, None, None, True),
        ([1, 2, 3], 3, None, "z", None, None, None, True),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 1, 5, None, None, True),
        ({0: 1, 2: 1, 4: 1}, 6, 0, 2, 0, 2, None, True),
        (None, 0, None, 1, None, None, None, True),
        ({0: 1, 2: 3}, 5, 0, 1, 10, None, None, True),
        ({0: 0, 2: 1, 4: 0}, 6, 0, 0, None, None, 0, False),
        ({0: 0, 2: 1, 4: 0}, 6, 0, 0, 1, None, 1, False),
        # Large dense test
        ({i: i for i in range(15000)}, 20000, 0, 100, None, None, 100, False),
        # Large sparse test
        ({i * 100: i for i in range(100)}, 20000, 0, 50, None, None, 5000, False),
    ],
    ids=[
        "find_explicit_x",
        "find_explicit_y",
        "find_dense",
        "find_default_first",
        "find_default_at_1",
        "with_start_find_at_2",
        "with_start_find_at_4",
        "with_start_find_at_neg_2",
        "with_start_find_at_neg_10",
        "with_start_find_default",
        "with_start_stop_find_at_0",
        "with_start_stop_find_at_4",
        "with_start_stop_find_at_neg_2",
        "with_start_stop_find_at_neg_10",
        "with_start_stop_find_at_neg_2_to_neg_1",
        "with_start_stop_find_at_neg_10_to_neg_1",
        "with_start_stop_default",
        "stop_beyond_range",
        "not_found",
        "not_found_dense",
        "not_found_after_start",
        "not_found_in_range",
        "empty_list",
        "start_out_of_range",
        "mixed_find_explicit_0",
        "mixed_find_implicit_0",
        "find_in_large_dense_list",
        "find_in_large_sparse_list",
    ],
)
def test_index(initial_data, size, default, value, start, stop, expected_index, should_raise):
    """Test index method finds first occurrence of value."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size

        if start is not None and stop is not None:
            with pytest.raises(ValueError):
                sl.index(value, start, stop)
        elif start is not None:
            with pytest.raises(ValueError):
                sl.index(value, start)
        else:
            with pytest.raises(ValueError):
                sl.index(value)

        assert list(sl) == before_list
        assert sl.size == before_size
    else:
        if start is not None and stop is not None:
            result = sl.index(value, start, stop)
        elif start is not None:
            result = sl.index(value, start)
        else:
            result = sl.index(value)

        assert result == expected_index


# --------------------------
# Sparselist method tests
# --------------------------


@pytest.mark.parametrize(
    "initial_data, size, default, expected_list, expected_size",
    [
        # Sparse list with gaps - should compact to contiguous
        ({0: "a", 5: "f", 10: "k"}, 15, None, ["a", "f", "k"], 3),
        # Dense list - should remain the same
        ([1, 2, 3, 4], 4, 0, [1, 2, 3, 4], 4),
        # All defaults - should become empty
        (None, 10, 0, [], 0),
        # Empty list - should remain empty
        (None, 0, None, [], 0),
        # Single explicit value at beginning
        ({0: "x"}, 5, "", ["x"], 1),
        # Single explicit value in middle
        ({5: "x"}, 10, "", ["x"], 1),
        # Single explicit value at end
        ({9: "x"}, 10, "", ["x"], 1),
        # Large sparse list
        ({0: 1, 100: 2, 500: 3, 999: 4}, 1000, 0, [1, 2, 3, 4], 4),
        # Mixed with explicit defaults (should keep explicit defaults)
        ({0: 0, 5: 1, 10: 0}, 15, 0, [0, 1, 0], 3),
        # Consecutive explicit values with gap at start
        ({5: "a", 6: "b", 7: "c"}, 10, None, ["a", "b", "c"], 3),
        # Consecutive explicit values with gap at end
        ({0: "a", 1: "b", 2: "c"}, 10, None, ["a", "b", "c"], 3),
        # Large dense test
        (
            {i: i for i in range(15000)},
            20000,
            0,
            list(range(15000)),
            15000,
        ),
        # Large sparse test
        ({i * 100: i for i in range(100)}, 20000, 0, list(range(100)), 100),
    ],
    ids=[
        "sparse_with_gaps",
        "dense",
        "all_defaults",
        "empty",
        "single_at_start",
        "single_in_middle",
        "single_at_end",
        "large_sparse",
        "explicit_defaults",
        "consecutive_gap_start",
        "consecutive_gap_end",
        "large_dense_list",
        "large_sparse_list",
    ],
)
def test_compact(initial_data, size, default, expected_list, expected_size):
    """Test compact method removes gaps and compacts explicit elements."""
    sl = sparselist(initial_data, size=size, default=default)

    result = sl.compact()
    assert result is None
    assert list(sl) == expected_list
    assert sl.size == expected_size
    assert sl.default == default
    assert _get_sparselist_key_count(sl) == expected_size

    # Verify all keys are contiguous starting from 0
    if expected_size > 0:
        keys = sorted(sl._explicit.keys())
        assert keys == list(range(expected_size))


@pytest.mark.parametrize(
    "initial_data, size, default, index, expected_return, expected_list, expected_key_count, should_raise",
    [
        # Unset explicit value at beginning
        ({0: "a", 2: "c"}, 5, None, 0, "a", [None, None, "c", None, None], 1, False),
        # Unset explicit value in middle
        ({0: "a", 2: "c", 4: "e"}, 5, None, 2, "c", ["a", None, None, None, "e"], 2, False),
        # Unset explicit value at end
        ({0: "a", 2: "c", 4: "e"}, 5, None, 4, "e", ["a", None, "c", None, None], 2, False),
        # Unset already default value (no-op)
        ({0: "a", 4: "e"}, 5, None, 2, None, ["a", None, None, None, "e"], 2, False),
        # Unset with negative index
        ({0: "a", 2: "c", 4: "e"}, 5, None, -1, "e", ["a", None, "c", None, None], 2, False),
        # Unset with negative index (middle)
        ({0: "a", 2: "c", 4: "e"}, 5, None, -3, "c", ["a", None, None, None, "e"], 2, False),
        # Unset explicit default value (should remove from explicit)
        ({0: 0, 2: 1, 4: 0}, 5, 0, 4, 0, [0, 0, 1, 0, 0], 2, False),
        # Unset in dense list
        ([1, 2, 3, 4], 4, 0, 1, 2, [1, 0, 3, 4], 3, False),
        # Unset only explicit value
        ({5: "x"}, 10, None, 5, "x", [None] * 10, 0, False),
        # Index out of range (positive)
        ({0: "a"}, 5, None, 10, None, None, 1, True),
        # Index out of range (negative)
        ({0: "a"}, 5, None, -10, None, None, 1, True),
        # Empty list
        (None, 0, None, 0, None, None, 0, True),
    ],
    ids=[
        "explicit_at_start",
        "explicit_in_middle",
        "explicit_at_end",
        "already_default",
        "negative_index",
        "negative_index_middle",
        "explicit_default_value",
        "dense_list",
        "only_explicit",
        "out_of_range_positive",
        "out_of_range_negative",
        "empty_list",
    ],
)
def test_unset(initial_data, size, default, index, expected_return, expected_list, expected_key_count, should_raise):
    """Test unset method removes explicit value and returns the old value."""
    sl = sparselist(initial_data, size=size, default=default)

    if should_raise:
        before_list = list(sl)
        before_size = sl.size
        before_key_count = _get_sparselist_key_count(sl)

        with pytest.raises(IndexError):
            sl.unset(index)

        assert list(sl) == before_list
        assert sl.size == before_size
        assert _get_sparselist_key_count(sl) == before_key_count
    else:
        result = sl.unset(index)
        assert result == expected_return
        assert list(sl) == expected_list
        assert sl.size == size  # Size should not change
        assert sl.default == default
        assert _get_sparselist_key_count(sl) == expected_key_count


# ---------------------
# Copy tests
# ---------------------
@pytest.mark.parametrize(
    "initial_data, size, default, copy_method",
    [
        ({0: 1, 5: 6}, 10, 0, "builtin"),
        ({0: 1, 5: 6}, 10, 0, "copy_module"),
        ({0: "a", 2: "c"}, 5, None, "builtin"),
        ({0: "a", 2: "c"}, 5, None, "copy_module"),
        ([1, 2, 3], 3, 0, "builtin"),
        ([1, 2, 3], 3, 0, "copy_module"),
        (None, 5, 0, "builtin"),
        (None, 5, 0, "copy_module"),
        (None, 0, None, "builtin"),
        (None, 0, None, "copy_module"),
    ],
    ids=[
        "sparse_builtin",
        "sparse_copy_module",
        "sparse_none_builtin",
        "sparse_none_copy_module",
        "dense_builtin",
        "dense_copy_module",
        "all_defaults_builtin",
        "all_defaults_copy_module",
        "empty_builtin",
        "empty_copy_module",
    ],
)
def test_copy_returns_sparselist(initial_data, size, default, copy_method):
    """Test copy returns independent sparselist with same properties."""
    sl = sparselist(initial_data, size=size, default=default)
    original_key_count = _get_sparselist_key_count(sl)

    sl_copy = sl.copy() if copy_method == "builtin" else copy.copy(sl)

    assert isinstance(sl_copy, sparselist)
    assert sl_copy == sl
    assert list(sl_copy) == list(sl)
    assert sl_copy is not sl
    assert sl_copy.default == sl.default
    assert sl_copy.size == sl.size
    assert _get_sparselist_key_count(sl_copy) == original_key_count

    if sl.size > 0:
        sl_copy[0] = "MODIFIED"
        assert sl[0] != "MODIFIED"

    if sl.size > 0:
        sl[0] = "ORIGINAL_MODIFIED"
        assert sl_copy[0] == "MODIFIED"


# ---------------------
# Complex behavior tests
# ---------------------
@pytest.mark.parametrize(
    "initial_data, size, default",
    [
        ({0: "start", 999_999: "end"}, 1_000_000, None),
        ({500_000: "middle"}, 1_000_001, 0),
        ({}, 2_000_000, None),
        ({0: 1, 100_000: 2, 999_999: 3}, 1_000_000, 0),
    ],
    ids=["sparse_first_last", "sparse_middle", "all_defaults", "sparse_multiple"],
)
def test_large_sparse_indexing(initial_data, size, default):
    """Test that large sparse lists handle indexing efficiently without materialization."""
    sl = sparselist(initial_data, size=size, default=default)

    assert sl.size == size

    for idx, expected_value in initial_data.items():
        assert sl[idx] == expected_value

    if size > 0:
        mid_idx = size // 2
        if mid_idx not in initial_data:
            assert sl[mid_idx] == default

    expected_key_count = len(initial_data)
    assert _get_sparselist_key_count(sl) == expected_key_count

    if size > 0:
        assert sl[-1] == initial_data.get(size - 1, default)


@pytest.mark.parametrize(
    "initial_data, size, default",
    [
        ({0: "start", 999_999: "end"}, 1_000_000, None),
        ({500_000: "middle"}, 1_000_001, 0),
        ({}, 2_000_000, None),
        ({0: 1, 100_000: 2, 999_999: 3}, 1_000_000, 0),
    ],
    ids=["sparse_first_last", "sparse_middle", "all_defaults", "sparse_multiple"],
)
def test_large_sparse_operations(initial_data, size, default):
    """Test that operations on large sparse lists maintain efficiency."""
    sl = sparselist(initial_data, size=size, default=default)
    original_key_count = _get_sparselist_key_count(sl)

    assert len(sl) == size

    small_slice = sl[-10:]
    assert isinstance(small_slice, sparselist)
    assert len(small_slice) == 10
    assert small_slice.default == default
    keys_in_range = [k for k in initial_data if k >= size - 10]
    assert _get_sparselist_key_count(small_slice) <= len(keys_in_range)

    mid_slice = sl[1000:1010]
    assert isinstance(mid_slice, sparselist)
    assert len(mid_slice) == 10
    assert mid_slice.default == default
    keys_in_mid_range = [k for k in initial_data if 1000 <= k < 1010]
    assert _get_sparselist_key_count(mid_slice) <= len(keys_in_mid_range)

    sl.size = min(sl.size, 1000)
    assert sl.size == 1000
    assert _get_sparselist_key_count(sl) <= len([k for k in initial_data if k < 1000])

    sl.size = 2000
    assert sl.size == 2000
    assert _get_sparselist_key_count(sl) == len([k for k in initial_data if k < 1000])

    sl2 = sparselist(initial_data, size=size, default=default)
    assert _get_sparselist_key_count(sl2) == original_key_count


def test_nested_sparselist():
    """Test that sparselists can contain other sparselists as elements."""
    inner = sparselist([1, 2], default=0)
    outer = sparselist({0: inner}, size=3, default=None)

    elem0 = outer[0]
    assert elem0 == inner
    assert elem0 is inner
    assert isinstance(elem0, sparselist)
    assert elem0[1] == 2
    assert outer[1] is None

    inner.append(3)
    assert elem0 == inner
    assert list(elem0) == [1, 2, 3]

    assert len(outer) == 3
    assert _get_sparselist_key_count(outer) == 1


def test_comprehensive_integration():
    """Test complex sequences of operations to ensure features work together."""
    # Phase 1: Initialize
    sl = sparselist({0: "a", 10: "k"}, size=15, default="_")
    assert sl[0] == "a"
    assert sl[5] == "_"
    assert sl[10] == "k"
    assert sl.size == 15
    assert _get_sparselist_key_count(sl) == 2

    # Phase 2: Modify size and default
    sl.size = 20
    assert sl.size == 20
    assert _get_sparselist_key_count(sl) == 2

    sl.default = "-"
    assert sl[15] == "-"
    assert sl[5] == "-"

    # Phase 3: List modifications
    sl[5] = "f"
    assert _get_sparselist_key_count(sl) == 3

    sl.append("end")
    assert sl.size == 21
    assert sl[-1] == "end"
    assert _get_sparselist_key_count(sl) == 4

    sl.insert(0, "start")
    assert sl.size == 22
    assert sl[0] == "start"
    assert _get_sparselist_key_count(sl) == 5

    # Phase 4: Slicing
    middle = sl[8:13]
    assert isinstance(middle, sparselist)
    assert middle.default == "-"
    assert len(middle) == 5
    assert "k" in list(middle)

    # Phase 5: Arithmetic
    sl2 = sparselist([1, 2, 3], default=0)
    combined = sl2 + sl
    assert isinstance(combined, sparselist)
    assert combined.default == 0
    assert combined.size == 3 + 22

    # Phase 6: Search
    assert "start" in sl
    assert "end" in sl
    assert "-" in sl
    assert sl.count("-") > 0
    assert sl.index("start") == 0

    # Phase 7: Deletion
    original_size = sl.size
    del sl[1]
    assert sl.size == original_size - 1
    assert "a" not in list(sl)

    # Phase 8: Copy independence
    sl_copy = sl.copy()
    assert sl_copy == sl
    sl_copy[0] = "MODIFIED"
    assert sl[0] == "start"

    # Phase 9: Final state verification
    assert isinstance(sl, sparselist)
    assert isinstance(sl, list)
    assert sl.default == "-"

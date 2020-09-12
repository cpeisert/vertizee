# Copyright 2020 The Vertizee Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RangeDict is a dictionary mapping numeric ranges to values."""

# pytype: disable=invalid-annotation

import bisect
from typing import Any, List, MutableMapping, Optional, TypeVar, Union

KT = TypeVar("KT")
VT = TypeVar("VT")


class RangeDict(dict, MutableMapping[KT, VT]):
    """RangeDict is a dictionary mapping numeric ranges to values.

    The keys are range objects, however, values may be retrieved using any number within the range.
    For example, if the dictionary had key = range(1, 3), then the value could be retrieved using:
    range_dict[1]
    range_dict[2]

    But range_dict[3] would raise a key error.
    """

    def __init__(self):
        self._sorted_range_endpoints: List[int] = []
        super(RangeDict, self).__init__()

    def __contains__(self, key: Union[range, int]) -> bool:
        if isinstance(key, range):
            return super(RangeDict, self).__contains__(key)
        rng_key = self.get_range_key_for_value_in_range(key)
        return super(RangeDict, self).__contains__(rng_key)

    def __delitem__(self, key: Union[range, int]):
        if isinstance(key, range):
            rng_key = key
        else:  # key is an integer
            rng_key = self.get_range_key_for_value_in_range(key)
        if rng_key is not None:
            super(RangeDict, self).__delitem__(rng_key)
            prev_rng = self.get_range_key_for_value_in_range(rng_key[0] - 1)
            next_rng = self.get_range_key_for_value_in_range(rng_key[-1] + 1)
            if prev_rng is None or (prev_rng is not None and prev_rng[-1] + 1 != rng_key[0]):
                self._sorted_range_endpoints.remove(rng_key[0])
            if next_rng is None or (next_rng is not None and next_rng[0] != rng_key[-1] + 1):
                self._sorted_range_endpoints.remove(rng_key[-1] + 1)

    def __getitem__(self, key: Union[range, int]) -> VT:
        return self._get_item(key)

    def __setitem__(self, key: Union[range, int], val: Any):
        if isinstance(key, range):
            if len(key) < 1:
                raise KeyError("range key must contain at least one value")
            if super(RangeDict, self).__contains__(key):
                super(RangeDict, self).__setitem__(key, val)
                return
            existing_rng = self._get_overlapping_range(key)
            if existing_rng is None:
                self._add_new_range(key, val)
            else:
                raise KeyError(
                    f"existing range {existing_rng} partially overlaps {key}; ranges "
                    " must either match or be disjoint"
                )
        else:  # key is an integer
            rng_key = self.get_range_key_for_value_in_range(key)
            if rng_key is None:
                raise KeyError(f'there is no existing range containing value "{key}"')
            super(RangeDict, self).__setitem__(rng_key, val)

    def __repr__(self):
        dictrepr = super(RangeDict, self).__repr__()
        return f"{type(self).__name__}({dictrepr})"

    def copy(self):
        new = RangeDict()
        new._sorted_range_endpoints = self._sorted_range_endpoints.copy()
        new.update(self)
        return new

    def get_range_key_for_value_in_range(self, value: int) -> Optional[range]:
        """Returns the range containing `value`, or None if there is no such range."""
        rng_left = self._find_left_range_endpoint(value)
        rng_right = self._find_right_range_endpoint(value)
        if rng_left is None or rng_right is None:
            return None
        rng_key = range(rng_left, rng_right)
        if super(RangeDict, self).__contains__(rng_key):
            return rng_key
        else:
            return None

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
            other = dict(args[0])
            for key in other:
                if not isinstance(key, range):
                    raise KeyError(f'key "{key}" was not a range object')
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def _add_new_range(self, rng: range, val: Any):
        if len(rng) < 1:
            raise ValueError("range must contain at least one value")
        left_point = self._find_left_range_endpoint(rng[0])
        right_point = self._find_left_range_endpoint(rng[-1] + 1)
        if left_point is None or (left_point is not None and left_point != rng[0]):
            i = bisect.bisect(self._sorted_range_endpoints, rng[0])
            self._sorted_range_endpoints.insert(i, rng[0])
        if right_point is None or (right_point is not None and right_point != rng[-1] + 1):
            i = bisect.bisect(self._sorted_range_endpoints, rng[-1] + 1)
            self._sorted_range_endpoints.insert(i, rng[-1] + 1)
        super(RangeDict, self).__setitem__(rng, val)

    def _find_left_range_endpoint(self, value: int) -> Optional[int]:
        i = bisect.bisect_right(self._sorted_range_endpoints, value)
        if i:
            return self._sorted_range_endpoints[i - 1]
        else:
            return None

    def _find_right_range_endpoint(self, value: int) -> Optional[int]:
        i = bisect.bisect_right(self._sorted_range_endpoints, value)
        if i != len(self._sorted_range_endpoints):
            return self._sorted_range_endpoints[i]
        else:
            return None

    def _get_item(self, key: Union[range, int]) -> VT:
        if isinstance(key, range):
            return super(RangeDict, self).__getitem__(key)
        rng_key = self.get_range_key_for_value_in_range(key)
        if rng_key is None:
            raise KeyError(f"{key}")
        else:
            return super(RangeDict, self).__getitem__(rng_key)

    def _get_overlapping_range(self, r: range) -> Optional[range]:
        """Checks if there is a range in the dictionary that overlaps `r`. If so, returns the
        existing overlapping range, otherwise returns None."""
        if len(r) < 1:
            return None
        overlapping = self.get_range_key_for_value_in_range(r[0])
        if overlapping is not None:
            return overlapping
        overlapping = self.get_range_key_for_value_in_range(r[-1])
        return overlapping

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

"""Tests for RangeDict container."""

import pytest

from vertizee.classes.collections.range_dict import RangeDict

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


@pytest.mark.usefixtures()
class TestRangeDict:
    def test_del_len_set(self):
        d1 = RangeDict()

        with pytest.raises(KeyError):
            # KeyError(f'there is no existing range containing value "{key}"')
            d1[1] = "one"
        assert len(d1) == 0, "d1 should have 0 items"

        d1[range(1, 3)] = "1 to 3"
        assert len(d1) == 1, "d1 should have 1 item"
        d1[range(3, 5)] = "3 to 5"
        assert len(d1) == 2, "d1 should have 2 items"

        with pytest.raises(KeyError):
            # KeyError(f'existing range {existing_rng} partially overlaps {key}; ranges '
            #          ' must either match or be disjoint')
            d1[range(3, 4)] = "3 to 4"

        del d1[range(1, 3)]
        del d1[range(3, 5)]
        assert len(d1) == 0, "after deletions, d1 should have 0 items"
        d1[range(3, 4)] = "3 to 4"
        assert len(d1) == 1, "d1 should have 1 item after insertion"
        d1[range(4, 10)] = "4 to 10"
        d1[range(10, 20)] = "10 to 20"
        d1[range(25, 30)] = "25 to 30"
        assert len(d1) == 4, "d1 should have 4 items"

    def test_contains_equals_get(self):
        d1 = RangeDict()
        d2 = RangeDict()

        d1[range(1, 3)] = "1 to 3"
        d2[range(1, 3)] = "1 to 3"
        assert d1 == d2, "d1 should equal d2 after inserting same key/value pairs"

        assert range(1, 3) in d1, "d1 should contain range(1, 3)"
        assert range(2, 3) not in d1, "d1 should contain range(2, 3)"
        assert 1 in d1, "d1 should contain 1, since 1 is in the range(1, 3)"
        assert 2 in d1, "d1 should contain 2, since 1 is in the range(1, 3)"
        assert 3 not in d1, "d1 should not contain 3, since 3 is not in the range(1, 3)"

        assert d1[1] == d1[range(1, 3)], "getitem by value in range should equal getitem by range"
        assert d1[1] == "1 to 3", "getitem should retrieve value for key within range"
        assert d1[2] == "1 to 3", "getitem should retrieve value for key within range"
        assert d1.get(3, "not found") == "not found", "d1 should not contain range with 3"

    def test_update(self):
        pairs = [(range(1, 3), "1 to 3"), (range(3, 6), "3 to 6")]
        d1 = RangeDict()

        d1.update(pairs)
        assert len(d1) == 2, "Dict d1 should have length 2 after update"

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

"""Tests for BKHeap data structure."""

from typing import Dict, List, Set, Tuple

import pytest

from vertizee.classes.collections.bk_tree import BKNode, BKNodeLabeled, BKTree

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip.")


def count_bits(n: int) -> int:
    """Counts the number of '1's in an integer. This algorithm exploits the assembly instruction
    __builtin_popcount(). See: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/ """
    return bin(n).count('1')


def hamming_dist(n1: int, n2: int) -> int:
    """Calculates the hamming distance between two numbers based on their binary representation,
    i.e., how many bits differ. Example: 7 (0b111) and 2 (0b010) have a distance of 2."""
    xor = n1 ^ n2
    return count_bits(xor)


def levenshtein(s1: str, s2: str) -> int:
    """Function to calculate the the Levenshtein distance between two strings, which is the minimum
    number of single-character edits (insertions, deletions or substitutions) required to change
    one string into the other.

    This implementation is taken from "Python Advanced Course Topics"
    (https://www.python-course.eu/levenshtein_distance.php).

    For a faster Python solution implemented in C, see:
    https://github.com/ztane/python-Levenshtein

    Args:
        s1 (str): The first string to compare.
        s2 (str): The second string to compare.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    memo: Dict[Tuple[str, str], int] = {}
    """Dictionary to memoize subproblem solutions during the calculation of the Levenshtein
    distance. The keys are Tuples of substrings of the original `s1` and `s2` and the values are
    the Levenshtein distances between the substrings."""

    return _memoized_levenshtein(s1, s2, memo)


def _memoized_levenshtein(s1: str, s2: str, memo: Dict[Tuple[str, str], int]) -> int:
    if s1 == "":
        return len(s2)
    if s2 == "":
        return len(s1)
    cost = 0 if s1[-1] == s2[-1] else 1

    i1 = (s1[:-1], s2)
    if i1 not in memo:
        memo[i1] = _memoized_levenshtein(*i1, memo)
    i2 = (s1, s2[:-1])
    if i2 not in memo:
        memo[i2] = _memoized_levenshtein(*i2, memo)
    i3 = (s1[:-1], s2[:-1])
    if i3 not in memo:
        memo[i3] = _memoized_levenshtein(*i3, memo)
    res = min([memo[i1] + 1, memo[i2] + 1, memo[i3] + cost])
    return res


@pytest.mark.usefixtures()
class TestBKTreeHammingDistance:
    """Test BK Tree over the metric space defined by integer keys and the hamming distance between
    their binary representations, i.e., the number of positions at which the corresponding bits
    are different."""

    def test_insert_search(self):
        keys = [
            (1, 189), (2, 361), (3, 130), (4, 245), (5, 87), (6, 148), (7, 188), (8, 121), (9, 66),
            (10, 103), (11, 220), (12, 157), (13, 247), (14, 106), (15, 152), (16, 112), (17, 14),
            (18, 204), (19, 14), (20, 365), (21, 385), (22, 288), (23, 113), (24, 356), (25, 214),
            (26, 293), (27, 355), (28, 350), (29, 374), (30, 8), (31, 393), (32, 344), (33, 34),
            (34, 292), (35, 36), (36, 352), (37, 216), (38, 10), (39, 212), (40, 366), (41, 276),
            (42, 123), (43, 338), (44, 36), (45, 149), (46, 24), (47, 362), (48, 353), (49, 256),
            (50, 62), (51, 182), (52, 307), (53, 71), (54, 55), (55, 196), (56, 55), (57, 130),
            (58, 57), (59, 268), (60, 325), (61, 85), (62, 246), (63, 17), (64, 296), (65, 154),
            (66, 117), (67, 16), (68, 82), (69, 108), (70, 65), (71, 223), (72, 85), (73, 326),
            (74, 51), (75, 128), (76, 21), (77, 371), (78, 41), (79, 358), (80, 136), (81, 342),
            (82, 31), (83, 377), (84, 24), (85, 301), (86, 74), (87, 182), (88, 80), (89, 294),
            (90, 313), (91, 21), (92, 369), (93, 188), (94, 198), (95, 71), (96, 360), (97, 10),
            (98, 168), (99, 8), (100, 243)]

        bk_tree = BKTree[int](distance_function=hamming_dist, labeled_nodes=True)
        for k_v in keys:
            bk_tree.insert(key_value=k_v[1], key_label=k_v[0])

        radius = 4
        bf_results = self._brute_force_search(
            keys, search_key=189, radius=radius, search_key_label=1)
        results: List[BKNodeLabeled] = bk_tree.search(key_value=189, radius=radius, key_label=1)

        # print(f'\n\nDEBUG BK Tree found {len(results)} keys within radius {radius} of key "1".')

        assert len(bf_results) == len(results), 'The brute-force and BK tree result lengths ' \
                                                'should be the same.'
        for node in results:
            assert (int(node.key_label), node.key_value) in bf_results, \
                f'BK Tree result node {node} should be in the brute-force result set.'

    def test_delete_and_garbage_collection(self):
        keys = [(i, i) for i in range(1, 5001)]

        bk_tree = BKTree[int](distance_function=hamming_dist, labeled_nodes=True)
        for k_v in keys:
            bk_tree.insert(key_value=k_v[1], key_label=k_v[0])
        assert len(bk_tree) == 5000, 'BK Tree should have 5000 nodes.'

        radius = 2

        # Garbage collection runs when deleted nodes >= 30% of total nodes (active + deleted).
        # 30% * 5000 = 1500 deleted nodes when garbage collection gets triggered.
        while len(bk_tree) > 3450 and len(keys) > 0:
            k_v = keys.pop()
            results = bk_tree.search(key_value=k_v[1], radius=radius, key_label=k_v[0])
            for node in results:
                bk_tree.delete_node(node)

        assert len(bk_tree) <= 3450, 'BK Tree should have fewer than 3400 nodes.'
        assert bk_tree._deleted_item_count < 100, \
            'After garbage collection was triggered upon 1,500 deleted nodes, the total number ' \
            'of deleted nodes should now be less than 100.'

    def _brute_force_search(self, keys: List[Tuple[int, int]], search_key: int, radius: int,
                            search_key_label: str) -> Set[Tuple[int, int]]:
        results: Set[Tuple[int, int]] = set()
        for k_v in keys:
            if str(k_v[0]) != str(search_key_label) and hamming_dist(search_key, k_v[1]) <= radius:
                results.add(k_v)

        return results


@pytest.mark.usefixtures()
class TestBKTreeLevenshteinDistance:
    """Test BK Tree over the metric space defined by integer keys and the hamming distance between
    their binary representations, i.e., the number of positions at which the corresponding bits
    are different."""

    def test_insert_search(self):
        dictionary = [
            'abseiler', 'accelerate', 'aileron', 'alert', 'alerted', 'allergen', 'allergic',
            'allergies', 'angler', 'antler', 'artillery', 'assembler',
            'simple', 'simpler', 'simplest', 'simplex', 'simplicity', 'simplify', 'simply']

        bk_tree = BKTree[str](distance_function=levenshtein, labeled_nodes=False)
        for word in dictionary:
            bk_tree.insert(key_value=word)

        radius = 2
        results: List[BKNode] = bk_tree.search(key_value='alerte', radius=radius)
        similar_words = set([n.key_value for n in results])
        assert len(similar_words) == 2, \
            'There should be 2 words in dictionary with Levenshtein distance 2 from the ' \
            'string "alerte".'
        assert 'alert' in similar_words, '"alert" should have matched "alerte"'
        assert 'alerted' in similar_words, '"alerted" should have matched "alerte"'

        radius = 3
        results: List[BKNode] = bk_tree.search(key_value='simpl', radius=radius)
        similar_words = set([n.key_value for n in results])
        assert len(similar_words) == 6, \
            'There should be 6 words in dictionary with Levenshtein distance 3 from the ' \
            'string "simpl".'
        assert 'simple' in similar_words, '"simple" should have matched "simpl"'
        assert 'simpler' in similar_words, '"simpler" should have matched "simpl"'
        assert 'simplest' in similar_words, '"simplest" should have matched "simpl"'
        assert 'simplex' in similar_words, '"simplex" should have matched "simpl"'
        assert 'simplify' in similar_words, '"simplify" should have matched "simpl"'
        assert 'simply' in similar_words, '"simply" should have matched "simpl"'

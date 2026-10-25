import numpy as np
import json

from experiment_selection import describe_candidate, select_indices
from summarize_selection_experiments import aggregate


def test_segment_distance_not_centroid():
    guide = np.array([[0., 0.], [10., 0.]])
    item = ('lane_marking', {'points': [[5., -100.], [5., 100.]]})
    result = describe_candidate(item, guide)
    assert result['distance_to_guide_m'] == 0.
    assert result['mean_abs_direction_alignment'] == 0.
    assert result['length_m'] == 200.
    item = ('lane_marking', {'points': [[0., 2.], [10., 2.]]})
    result = describe_candidate(item, guide)
    assert result['distance_to_guide_m'] == 2.
    assert result['mean_abs_direction_alignment'] == 1.


def test_same_count_nested_subsets_and_stable_random():
    descriptors = [{'distance_to_guide_m': i, 'mean_point_distance_to_guide_m': i} for i in range(10)]
    assert select_indices(descriptors, 'near', 4, 's', 42) == [0, 1, 2, 3]
    assert select_indices(descriptors, 'far', 4, 's', 42) == [6, 7, 8, 9]
    for mode in ('near', 'far', 'random'):
        small = select_indices(descriptors, mode, 4, 's', 42)
        large = select_indices(descriptors, mode, 8, 's', 42)
        assert len(small) == 4 and set(small) <= set(large)
        assert small == select_indices(descriptors, mode, 4, 's', 42)
        assert select_indices(descriptors, mode, 12, 's', 42) == list(range(10))
        assert select_indices([], mode, 4, 's', 42) == []
    assert select_indices(descriptors, 'baseline', 0, 's', 42) == []
    assert select_indices(descriptors, 'full', 32, 's', 42) == list(range(10))


def test_multi_seed_aggregation(tmp_path):
    paths = []
    for seed, error in [(724, .7), (725, .8)]:
        path = tmp_path / str(seed)
        path.mkdir()
        stats = {'samples': 2, 'mean_added_tokens': 4., 'worlds_improved_over_1cm': 0,
                 'worlds_worsened_over_1cm': 1,
                 **{key: {'after': error} for key in ('ade_m', 'fde_m', 'wrong_direction', 'outside_target_lane')}}
        report = {'checkpoint': 'same.pt', 'input': 'same.jsonl', 'selection_seed': 42,
                  'supplement_seed': seed, 'budgets': [4],
                  'all': {name: stats for name in ('baseline', 'full', 'near_4', 'far_4', 'random_4')}}
        (path / 'summary.json').write_text(json.dumps(report))
        np.savez(path / 'predictions.npz', ids=np.array(['a', 'b']))
        paths.append(path)
    report = aggregate(paths, tmp_path / 'summary')
    assert report['aggregate']['near_4']['ade_m']['mean'] == .75
    assert (tmp_path / 'summary' / 'comparison.png').exists()

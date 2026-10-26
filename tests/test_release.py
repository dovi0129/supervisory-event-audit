"""Release-specific checks of published aggregate relationships and synthetic data."""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open() as handle:
        return list(csv.DictReader(handle))


def test_published_probability_differences():
    for row in read('artifacts/stage1_audit/q1_summary.csv'):
        assert abs(float(row['p_observed'])-float(row['p_chance'])-float(row['delta'])) < 1e-12
    for row in read('artifacts/stage1_audit/s89/q1_elapsed_controls.csv'):
        assert abs(float(row['p_observed'])-float(row['p_reference'])-float(row['delta'])) < 1e-12


def test_published_information_increments():
    metrics = read('artifacts/stage1_audit/q2_metrics.csv')
    for row in read('artifacts/stage1_audit/q2_paired_comparisons.csv'):
        first, second = row['contrast'].split('-')
        values = {r['model']: float(r['accuracy']) for r in metrics
                  if r['split'] == row['split'] and r['stratum'] == row['stratum']}
        assert abs(values[first]-values[second]-float(row['accuracy_difference'])) < 1e-12


def test_synthetic_example():
    spec = importlib.util.spec_from_file_location('synthetic_demo', ROOT / 'examples/synthetic_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.demonstrate()['within_group_event_counts_preserved']

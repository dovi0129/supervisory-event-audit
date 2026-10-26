from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from s74_episodes import SEED, variance_parts

@njit(cache=True, parallel=True)
def distance_matrix(padded, lengths):
    n = len(lengths)
    raw = np.zeros((n, n), np.float64)
    normalized = raw.copy()
    for i in prange(n):
        for j in range(i + 1, n):
            m, k = (lengths[i], lengths[j])
            previous = np.arange(k + 1)
            for a in range(m):
                current = np.empty(k + 1, np.int64)
                current[0] = a + 1
                for b in range(k):
                    current[b + 1] = min(previous[b + 1] + 1, current[b] + 1, previous[b] + (padded[i, a] != padded[j, b]))
                previous = current
            d = previous[k]
            raw[i, j] = raw[j, i] = d
            normalized[i, j] = normalized[j, i] = d / max(m, k) if max(m, k) else 0.0
    return (raw, normalized)

def cv(v):
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 and np.mean(v) > 1e-15 else None

def sequence_mean(matrix, ids):
    if len(ids) < 2:
        return None
    subset = matrix[np.ix_(ids, ids)]
    return float((subset.sum() - np.trace(subset)) / (len(ids) * (len(ids) - 1)))

def make_table(episodes, actions, roles):
    e = episodes[episodes.cohort.eq('transition')].copy()
    eligible = e.pair.value_counts()
    e = e[e.pair.isin(eligible[eligible.ge(3)].index)].reset_index(drop=True)
    target = actions[actions.kind.eq('target')]
    grouped = {k: g for k, g in target.groupby('episode_id')}
    sequences, rows = ([], [])
    for row in e.itertuples():
        g = grouped.get(row.episode_id, target.iloc[:0])
        tokens = tuple(('+'.join(sorted(t.loop.tolist())) for _, t in g.groupby('relative_min', sort=True)))
        sequences.append(tokens)
        r = {'episode_id': row.episode_id, 'pair': row.pair, 'first_action': row.first_action_min, 'first_target': float(g.relative_min.min()) if len(g) else np.nan, 'sequence_length': len(tokens)}
        for i, role in enumerate(roles):
            values = g.loc[g.loop.eq(role), 'delta']
            r[f'net_{i}'] = float(values.sum())
            r[f'abs_{i}'] = float(values.abs().sum())
            r[f'first_{i}'] = float(g.loc[g.loop.eq(role), 'relative_min'].min()) if len(values) else np.nan
        rows.append(r)
    return (e, pd.DataFrame(rows), sequences)

def analyze(dest):
    metadata = json.loads((dest / 'metadata.json').read_text())
    roles = metadata['roles']
    episodes = pd.read_parquet(dest / 'episodes.parquet')
    actions = pd.read_parquet(dest / 'episode_actions.parquet')
    e, table, sequences = make_table(episodes, actions, roles)
    if not len(e):
        (dest / 'recipe_summary.json').write_text(json.dumps({'eligible_episodes': 0, 'eligible_pairs': 0, 'permutations': 50}))
        pd.DataFrame(columns=['pair', 'episodes']).to_csv(dest / 'sequence_consistency.csv', index=False)
        pd.DataFrame(columns=['measurement', 'ratio']).to_csv(dest / 'variance_ratios.csv', index=False)
        return
    unique = sorted(set(sequences))
    token_map = {t: i for i, t in enumerate(sorted({s for seq in unique for s in seq}))}
    seq_map = {seq: i for i, seq in enumerate(unique)}
    lengths = np.array([len(s) for s in unique])
    padded = np.full((len(unique), max(1, int(lengths.max()))), -1, np.int64)
    for i, seq in enumerate(unique):
        padded[i, :len(seq)] = [token_map[t] for t in seq]
    set_num_threads(16)
    raw, normalized = distance_matrix(padded, lengths)
    ids = np.array([seq_map[s] for s in sequences])
    labels = table.pair.to_numpy()
    names = sorted(np.unique(labels))
    metrics = ['first_action', 'first_target'] + [p + str(i) for i in range(len(roles)) for p in ['net_', 'abs_', 'first_']]
    null_sequence, null_pair, null_variance = ([], [], [])
    rng = np.random.default_rng(SEED)
    for repeat in range(50):
        shuffled = rng.permutation(labels)
        for name in names:
            ix = np.flatnonzero(shuffled == name)
            g = table.iloc[ix]
            null_sequence.append({'repeat': repeat, 'pair': name, 'edit_mean': sequence_mean(raw, ids[ix]), 'normalized_edit_mean': sequence_mean(normalized, ids[ix]), 'first_action_variance': float(g.first_action.var(ddof=1)), 'first_target_variance': float(g.first_target.var(ddof=1))})
            for i, role in enumerate(roles):
                timing = g[f'first_{i}'].dropna()
                null_pair.append({'repeat': repeat, 'pair': name, 'target': role, 'amplitude_cv': cv(g[f'abs_{i}']), 'first_target_variance': float(timing.var(ddof=1)) if len(timing) > 1 else None})
        for metric in metrics:
            null_variance.append({'repeat': repeat, 'measurement': metric, **variance_parts(table[metric], shuffled)})
    null_sequence = pd.DataFrame(null_sequence)
    null_pair = pd.DataFrame(null_pair)
    null_variance = pd.DataFrame(null_variance)
    null_sequence.to_csv(dest / 'sequence_permutations.csv', index=False)
    null_pair.to_csv(dest / 'pair_permutations.csv', index=False)
    null_variance.to_csv(dest / 'variance_permutations.csv', index=False)
    sequence_rows, pair_rows, ratio_rows = ([], [], [])
    for name in names:
        ix = np.flatnonzero(labels == name)
        g = table.iloc[ix]
        null = null_sequence[null_sequence.pair.eq(name)]
        seqrow = {'pair': name, 'episodes': len(ix), 'pairwise_comparisons': len(ix) * (len(ix) - 1) // 2, 'empty_sequences': int(g.sequence_length.eq(0).sum()), 'edit_mean': sequence_mean(raw, ids[ix]), 'normalized_edit_mean': sequence_mean(normalized, ids[ix]), 'first_action_variance': float(g.first_action.var(ddof=1)), 'first_target_variance': float(g.first_target.var(ddof=1))}
        for col in ['edit_mean', 'normalized_edit_mean', 'first_action_variance', 'first_target_variance']:
            seqrow[col + '_null_mean'] = float(null[col].mean())
            seqrow[col + '_null_p05'] = float(null[col].quantile(0.05))
            seqrow[col + '_null_p95'] = float(null[col].quantile(0.95))
        sequence_rows.append(seqrow)
        for i, role in enumerate(roles):
            timing = g[f'first_{i}'].dropna()
            nn = null_pair[null_pair.pair.eq(name) & null_pair.target.eq(role)]
            pair_rows.append({'pair': name, 'target': role, 'episodes': len(g), 'nonzero_amplitude_episodes': int(g[f'abs_{i}'].gt(0).sum()), 'net_change_mean': float(g[f'net_{i}'].mean()), 'absolute_change_mean': float(g[f'abs_{i}'].mean()), 'amplitude_cv': cv(g[f'abs_{i}']), 'first_target_n': len(timing), 'first_target_variance': float(timing.var(ddof=1)) if len(timing) > 1 else None, 'amplitude_cv_null_mean': float(nn.amplitude_cv.mean()) if nn.amplitude_cv.notna().any() else None, 'first_target_variance_null_mean': float(nn.first_target_variance.mean()) if nn.first_target_variance.notna().any() else None})
    for metric in metrics:
        null = null_variance[null_variance.measurement.eq(metric)].ratio.dropna()
        label = metric
        for i, role in enumerate(roles):
            if metric in [f'net_{i}', f'abs_{i}', f'first_{i}']:
                label = metric.split('_')[0] + '_' + role
        ratio_rows.append({'measurement': label, 'code': metric, **variance_parts(table[metric], labels), 'null_ratio_n': len(null), 'null_ratio_mean': float(null.mean()) if len(null) else None, 'null_ratio_p05': float(null.quantile(0.05)) if len(null) else None, 'null_ratio_p95': float(null.quantile(0.95)) if len(null) else None})
    pd.DataFrame(sequence_rows).to_csv(dest / 'sequence_consistency.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(dest / 'recipe_by_pair.csv', index=False)
    pd.DataFrame(ratio_rows).to_csv(dest / 'variance_ratios.csv', index=False)
    table.to_parquet(dest / 'recipe_episode_metrics.parquet', index=False)
    examples = []
    for name in e.pair.value_counts().index[:3]:
        ix = np.flatnonzero(labels == name)
        dist = normalized[np.ix_(ids[ix], ids[ix])].sum(axis=1)
        candidate = np.flatnonzero(table.iloc[ix].sequence_length.to_numpy() > 0)
        candidate = candidate if len(candidate) else np.arange(len(ix))
        chosen = ix[candidate[int(np.argmin(dist[candidate]))]]
        examples.append({'pair': name, 'episode_id': table.iloc[chosen].episode_id, 'episodes_in_pair': len(ix), 'selection': 'nonempty target sequence; minimum distance to entire pair; first index breaks ties'})
    pd.DataFrame(examples).to_csv(dest / 'sequence_examples.csv', index=False)
    (dest / 'recipe_summary.json').write_text(json.dumps({'eligible_episodes': len(e), 'eligible_pairs': len(names), 'unique_sequences': len(unique), 'permutations': 50, 'seed': SEED}, indent=2))
    print(json.dumps({'recipe_episodes': len(e), 'recipe_pairs': len(names), 'permutations': 50}), flush=True)

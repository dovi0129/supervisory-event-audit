from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import re
import numpy as np
import pandas as pd
from s70_features import ROOT, future_labels
from s70_fit import event_hits, lot_windows
BASE = ROOT / 'artifacts/intervention'
OUT = BASE / 's73'
AGES = ['moisture_target_age', 'size_moisture_target_age', 'weight_target_age']
REGIONS = ['all', 'normal', 'transition', 'other']
REFERENCE = {'normal': [17, 40, 2, 83], 'transition': [245, 364, 17, 363]}

def parse_paths(text):
    stack, leaves = ({}, [])
    for line in text.splitlines():
        depth = line.count('|') - 1
        token = line.split('--- ', 1)[1]
        if token.startswith('class:'):
            leaves.append([stack[d] for d in range(depth)])
            continue
        match = re.fullmatch('(.+?)\\s+(<=|>)\\s+([-+\\d.eE]+)', token)
        if not match:
            raise ValueError('Unsupported saved tree predicate')
        stack[depth] = (match[1], match[2], float(match[3]))
    return leaves

def count_between(positions, start, end):
    return int(np.searchsorted(positions, end, side='right') - np.searchsorted(positions, start, side='left')) if end >= start else 0

def signed_distance(positions, e, maximum=120):
    at = np.searchsorted(positions, e)
    nearby = positions[max(0, at - 1):at + 1]
    if not len(nearby):
        return None
    delta = nearby[np.argmin(np.abs(nearby - e))] - e
    return int(delta) if abs(delta) <= maximum else None

def sequence(candidates, e, loop, index):
    t = index[e]
    lo = candidates.time.searchsorted(t - pd.Timedelta(minutes=10))
    hi = candidates.time.searchsorted(t + pd.Timedelta(minutes=10), side='right')
    local = candidates.iloc[lo:hi]
    local = local[~(local.kind.eq('mode') & local.loop.eq(loop))]
    steps = {0: [loop + ':aut→man']}
    records = []
    for r in local.itertuples():
        offset = int((r.time - t).total_seconds() / 60)
        action = r.loop + ':' + (r.shape if r.kind == 'mode' else '목표 변경' if r.kind == 'target' else '수동 출력 변경')
        steps.setdefault(offset, []).append(action)
        records.append({'offset_min': offset, 'action': action, 'kind': r.kind})
    ordered = sorted(steps)
    tokens = [' + '.join(sorted(set(steps[o]))) for o in ordered]
    timed = tokens[0]
    for i in range(1, len(tokens)):
        timed += f' → {ordered[i] - ordered[i - 1]}분 → ' + tokens[i]
    return (' → '.join(tokens), timed, records)

def route_paths(pred, x, rule, mode_names):
    leaves = parse_paths(rule['text'])
    values = x.iloc[pred.position.to_numpy()]
    assignment = np.full(len(pred), -1, int)
    method = np.full(len(pred), 'unresolved', object)
    probabilities, conditions = ({}, {})
    for leaf_id, path in enumerate(leaves):
        clear = np.ones(len(pred), bool)
        readable = []
        for feature, operator, threshold in path:
            if feature.startswith('num__'):
                feature = feature[5:]
                v = values[feature].to_numpy(float)
                name = mode_names.get(feature, feature)
            elif feature.startswith('grade__grade_'):
                grade = feature[len('grade__grade_'):]
                v = values.grade.eq(grade).to_numpy(float)
                name = 'grade == ' + grade
            else:
                raise ValueError('Unrecognized anonymized feature')
            finite = np.isfinite(v)
            clear &= finite & (np.abs(v - threshold) > 5.1e-06)
            clear &= v <= threshold if operator == '<=' else v > threshold
            readable.append(f'{name} {operator} {threshold:.5f}')
        if np.any(assignment[clear] >= 0):
            raise ValueError('Overlapping saved leaf paths')
        assignment[clear] = leaf_id
        method[clear] = 'finite_conditions'
        scores = np.unique(pred.loc[clear, 'tree'])
        if len(scores) > 1:
            raise ValueError('Saved scores disagree within a reconstructed finite leaf')
        probabilities[leaf_id] = float(scores[0]) if len(scores) else None
        conditions[leaf_id] = ' AND '.join(readable)
    for score in np.unique(pred.loc[assignment < 0, 'tree']):
        matching = [leaf for leaf, probability in probabilities.items() if probability is not None and abs(probability - float(score)) < 1e-08]
        if len(matching) == 1:
            mask = (assignment < 0) & np.isclose(pred.tree.to_numpy(), score, atol=1e-08, rtol=0)
            assignment[mask] = matching[0]
            method[mask] = 'unique_saved_score'
    pred = pred.copy()
    pred['path_id'] = assignment
    pred['path_assignment'] = method
    assert np.all(pred.loc[pred.tree_alarm, 'tree'] >= rule['threshold'])
    rows = []
    for leaf_id in list(range(len(leaves))) + [-1]:
        for region in REGIONS:
            mask = pred.path_id.eq(leaf_id) & (True if region == 'all' else pred.region.eq(region))
            p = pred[mask]
            rows.append({'loop': rule['loop'], 'path_id': leaf_id, 'region': region, 'conditions': conditions.get(leaf_id, 'unresolved'), 'saved_probability': probabilities.get(leaf_id), 'tau': rule['threshold'], 'probability_ge_tau': probabilities.get(leaf_id) >= rule['threshold'] if probabilities.get(leaf_id) is not None else None, 'test_minutes': len(p), 'score_ge_tau_minutes': int(p.tree.ge(rule['threshold']).sum()), 'alarms': int(p.tree_alarm.sum()), 'hit_alarms': int((p.tree_alarm & p.y).sum()), 'finite_condition_minutes': int(p.path_assignment.eq('finite_conditions').sum()), 'unique_score_minutes': int(p.path_assignment.eq('unique_saved_score').sum())})
    return (pred, rows)

def markdown_table(frame):

    def cell(v):
        if pd.isna(v):
            return '—'
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else f'{v:.6f}'
        return str(v).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] + ['| ' + ' | '.join((cell(v) for v in row)) + ' |' for row in frame.itertuples(index=False, name=None)])

def main():
    if (OUT / 'DONE_s73').exists():
        raise FileExistsError('Completed S73 exists; preserve it before a new version')
    meta = json.loads((BASE / 'data/metadata.json').read_text())
    rules = [r for r in json.loads((BASE / 'full/tree_rules.json').read_text()) if r['k'] == 5 and r['variant'] == 'main' and (r['direction'] == 'aut→man') and (r['fold'] == 'time') and (r['model'] == 'tree')]
    cols = set(AGES + ['grade', 'grade_age'])
    for r in rules:
        for path in parse_paths(r['text']):
            cols.update((f[5:] for f, _, _ in path if f.startswith('num__')))
    cols.update((c + suffix for c in ['moisture', 'size_moisture', 'weight'] for suffix in ['_error', '_duration']))
    x = pd.read_parquet(BASE / str(record_path('data_features_private_parquet')), columns=sorted(cols))
    minutes = pd.read_parquet(BASE / str(record_path('data_minutes_private_parquet')))
    modes = pd.read_parquet(BASE / str(record_path('data_modes_private_parquet')))
    events = pd.read_parquet(BASE / str(record_path('data_events_private_parquet')))
    assert x.index.equals(minutes.index) and x.index.equals(modes.index)
    candidates = pd.read_parquet(ROOT / 'artifacts/operator_policy/events.parquet', columns=['time', 'loop', 'kind', 'shape'], filters=[('class', '==', '운전자 후보'), ('kind', 'in', ['target', 'mode', 'op'])]).sort_values(['time', 'loop'])
    setting_roles = [loop.replace('모드', '설정값') for loop in meta['targets']]
    settings = pd.read_parquet(ROOT / 'artifacts/operator_policy/events.parquet', columns=['time', 'loop'], filters=[('loop', 'in', setting_roles)])
    setting_positions = {role: np.sort(x.index.get_indexer(g.time)) for role, g in settings.groupby('loop')}
    output_positions = {role: np.sort(x.index.get_indexer(g.time)) for role, g in candidates[candidates.kind.eq('op')].groupby('loop')}
    target_positions = np.sort(x.index.get_indexer(candidates[candidates.kind.eq('target')].time))
    brk = np.flatnonzero(minutes.brk.fillna(False).to_numpy(bool))
    grade_changes = np.flatnonzero(x.grade_age.eq(0).to_numpy())
    same_lot = lot_windows(minutes.lot.fillna('__missing__').astype(str), 5)
    all_events, memberships, sequence_rows, path_rows, path_predictions, contrasts = ([], [], [], [], [], [])
    reverse_names = {v: k for k, v in meta['mode_names'].items()}
    for loop_i, loop in enumerate(meta['targets']):
        ev = events[events.loop.eq(loop) & events['shape'].eq('aut→man')].copy()
        wanted = ~ev.flag_brk_near & ev.return_observed & ~ev.quick & ~ev.short_previous
        positions = x.index.get_indexer(ev.time)
        accepted = np.zeros(len(x), bool)
        rejected = accepted.copy()
        accepted[positions[wanted.to_numpy()]] = True
        rejected[positions[~wanted.to_numpy()]] = True
        accepted_positions = np.flatnonzero(accepted)
        full_domain = modes[meta['mode_names'][loop]].eq(1.0).to_numpy() & ~future_labels(rejected, 5) & ~rejected & ~minutes.brk_near.to_numpy(bool) & same_lot
        pred = pd.read_parquet(BASE / f'full/predictions_PRIVATE/{loop_i}_5_main_aut→man/time.parquet')
        rule = next((r for r in rules if r['loop'] == loop))
        routed, paths = route_paths(pred, x, rule, reverse_names)
        routed['loop'] = loop
        path_predictions.append(routed)
        path_rows.extend(paths)
        for scope in ['full', 'time']:
            for region in REGIONS:
                domain = np.zeros(len(x), bool)
                alarm = domain.copy()
                if scope == 'time':
                    pp = pred if region == 'all' else pred[pred.region.eq(region)]
                    domain[pp.position] = True
                    alarm[pp.position] = pp.tree_alarm
                else:
                    domain = full_domain & (True if region == 'all' else minutes.region.eq(region).to_numpy())
                ee, hit, _ = event_hits(alarm, accepted_positions, domain, 5)
                recent = x.iloc[ee - 1][AGES].min(axis=1).le(4).to_numpy()
                contrasts.append({'scope': scope, 'loop': loop, 'region': region, 'recent_hits': int(hit[recent].sum()) if scope == 'time' else None, 'recent_events': int(recent.sum()), 'no_recent_hits': int(hit[~recent].sum()) if scope == 'time' else None, 'no_recent_events': int((~recent).sum())})
                for e, h, r in zip(ee, hit, recent):
                    memberships.append({'scope': scope, 'loop': loop, 'region': region, 'event_position': int(e), 'recent_target': bool(r), 'hit': bool(h) if scope == 'time' else None})
        loop_events = events[events.loop.eq(loop)].sort_values('time')
        lp = x.index.get_indexer(loop_events.time)
        next_return = {int(p): int(lp[j + 1]) for j, p in enumerate(lp[:-1]) if loop_events.iloc[j]['shape'] == 'aut→man'}
        for e in accepted_positions:
            structural, timed, neighbors = sequence(candidates, e, loop, x.index)
            row = {'loop': loop, 'event_position': int(e), 'event_time': x.index[e], 'structural_pattern': structural, 'timed_pattern': timed, 'recent_target': bool(x.iloc[e - 1][AGES].min() <= 4), 'manual_duration_min': next_return[int(e)] - int(e), 'nearest_break_delta_min': signed_distance(brk, e), 'prior120_break': count_between(brk, max(0, e - 120), e - 1) > 0, 'next120_break': count_between(brk, e + 1, min(len(x) - 1, e + 120)) > 0, 'next60_break': count_between(brk, e + 1, min(len(x) - 1, e + 60)) > 0, 'next60_grade_change': count_between(grade_changes, e + 1, min(len(x) - 1, e + 60)) > 0, 'pre60_minutes': min(60, int(e)), 'next60_complete': bool(e + 60 < len(x)), 'sequence_window_complete': bool(e >= 10 and e + 10 < len(x))}
            end = next_return[int(e)] - 1
            row['own_manual_output_changes'] = count_between(output_positions.get(loop.replace('모드', '출력'), np.array([], int)), e, end)
            row['own_manual_setpoint_changes'] = count_between(setting_positions.get(loop.replace('모드', '설정값'), np.array([], int)), e, end)
            row['manual_quality_target_changes'] = count_between(target_positions, e, end)
            before = x.iloc[max(0, e - 60):e]
            for name in ['moisture', 'size_moisture', 'weight']:
                error = before[name + '_error'].abs()
                row[name + '_abs_pre60_n'] = int(error.notna().sum())
                row[name + '_abs_pre60_median'] = float(error.median()) if error.notna().any() else None
                row[name + '_abs_pre60_max'] = float(error.max()) if error.notna().any() else None
                duration = before[name + '_duration']
                row[name + '_sign_duration_pre60_max'] = float(duration.max()) if duration.notna().any() else None
            all_events.append(row)
            sequence_rows.extend(({'loop': loop, 'event_position': int(e), **r} for r in neighbors))
        print(json.dumps({'loop': loop, 'accepted_events': len(accepted_positions), 'test_minutes': len(pred)}), flush=True)
    event_frame = pd.DataFrame(all_events)
    member_frame = pd.DataFrame(memberships)
    detail = member_frame.merge(event_frame, on=['loop', 'event_position', 'recent_target'], validate='many_to_one')
    event_frame.to_parquet(OUT / str(record_path('event_conditions_private_parquet')), index=False)
    member_frame.to_parquet(OUT / str(record_path('event_membership_private_parquet')), index=False)
    pd.DataFrame(sequence_rows).to_parquet(OUT / str(record_path('sequences_private_parquet')), index=False)
    pd.concat(path_predictions, ignore_index=True).to_parquet(OUT / str(record_path('path_predictions_private_parquet')), index=False)
    contrast = pd.DataFrame(contrasts)
    pooled = contrast.groupby(['scope', 'region'], dropna=False)[['recent_hits', 'recent_events', 'no_recent_hits', 'no_recent_events']].sum(min_count=1).reset_index()
    pooled['loop'] = 'pooled'
    contrast = pd.concat([contrast, pooled], ignore_index=True)
    contrast.to_csv(OUT / 'recent_target_counts.csv', index=False)
    checks = []
    for region, expected in REFERENCE.items():
        row = pooled[pooled.scope.eq('time') & pooled.region.eq(region)].iloc[0]
        observed = [int(row[k]) for k in ['recent_hits', 'recent_events', 'no_recent_hits', 'no_recent_events']]
        if observed != expected:
            raise ValueError('S71 target-history reproduction mismatch')
        checks.append({'region': region, 'expected': expected, 'observed': observed, 'matched': True})
    relative = pd.DataFrame(sequence_rows).merge(member_frame[['scope', 'loop', 'region', 'event_position']], on=['loop', 'event_position'], validate='many_to_many')
    relative[relative.kind.eq('target')].groupby(['scope', 'loop', 'region', 'action', 'offset_min']).size().rename('records').reset_index().to_csv(OUT / 'target_relative_minutes.csv', index=False)
    for pattern in ['structural_pattern', 'timed_pattern']:
        patterns = detail.groupby(['scope', 'loop', 'region', pattern]).size().rename('events').reset_index()
        patterns.sort_values(['scope', 'loop', 'region', 'events'], ascending=[True, True, True, False]).to_csv(OUT / (pattern + '_counts.csv'), index=False)
    no_recent = detail[~detail.recent_target].copy()
    no_recent.to_parquet(OUT / str(record_path('no_recent_conditions_private_parquet')), index=False)
    expanded = pd.concat([no_recent, no_recent.assign(loop='pooled')], ignore_index=True)
    condition_summary, distributions = ([], [])
    numerical = ['manual_duration_min', 'nearest_break_delta_min'] + [c for c in event_frame if c.endswith(('_pre60_median', '_pre60_max'))]
    for key, g in expanded.groupby(['scope', 'loop', 'region']):
        row = dict(zip(['scope', 'loop', 'region'], key))
        row.update({'events': len(g), 'hit_events': int(g.hit.eq(True).sum()) if key[0] == 'time' else None, 'prior120_break': int(g.prior120_break.sum()), 'next120_break': int(g.next120_break.sum()), 'next60_break': int(g.next60_break.sum()), 'next60_grade_change': int(g.next60_grade_change.sum()), 'own_manual_output_changed': int(g.own_manual_output_changes.gt(0).sum()), 'own_manual_setpoint_changed': int(g.own_manual_setpoint_changes.gt(0).sum()), 'manual_quality_target_changed': int(g.manual_quality_target_changes.gt(0).sum()), 'next60_complete': int(g.next60_complete.sum())})
        condition_summary.append(row)
        for name in numerical:
            values = g[name].dropna()
            distributions.append(dict(zip(['scope', 'loop', 'region'], key)) | {'measurement': name, 'n': len(values)} | dict(zip(['p10', 'median', 'p90', 'max'], values.quantile([0.1, 0.5, 0.9, 1.0]).tolist() if len(values) else [None] * 4)))
    condition_summary = pd.DataFrame(condition_summary)
    condition_summary.to_csv(OUT / 'no_recent_condition_counts.csv', index=False)
    pd.DataFrame(distributions).to_csv(OUT / 'no_recent_distributions.csv', index=False)
    paths = pd.DataFrame(path_rows)
    paths.to_csv(OUT / 'all_tree_paths.csv', index=False)
    paths[paths.probability_ge_tau.eq(True) | paths.alarms.gt(0)].to_csv(OUT / 'active_tree_paths.csv', index=False)
    existing_metrics = pd.read_csv(BASE / 'full/metrics.csv')
    for loop in meta['targets']:
        for region in REGIONS:
            expected = existing_metrics[existing_metrics.loop.eq(loop) & existing_metrics.k.eq(5) & existing_metrics.variant.eq('main') & existing_metrics.direction.eq('aut→man') & existing_metrics.split.eq('time') & existing_metrics.model.eq('tree') & existing_metrics.region.eq(region)].iloc[0]
            actual = paths[paths.loop.eq(loop) & paths.region.eq(region)]
            assert int(actual.alarms.sum()) == int(expected.alarms)
            assert int(actual.hit_alarms.sum()) == int(expected.true_alarm_minutes)
    unresolved = paths[paths.path_id.eq(-1) & paths.region.eq('all')]
    status = {'s71_reference': checks, 'accepted_unique_events': len(event_frame), 'membership_rows': len(detail), 'time_tree_unresolved_minutes': int(unresolved.test_minutes.sum()), 'time_tree_unresolved_alarms': int(unresolved.alarms.sum()), 'saved_prediction_alarm_count_match': True, 'new_training': False, 'scope': 'existing S70/S68 artifacts only'}
    (OUT / 'verification.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    write_report(contrast, condition_summary, paths, status)
    (OUT / 'DONE_s73').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(json.dumps(status, ensure_ascii=False), flush=True)

def write_report(contrast, conditions, paths, status):
    table = contrast[contrast.scope.eq('time') & contrast.loop.eq('pooled') & contrast.region.isin(['normal', 'transition'])][['region', 'recent_hits', 'recent_events', 'no_recent_hits', 'no_recent_events']].copy()
    table.columns = ['구간', '최근 변경 있음 적중', '사건', '최근 변경 없음 적중', '사건 수']
    cond = conditions[conditions.scope.eq('time') & conditions.loop.eq('pooled') & conditions.region.isin(['normal', 'transition'])].drop(columns=['scope', 'loop', 'next60_complete'])
    active = paths[paths.region.eq('all') & paths.probability_ge_tau.eq(True)][['loop', 'path_id', 'saved_probability', 'tau', 'alarms', 'hit_alarms', 'conditions']]
    table['구간'] = table['구간'].map({'normal': '정상 생산', 'transition': '지종 전환'})
    cond.columns = ['구간', '사건', '적중', '앞 120분 지절', '뒤 120분 지절', '뒤 60분 지절', '뒤 60분 지종 변경', '수동 출력 변경', '자기 설정값 변화', '품질 목표 변경']
    cond['구간'] = cond['구간'].map({'normal': '정상 생산', 'transition': '지종 전환'})
    active.columns = ['루프', '경로', '저장 점수', '실제 τ', '발화', '적중 발화', '조건식']
    pattern = pd.read_csv(OUT / 'timed_pattern_counts.csv')
    pattern = pattern[pattern.region.isin(['normal', 'transition'])].sort_values(['scope', 'loop', 'region', 'events'], ascending=[True, True, True, False]).groupby(['scope', 'loop', 'region'], sort=False).head(1)
    pattern = pattern[['scope', 'loop', 'region', 'events', 'timed_pattern']]
    pattern.columns = ['기간', '루프', '구간', '사건', '최빈 순서와 간격']
    pattern['기간'] = pattern['기간'].map({'full': '전체', 'time': '시간순 시험'})
    pattern['구간'] = pattern['구간'].map({'normal': '정상 생산', 'transition': '지종 전환'})
    dist = pd.read_csv(OUT / 'no_recent_distributions.csv')
    dist = dist[dist.scope.eq('time') & dist.loop.eq('pooled') & dist.region.isin(['normal', 'transition'])]
    short_stats = []
    for region in ['normal', 'transition']:
        g = dist[dist.region.eq(region)].set_index('measurement')
        short_stats.append({'구간': '정상 생산' if region == 'normal' else '지종 전환', '수동 지속 중앙값(분)': g.loc['manual_duration_min', 'median'], '수동 지속 90백분위(분)': g.loc['manual_duration_min', 'p90'], 'reel 수분 최대편차 중앙값': g.loc['moisture_abs_pre60_max', 'median'], 'size 수분 최대편차 중앙값': g.loc['size_moisture_abs_pre60_max', 'median'], '평량 최대편차 중앙값': g.loc['weight_abs_pre60_max', 'median']})
    lines = ['# 모드 전환의 행동 순서와 시험 발화 경로', '', '## 대상과 정의', '', '전체 기간은 2022년 1~11월이고 시간순 시험은 9~11월이다. 자동→수동 주 분석의 네 루프를 대상으로 k=5의 사건 정의와 제외 조건을 유지했다. 시간순 시험은 기존 저장 예측에 포함된 분을 그대로 사용했다. 전체 기간은 동일한 적격·제외 조건과 LOT 경계 제거를 적용했다. 구간은 예측 분 기준이며 한 사건이 여러 구간에 들어갈 수 있다.', '', '최근 목표 변경은 사건 직전 [e−5,e−1]의 reel 수분·size 수분·reel 평량 목표 변경이다. 기존 세 목표 변경 경과 시간의 e−1 값 중 최솟값이 4분 이하면 해당한다. 행동 순서의 목표 변경은 여섯 품질 목표의 S68 운전자 후보 사건을 사용한다. 전자는 모든 관측 차분이고 후자는 S68 후보 기준으로 분류된 기록이다.', '', '## 최근 목표 변경에 따른 사건과 적중', '', markdown_table(table), '', 'S71의 네 적중·분모 조합을 재현했다. 루프별·전체 기간 표는 recent_target_counts.csv에 제시했다. 전체 기간 표의 적중 값은 저장된 시간순 시험 밖의 예측 범위와 구분해 빈값으로 둔다.', '', '## 행동 묶음 순서', '', '사건 전후 ±10분의 목표 변경, 다른 루프의 양방향 모드 전환, 수동 출력 변경을 시각순으로 묶었다. 같은 분의 기록은 +로 묶고 순서를 부여하지 않았다. structural_pattern_counts.csv는 행동 순서, timed_pattern_counts.csv는 인접 기록 사이의 분 간격을 포함한 패턴별 빈도를 루프·구간·기간별로 제시한다. 개별 사건의 순서와 시간은 sequences_PRIVATE.parquet에 저장했다.', '', markdown_table(pattern), '', '표는 루프·구간·기간마다 가장 빈도가 높은 정확한 시간 순서 하나를 제시하며 동률은 저장 정렬의 첫 패턴이다. 목표별 상대 시각 빈도는 target_relative_minutes.csv에 제시했다.', '', '## 최근 목표 변경이 없는 전환의 기록 조건', '', markdown_table(cond), '', markdown_table(pd.DataFrame(short_stats)), '', '사건 직전 60분의 기존 편차 절댓값 중앙값·최댓값과 부호 지속 길이를 집계했다. no_recent_distributions.csv는 각 요약량의 유효 표본 수와 10·50·90백분위수·최댓값을 제시한다. 지절 거리의 부호는 음수가 과거, 양수가 미래이며 ±120분 안에서 가장 가까운 지절 분까지의 거리다. 수동 지속 시간은 사건부터 다음 자동 복귀까지의 분이다.', '', '수동 중 자기 루프 출력 변경은 S68 수동 출력 후보, 자기 설정값 변경은 동일 루프의 기록된 설정값 변화, 품질 목표 변경은 여섯 목표 후보를 센다. 자기 설정값은 PV 추종으로 변할 수 있다. 뒤 60분의 지종 변경은 기존 지종 변경 경과 시간이 0인 분, 지절은 기존 지절 플래그로 집계했다.', '', '## 실제 발화 문턱에 대응하는 말단 경로', '', markdown_table(active), '', '저장된 시험 점수를 조건 경로에 연결해 말단 확률을 확인했다. 표의 적중은 발화 후 5분 안에 유효 사건이 있었던 발화 수다. class 표기 대신 저장 확률과 실제 τ를 비교했다. active_tree_paths.csv는 구간별 발화·적중, all_tree_paths.csv는 비활성·미관측 경로까지 제시한다. mode 특성은 비식별 루프 역할명으로 바꿨다.', '', '## 가정과 확인 범위', '', f"텍스트 조건식은 소수 다섯 자리이며 경계 ±0.0000051 안의 값과 결측값은 조건만으로 배정하지 않았다. 기존 점수가 한 경로에만 대응하면 그 점수로 경로를 연결했다. 최종 미확정 표본은 {status['time_tree_unresolved_minutes']}분이고, 미확정 발화는 {status['time_tree_unresolved_alarms']}건이다. 실제 관측된 점수가 없는 말단의 확률은 빈값이다.", '', '목표 변경 차분과 운전자 후보 분류는 정의가 다르다. 기록은 분 단위이므로 같은 분의 조작 순서와 행동 주체는 구별되지 않는다. 부호 지속 길이는 편차의 크기가 일정 문턱 이상인 지속 시간과 다르다.', '', '대상 버전: S70 v0.2의 저장 산출물과 S73 브리프. 이번 공백: 행동 순서·목표 변경 없는 조건·실제 τ 경로. 수행 검사: S71 네 수치 재현, 경로별 저장 확률 일치, 발화·적중의 기존 표 대조. 재사용 검증: S70·S71의 시점·라벨·시험 집계 검증. 결론과 한계: 기록 조건과 저장 발화를 재집계했으며 경로 미확정 수는 위와 같다.', '']
    (ROOT / 'generated/notes/S73_sequence_W4.md').write_text('\n'.join(lines))
if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main()

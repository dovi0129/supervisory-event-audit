from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from s70_features import OUT, ROOT
KEYS = ['loop', 'k', 'variant', 'direction', 'split', 'model', 'region']

def category(r, p, c):
    if r is None or c is None or (not np.isfinite(r)) or (not np.isfinite(c)) or (c >= 1):
        return '보류'
    e = (r - c) / (1 - c)
    p = p if p is not None and np.isfinite(p) else 0.0
    if r >= 0.5 and p >= 0.3 and (e >= 0.3):
        return '설명 가능'
    if r >= 0.3 and e >= 0.15:
        return '부분'
    return '미설명'

def mechanical(row):
    r, p, c = (row['event_recall'], row['minute_precision'], row['chance_recall'])
    label = category(r, p, c)
    boundary = False
    if row['events'] and label != '보류':
        for hit in [row['event_hits'] - 1, row['event_hits'] + 1]:
            if 0 <= hit <= row['events'] and category(hit / row['events'], p, c) != label:
                boundary = True
    return {'mechanical_category': label, 'one_event_boundary': boundary, 'recall_ge_05': bool(r >= 0.5) if pd.notna(r) else None, 'precision_ge_03': bool(p >= 0.3) if pd.notna(p) else None, 'chance_excess_ge_03': bool(row['chance_excess'] >= 0.3) if pd.notna(row['chance_excess']) else None}

def aggregate(mode='smoke'):
    base = OUT / mode
    paths = sorted((base / 'jobs').glob('*.json'))
    if not paths:
        raise ValueError('No completed jobs')
    rows, support, rules, event_region = ([], [], [], [])
    predictions = []
    for path in paths:
        job = json.loads(path.read_text())
        meta = {c: job[c] for c in ['loop', 'k', 'variant', 'direction']}
        event_region.extend((meta | r for r in job.get('event_region_sensitivity', [])))
        for row in job['folds']:
            rows.append(meta | row | {'split': 'time' if row['fold'] == 'time' else 'lot'})
        support.extend((meta | r for r in job['support']))
        rules.extend((meta | r for r in job['rules']))
        for pp in sorted((base / str(record_path('predictions_private')) / job['key'].replace(':', '_')).glob('*.parquet')):
            pred = pd.read_parquet(pp)
            pred['split'] = 'time' if pp.stem == 'time' else 'lot'
            for key, val in meta.items():
                pred[key] = val
            predictions.append(pred)
    folds = pd.DataFrame(rows)
    if 'status' not in folds:
        folds['status'] = 'evaluated'
    folds['status'] = folds.status.fillna('evaluated')
    unsupported = folds[folds.status.ne('evaluated')].drop_duplicates(['loop', 'k', 'variant', 'direction', 'fold'])
    unsupported.drop(columns=['lead_minutes', 'calibration', 'split_counts']).to_csv(base / 'unsupported_folds.csv', index=False)
    folds.drop(columns=['lead_minutes', 'calibration', 'split_counts']).to_csv(base / 'fold_metrics.csv', index=False)
    pd.DataFrame(event_region).to_csv(base / 'event_region_sensitivity.csv', index=False)
    pd.DataFrame(support).to_csv(base / 'support.csv', index=False)
    (base / 'tree_rules.json').write_text(json.dumps(rules, ensure_ascii=False, indent=1))
    if not predictions:
        raise ValueError('Saved predictions required for exact pooled PR-AUC')
    pred = pd.concat(predictions, ignore_index=True)
    summary = []
    prediction_keys = ['loop', 'k', 'variant', 'direction', 'split']
    group_positions = pred.groupby(prediction_keys, sort=False).indices
    pooled_positions = pred.groupby(prediction_keys[1:], sort=False).indices
    expanded = pd.concat([folds, folds.assign(loop='pooled')], ignore_index=True)
    for key, g in expanded.groupby(KEYS, dropna=False):
        row = dict(zip(KEYS, key))
        for c in ['n_minutes', 'positive_minutes', 'alarms', 'true_alarm_minutes', 'events', 'event_hits', 'candidate_events']:
            row[c] = int(g[c].sum())
        row['event_recall'] = row['event_hits'] / row['events'] if row['events'] else None
        row['minute_precision'] = row['true_alarm_minutes'] / row['alarms'] if row['alarms'] else None
        valid = g.chance_recall.notna() & g.events.gt(0)
        row['chance_recall'] = float(np.average(g.loc[valid, 'chance_recall'], weights=g.loc[valid, 'events'])) if valid.any() else None
        row['chance_excess'] = (row['event_recall'] - row['chance_recall']) / (1 - row['chance_recall']) if row['event_recall'] is not None and row['chance_recall'] is not None and (row['chance_recall'] < 1) else None
        if row['loop'] == 'pooled':
            positions = pooled_positions.get(tuple((row[c] for c in prediction_keys[1:])), [])
        else:
            positions = group_positions.get(tuple((row[c] for c in prediction_keys)), [])
        p = pred.iloc[positions]
        if row['region'] != 'all':
            p = p[p.region.eq(row['region'])]
        assert len(p) == row['n_minutes'], 'Prediction/metric minute alignment mismatch'
        row['pr_auc'] = float(average_precision_score(p.y, p[row['model']])) if p.y.any() else None
        leads = [v for array in g.lead_minutes for v in array]
        row.update(dict(zip(['lead_p10', 'lead_p50', 'lead_p90'], np.percentile(leads, [10, 50, 90]) if leads else [None] * 3)))
        row.update(mechanical(row))
        summary.append(row)
    table = pd.DataFrame(summary)
    pair_keys = [c for c in KEYS if c != 'model']
    gap = table[table.model.isin(['tree', 'gbm'])].pivot(index=pair_keys, columns='model', values='pr_auc')
    gap['gbm_minus_tree_pr_auc'] = gap['gbm'] - gap['tree']
    gap['description'] = 'selected-model predictive difference'
    gap.reset_index().to_csv(base / 'gbm_tree_gap.csv', index=False)
    comparison = table[table.model.eq('tree')].pivot(index=['loop', 'k', 'variant', 'direction', 'region'], columns='split', values='mechanical_category')
    comparison['split_disagreement'] = comparison['time'].ne(comparison['lot'])
    comparison['combined_status'] = np.where(comparison.split_disagreement, '보류', comparison['time'])
    comparison.to_csv(base / 'split_threshold_comparison.csv')
    table.to_csv(base / 'metrics.csv', index=False)
    plot_data = pred[(pred.k == 5) & pred.variant.eq('main') & pred.direction.eq('aut→man') & pred.split.eq('time')]
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout='constrained')
    calibration = []
    styles = {'rule': ('#0072B2', 'o', '-'), 'tree': ('#D55E00', 's', '--'), 'gbm': ('#009E73', '^', '-.')}
    for model, (color, marker, line) in styles.items():
        points = []
        for low in np.linspace(0, 0.9, 10):
            sel = plot_data[model].ge(low) & (plot_data[model].lt(low + 0.1) if low < 0.85 else plot_data[model].le(1.0))
            p = plot_data[sel]
            if len(p):
                points.append((float(p[model].mean()), float(p.y.mean())))
                calibration.append({'model': model, 'bin': float(low), 'n': len(p), 'predicted': points[-1][0], 'observed': points[-1][1]})
        if points:
            xx, yy = np.asarray(points).T
            ax.plot(xx, yy, color=color, marker=marker, linestyle=line, label=model)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)
    ax.set(xlabel='Predicted event probability', ylabel='Observed event fraction', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.savefig(base / 'calibration.png', dpi=180)
    fig.savefig(base / 'calibration.pdf')
    plt.close(fig)
    pd.DataFrame(calibration).to_csv(base / 'calibration.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout='constrained')
    selection = table[table.loop.eq('pooled') & table.k.eq(5) & table.variant.eq('main') & table.direction.eq('aut→man') & table.split.eq('time')]
    for model, (color, marker, line) in styles.items():
        p = selection[selection.model.eq(model) & selection.region.isin(['normal', 'transition'])].set_index('region').reindex(['normal', 'transition'])
        ax.plot([0, 1], p.event_recall, color=color, marker=marker, linestyle=line, label=model)
    ax.set(xticks=[0, 1], xticklabels=['Normal', 'Grade transition'], ylabel='Event recall', ylim=(-0.02, 1.02))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.savefig(base / 'event_recall.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout='constrained')
    primary = expanded[expanded.loop.eq('pooled') & expanded.k.eq(5) & expanded.variant.eq('main') & expanded.direction.eq('aut→man') & expanded.split.eq('time') & expanded.region.eq('all')]
    for model, (color, _, line) in styles.items():
        values = [v for array in primary.loc[primary.model.eq(model), 'lead_minutes'] for v in array]
        if values:
            ax.hist(values, bins=np.arange(0.5, 6.5), histtype='step', color=color, linestyle=line, label=model)
    ax.set(xlabel='First alarm lead time (min)', ylabel='Detected events', xticks=np.arange(1, 6))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.savefig(base / 'lead_time.png', dpi=180)
    plt.close(fig)
    status = {'mode': mode, 'completed_jobs': len(paths), 'fold_rows': len(folds), 'summary_rows': len(table), 'prediction_rows': len(pred), 'unsupported_folds': len(unsupported), 'report_scope': 'numerical diagnostics only'}
    (base / 'summary.json').write_text(json.dumps(status, indent=2))
    report = ROOT / 'generated/notes/S70_intervention_W4.md'
    lines = [f"# S70 개입 결정 분석: {('1개월 스모크' if mode == 'smoke' else '전체 평가')}", '', '## 정의와 평가 방법', '', '분 단위 라벨은 [t+1,t+k]의 전환 유무다. 주 분석은 k=5이고 k=15, 빠른 왕복·짧은 자동 포함 조건을 민감도로 계산한다. 적격 분은 모드[t]가 자동인 분이며 복귀 분석은 대칭적으로 수동인 분이다. 지속 ≤2분의 빠른 왕복·짧은 자동, 복귀 미관측, 지절 ±30분 사건과 그 창 [e−k,e]를 주 분석에서 제거한다.', '', '분 t가 끝난 시점의 관측값까지 입력한다. 측정값과 목표값은 같은 시각이며 평량 편차는 현재까지 60분 중앙값으로 중심화한다. 목표 변경 경과 시간은 목표값 차분으로 계산한다. 대치값과 지종 범주는 학습 자료에서만 적합한다. 60개 단일 규칙과 나무 깊이 2·4, 발화 문턱은 검증 사건 F1으로 선택한다. 문턱 후보는 나무의 모든 고유 확률, 부스팅 고유 확률의 101개 분위수다. 부스팅은 60회·최대 잎 7개로 고정한다.', '', '시간순 분할은 1~6월 학습·7~8월 검증·9~11월 시험이다. 1개월 스모크는 날짜 순서의 55%·20%·25%를 사용한다. LOT 그룹 5겹의 각 학습 풀에서 검증 LOT 20%를 분리한다. [t,t+k]가 실제 LOT 경계 또는 분할 경계를 넘는 분을 제거해 한 사건의 양성 분을 같은 겹에 둔다.', '', '발화는 점수가 문턱 이상이 되는 상승 에지이며 같은 자동 구간에서 뒤 k분을 억제한다. 사건의 [e−k,e−1] 중 적격 시험 예측 분이 하나라도 있는 사건을 분모로 삼고, 구간은 그 예측 분으로 나눈다. 따라서 경계 사건은 두 구간에 속할 수 있다. 발화 정밀도는 적중 발화/전체 발화이며 PR-AUC는 분 단위 average precision이다. 적중 수와 사건 수, 첫 발화에서 사건까지의 분을 저장한다. 사건 시각 기준 구간 재현율은 event_region_sensitivity.csv에 별도 저장한다.', '', '우연 기준 C는 같은 루프·시험 기간·예측 구간의 적격 평가 분에서 사건 수를 보존해 무작위 재배치한 50회 재현율 평균이다. 합산 R과 C는 사건 수로 가중하고 E=(R−C)/(1−C)를 다시 계산한다. PR-AUC는 저장된 시험 예측을 합쳐 계산한다.', '', '## 실행 산출물', '', f'완료 작업은 {len(paths)}개이고 분할·모형·구간 조합은 {len(folds)}개다. 합산표는 `{base.relative_to(ROOT)}/metrics.csv`, 분할별 표는 `fold_metrics.csv`, 지지 영역은 `support.csv`, 나무 규칙은 `tree_rules.json`에 저장했다.', '', '교정 곡선은 calibration.png, 구간별 사건 재현율은 event_recall.png, 첫 발화 시간차는 lead_time.png에 제시했다. 교정 자료는 calibration.csv에 저장했다. 지지 영역 표는 학습 사건 직전 특성의 최솟값·최댓값과 시험 범위, 그 범위 밖의 시험 분 비율을 제시한다.', '', '## 기계적 대조', '', 'metrics.csv는 등록 범주를 위에서부터 차례 적용한 mechanical_category와 경계 여부를 기록한다. 경계는 P·C를 고정하고 사건 적중 수를 ±1 바꿨을 때 범주가 바뀌는 경우다. gbm_tree_gap.csv는 선택한 부스팅과 나무의 PR-AUC 차를 기록한다. split_threshold_comparison.csv는 두 분할의 기계적 범주가 다르면 보류로 표시한다.', '', '## 가정과 한계', '', f'학습·검증 지지가 부족한 분할은 {len(unsupported)}개이며 unsupported_folds.csv에 분모와 사유를 기록했다. 해당 분할은 공통 비교 모집단을 유지하기 위해 세 모형 모두의 합산에서 제외했다.', '', '모드의 마지막 기록을 다음 기록까지 유지한다고 가정한다. 모드 전환은 수동 개입 후보이며 자동 해제를 포함할 수 있다. 교대는 06·14·22시로 가정한다. 정상·지종 전환 구간은 S68의 사후 구간 표지로 층화하며 예측 입력에는 넣지 않는다. 지지 영역의 변수별 범위는 다변량 결합 지지를 보장하지 않는다.', '', '## 구현 인수인계', '', '대상 버전은 S70 명세 v0.2이며 실행 provenance.json에 코드·명세 고정 정보를 저장한다. 이번 구현은 과거 특성·미래 라벨·두 분할·모형·사건 지표·집계 경로를 추가했다. 합성 입력 시험으로 미래 값 불변성, 라벨 경계, 분할 purge, 사건 재현율을 검사했다. 입력은 기존 S68 캐시와 비식별 대응표를 재사용했다. 전체 실행은 동일 런처의 full 모드로 수행한다.', '']
    report.write_text('\n'.join(lines))
    print(json.dumps(status), flush=True)
    return status
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    aggregate(args.mode)

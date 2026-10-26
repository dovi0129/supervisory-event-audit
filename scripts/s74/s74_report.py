from __future__ import annotations
import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paper_figstyle as ps
from s74_episodes import ROOT
sys.path.insert(0, str(PACKAGE_ROOT / 'scripts/paper1_figs'))
from bbox_overlap_check import audit_fig
COLORS = {'target': '#0072B2', 'mode': '#D55E00', 'op': '#009E73', 'dcs': '#555555'}
LABELS = {'target': 'Quality targets', 'mode': 'Mode switches', 'op': 'Manual outputs', 'dcs': 'DCS candidates'}

def table(frame):

    def cell(v):
        if pd.isna(v):
            return '—'
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else f'{v:.3f}'
        return str(v).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] + ['| ' + ' | '.join((cell(x) for x in row)) + ' |' for row in frame.itertuples(index=False, name=None)])

def save_figure(fig, path, audits):
    findings = audit_fig(fig, verbose=False)
    critical = [f for f in findings if f.severity.lower() == 'critical']
    if critical:
        raise ValueError('Figure layout critical: ' + str(critical))
    text_sizes = [float(t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    if min(text_sizes) < 7:
        raise ValueError('Figure font smaller than 7 pt')
    audits.append({'figure': path.name, 'critical': len(critical), 'warnings': len(findings), 'minimum_font_pt': min(text_sizes), 'width_in': float(fig.get_figwidth()), 'layout_check': 'matplotlib renderer bounds and overlap'})
    plt.rcParams['savefig.pad_inches'] = 0.05
    ps.save(fig, str(path), dpi=240)
    plt.close(fig)

def report_and_figures(dest):
    meta = json.loads((dest / 'metadata.json').read_text())
    episodes = pd.read_parquet(dest / 'episodes.parquet')
    actions = pd.read_parquet(dest / 'episode_actions.parquet')
    distributions = pd.read_csv(dest / 'distributions.csv')
    sequence = pd.read_csv(dest / 'sequence_consistency.csv')
    ratios = pd.read_csv(dest / 'variance_ratios.csv')
    audits = []
    ps.apply(width='text', aspect=0.62)
    fig, ax = plt.subplots()
    for cohort, color, style in [('transition', '#0072B2', '-'), ('normal_manual', '#D55E00', '--')]:
        values = episodes.loc[episodes.cohort.eq(cohort), 'duration_min'].sort_values().to_numpy()
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post', label=cohort, color=color, linestyle=style)
    ax.set(xlabel='Episode duration (min)', ylabel='Cumulative fraction', xlim=(0, 240), ylim=(0, 1.02))
    ax.legend(loc='lower right')
    save_figure(fig, dest / 'duration', audits)
    ps.apply(width='text', aspect=0.62)
    fig, ax = plt.subplots()
    trans_ids = episodes.loc[episodes.cohort.eq('transition'), 'episode_id']
    selected = actions[actions.episode_id.isin(trans_ids)]
    numeric = []
    bins = np.arange(0, 241, 5)
    for kind in ['target', 'mode', 'op', 'dcs']:
        values = selected.loc[selected.kind.eq(kind), 'relative_min']
        counts, _ = np.histogram(values, bins=bins)
        numeric.extend(({'kind': kind, 'left_min': int(a), 'right_min': int(b), 'count': int(n)} for a, b, n in zip(bins[:-1], bins[1:], counts)))
        if counts.sum() != len(values):
            raise ValueError('Action figure bin counts do not conserve actions')
        ax.stairs(counts, bins, color=COLORS[kind], label=LABELS[kind], linestyle={'target': '-', 'mode': '--', 'op': '-.', 'dcs': ':'}[kind])
    ax.set(xlabel='Time from grade change (min)', ylabel='Recorded candidate actions', xlim=(0, 240))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    save_figure(fig, dest / 'action_timing', audits)
    pd.DataFrame(numeric).to_csv(dest / 'figure_action_timing.csv', index=False)
    example_path = dest / 'sequence_examples.csv'
    if example_path.exists():
        examples = pd.read_csv(example_path)
        ps.apply(width='text', aspect=0.55, nrows=max(1, len(examples)))
        fig, axs = plt.subplots(len(examples), 1, squeeze=False, sharex=True)
        role_labels = {'reel 평량 목표': 'Reel basis weight', 'size 평량 목표': 'Size basis weight', 'reel 수분 목표': 'Reel moisture', 'size 수분 목표': 'Size moisture', '회분 목표': 'Ash target', '분사/와이어 속도비 목표': 'Jet/wire target'}
        role_labels['reel 평량 목표'] = 'Reel basis weight'
        roles = meta['roles']
        ylabels = [role_labels.get(r, r) for r in roles] + ['Other modes', 'Manual output', 'DCS candidate']
        for i, r in enumerate(examples.itertuples()):
            ax = axs[i, 0]
            a = actions[actions.episode_id.eq(r.episode_id)]
            for kind in ['target', 'mode', 'op', 'dcs']:
                g = a[a.kind.eq(kind)]
                y = [roles.index(loop) if kind == 'target' else len(roles) + {'mode': 0, 'op': 1, 'dcs': 2}[kind] for loop in g.loop]
                ax.scatter(g.relative_min, y, s=13, color=COLORS[kind], marker={'target': 'o', 'mode': 's', 'op': '^', 'dcs': 'x'}[kind])
            ax.set(yticks=np.arange(len(ylabels)), yticklabels=ylabels, ylim=(-0.8, len(ylabels) + 0.2), xlim=(-2, 240))
            ax.text(0.98, 0.96, f'({chr(97 + i)}) {r.pair}', transform=ax.transAxes, ha='right', va='top', fontsize=8)
        axs[-1, 0].set_xlabel('Time from grade change (min)')
        save_figure(fig, dest / 'sequence_examples', audits)
    (dest / 'figure_audit.json').write_text(json.dumps(audits, indent=2))
    reason = pd.read_csv(dest / 'end_reasons.csv')
    overview = episodes.groupby('cohort').agg(episodes=('episode_id', 'size'), break_episodes=('break_included', 'sum'), median_duration=('duration_min', 'median'), median_actions=('action_count', 'median'), power_missing_fraction=('power_missing_fraction', 'mean')).reset_index()
    stats = distributions[distributions.measurement.isin(['duration_min', 'outside_band_min', 'weight_iae_g_m2_min', 'moisture_iae_pp_min', 'action_count', 'first_action_min'])]
    sensitivity = pd.read_parquet(dest / 'sensitivity_tight/episodes.parquet')
    sens = pd.concat([episodes.assign(band_label='BW ±2, moisture ±0.3'), sensitivity.assign(band_label='BW ±1, moisture ±0.2')])
    sens = sens.groupby(['band_label', 'cohort']).agg(episodes=('episode_id', 'size'), confirmed=('end_reason', lambda x: int(x.eq('stable_band').sum())), median_duration=('duration_min', 'median')).reset_index()
    recipe = json.loads((dest / 'recipe_summary.json').read_text())
    sequence_columns = ['pair', 'episodes', 'empty_sequences', 'normalized_edit_mean', 'normalized_edit_mean_null_p05', 'normalized_edit_mean_null_p95', 'first_action_variance']
    representative = sequence.sort_values('episodes', ascending=False).head(12)[sequence_columns] if 'episodes' in sequence else sequence
    ratio_display = ratios[[c for c in ['measurement', 'n', 'groups', 'within', 'between', 'ratio', 'null_ratio_p05', 'null_ratio_p95'] if c in ratios]]
    lines = ['# 지종 전환과 정상 생산 목표 조정 에피소드', '', '## 정의와 집계 방법', '', '지종|평량 결합키가 바뀐 분을 전환 시작으로 삼았다. 결합키는 최대 30분 앞값 채움 뒤 인접한 두 키가 모두 관측된 경우에 비교했다. 이 표의 grade n은 결합키를 다시 비식별화한 이름이며 S68의 단일 지종 번호와 구분한다.', '', '주 밴드는 reel 평량 측정−목표 ±2 g/m², reel 수분 측정−목표 ±0.3 %p이며 민감도는 ±1 g/m²·±0.2 %p다. 중심화 전 평량 편차를 사용했다. 기존 유효 분 플래그와 두 편차의 관측이 모두 있어야 밴드 체류로 센다. 결측·무효 분은 연속 체류를 끊는다.', '', '전환 종료는 15분 연속 밴드 체류를 확인한 분의 끝, 다음 결합키 변경, 시작 후 240분 중 가장 이른 경계다. 표본 끝은 record_end로 기록한다. [start,end_exclusive) 안의 1분 표본을 포함한다. band_entry_min은 연속 체류 시작, band_confirmed_min은 확인까지 걸린 시간이다. 종료 경계가 같으면 밴드 확인을 우선하고, 다음 지종과 240분이 같으면 240분으로 기록한다.', '', '정상 생산 비교군은 전환 구간 밖 정상 생산 분에서 시작한 자동→수동 구간이다. 밴드 확인·자동 복귀·다음 결합키 변경·240분 중 가장 이른 경계로 자르고, 잘린 구간 안에 여섯 품질 목표 후보 변경이 있는 경우만 포함한다. 자동 복귀를 관측하지 못한 구간은 제외한다. 서로 다른 루프의 수동 구간은 겹칠 수 있으며 별도 에피소드로 센다.', '', '행동은 기존 S68의 운전자 후보인 여섯 품질 목표 변경, 11개 모드 전환, 수동 출력, DCS 설정값 후보를 사용했다. 방향·크기는 S68의 기록된 차분이고 시각은 시작에 대한 상대 분이다. 동시에 기록된 품질 목표는 집합 한 개로 묶어 순서를 부여하지 않았다.', '', '## 에피소드와 종료 사유', '', table(overview), '', table(reason), '', table(stats), '', '밴드 밖 분 수는 유효·관측된 분 중 밴드 밖인 분이다. 무효 분은 quality_invalid_min으로 따로 기록했다. 편차 적분은 같은 유효 분에서 절댓값을 1분 간격으로 더했다. 속도 변화는 마지막 관측 속도−시작 직전 속도다. 전력은 여섯 채널이 모두 관측된 분의 채널 합 평균과 관측 적분을 저장하고 결측률을 병기했다.', '', '## 밴드 민감도', '', table(sens), '', '## 반복된 지종 쌍', '', f"에피소드 3개 이상인 결합키 쌍은 {recipe['eligible_pairs']}개이며 해당 에피소드는 {recipe['eligible_episodes']}개다. 모든 쌍과 반복 쌍 목록은 grade_pairs.csv·grade_pairs_at_least3.csv에 저장했다. 다음 표는 빈도가 높은 쌍부터 최대 12개다.", '', table(representative), '', '## 목표 변경 크기와 타이밍의 분산', '', '순서는 동시 변경 집합을 토큰으로 한 Levenshtein 거리이며, 정규화 거리는 두 순서의 최대 길이로 나눈 값이다. 둘 다 비어 있으면 0이다. empty_sequences와 비교 수를 병기했다. 목표 크기는 에피소드별 여섯 목표의 절대 변경량 합이며 CV는 표본 표준편차/평균이다. 목표 변경이 없는 에피소드도 크기 0으로 포함하고 평균 0이면 CV를 빈값으로 둔다.', '', '같은 쌍 안 분산 W는 각 에피소드와 해당 쌍 평균의 제곱 차 합/N, 쌍 간 분산 B는 쌍 표본 수로 가중한 쌍 평균과 전체 평균의 제곱 차 합/N이다. 표는 W/B다. net은 부호 있는 변경량 합, abs는 절대 변경량 합, first는 해당 목표 첫 변경 시각이다. 타이밍에서 행동이 없는 값은 결측으로 두었다. 레시피별 CV와 첫 목표 변경 시각 분산은 recipe_by_pair.csv에 저장했다.', '', table(ratio_display), '', '무작위 기준은 반복 쌍 안의 에피소드들 사이에서 쌍 이름을 50회 섞으며 각 쌍의 표본 수를 보존한다. 순서·변경량·타이밍은 함께 이동한다. 난수 시드는 7401이다. 기준 분포는 sequence_permutations.csv·pair_permutations.csv·variance_permutations.csv에 저장했다.', '', '## 그림', '', 'duration.png: 주 밴드에서 전환·정상 비교군의 에피소드 지속 시간 누적 분포다. 240분과 다음 지종 때문에 종료된 경우도 포함한다.', '', 'action_timing.png: 주 밴드 전환 에피소드 안의 행동을 시작 후 5분 간격으로 센 분포다. 각 막대 구간은 [왼쪽,오른쪽)이며 마지막 구간만 오른쪽 끝을 포함한다.', '', 'sequence_examples.png: 빈도가 가장 높은 세 반복 지종 쌍에서 목표 변경이 있는 에피소드 중 해당 쌍 전체에 대한 정규화 편집 거리 합이 최소인 에피소드를 하나씩 보였다. 같은 시각의 점에는 선후관계를 부여하지 않았다. 선택된 쌍·에피소드는 sequence_examples.csv에 저장했다. 모든 그림은 폭 5.5인치이며 축 단위는 분이다.', '', '## 가정과 확인 범위', '', '전력 채널의 물리 단위와 채널 간 포함 관계는 확인되지 않아 채널 합의 관측 단위로 기록했다. 관측 적분은 결측 구간을 보간하거나 1시간 에너지로 환산하지 않았다. 평균 전력 결측률 표는 에피소드별 결측률의 평균이고 각 에피소드 표에는 유효 관측 수를 함께 제공한다.', '', '행동은 기록된 운전자 후보이며 자동 해제·레시피·추종 기록의 혼입을 분리하지 않는다. 같은 분의 조작 순서는 관측되지 않는다. 같은 결합키 쌍 안의 반복도 겹치는 외부 조건을 공유할 수 있다. 이 표는 에피소드와 기록의 서술 통계다.', '', '대상 버전: S74 브리프와 기존 S68·S70 캐시. 이번 구현: 에피소드 경계·행동 표·반복 쌍 요약. 수행 검사: 종료 경계 정답 예제, 편집 거리·분산 기대값, 행동 수와 구간 소속 대조. 재사용 검증: 기존 분 정렬·비식별 대응과 S70 시점 검증. 결론과 한계: 요청한 서술 표와 그림을 저장했고 물리 전력 단위는 위 범위에 따른다.', '']
    if not meta['month']:
        lines.extend(['![에피소드 지속 시간](../artifacts/transition/duration.png)', '', '![행동 시각 분포](../artifacts/transition/action_timing.png)', '', '![반복 지종 쌍의 행동 순서](../artifacts/transition/sequence_examples.png)', ''])
    if meta['month']:
        (dest / 'smoke_report.md').write_text('\n'.join(lines))
    else:
        (ROOT / 'generated/notes/S74_transition_episodes_W4.md').write_text('\n'.join(lines))
    print(json.dumps({'figures': len(audits), 'critical_layout_findings': 0}), flush=True)

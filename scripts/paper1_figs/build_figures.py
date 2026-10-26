import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
from pathlib import Path
import csv
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_figstyle as ps
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures'
HERE = Path(__file__).resolve().parent
from bbox_overlap_check import audit_fig

def read(relative):
    with (ROOT / relative).open() as handle:
        return list(csv.DictReader(handle))

def select(rows, **criteria):
    matches = [r for r in rows if all((r[k] == v for k, v in criteria.items()))]
    assert len(matches) == 1, criteria
    return matches[0]

def axes(height=3.3, left=0.34):
    ps.apply(height)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=left, right=0.97, bottom=0.19, top=0.8)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.spines['left'].set_visible(False)
    return (fig, ax)

def point(ax, x, y, key, interval=None):
    style = ps.style(key)
    if interval is not None:
        lo, hi = interval
        assert lo <= x <= hi
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=style.pop('marker'), linestyle='none', capsize=2.5, markersize=4.5, **style)
    else:
        ax.plot(x, y, linestyle='none', markersize=4.5, **style)

def legend(fig, keys, criterion=None):
    handles = [Line2D([], [], linestyle='none', markersize=4.5, **ps.style(k)) for k in keys]
    if criterion is not None:
        handles += [Line2D([], [], color='#666666', ls='--', lw=0.8, label=f'Pre-specified criterion: {criterion:.2f}')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 0.99), ncol=2, columnspacing=1.3, handlelength=1.6)

def finish(fig, name):
    findings = audit_fig(fig)
    ps.save(fig, OUT / name)
    plt.close(fig)
    return [str(f) for f in findings]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    layout = {}
    q1 = read('artifacts/stage1_audit/q1_summary.csv')
    fig, ax = axes(3.7)
    regions = [('정상 생산', 'Normal production', 327), ('지종 전환', 'Grade transition', 1520)]
    labels = []
    expected = [0.0993, 0.1715, 0.3043, 0.3014]
    for i, (region, title, n) in enumerate(regions):
        for j, side in enumerate(('before', 'inside')):
            y = 3 - (2 * i + j)
            labels.append(f'{title}\n' + ('Before onset' if side == 'before' else 'Within window') + f' (n={n:,})')
            for cap, offset in (('full', 0.18), ('cap30', 0.0)):
                r = select(q1, population='main_no_cross', loop='pooled', region=region, cap=cap, side=side)
                value, lo, hi = [float(r[k]) for k in ('delta', 'ci_low', 'ci_high')]
                assert int(r['windows']) == n
                recomputed = float(r['p_observed']) - float(r['p_chance'])
                assert abs(value - recomputed) < 1e-12
                if cap == 'full':
                    assert round(value, 4) == expected[2 * i + j]
                point(ax, value, y + offset, cap, (lo, hi))
                checks.append(dict(figure=1, row=f'{title}; {side}; {cap}', n=n, csv=value, independent=recomputed, low=lo, high=hi))
    elapsed = read('artifacts/stage1_audit/s89/q1_elapsed_controls.csv')
    for side, y in [('before', 1), ('inside', 0)]:
        r = select(elapsed, method='other_episode_elapsed', cap='full', side=side, elapsed_phase='all', loop='pooled')
        v, lo, hi = [float(r[k]) for k in ['delta', 'ci_low', 'ci_high']]
        point(ax, v, y - 0.18, 'elapsed', (lo, hi))
        calc = float(r['p_observed']) - float(r['p_reference'])
        assert abs(v - calc) < 1e-12
        checks.append(dict(figure=1, row='Grade transition; ' + side + '; elapsed', n=int(r['matched_windows']), csv=v, independent=calc, low=lo, high=hi))
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.set_ylim(-0.5, 3.55)
    ax.set_xlim(-0.015, 0.36)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.axvline(0, color='#888888', lw=0.7, ls=':')
    ax.axvline(0.1, color='#666666', lw=0.8, ls='--')
    ax.text(0.175, 3.18, '0.0993 < 0.10', fontsize=8, va='center')
    ax.set_xlabel(ps.UNITS['excess'])
    legend(fig, ['full', 'cap30', 'elapsed'], 0.1)
    layout['fig1_bundle_excess'] = finish(fig, 'fig1_bundle_excess')
    q2 = read('artifacts/stage1_audit/q2_paired_comparisons.csv')
    metrics = read('artifacts/stage1_audit/q2_metrics.csv')
    fig, ax = axes(3.15, 0.3)
    for base_y, contrast, a, b in ((1.8, 'R1-R0p', 'R1', 'R0p'), (0.3, 'R2-R1', 'R2', 'R1')):
        for split, offset in (('time', 0.17), ('group', -0.17)):
            r = select(q2, split=split, stratum='seen', contrast=contrast)
            v, lo, hi = [float(r[k]) for k in ('accuracy_difference', 'ci_low', 'ci_high')]
            ma = select(metrics, split=split, stratum='seen', model=a)
            mb = select(metrics, split=split, stratum='seen', model=b)
            calc = float(ma['accuracy']) - float(mb['accuracy'])
            assert abs(v - calc) < 1e-12
            assert ma['decisions'] == mb['decisions'] == r['decisions']
            point(ax, v * 100, base_y + offset, split, (lo * 100, hi * 100))
            checks.append(dict(figure=2, row=f'{split}; {contrast}', n=int(r['decisions']), episodes=int(r['episodes']), csv=v, independent=calc, low=lo, high=hi))
    ax.set_yticks([1.8, 0.3], ['Setpoint/action\nR1 − R0′', 'Measured state\nR2 − R1'])
    ax.set_ylim(-0.4, 2.5)
    ax.set_xlim(-4, 18)
    ax.set_xticks([0, 5, 10, 15])
    ax.axvline(0, color='#888888', lw=0.7, ls=':')
    ax.axvline(5, color='#666666', lw=0.8, ls='--')
    ax.set_xlabel(ps.UNITS['accuracy'])
    legend(fig, ['time', 'group'], 5)
    layout['fig2_information_increment'] = finish(fig, 'fig2_information_increment')
    q3 = read('artifacts/intervention/s73/recent_target_counts.csv')
    fig, ax = axes(2.85, 0.27)
    for y, region, label in ((1.5, 'normal', 'Normal production'), (0, 'transition', 'Grade transition')):
        r = select(q3, scope='time', loop='pooled', region=region)
        for key, offset in (('recent', 0.17), ('no_recent', -0.17)):
            h, n = (int(float(r[key + '_hits'])), int(r[key + '_events']))
            assert (h, n) == {('normal', 'recent'): (17, 40), ('normal', 'no_recent'): (2, 83), ('transition', 'recent'): (245, 364), ('transition', 'no_recent'): (17, 363)}[region, key]
            v = h / n
            point(ax, v, y + offset, key)
            ax.text(v + 0.025, y + offset, f'{h}/{n}', fontsize=8, va='center')
            checks.append(dict(figure=3, row=f'{label}; {key}', n=n, hits=h, csv=f'{h}/{n}', independent=v))
    ax.set_yticks([1.5, 0], ['Normal production', 'Grade transition'])
    ax.set_ylim(-0.65, 2.15)
    ax.set_xlim(0, 0.88)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(ps.UNITS['recall'])
    legend(fig, ['recent', 'no_recent'])
    layout['fig3_onset_recall'] = finish(fig, 'fig3_onset_recall')
    (HERE / 'numeric_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    (HERE / 'layout_checks.json').write_text(json.dumps(layout, indent=2) + '\n')
    print(f'Built 3 PDF/PNG pairs at {ps.WIDTH_CM} cm, 600 dpi. Checked {len(checks)} aggregate rows.')
if __name__ == '__main__':
    main()

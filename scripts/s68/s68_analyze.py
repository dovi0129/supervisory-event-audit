import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
WS = str(WORKSPACE)
PRIV = os.path.join(WS, str(record_path('artifacts_private')))
OUT = os.path.join(WS, 'artifacts/operator_policy')
W = 60
SEED = 0
CAT = {'target': '목표 설정값', 'op': '수동 출력', 'mode': '모드 전환', 'dcs': 'DCS 설정값(모드 없음)'}
CAT_EN = {'목표 설정값': 'target SP step', '수동 출력': 'manual OP step', '모드 전환': 'mode switch', 'DCS 설정값(모드 없음)': 'DCS SP step (no mode tag)'}
DEV_TH = 0.3
RULE_GRID = {'dY': [0.1, 0.2, 0.3, 0.5, 0.8], 'dYs': [0.2, 0.3, 0.5, 0.8, 1.0], 'dGc': [0.5, 1.0, 2.0, 3.0, 5.0]}
RULE_K = [1, 3, 5, 10]
TOLS = [5, 15]
N_PERM = 50

def q(x, ps):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 4) for v in np.percentile(x, ps)] if len(x) else [None] * len(ps)

def run_end_len(cond):
    out = np.zeros(len(cond), int)
    c = 0
    for i, v in enumerate(cond):
        c = c + 1 if v else 0
        out[i] = c
    return out

def run_start_dist(cond):
    rl = run_end_len(cond)
    return np.where(cond, rl - 1, -1)

def hit_rate(src, dst, lo_off, hi_off):
    if len(src) == 0:
        return None
    if len(dst) == 0:
        return 0.0
    dst = np.sort(dst)
    lo = np.searchsorted(dst, src + lo_off, 'left')
    hi = np.searchsorted(dst, src + hi_off, 'right')
    return float(np.mean(hi > lo))

def main():
    rng = np.random.default_rng(SEED)
    cols = json.load(open(os.path.join(PRIV, str(record_path('s68_columns_private_json')))))
    need = ['time', 'moisture', 'moisture_target', 'weight', 'weight_target', 'speed', 'pressure_after', 'size_moisture', 'size_moisture_target'] + cols['kw']
    df = pd.read_parquet(os.path.join(PRIV, 's68_cache.parquet'), columns=need)
    M = pd.read_parquet(os.path.join(PRIV, 's68_minutes.parquet'))
    assert (M['time'].to_numpy() == df['time'].to_numpy()).all()
    E = pd.read_parquet(os.path.join(OUT, 'events.parquet'))
    LP = json.load(open(os.path.join(OUT, 'loops.json')))
    n = len(df)
    t = pd.DatetimeIndex(df['time'])
    valid = M['valid'].to_numpy(bool)
    region = M['region'].to_numpy(object)
    normal = region == '정상 생산'

    def sig(c):
        x = df[c].to_numpy(float).copy()
        x[~valid] = np.nan
        return x
    Y, R, G, Gs, V, P = (sig(c) for c in ('moisture', 'moisture_target', 'weight', 'weight_target', 'speed', 'pressure_after'))
    Ys, Rs = (sig('size_moisture'), sig('size_moisture_target'))
    KW = df[cols['kw']].sum(axis=1, min_count=len(cols['kw'])).to_numpy(float)
    KW[~valid] = np.nan
    dY, dYs, dG = (Y - R, Ys - Rs, G - Gs)
    lot = M['lot'].to_numpy(object)
    lot_med = pd.Series(dG).groupby(pd.Series(lot)).transform('median').to_numpy(float)
    dGc = dG - lot_med
    S = dict(n_minutes=int(n), days=round(n / 1440, 2), n_lots=int(pd.Series(lot).nunique()), region_minute_share={k: round(float(v), 4) for k, v in pd.Series(region).value_counts(normalize=True).items()}, dG_offset_normal=q(dG[normal], [10, 50, 90]), kw_missing_normal=round(float(np.isnan(KW[normal]).mean()), 4))
    O = E[E['class'] == '운전자 후보'].copy()
    O['cat'] = O['kind'].map(CAT)
    O['i'] = np.searchsorted(t.values, O['time'].values)
    A = E[E['class'] == '자동']
    S['step1'] = dict(counts=E.groupby(['kind', 'class']).size().rename('n').reset_index().to_dict('records'), by_loop=O.groupby(['cat', 'loop']).size().rename('n').reset_index().to_dict('records'), auto_by_basis=A.groupby('basis').size().to_dict(), flag_share={c: round(float(O[c].mean()), 4) for c in O.columns if c.startswith('flag_')}, flag_share_by_cat={cat: {c: round(float(g[c].mean()), 4) for c in g.columns if c.startswith('flag_')} for cat, g in O.groupby('cat')}, mode_dir=O[O.kind == 'mode'].groupby('shape').size().to_dict(), thresholds={k: dict(thr=v.get('thr'), n_zero_dropped=v.get('n_zero_dropped'), kind=v['kind'], frac_missing=None if v.get('frac_missing') is None else round(v['frac_missing'], 4)) for k, v in LP['loops'].items()}, params=LP['params'])
    mo = O[O.kind == 'mode'].sort_values(['loop', 'time'])
    back = mo.groupby('loop')['i'].diff().le(2)
    S['step1']['mode_flip_within_2min'] = int(back.sum())
    days = n / 1440
    nlots = S['n_lots']
    rows = []
    for (cat, lp), g in O.groupby(['cat', 'loop']):
        gi = np.sort(g['i'].to_numpy())
        per_lot = g.groupby('lot').size()
        rows.append(dict(cat=cat, loop=lp, n=len(g), per_day=round(len(g) / days, 2), per_lot=round(len(g) / nlots, 3), lots_with_event=round(len(per_lot) / nlots, 4), abs_size_q25_50_75_90=None if cat == '모드 전환' else q(np.abs(g['size']), [25, 50, 75, 90]), gap_min_q10_50_90=q(np.diff(gi), [10, 50, 90]) if len(gi) > 1 else None, share_normal=round(float((g.region == '정상 생산').mean()), 3), share_gc=round(float((g.region == '지종 전환').mean()), 3)))
    S['step2_loops'] = rows
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 's68_step2_loops.csv'), index=False)

    def minute_frac(sub):
        m = np.zeros(n, bool)
        m[sub['i'].to_numpy()] = True
        return m
    H1 = {}
    sets = {'전체 후보': O, **{c: g for c, g in O.groupby('cat')}, '모드 전환 중 지절 ±30분 제외': O[~((O.kind == 'mode') & O.flag_brk_near)], '모드 전환 제외': O[O.kind != 'mode'], '목표 설정값 단독 변경(동시 변경 제외)': O[(O.kind == 'target') & ~O.flag_cochange], '전체 후보에서 목표 동시 변경 제외': O[~((O.kind == 'target') & O.flag_cochange)]}
    for name, sub in sets.items():
        m = minute_frac(sub)
        H1[name] = {'전체 분': round(float(m.mean()), 5), **{r: round(float(m[region == r].mean()), 5) for r in ('정상 생산', '지종 전환', '지절', '기타 무효')}, 'n_events': int(len(sub))}
    S['step2_H1_minute_fraction'] = H1
    reg_rows = []
    for cat, g in [('전체 후보', O)] + list(O.groupby('cat')):
        es = g['region'].value_counts(normalize=True)
        reg_rows.append({'cat': cat, **{r: round(float(es.get(r, 0) / S['region_minute_share'][r]), 2) for r in S['region_minute_share']}})
    S['step2_region_rate_ratio'] = reg_rows
    mins_hour = pd.Series(t.hour).value_counts().sort_index()
    hour_rate = {}
    for cat, g in [('전체 후보', O)] + list(O.groupby('cat')):
        c = g['time'].dt.hour.value_counts().reindex(range(24), fill_value=0)
        hour_rate[cat] = (c / mins_hour * 1000).round(2).tolist()
    S['step2_hour_rate_per_1000min'] = hour_rate
    shift = M['shift'].to_numpy(object)
    crew = M['crew'].to_numpy(object)

    def rate_by(lab_min, lab_ev, sub):
        mins = pd.Series(lab_min).value_counts()
        ev = pd.Series(lab_ev).value_counts()
        return {str(k): round(float(ev.get(k, 0) / mins[k] * 1000), 3) for k in sorted(mins.index, key=str)}
    S['step2_shift_rate'] = {cat: rate_by(shift, shift[g['i']], g) for cat, g in [('전체 후보', O)] + list(O.groupby('cat'))}
    S['step2_shift_rate_normal'] = {cat: rate_by(shift[normal], shift[g['i'][g.region == '정상 생산']], g) for cat, g in [('전체 후보', O)] + list(O.groupby('cat'))}
    S['step2_crew_rate_normal'] = {cat: rate_by(np.asarray(crew[normal], str), np.asarray(crew[g['i'][g.region == '정상 생산']], str), g) for cat, g in [('전체 후보', O)] + list(O.groupby('cat'))}
    wd = t.dayofweek.to_numpy()
    S['step2_weekday_rate'] = rate_by(wd, wd[O['i']], O)
    mod = t.hour.to_numpy() * 60 + t.minute.to_numpy()
    near_shift = np.zeros(n, bool)
    for h in (6, 14, 22):
        dd = np.abs((mod - h * 60 + 720) % 1440 - 720)
        near_shift |= dd <= 30
    S['step2_shift_change_rate'] = {cat: dict(near=round(float(np.isin(np.arange(n)[near_shift & normal], g['i']).sum() / (near_shift & normal).sum() * 1000), 3), other=round(float(np.isin(np.arange(n)[~near_shift & normal], g['i']).sum() / (~near_shift & normal).sum() * 1000), 3)) for cat, g in [('전체 후보', O)] + list(O.groupby('cat'))}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    wk = O.set_index('time').groupby('cat').resample('W')['loop'].size().unstack(0).fillna(0)
    for c in wk.columns:
        ax[0].plot(wk.index, wk[c], label=CAT_EN[c])
    ax[0].set_ylabel('events per week')
    ax[0].legend(fontsize=7)
    ax[0].tick_params(axis='x', labelrotation=30)
    for c in CAT:
        cc = CAT[c]
        if cc in hour_rate:
            ax[1].plot(range(24), hour_rate[cc], marker='o', ms=3, label=CAT_EN[cc])
    for h in (6, 14, 22):
        ax[1].axvline(h, color='0.6', ls=':', lw=0.8)
    ax[1].set_xlabel('hour of day')
    ax[1].set_ylabel('events per 1000 min')
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F1_sparsity.png'), dpi=150)
    plt.close(fig)
    off = np.arange(-W, W + 1)

    def win(x, idx):
        idx = idx[(idx - W >= 0) & (idx + W < n)]
        return (x[idx[:, None] + off[None, :]], idx)

    def rel(x, idx):
        m, idx2 = win(x, idx)
        return m - x[idx2 - 1][:, None]
    norm_idx = np.where(normal)[0]
    rnd_normal = rng.choice(norm_idx[(norm_idx > W) & (norm_idx < n - W - 1)], 5000, replace=False)
    groups3 = {c: g['i'].to_numpy()[g.region.to_numpy() == '정상 생산'] for c, g in O.groupby('cat')}
    vars3 = [('dY', dY, False, 'reel moisture − target (%)'), ('dGc', dGc, False, 'reel BW dev., LOT-centred (g/m²)'), ('V', V, True, 'Δ reel speed (m/min)'), ('P', P, True, 'Δ after-dryer pressure'), ('KW', KW, True, 'Δ power sum (unit n/a)')]
    fig, ax = plt.subplots(len(vars3), len(groups3), figsize=(3.2 * len(groups3), 2.1 * len(vars3)), sharex=True)
    traj = {}
    for j, (cname, gi) in enumerate(groups3.items()):
        for k, (vn, x, relative, lab) in enumerate(vars3):
            m = rel(x, gi) if relative else win(x, gi)[0]
            mr = rel(x, rnd_normal) if relative else win(x, rnd_normal)[0]
            a = ax[k, j]
            if len(m):
                with np.errstate(all='ignore'):
                    med = np.nanmedian(m, 0)
                    lo = np.nanpercentile(m, 10, 0)
                    hi = np.nanpercentile(m, 90, 0)
                    mean = np.nanmean(m, 0)
                a.fill_between(off, lo, hi, color='C0', alpha=0.2, lw=0)
                a.plot(off, med, color='C0', lw=1.2, label='median (10–90%)')
                a.plot(off, mean, color='C0', lw=0.8, ls='--', label='mean')
                traj[f'{cname}|{vn}'] = dict(n=int(m.shape[0]), median_at=dict(zip(['-60', '-30', '-10', '-1', '+10', '+30', '+60'], [round(float(med[W + d]), 4) if np.isfinite(med[W + d]) else None for d in (-60, -30, -10, -1, 10, 30, 60)])))
            with np.errstate(all='ignore'):
                a.plot(off, np.nanmedian(mr, 0), color='0.4', lw=0.8, ls=':', label='random minutes (median)')
            a.axvline(0, color='k', lw=0.5)
            if j == 0:
                a.set_ylabel(lab, fontsize=7)
            if k == 0:
                a.set_title(f'{CAT_EN[cname]} (n={m.shape[0]})', fontsize=8)
            if k == len(vars3) - 1:
                a.set_xlabel('minutes from event')
            a.tick_params(labelsize=7)
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F2_event_aligned_normal.png'), dpi=150)
    plt.close(fig)
    S['step3_traj_normal'] = traj
    pre = {}
    for cname, gi in list(groups3.items()) + [('무작위 정상 생산 분', rnd_normal)]:
        gi = gi[gi >= 1]
        a = np.abs(dY[gi - 1])
        b = np.abs(dGc[gi - 1])
        pre[cname] = dict(n=int(len(gi)), absdY_q50_90=q(a, [50, 90]), frac_absdY_gt_0_3=round(float(np.nanmean(a > 0.3)), 3), absdGc_q50_90=q(b, [50, 90]), frac_absdGc_gt_2=round(float(np.nanmean(b > 2)), 3))
    S['step3_pre_event_dev'] = pre
    devs = {'dY': dY, 'dYs': dYs, 'dGc': dGc}
    fires = {}
    for vn, x in devs.items():
        for th in RULE_GRID[vn]:
            cond = np.nan_to_num(np.abs(x)) > th
            rl = run_end_len(cond)
            for k in RULE_K:
                f = np.where((rl == k) & normal)[0]
                fires[vn, th, k] = f
    shifts = rng.integers(2000, n - 2000, N_PERM)
    rule_rows = []
    best = {}
    for cname, sub in [('전체 후보', O)] + list(O.groupby('cat')) + [('목표 설정값 단독 변경', O[(O.kind == 'target') & ~O.flag_cochange])]:
        ev = sub['i'].to_numpy()[sub.region.to_numpy() == '정상 생산']
        if len(ev) < 20:
            continue
        for tol, win_kind in [(tl, wk) for tl in TOLS for wk in ('before', 'sym')]:
            e_lo, e_hi = (-tol, -1) if win_kind == 'before' else (-tol, tol)
            res = []
            for key, f in fires.items():
                rec = hit_rate(ev, f, e_lo, e_hi)
                prec = hit_rate(f, ev, -e_hi, -e_lo) if len(f) else 0.0
                f1 = 0 if rec + prec == 0 else 2 * rec * prec / (rec + prec)
                res.append((f1, key, prec, rec, len(f)))
            res.sort(key=lambda r: -r[0])
            f1, key, prec, rec, nf = res[0]
            ch = []
            for s in shifts:
                sh = (ev + s) % n
                sh = sh[normal[sh]]
                ch.append(hit_rate(sh, fires[key], e_lo, e_hi))
            per_var = {}
            for r in res:
                per_var.setdefault(r[1][0], r)
            orf = np.unique(np.concatenate([fires[per_var[v][1]] for v in per_var]))
            rec_or = hit_rate(ev, orf, e_lo, e_hi)
            prec_or = hit_rate(orf, ev, -e_hi, -e_lo)
            ch_or = np.mean([hit_rate(((ev + s) % n)[normal[(ev + s) % n]], orf, e_lo, e_hi) for s in shifts])
            ok = [r for r in res if r[2] >= 0.3]
            mr = max(ok, key=lambda r: r[3]) if ok else None
            row = dict(cat=cname, tol=tol, window=win_kind, n_events=int(len(ev)), best_rule=f'|{key[0]}|>{key[1]} for {key[2]} min', n_fires=int(nf), precision=round(prec, 3), recall=round(rec, 3), F1=round(f1, 3), recall_chance=round(float(np.mean(ch)), 3), unexplained=round(1 - rec, 3), recall_excess=round(float((rec - np.mean(ch)) / (1 - np.mean(ch))), 3), or_rule=' OR '.join((f'|{per_var[v][1][0]}|>{per_var[v][1][1]}/{per_var[v][1][2]}m' for v in per_var)), or_precision=round(prec_or, 3), or_recall=round(rec_or, 3), or_recall_chance=round(float(ch_or), 3), or_unexplained=round(1 - rec_or, 3), or_recall_excess=round(float((rec_or - ch_or) / (1 - ch_or)), 3), maxrecall_prec_ge_0_3=None if mr is None else dict(rule=f'|{mr[1][0]}|>{mr[1][1]}/{mr[1][2]}m', precision=round(mr[2], 3), recall=round(mr[3], 3)))
            rule_rows.append(row)
            best[cname, tol, win_kind] = row
    S['step3_rules'] = rule_rows
    pd.DataFrame(rule_rows).to_csv(os.path.join(OUT, 's68_step3_rules.csv'), index=False)
    S['step3_share_in_gc_window'] = {cat: round(float(g.flag_gc.mean()), 3) for cat, g in [('전체 후보', O)] + list(O.groupby('cat'))}
    grade = M['grade'].to_numpy(object)
    bins = [-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf]
    in_dev = np.abs(np.nan_to_num(dY)) > DEV_TH
    dstart = run_start_dist(in_dev)

    def decomp(df4, y):
        tot = sg = sl = sw = sc = ss = 0.0
        for _, b in df4.groupby('bin'):
            yy = b[y].to_numpy(float)
            if len(yy) < 2:
                continue
            m0 = yy.mean()
            tot += ((yy - m0) ** 2).sum()
            gm = b.groupby('grade')[y].transform('mean').to_numpy(float)
            lm = b.groupby('lot')[y].transform('mean').to_numpy(float)
            sg += ((gm - m0) ** 2).sum()
            sl += ((lm - gm) ** 2).sum()
            sw += ((yy - lm) ** 2).sum()
            cm = b.groupby('crew')[y].transform('mean').to_numpy(float)
            sm = b.groupby('shift')[y].transform('mean').to_numpy(float)
            sc += ((cm - m0) ** 2).sum()
            ss += ((sm - m0) ** 2).sum()
        if tot <= 0:
            return None
        return dict(grade=sg / tot, lot_in_grade=sl / tot, within_lot=sw / tot, crew_eta2=sc / tot, shift_eta2=ss / tot)

    def perm_null(df4, y, col, level_col=None):
        vals = []
        for _ in range(N_PERM):
            d2 = df4.copy()
            if level_col:
                for _, b in d2.groupby('bin'):
                    lots = b[level_col].unique()
                    lab = b.groupby(level_col)[col].first()
                    perm = dict(zip(lots, rng.permutation(lab.reindex(lots).to_numpy())))
                    d2.loc[b.index, col] = b[level_col].map(perm)
            else:
                d2[col] = d2.groupby('bin')[col].transform(lambda s: rng.permutation(s.to_numpy()))
            r = decomp(d2, y)
            vals.append(r)
        key = {'grade': 'grade', 'crew': 'crew_eta2', 'shift': 'shift_eta2'}[col]
        return round(float(np.mean([v[key] for v in vals if v])), 4)
    het = []
    sel = O[O.region == '정상 생산'].copy()
    sel['dY_pre'] = dY[np.maximum(sel['i'].to_numpy() - 1, 0)]
    sel['bin'] = pd.cut(sel['dY_pre'], bins, labels=False)
    sel = sel[sel['bin'].notna()]
    sel['grade'] = grade[sel['i']]
    sel['crew'] = np.asarray(crew[sel['i']], str)
    sel['shift'] = shift[sel['i']]
    sel = sel[pd.notna(sel['grade']) & pd.notna(sel['lot'])]
    scale = {k: v['q99'] - v['q1'] for k, v in LP['loops'].items() if v.get('q99') is not None}
    sel['size_std'] = sel['size'] / sel['loop'].map(scale)
    sel['dir'] = np.sign(sel['size'])
    ds = dstart[np.maximum(sel['i'].to_numpy() - 1, 0)]
    sel['delay'] = np.where(ds >= 0, ds + 1, np.nan)
    specs = [('목표 설정값(전체, 표준화 크기)', sel[sel.kind == 'target'], 'size_std'), ('목표 설정값(전체, 방향)', sel[sel.kind == 'target'], 'dir'), ('목표 설정값(전체, 지연)', sel[(sel.kind == 'target') & sel.delay.notna()], 'delay'), ('수동 출력(표준화 크기)', sel[sel.kind == 'op'], 'size_std'), ('수동 출력(방향)', sel[sel.kind == 'op'], 'dir'), ('모드 전환(방향: aut→man = −1)', sel[sel.kind == 'mode'], 'dir'), ('모드 전환(지연)', sel[(sel.kind == 'mode') & sel.delay.notna()], 'delay')]
    for lp in ('reel 평량 목표', 'size 수분 목표', '회분 목표', 'reel 수분 목표'):
        specs.append((f'{lp}(크기)', sel[sel.loop == lp], 'size'))
    for name, d4, y in specs:
        d4 = d4.dropna(subset=[y])
        if len(d4) < 30:
            continue
        r = decomp(d4, y)
        if r is None:
            continue
        lots_n = d4.groupby('lot').size()
        het.append(dict(spec=name, n=int(len(d4)), n_grades=int(d4.grade.nunique()), n_lots=int(len(lots_n)), n_lots_ge2=int((lots_n >= 2).sum()), n_crews=int(d4.crew.nunique()), **{k: round(v, 4) for k, v in r.items()}, grade_null=perm_null(d4, y, 'grade', 'lot'), crew_null=perm_null(d4, y, 'crew'), shift_null=perm_null(d4, y, 'shift')))
    S['step4_heterogeneity'] = het
    pd.DataFrame(het).to_csv(os.path.join(OUT, 's68_step4_heterogeneity.csv'), index=False)
    g5 = {}
    for lp in ('reel 수분 목표', 'reel 평량 목표', 'size 수분 목표', '회분 목표'):
        g5[f'{lp} 계단'] = O[(O.loop == lp) & (O.region == '정상 생산')]['i'].to_numpy()
    g5['수동 출력 계단'] = O[(O.kind == 'op') & (O.region == '정상 생산')]['i'].to_numpy()
    g5['모드 aut→man'] = O[(O.kind == 'mode') & (O['shape'] == 'aut→man') & (O.region == '정상 생산')]['i'].to_numpy()
    g5['모드 man→aut'] = O[(O.kind == 'mode') & (O['shape'] == 'man→aut') & (O.region == '정상 생산')]['i'].to_numpy()
    g5['무작위 정상 생산 분'] = rnd_normal
    absdY, absdGc = (np.abs(dY), np.abs(dGc))
    vars5 = [('|dY|', absdY), ('|dGc|', absdGc), ('Y', Y), ('G', G), ('V', V), ('P', P), ('KW', KW)]
    obj = {}
    fig, ax = plt.subplots(1, len(vars5), figsize=(2.6 * len(vars5), 3.2))
    for j, (gname, gi) in enumerate(g5.items()):
        rec = dict(n=int(len(gi)))
        for k, (vn, x) in enumerate(vars5):
            m = rel(x, gi)
            if not len(m):
                continue
            with np.errstate(all='ignore'):
                med = np.nanmedian(m, 0)
            rec[vn] = {f'+{d}': round(float(med[W + d]), 4) if np.isfinite(med[W + d]) else None for d in (15, 30, 60)}
            if vn in ('|dY|', '|dGc|'):
                a0 = x[gi[(gi - W >= 0) & (gi + W < n)] - 1]
                a30 = x[gi[(gi - W >= 0) & (gi + W < n)] + 30]
                okk = np.isfinite(a0) & np.isfinite(a30)
                rec[vn]['frac_decrease_30'] = round(float(np.mean(a30[okk] < a0[okk])), 3) if okk.any() else None
            ls = ':' if gname.startswith('무작위') else '-'
            ax[k].plot(np.arange(0, W + 1), med[W:], ls=ls, lw=1, label=gname if k == 0 else None)
            ax[k].set_title(f'Δ{vn} after event', fontsize=8)
            ax[k].tick_params(labelsize=7)
        obj[gname] = rec
    ax[0].legend(fontsize=6, labels=[{'reel 수분 목표 계단': 'reel moisture target', 'reel 평량 목표 계단': 'reel BW target', 'size 수분 목표 계단': 'size moisture target', '회분 목표 계단': 'ash target', '수동 출력 계단': 'manual OP', '모드 aut→man': 'mode aut→man', '모드 man→aut': 'mode man→aut', '무작위 정상 생산 분': 'random minutes'}[g] for g in g5])
    for a in ax:
        a.set_xlabel('minutes after event', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F3_post_event.png'), dpi=150)
    plt.close(fig)
    S['step5_objective'] = obj
    al = {}
    for gname, sub in [(f'{lp} 계단', O[(O.loop == lp) & (O.region == '정상 생산')]) for lp in ('reel 수분 목표', 'reel 평량 목표', 'size 수분 목표', '회분 목표')] + [('수동 출력 계단', O[(O.kind == 'op') & (O.region == '정상 생산')])]:
        gi = sub['i'].to_numpy()
        sg = np.sign(sub['size'].to_numpy())
        keep = (gi - W >= 0) & (gi + W < n)
        rec = dict(n=int(keep.sum()), n_up=int((sg[keep] > 0).sum()), n_down=int((sg[keep] < 0).sum()))
        for vn, x in (('Y', Y), ('G', G), ('V', V), ('P', P), ('KW', KW), ('|dY|', absdY), ('|dGc|', absdGc)):
            m = rel(x, gi) * sg[keep][:, None]
            with np.errstate(all='ignore'):
                med = np.nanmedian(m, 0)
                c30 = m[:, W + 30] if len(m) else np.array([])
                c30 = c30[np.isfinite(c30)]
                frac_pos = np.mean(c30 > 0) if len(c30) else np.nan
            rec[vn] = {**{f'+{d}': round(float(med[W + d]), 4) if np.isfinite(med[W + d]) else None for d in (15, 30, 60)}, 'frac_same_dir_30': None if not np.isfinite(frac_pos) else round(float(frac_pos), 3)}
        al[gname] = rec
    S['step5_sign_aligned'] = al
    gb = [0, 60, 80, 100, 120, 150, 180, 250, 1000]
    yb = [0, 4.0, 4.5, 5.0, 5.5, 20]
    vb = [0, 800, 950, 1100, 1250, 3000]
    nm = normal & np.isfinite(G) & np.isfinite(Y) & np.isfinite(V)
    gi_ = np.digitize(G, gb) - 1
    yi_ = np.digitize(Y, yb) - 1
    vi_ = np.digitize(V, vb) - 1
    ev_min = np.zeros(n, int)
    np.add.at(ev_min, O['i'].to_numpy(), 1)
    cell = pd.DataFrame(dict(g=gi_[nm], y=yi_[nm], v=vi_[nm], ev=ev_min[nm], grade=grade[nm]))
    C3 = cell.groupby(['g', 'y', 'v']).agg(minutes=('ev', 'size'), events=('ev', 'sum')).reset_index()
    obs = C3[C3.minutes >= 60]
    S['step6_cells'] = dict(n_cells_total=int(len(gb) - 1) * (len(yb) - 1) * (len(vb) - 1), n_cells_any_minute=int(len(C3)), n_cells_ge60min=int(len(obs)), n_cells_ge60min_with_event=int((obs.events > 0).sum()), n_cells_ge60min_no_event=int((obs.events == 0).sum()), minute_share_in_no_event_cells=round(float(obs[obs.events == 0].minutes.sum() / C3.minutes.sum()), 4), minute_share_in_cells_lt60=round(float(C3[C3.minutes < 60].minutes.sum() / C3.minutes.sum()), 4), bins=dict(G=gb, Y=yb, V=vb))
    GV = cell.groupby(['g', 'v']).agg(minutes=('ev', 'size'), events=('ev', 'sum')).reset_index()
    GV['per1000'] = (GV.events / GV.minutes * 1000).round(2)
    GV.to_csv(os.path.join(OUT, 's68_step6_G_by_V.csv'), index=False)
    S['step6_G_by_V'] = GV.to_dict('records')
    top = pd.Series(cell.grade).value_counts().index[:10]
    cg = cell.assign(gr=np.where(cell.grade.isin(top), cell.grade, '기타'))
    rows6 = []
    for gr, b in cg.groupby('gr'):
        cc = b.groupby(['g', 'y', 'v']).agg(minutes=('ev', 'size'), events=('ev', 'sum'))
        cc = cc[cc.minutes >= 60]
        rows6.append(dict(grade=gr, minutes=int(len(b)), events=int(b.ev.sum()), per1000=round(float(b.ev.sum() / len(b) * 1000), 2), cells_ge60=int(len(cc)), cells_with_event=int((cc.events > 0).sum())))
    S['step6_by_grade'] = rows6
    gbf = list(range(0, 410, 10))
    ybf = list(np.round(np.arange(0, 10.01, 0.25), 2))
    vbf = list(range(0, 2050, 50))
    cf = pd.DataFrame(dict(g=np.digitize(G, gbf)[nm], y=np.digitize(Y, ybf)[nm], v=np.digitize(V, vbf)[nm]))
    per_cat = {}
    for cat, g in [('전체 후보', O)] + list(O.groupby('cat')):
        em = np.zeros(n, int)
        np.add.at(em, g['i'].to_numpy(), 1)
        cf['ev'] = em[nm]
        for label, d3 in (('거친 격자', cell.assign(ev=em[nm])), ('촘촘한 격자', cf)):
            cc = d3.groupby(['g', 'y', 'v']).agg(minutes=('ev', 'size'), events=('ev', 'sum'))
            ob = cc[cc.minutes >= 60]
            per_cat.setdefault(label, {})[cat] = dict(cells_ge60=int(len(ob)), cells_with_event=int((ob.events > 0).sum()), minute_share_no_event=round(float(ob[ob.events == 0].minutes.sum() / cc.minutes.sum()), 4), minute_share_lt60=round(float(cc[cc.minutes < 60].minutes.sum() / cc.minutes.sum()), 4), n_cells_any=int(len(cc)))
    S['step6_cells_by_cat'] = per_cat
    mat = np.full((len(gb) - 1, len(vb) - 1), np.nan)
    mins = np.zeros_like(mat)
    for r in GV.itertuples():
        mat[r.g, r.v] = r.per1000
        mins[r.g, r.v] = r.minutes
    fig, axh = plt.subplots(figsize=(5.6, 4.2))
    im = axh.imshow(np.where(mins >= 60, mat, np.nan), origin='lower', cmap='viridis', aspect='auto')
    for a_ in range(mat.shape[0]):
        for b_ in range(mat.shape[1]):
            if mins[a_, b_] > 0:
                txt = f'{mat[a_, b_]:.1f}\n({int(mins[a_, b_])})' if mins[a_, b_] >= 60 else f'<60 min\n({int(mins[a_, b_])})'
                axh.text(b_, a_, txt, ha='center', va='center', fontsize=5.5, color='w' if mins[a_, b_] >= 60 else 'k')
    axh.set_xticks(range(len(vb) - 1))
    axh.set_xticklabels([f'{vb[k]}–{vb[k + 1]}' for k in range(len(vb) - 1)], fontsize=6)
    axh.set_yticks(range(len(gb) - 1))
    axh.set_yticklabels([f'{gb[k]}–{gb[k + 1]}' for k in range(len(gb) - 1)], fontsize=6)
    axh.set_xlabel('reel speed (m/min)')
    axh.set_ylabel('reel basis weight (g/m²)')
    fig.colorbar(im, ax=axh, label='operator-candidate events per 1000 min')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F4_support_G_by_V.png'), dpi=150)
    plt.close(fig)
    json.dump(S, open(os.path.join(OUT, 's68_summary.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(dict(H1=S['step2_H1_minute_fraction'], rules=[{k: r[k] for k in ('cat', 'tol', 'window', 'best_rule', 'precision', 'recall', 'recall_chance', 'or_recall', 'or_recall_chance')} for r in rule_rows]), ensure_ascii=False, indent=1))
if __name__ == '__main__':
    main()

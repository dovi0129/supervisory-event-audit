import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
WS = str(WORKSPACE)
from audit_runtime import load_roles
from audit_runtime import fill_and_mask
from audit_runtime import plant_events
PRIV = os.path.join(WS, str(record_path('artifacts_private')))
OUT = os.path.join(WS, 'artifacts/operator_policy')
REL = 0.002
RAMP_MIN = 3
SETTLE = 3
DCS_CONT = 0.05
DENSE_FILL = 5
GC_PRE, GC_POST = (30, 60)
BRK_NEAR = 30
COCHANGE = 2
QCS = {'signal_033': ('회분 목표', 'target', 'signal_067'), 'signal_036': ('size 평량 목표', 'target', 'signal_056'), 'weight_target': ('reel 평량 목표', 'target', 'signal_065'), 'signal_040': ('분사/와이어 속도비 목표', 'target', 'signal_041'), 'size_moisture_target': ('size 수분 목표', 'target', 'signal_057'), 'moisture_target': ('reel 수분 목표', 'target', 'signal_066'), 'signal_051': ('수분 계산 설정값', 'calc', None), 'signal_028': ('후건조 압력 설정값', 'manip', 'signal_025'), 'signal_049': ('전건조 압력 설정값', 'manip', 'signal_046'), 'signal_062': ('지료 유량 설정값', 'manip', 'signal_058'), 'signal_031': ('회분 유량 설정값', 'manip', 'signal_029'), 'signal_026': ('후건조 압력 출력', 'op', 'signal_025'), 'signal_047': ('전건조 압력 출력', 'op', 'signal_046'), 'signal_061': ('지료 유량 출력', 'op', 'signal_058'), 'signal_030': ('회분 유량 출력', 'op', 'signal_029'), 'signal_039': ('분사/와이어 속도비 출력', 'op', 'signal_041')}
MODE_NAME = {'signal_025': '후건조 압력 모드', 'signal_046': '전건조 압력 모드', 'signal_058': '지료 유량 모드', 'signal_029': '회분 유량 모드', 'signal_041': '분사/와이어 속도비 모드', 'signal_067': '회분 모드', 'signal_065': 'reel 평량 모드', 'signal_066': 'reel 수분 모드', 'signal_056': 'size 평량 모드', 'signal_057': 'size 수분 모드', 'signal_060': '지료 농도 모드'}
DUP = {'signal_055': 'signal_051'}
DCS_KIND = {'PIC': '압력', 'PDC': '차압', 'TIC': '온도'}

def dcs_names(sp_tags):
    from audit_runtime import configuration
    names = configuration().get('dcs_role_names', {})
    needed = [key for key in sp_tags if key not in QCS and key not in DUP]
    missing = [key for key in needed if key not in names]
    if missing:
        raise ValueError('External dcs_role_names must describe all DCS role keys')
    return {key: names[key] for key in needed}


def runs(mask):
    a = np.r_[0, mask.astype(np.int8), 0]
    d = np.diff(a)
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return (s, e - s)

def episodes(x, chg, sign):
    out = []
    s, L = runs(chg)
    for a, n in zip(s, L):
        b = a
        for k in range(a + 1, a + n):
            if sign[k] != sign[k - 1]:
                out.append((b, k - b))
                b = k
        out.append((b, a + n - b))
    return out

def linear_flag(x, a, n):
    if n < RAMP_MIN:
        return False
    inc = np.diff(x[a - 1:a + n])
    return bool(np.all(np.abs(inc - inc.mean()) <= 0.02 * abs(inc.mean()) + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default=None, help='예: 2022-01(시험용 한 달)')
    ap.add_argument('--tag', default='full')
    a = ap.parse_args()
    cols = json.load(open(os.path.join(PRIV, str(record_path('s68_columns_private_json')))))
    df = pd.read_parquet(os.path.join(PRIV, 's68_cache.parquet')).set_index('time')
    raw = load_roles(smoke=False)
    mask_config = json.loads(record_path('mask_configuration').read_text())
    g0df, mask = fill_and_mask(raw, mask_config['procedure']['mask'])
    mask = mask.reindex(df.index)
    for c in mask.columns:
        mask[c] = mask[c].fillna(c == 'missing').astype(bool)
    gkey = g0df['GKEY'].reindex(df.index).to_numpy(object)
    if a.month:
        sel = df.index.strftime('%Y-%m') == a.month
        df, mask, gkey = (df[sel], mask[sel], gkey[sel])
    n = len(df)
    t = df.index
    gpos = plant_events.grade_change_positions(gkey)
    gc = np.zeros(n, bool)
    for p in gpos:
        gc[max(0, p - GC_PRE):min(n, p + GC_POST + 1)] = True
    hist = df['이력TYPE'].astype('object').ffill(limit=30)
    hist = hist.where(hist.notna(), None).to_numpy(object)
    nonprod = np.array([h is not None and h != '생산' for h in hist])
    brk = mask['brk'].to_numpy(bool)
    valid = mask['valid'].to_numpy(bool)
    region = np.where(brk, '지절', np.where(gc, '지종 전환', np.where(valid, '정상 생산', '기타 무효')))
    hour = t.hour.to_numpy()
    shift = np.where((hour >= 6) & (hour < 14), '주간(06-14)', np.where((hour >= 14) & (hour < 22), '오후(14-22)', '야간(22-06)'))
    tmp_path = os.path.join(OUT, str(record_path('tag_map_private_json')))
    grade_raw = df['지종'].astype('object').ffill(limit=30)
    full_grade_rank = pd.read_parquet(os.path.join(PRIV, 's68_cache.parquet'), columns=['지종'])['지종'].value_counts()
    gmap = {g: f'grade {i + 1}' for i, g in enumerate(full_grade_rank.index)}
    crew_all = pd.read_parquet(os.path.join(PRIV, 's68_cache.parquet'), columns=['작업자'])['작업자'].dropna().unique()
    cmap = {c: f'C{i + 1}' for i, c in enumerate(crew_all)}
    lot_all = pd.read_parquet(os.path.join(PRIV, 's68_cache.parquet'), columns=['LOT'])['LOT'].dropna().unique()
    lmap = {v: f'L{i + 1:04d}' for i, v in enumerate(lot_all)}
    grade = grade_raw.map(gmap).to_numpy(object)
    crew = df['작업자'].astype('object').ffill(limit=30).map(cmap).to_numpy(object)
    lot = df['LOT'].astype('object').ffill(limit=30).map(lmap).to_numpy(object)
    names = {}
    for k, v in QCS.items():
        names[k] = v[0]
    names.update(MODE_NAME)
    names.update(dcs_names(cols['sp']))
    loops_meta = {}
    rows = []

    def add(time_idx, loop, prev, new, mode_val, cls, basis, shape, dur, lin=False, filled=False, mval=(None, None)):
        rows.append((time_idx, loop, prev, new, mode_val, cls, basis, shape, dur, lin, filled, mval[0], mval[1]))

    def mode_series(mt):
        if mt is None:
            return np.full(n, None, object)
        s = df[mt].astype('object').ffill()
        return s.where(s.notna(), None).to_numpy(object)
    sp_tags = [s for s in cols['sp'] if s not in DUP]
    op_tags = [k for k, v in QCS.items() if v[1] == 'op']
    for tag in sp_tags + op_tags:
        info = QCS.get(tag)
        if info is None:
            kind = 'dcs'
            mt = None
        else:
            kind, mt = (info[1], info[2])
        x0 = df[tag].to_numpy(float).copy()
        n_zero = 0
        if kind in ('target', 'calc', 'dcs'):
            nz = x0[np.isfinite(x0) & (x0 != 0)]
            if len(nz) >= 10:
                lo_, hi_ = np.percentile(nz, [1, 99])
                if not lo_ <= 0 <= hi_:
                    zero = x0 == 0
                    n_zero = int(zero.sum())
                    x0[zero] = np.nan
        isnan = np.isnan(x0)
        if kind == 'dcs':
            x = pd.Series(x0).ffill().to_numpy(float)
        else:
            x = pd.Series(x0).ffill(limit=DENSE_FILL).to_numpy(float)
        md = mode_series(mt)
        if kind == 'op':
            man = np.array([m == 'man' for m in md])
            last_change = np.zeros(n, int)
            c = 10 ** 6
            for i in range(n):
                c = 0 if i > 0 and md[i] != md[i - 1] else c + 1
                last_change[i] = c
            pct = (x >= -1) & (x <= 101)
            ok = man & np.r_[False, man[:-1]] & (last_change >= SETTLE) & pct & np.r_[False, pct[:-1]]
            ref = x[ok]
        else:
            ok = np.ones(n, bool)
            ref = x0[~isnan]
        if len(ref) < 10:
            loops_meta[names[tag]] = dict(kind=kind, n_recorded=int(len(ref)), thr=None, note='기록 부족')
            continue
        q1, q99 = np.percentile(ref, [1, 99])
        thr = max(REL * (q99 - q1), 1e-09)
        d = np.r_[np.nan, np.diff(x)]
        chg = ok & np.isfinite(d) & (np.abs(d) > thr)
        filled_now = isnan & np.isfinite(x)
        rec = ~isnan
        rate = float(chg[rec].mean()) if rec.any() else 0.0
        meta = dict(kind=kind, thr=float(thr), n_zero_dropped=n_zero, q1=float(q1), q99=float(q99), n_recorded=int(rec.sum()), frac_missing=float(isnan.mean()), change_rate_recorded=rate, mode_tag=MODE_NAME.get(mt))
        sign = np.sign(np.nan_to_num(d))
        if kind == 'manip' or (kind == 'dcs' and rate > DCS_CONT and (rec.sum() >= 0.01 * n)):
            for i in np.where(chg)[0]:
                if kind == 'manip':
                    basis = 'QCS 출력(자동 모드)' if md[i] == 'aut' else 'PV 추종(수동 모드)' if md[i] == 'man' else '모드 미상'
                else:
                    basis = '캐스케이드 추정(모드 태그 없음)'
                add(i, names[tag], x[i - 1], x[i], md[i], '자동', basis, '연속', 1, False, bool(filled_now[i] or filled_now[i - 1]))
            meta['rule'] = '분 단위 변화 전부 자동'
        else:
            n_step = 0
            for s0, L in episodes(x, chg, sign):
                e = s0 + L - 1
                shape = '계단' if L < RAMP_MIN else '경사'
                lin = linear_flag(x, s0, L)
                if kind == 'calc':
                    cls, basis = ('자동', '계산형 설정값 추정')
                    n_step += shape == '계단'
                elif kind == 'dcs':
                    cls = '운전자 후보' if shape == '계단' else '자동'
                    basis = '계단(모드 태그 없음)' if shape == '계단' else '경사 로직 또는 보간'
                elif kind == 'op':
                    cls = '운전자 후보' if shape == '계단' else '자동'
                    basis = '수동 모드 출력 계단' if shape == '계단' else '수동 모드 출력 경사'
                else:
                    cls = '운전자 후보' if shape == '계단' else '자동'
                    basis = '목표 설정값 계단' if shape == '계단' else '목표 설정값 경사'
                add(s0, names[tag], x[s0 - 1], x[e], md[s0], cls, basis, shape, L, lin, bool(filled_now[s0 - 1:e + 1].any()))
            meta['rule'] = '같은 부호 묶음: 길이 ≤ 2 계단, ≥ 3 경사'
            if kind == 'calc':
                meta['n_step_like'] = int(n_step)
        loops_meta[names[tag]] = meta
    brk_near = np.zeros(n, bool)
    bs, bl = runs(brk)
    for s0, L in zip(bs, bl):
        brk_near[max(0, s0 - BRK_NEAR):min(n, s0 + L + BRK_NEAR)] = True
    for mt in cols['mode']:
        v = mode_series(mt)
        nn = np.array([x is not None for x in v])
        ch = np.zeros(n, bool)
        ch[1:] = nn[1:] & nn[:-1] & (v[1:] != v[:-1])
        for i in np.where(ch)[0]:
            add(i, MODE_NAME[mt], 1.0 if v[i - 1] == 'aut' else 0.0, 1.0 if v[i] == 'aut' else 0.0, v[i], '운전자 후보', '모드 전환', f'{v[i - 1]}→{v[i]}', 1, False, False)
        loops_meta[MODE_NAME[mt]] = dict(kind='mode', n_transitions=int(ch.sum()), values=sorted({str(x) for x in v[nn]}), rule='aut↔man 전환 전부 운전자 후보')
    E = pd.DataFrame(rows, columns=['i', 'loop', 'prev', 'new', 'mode', 'class', 'basis', 'shape', 'dur_min', 'linear', 'own_filled', '_a', '_b']).drop(columns=['_a', '_b'])
    i = E['i'].to_numpy(int)
    E.insert(0, 'time', t[i])
    E['size'] = E['new'] - E['prev']
    E['kind'] = E['loop'].map({k: v['kind'] for k, v in loops_meta.items()})
    for c, arr in (('region', region), ('shift', shift), ('crew', crew), ('lot', lot), ('grade', grade)):
        E[c] = arr[i]
    E['flag_brk'] = brk[i]
    E['flag_brk_near'] = brk_near[i]
    E['flag_gc'] = gc[i]
    E['flag_missing'] = mask['missing'].to_numpy(bool)[i] | mask['interp'].to_numpy(bool)[i]
    E['flag_frozen'] = mask['frozen'].to_numpy(bool)[i]
    E['flag_abnormal'] = mask['abnormal'].to_numpy(bool)[i]
    E['flag_down'] = mask['down'].to_numpy(bool)[i]
    E['flag_rule150'] = mask['rule150'].to_numpy(bool)[i]
    E['flag_scanner'] = ~mask['scanner_ok'].to_numpy(bool)[i]
    E['flag_nonprod'] = nonprod[i]
    E['flag_cochange'] = False
    tg = E[(E['kind'] == 'target') & (E['class'] == '운전자 후보')]
    ti = tg['i'].to_numpy()
    tl = tg['loop'].to_numpy()
    co = np.zeros(len(tg), bool)
    for lp in np.unique(tl):
        oth = np.sort(ti[tl != lp])
        me = tl == lp
        lo = np.searchsorted(oth, ti[me] - COCHANGE, 'left')
        hi = np.searchsorted(oth, ti[me] + COCHANGE, 'right')
        co[me] = hi > lo
    E.loc[tg.index, 'flag_cochange'] = co
    E = E.drop(columns=['i']).sort_values(['time', 'loop']).reset_index(drop=True)
    sfx = '' if a.tag == 'full' else f'_{a.tag}'
    os.makedirs(OUT, exist_ok=True)
    E.to_parquet(os.path.join(OUT, f'events{sfx}.parquet'))
    M = pd.DataFrame(dict(time=t, region=region, shift=shift, crew=crew, lot=lot, grade=grade, brk=brk, gc=gc, valid=valid, nonprod=nonprod, down=mask['down'].to_numpy(bool)))
    M.to_parquet(os.path.join(PRIV, f's68_minutes{sfx}.parquet'))
    json.dump(dict(params=dict(REL=REL, RAMP_MIN=RAMP_MIN, SETTLE=SETTLE, DCS_CONT=DCS_CONT, DENSE_FILL=DENSE_FILL, GC_PRE=GC_PRE, GC_POST=GC_POST, BRK_NEAR=BRK_NEAR, COCHANGE=COCHANGE), n_minutes=int(n), n_grade_changes=int(len(gpos)), loops=loops_meta), open(os.path.join(OUT, f'loops{sfx}.json'), 'w'), ensure_ascii=False, indent=1)
    if a.tag == 'full':
        json.dump(dict(loops={k: names[k] for k in names}, grades=gmap, crews=cmap, note='원 태그·지종·작업자 → 비식별 이름. 비공개.'), open(tmp_path, 'w'), ensure_ascii=False, indent=1)
        json.dump(lmap, open(os.path.join(PRIV, str(record_path('s68_lot_map_private_json'))), 'w'), ensure_ascii=False)
    op_min = np.zeros(n, bool)
    op_min[np.searchsorted(t, E.loc[E['class'] == '운전자 후보', 'time'].to_numpy())] = True
    print(f"minutes {n} events {len(E)} 운전자 후보 {int((E['class'] == '운전자 후보').sum())} 자동 {int((E['class'] == '자동').sum())}")
    print(f"운전자 후보가 있는 분의 비율 {op_min.mean():.4f}; 정상 생산 분 중 {op_min[region == '정상 생산'].mean():.4f}")
    print(E.groupby(['kind', 'loop', 'class']).size().to_string())
if __name__ == '__main__':
    main()

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest, wilcoxon

R = Path('diagnostics/results')
O = R / 'acc_shape_mechanism_final_20261011'
O.mkdir(parents=True, exist_ok=True)
TASKS = ('0000','0001','0006','0008','0010')
K = ['task','clip']
PA = 'pa_mpjpe_mm'

ALIASES = {
    'acc': ['original_acc_si','rms_acc_si','rms_acceleration_si','rms_acc_mps2','rms_acc_m_s2','rms_acceleration','rms_acc','acc_rms'],
    'speed': ['original_speed_si','original_speed_rms','original_rms_speed','original_rms_velocity','rms_speed_si','rms_velocity_si','rms_speed','rms_velocity','rms_vel','speed_rms'],
    'rotation': ['original_rotation_step_p95','robot_rotation_step_p95','rotation_step_p95','rot_step_p95','robot_rot_step_p95','original_rotation_range','rotation_range','angular_speed_rms','rms_angular_speed','angular_velocity_rms'],
}

def feature_table(df, source):
    cols = {c.lower(): c for c in df.columns}
    out = df[K].drop_duplicates().copy()
    for metric, candidates in ALIASES.items():
        found = next((cols[n] for n in candidates if n in cols), None)
        if found is None:
            print(f'{source}: {metric} NOT FOUND')
            continue
        print(f'{source}: {metric} <- {found}')
        data = df[K + [found]].copy()
        data[found] = pd.to_numeric(data[found], errors='coerce')
        agg = data.groupby(K, as_index=False)[found].median().rename(columns={found:metric})
        out = out.merge(agg, on=K, validate='one_to_one')
    return out

orig_frames = []
orig_feature_tables = []
audit_frames = []

for t in TASKS:
    name = f'task-{t}'
    path = R/'object_pa_mpjpe_acc_5task_current10457'/name/'per_frame_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing original geometry: {path}')
    g = pd.read_csv(path)
    needed = {'clip','frame',PA}
    if not needed <= set(g.columns):
        raise RuntimeError(f'Missing required columns in {path}: {sorted(needed-set(g.columns))}; available={list(g.columns)}')
    g['task'] = name
    g['frame'] = pd.to_numeric(g['frame'], errors='coerce')
    g[PA] = pd.to_numeric(g[PA], errors='coerce')
    orig_frames.append(g[K+['frame',PA]].groupby(K+['frame'],as_index=False)[PA].mean())
    orig_feature_tables.append(feature_table(g, name+' original'))

    af = R/'acc_causal_geometry_boundary_20261010'/name/'mechanism_action_audit.csv'
    if af.exists():
        a = pd.read_csv(af)
        a['task'] = name
        audit_frames.append(a)

F = pd.concat(orig_frames, ignore_index=True)
# Group objects per clip-frame before calculating temporal dynamics.
C = F.pivot_table(index=K, columns='frame', values=PA, aggfunc='first')
if 1 not in C.columns or 10 not in C.columns:
    raise RuntimeError(f'Expected 1..10 frame numbers, got {list(C.columns)}')
C = C.dropna(subset=list(range(1,11)))
Z = pd.DataFrame(index=C.index)
Z['pa_initial_mm'] = C[1]
Z['pa_final_mm'] = C[10]
Z['pa_growth_mm'] = C[10]-C[1]
Z['pa_late_early_mm'] = C[[8,9,10]].mean(axis=1)-C[[1,2,3]].mean(axis=1)
Z = Z.reset_index()
print('\nORIGINAL complete 10-frame clips:',len(Z))

base_features = pd.concat(orig_feature_tables, ignore_index=True)
assert not base_features.duplicated(K).any()
for metric in ALIASES:
    if metric in base_features.columns:
        Z = Z.merge(base_features[K+[metric]], on=K, how='left', validate='one_to_one')

if audit_frames:
    A = pd.concat(audit_frames,ignore_index=True)
    A = A.drop_duplicates(K)
    audit_features = feature_table(A,'boundary action audit')
    for metric in ALIASES:
        if metric not in audit_features.columns:
            continue
        Z = Z.merge(audit_features[K+[metric]].rename(columns={metric:metric+'_audit'}),on=K,how='left',validate='one_to_one')
        if metric in Z.columns:
            u,v = Z[metric], Z[metric+'_audit']
            both = u.notna() & v.notna() & (u.abs()+v.abs()>1e-6)
            if both.sum()>=20:
                diff = (u[both]-v[both]).abs()/np.maximum(v[both].abs(),1e-8)
                if diff.median()>0.05:
                    print(f'WARNING: {metric} original vs audit mismatch: median relative {diff.median():.3g}; keeping original only')
                else:
                    Z[metric] = u.combine_first(v)
            else:
                Z[metric] = u.combine_first(v)
        else:
            Z[metric]=Z[metric+'_audit']
        Z = Z.drop(columns=[metric+'_audit'])

for metric in ALIASES:
    print(f'FEATURE {metric}:', int(Z[metric].notna().sum()) if metric in Z.columns else 'MISSING', 'of',len(Z))

Z['episode']=Z['clip'].astype(str).str.replace(r'-\d+:\d+$','',regex=True)
Z.to_csv(O/'original_clip_data.csv',index=False)
if 'acc' not in Z or Z['acc'].notna().sum()<100:
    raise RuntimeError('No reliable acceleration column. Inspect printed aliases and add the EXACT ORIGINAL acceleration column to ALIASES. No causal conclusion produced.')


def partial_rank_corr(data, controls, output='pa_late_early_mm'):
    x=rankdata(data['acc'].to_numpy(float))
    y=rankdata(data[output].to_numpy(float))
    design=[np.ones(len(data))]
    tasks=pd.get_dummies(data['task'],drop_first=True,dtype=float)
    for j in range(tasks.shape[1]):
        design.append(tasks.iloc[:,j].to_numpy(float))
    for control in controls:
        z=rankdata(data[control].to_numpy(float))
        design.append((z-z.mean())/(z.std()+1e-12))
    X=np.column_stack(design)
    rx=x-X@np.linalg.lstsq(X,x,rcond=None)[0]
    ry=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    if rx.std()<1e-10 or ry.std()<1e-10:
        return np.nan
    return float(np.corrcoef(rx,ry)[0,1])


def cluster_ci(data,controls,seed=20261011,reps=300):
    rng=np.random.default_rng(seed)
    groups = [v.index.to_numpy() for _,v in data.reset_index(drop=True).groupby(['task','episode'],sort=False)]
    # Resample entire episodes, not overlapping clips individually.
    gt=[[] for _ in TASKS]
    dt=data.reset_index(drop=True)
    for arr in groups:
        task=dt.loc[arr[0],'task']
        gt[TASKS.index(task.split('-')[-1])].append(arr)
    vals=[]
    for _ in range(reps):
        parts=[]
        for cluster_ids in gt:
            if not cluster_ids:
                continue
            sel=rng.integers(0,len(cluster_ids),size=len(cluster_ids))
            parts.extend(cluster_ids[i] for i in sel)
        take=np.concatenate(parts)
        vals.append(partial_rank_corr(dt.iloc[take],controls))
    return np.nanpercentile(vals,[2.5,97.5])

print('\n===== CAUSE CONTROL: ORIGINAL ACTION =====')
print('Outcome = ORIGINAL PA-MPJPE (GT future shape) late-frame minus early-frame, mm')
model_controls=[('task+initial', ['pa_initial_mm'])]
if 'speed' in Z:
    model_controls.append(('task+initial+speed',['pa_initial_mm','speed']))
if 'rotation' in Z:
    model_controls.append(('task+initial+rotation',['pa_initial_mm','rotation']))
if 'speed' in Z and 'rotation' in Z:
    model_controls.append(('FULL: task+initial+speed+rotation',['pa_initial_mm','speed','rotation']))

for label, controls in model_controls:
    q=Z.dropna(subset=['acc','pa_late_early_mm']+controls).copy()
    if len(q)<100:
        print(label,'SKIPPED insufficient observations:',len(q))
        continue
    r=partial_rank_corr(q,controls)
    ci=cluster_ci(q,controls,reps=250)
    print(f'{label}: clips={len(q)} episodes={q.episode.nunique()} partial_rank_rho={r:+.4f} episode_boot95=[{ci[0]:+.4f},{ci[1]:+.4f}]')
    for task,v in q.groupby('task'):
        if len(v)>=30:
            # Within-task conditioning already contains task dummies; equivalent after singular-safe lstsq.
            print(f'  {task} n={len(v)} rho={partial_rank_corr(v,controls):+.4f}')

# Observational, within-task acc quantiles for an interpretable effect scale.
obs=[]
for task,v in Z.dropna(subset=['acc','pa_late_early_mm']).groupby('task'):
    q1,q3=v['acc'].quantile([0.25,0.75])
    lo=v[v['acc']<=q1]
    hi=v[v['acc']>=q3]
    obs.append({'task':task,'n_low':len(lo),'n_high':len(hi),'acc_median_low':lo.acc.median(),'acc_median_high':hi.acc.median(),'pa_growth_low_mm':lo.pa_late_early_mm.median(),'pa_growth_high_mm':hi.pa_late_early_mm.median(),'high_minus_low_growth_mm':hi.pa_late_early_mm.median()-lo.pa_late_early_mm.median()})
OBS=pd.DataFrame(obs)
OBS.to_csv(O/'original_acc_quartiles.csv',index=False)
print('\n===== UNADJUSTED ACC QUARTILES (NOT causal) =====')
print(OBS.round(5).to_string(index=False))

# Within-clip dose intervention: same action spatial path; temporal schedule changes.
print('\n===== TEMPORAL RETIMING DOSE REPEATED-MEASURES =====')
dose_rows=[]
for dose in (25,50,75,100):
    name = ('acc_causal_geometry_boundary_20261010' if dose==100 else f'acc_causal_geometry_dose{dose}_20261011')
    for t in TASKS:
        root=R/name/f'task-{t}'
        af=root/'mechanism_action_audit.csv'
        gf=root/'mechanism_geometry_per_frame.csv'
        if not (af.exists() and gf.exists()):
            print('MISSING DOSE',dose,t)
            continue
        a=pd.read_csv(af)
        g=pd.read_csv(gf)
        if not {'clip','modified_acc_si'}<=set(a.columns):
            raise RuntimeError(f'Action columns missing: {af}: {list(a.columns)}')
        if not {'clip','frame','variant',PA}<=set(g.columns):
            raise RuntimeError(f'Geometry columns missing: {gf}: {list(g.columns)}')
        if a['clip'].duplicated().any():
            raise RuntimeError(f'Duplicate clip in {af}')
        g=g.groupby(['clip','frame','variant'],as_index=False)[PA].mean()
        g=g.groupby(['clip','variant'],as_index=False)[PA].mean()
        wide=g.pivot(index='clip',columns='variant',values=PA)
        needed={'base','base_repeat','corrected'}
        if not needed<=set(wide.columns):
            raise RuntimeError(f'Missing variants in {gf}: {list(wide.columns)}')
        wide=wide.reset_index()
        q=a[['clip','modified_acc_si']].merge(wide,on='clip',validate='one_to_one')
        q['task']=f'task-{t}'
        q['dose']=dose
        q['delta_pa_from_same_run_base_mm']=q['corrected']-q['base']
        q['repeat_noise_mm']=(q['base']-q['base_repeat']).abs()
        dose_rows.append(q[K+['dose','modified_acc_si','delta_pa_from_same_run_base_mm','repeat_noise_mm','base']])

D=pd.concat(dose_rows,ignore_index=True)
D.to_csv(O/'dose_clip_data.csv',index=False)
W=D.pivot_table(index=K,columns='dose',values=['modified_acc_si','delta_pa_from_same_run_base_mm','repeat_noise_mm','base'],aggfunc='first')
W=W.dropna(subset=[('modified_acc_si',25),('modified_acc_si',100),('delta_pa_from_same_run_base_mm',25),('delta_pa_from_same_run_base_mm',100)])
acc_gap=W[('modified_acc_si',25)]-W[('modified_acc_si',100)]
# Positive: lower (100% retiming) acceleration coincides with lower geometry error.
benefit=W[('delta_pa_from_same_run_base_mm',25)]-W[('delta_pa_from_same_run_base_mm',100)]
noise=np.maximum(W[('repeat_noise_mm',25)],W[('repeat_noise_mm',100)])
base_spread=(W[('base',25)]-W[('base',100)]).abs()
Q=pd.DataFrame({'acc_gap_25minus100':acc_gap,'pa_benefit_full_vs25_mm':benefit,'noise':noise,'base_spread':base_spread})
Q=Q[(Q['acc_gap_25minus100']>1e-6) & Q['pa_benefit_full_vs25_mm'].notna()]
Q['exceeds_noise'] = Q.pa_benefit_full_vs25_mm>2*np.maximum(Q.noise,Q.base_spread)
Q.to_csv(O/'dose_pair_25_vs100.csv')
print('COMPARABLE CLIPS:',len(Q))
if len(Q)>20:
    x=Q['pa_benefit_full_vs25_mm'].to_numpy(float)
    print(f'25%-vs-100% dose: acc gap median={Q.acc_gap_25minus100.median():+.6f} m/s^2, PA benefit median={np.median(x):+.6f} mm, beneficial={100*(x>0).mean():.2f}%, >2x repeat/cross-run noise={100*Q.exceeds_noise.mean():.2f}%')
    print('Wilcoxon p=',wilcoxon(x).pvalue)
    for task,v in Q.groupby(level='task'):
        print(f'  {task} N={len(v)} median={v.pa_benefit_full_vs25_mm.median():+.6f}mm positive={100*(v.pa_benefit_full_vs25_mm>0).mean():.1f}%')
    # Complete 4-dose check: positive response to increased acceleration.
    complete=W.dropna(subset=[('modified_acc_si',v) for v in (25,50,75,100)]+[('delta_pa_from_same_run_base_mm',v) for v in (25,50,75,100)])
    pos=0; valid=0
    for _,v in complete.iterrows():
        xx=np.array([v[('modified_acc_si',t)] for t in (25,50,75,100)],float)
        yy=np.array([v[('delta_pa_from_same_run_base_mm',t)] for t in (25,50,75,100)],float)
        if np.var(xx)>1e-12:
            pos+=np.cov(xx,yy,bias=True)[0,1]>0
            valid+=1
    print(f'FOUR DOSES within-clip positive accel-to-PA slope = {pos}/{valid} ({100*pos/max(valid,1):.2f}%)')

print('\n===== INTERPRETATION RULES =====')
print('FULL control only if the FULL line appears above; missing speed/rotation means NOT controlled.')
print('Original Action PA uses future GT shape; modified retiming PA uses initial rigid shape, NOT counterfactual future GT.')
print('Even if dose effect is positive: changed timing also changes speed and other dynamics; cannot isolate acceleration as sole cause.')
print('PA is an outcome, not direct evidence of internal per-point correspondence/model mechanism.')
print('RESULTS:',O)

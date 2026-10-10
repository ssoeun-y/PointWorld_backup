#!/usr/bin/env python3
"""Cached-data audit/analysis of absolute grasp gaps; never runs PointWorld."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

COND = ('Original', 'Free', 'Anchor')
BAD = ('dev', 'delta', 'diff', 'drift', 'change', 'relative')


def audit(root, limit):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    print('[AUDIT] root:', root)
    files = sorted(
        p for p in root.rglob('*')
        if p.is_file()
        and p.suffix.lower() in ('.csv', '.npz', '.json', '.parquet')
        and any(s in str(p).lower()
                for s in ('anchor', 'grasp', 'gap', 'coupling'))
    )
    print('[AUDIT] candidate files:', len(files))
    for p in files[:limit]:
        print('\nFILE', p,
              '(%.2f MB)' % (p.stat().st_size / 1024**2))
        try:
            if p.suffix == '.csv':
                with p.open(encoding='utf-8-sig', newline='') as f:
                    cols = next(csv.reader(f), [])
                print('  columns:', cols)
            elif p.suffix == '.npz':
                with np.load(p, allow_pickle=False) as a:
                    print('  keys:', a.files)
            elif p.suffix == '.json':
                if p.stat().st_size > 2_000_000:
                    print('  large JSON: skip reading')
                else:
                    with p.open(encoding='utf-8') as f:
                        data = json.load(f)
                    print('  keys:', list(data)[:40]
                          if isinstance(data, dict)
                          else type(data).__name__)
            else:
                print('  parquet: see schema manually')
        except Exception as e:
            print('  cannot inspect:', e)
    print('\n[AUDIT] Deviation is NOT absolute gap.')


def snapshot(root, cols, expected):
    root = Path(root)
    paths = sorted(root.glob('task-*/per_clip_causal_metrics.csv'))
    if not paths:
        print('[SNAPSHOT] No per_clip_causal_metrics.csv in:', root)
        print('Use audit to locate actual result files.')
        return

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df['task'] = path.parent.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print('[SNAPSHOT] rows:', len(df), 'expected:', expected)
    print('[SNAPSHOT] gap-related columns:')
    for col in df:
        if 'gap' in col.lower() or 'drift' in col.lower():
            print(' ', col)

    if len(df) != expected:
        print('[STOP] Verify N=524 matched filtering first.')
        return

    if not any(cols):
        print('Specify VERIFIED absolute gap columns if available.')
        return

    if not all(cols):
        raise ValueError('All three gap columns are required')

    for col in cols:
        if col not in df or any(x in col.lower() for x in BAD):
            raise ValueError(f'Not a verified absolute gap: {col}')
        val = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(val).all() or (val < 0).any():
            raise ValueError(f'Invalid gap(mm): {col}')
        print(col, 'median absolute gap (mm):',
              float(val.median()))

    orig = pd.to_numeric(df[cols[0]])
    for name, col in zip(('Free', 'Anchor'), cols[1:]):
        dev = (pd.to_numeric(df[col]) - orig).abs()
        print(name, 'median |gap - Original gap| (mm):',
              float(dev.median()))

    print('Confirm same frame range and aggregation as old result.')


def load_long(path):
    df = pd.read_csv(path)
    alias = {
        'clip_id': ('clip', 'clip_key', 'clip_name'),
        'frame': ('t', 'frame_idx', 'timestep'),
        't_close': ('close_t', 'grasp_frame', 'close_frame'),
        'condition': ('method', 'variant'),
    }
    for dest, variants in alias.items():
        if dest not in df:
            matches = [x for x in variants if x in df]
            if len(matches) == 1:
                df = df.rename(columns={matches[0]: dest})

    ids = [x for x in
           ('task', 'clip_id', 'object_id', 'gripper_side')
           if x in df]

    required = ('clip_id', 'frame', 't_close')
    if any(x not in df for x in required):
        raise ValueError(f'Missing {required}; got {list(df)}')

    if 'condition' in df:
        if 'absolute_gap_mm' not in df:
            raise ValueError(
                'Long CSV needs VERIFIED absolute_gap_mm, not deviation'
            )
        data = df.copy()
        data['condition'] = (
            data['condition'].astype(str).str.lower().map({
                'original': 'Original',
                'orig': 'Original',
                'free': 'Free',
                'anchor': 'Anchor',
            })
        )
        if data['condition'].isna().any():
            raise ValueError('Unknown condition label')
    else:
        names = (
            'original_gap_mm',
            'free_gap_mm',
            'anchor_gap_mm',
        )
        if any(x not in df for x in names):
            raise ValueError(f'Wide CSV needs {names}; got {list(df)}')
        data = df.melt(
            id_vars=ids + ['frame', 't_close'],
            value_vars=list(names),
            var_name='condition',
            value_name='absolute_gap_mm',
        )
        data['condition'] = (
            data['condition']
            .str.replace('_gap_mm', '', regex=False)
            .str.capitalize()
        )
    return data


def analyze(path, out, expected):
    data = load_long(path)
    data['absolute_gap_mm'] = pd.to_numeric(
        data['absolute_gap_mm'], errors='raise'
    )

    gap = data['absolute_gap_mm']
    if not np.isfinite(gap).all() or (gap < 0).any():
        raise ValueError('Gap must be finite, >= 0, in millimeters')

    for col in ('frame', 't_close'):
        data[col] = pd.to_numeric(data[col], errors='raise')
        if (not np.isfinite(data[col]).all()
                or (data[col] != data[col].astype(int)).any()):
            raise ValueError(f'{col} must be integer')
        data[col] = data[col].astype(int)

    if (not data['frame'].between(0, 10).all()
            or not data['t_close'].between(1, 9).all()):
        raise ValueError('Expected frame 0..10, t_close 1..9')

    keys = [x for x in
            ('task', 'clip_id', 'object_id', 'gripper_side')
            if x in data]

    if (data[keys].isna().any().any()
            or data.duplicated(
                keys + ['frame', 't_close', 'condition']
            ).any()):
        raise ValueError('Missing identifiers or duplicate rows')

    if data.groupby(keys)['t_close'].nunique().max() != 1:
        raise ValueError('t_close mismatch across conditions')

    wide = data.pivot(
        index=keys + ['frame', 't_close'],
        columns='condition',
        values='absolute_gap_mm',
    ).reset_index()
    wide.columns.name = None

    if any(x not in wide or wide[x].isna().any() for x in COND):
        raise ValueError('Missing matched Original/Free/Anchor frames')

    clip_keys = [x for x in ('task', 'clip_id') if x in wide]
    n = len(wide[clip_keys].drop_duplicates())

    if n != expected:
        raise ValueError(
            f'Matched clips N={n}, expected {expected}'
        )
    if len(wide[keys].drop_duplicates()) != n:
        raise ValueError('Multiple targets per clip: verify selection')

    if not all(
        tuple(sorted(g['frame'])) == tuple(range(11))
        for _, g in wide.groupby(keys)
    ):
        raise ValueError('Each clip must have frames 0..10')

    wide['rel_t'] = wide['frame'] - wide['t_close']
    wide = wide.sort_values(keys + ['frame'])
    post = wide[wide['rel_t'] >= 0].copy()

    for v in ('Free', 'Anchor'):
        post[f'{v}_dev_mm'] = (
            post[v] - post['Original']
        ).abs()

    byclip = []
    for key, g in post.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        r = dict(zip(keys, key))
        start = g.loc[g['rel_t'] == 0].iloc[0]
        end = g.loc[g['frame'] == 10].iloc[0]

        for v in COND:
            r[f'{v}_grasp_mm'] = start[v]
            r[f'{v}_post_median_mm'] = g[v].median()
            r[f'{v}_final_mm'] = end[v]
            r[f'{v}_final_minus_grasp_mm'] = (
                end[v] - start[v]
            )

        for v in ('Free', 'Anchor'):
            r[f'{v}_dev_median_mm'] = (
                g[f'{v}_dev_mm'].median()
            )
        byclip.append(r)

    pc = pd.DataFrame(byclip)
    rows = []

    metrics = [
        ('Grasp absolute gap (mm)', 'grasp_mm'),
        ('Post-grasp absolute gap median (mm)', 'post_median_mm'),
        ('Final absolute gap (mm)', 'final_mm'),
        ('Final minus grasp gap (mm)', 'final_minus_grasp_mm'),
    ]
    for label, suffix in metrics:
        rows.append({
            'Metric': label,
            **{
                v: pc[f'{v}_{suffix}'].median()
                for v in COND
            }
        })

    rows.append({
        'Metric': 'Post-grasp |gap - Original gap| median (mm)',
        'Original': 0.,
        'Free': pc['Free_dev_median_mm'].median(),
        'Anchor': pc['Anchor_dev_median_mm'].median(),
    })

    table = pd.DataFrame(rows)
    curve = post.groupby('rel_t').agg(
        N=('Original', 'size'),
        Original=('Original', 'median'),
        Free=('Free', 'median'),
        Anchor=('Anchor', 'median'),
    ).reset_index()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    pc.to_csv(out / 'per_clip_gap.csv', index=False)
    post.to_csv(out / 'per_frame_gap.csv', index=False)
    table.to_csv(out / 'comparison_table.csv', index=False)
    curve.to_csv(out / 'gap_curve.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in COND:
        ax.plot(
            curve['rel_t'], curve[v],
            marker='o', label=v
        )
    ax.set(
        xlabel='Frames since first gripper close',
        ylabel='Absolute min gap (mm)',
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'gap_curve.png', dpi=200)
    plt.close(fig)

    print(f'[OK] matched clips N={n}')
    print(table.to_string(
        index=False,
        float_format=lambda x: f'{x:.6f}'
    ))
    print('[OK] saved to:', out)
    print('NOTE: gap alone does not prove full coupling.')
    print('Check old deviation aggregation before comparing 0.379.')


def main():
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest='mode', required=True)

    p = subs.add_parser('audit')
    p.add_argument('--root', default='diagnostics/results')
    p.add_argument('--limit', type=int, default=40)

    p = subs.add_parser('snapshot')
    p.add_argument(
        '--root',
        default='diagnostics/results/close_anchor_causal_5task_current10457'
    )
    p.add_argument('--original-col')
    p.add_argument('--free-col')
    p.add_argument('--anchor-col')
    p.add_argument('--expected', type=int, default=524)

    p = subs.add_parser('analyze')
    p.add_argument('--input', required=True)
    p.add_argument(
        '--out',
        default='diagnostics/results/grasp_absolute_gap_current524'
    )
    p.add_argument('--expected', type=int, default=524)

    args = ap.parse_args()

    if args.mode == 'audit':
        audit(args.root, args.limit)
    elif args.mode == 'snapshot':
        snapshot(
            args.root,
            (args.original_col, args.free_col, args.anchor_col),
            args.expected,
        )
    else:
        analyze(args.input, args.out, args.expected)


if __name__ == '__main__':
    main()

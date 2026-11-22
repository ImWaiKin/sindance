"""Consolidate the per-participant HUR Balance Platform exports (E01.xls ... E81.xls, tab-separated text)
into one table: ../data/hur_performance.csv (one row per participant x test condition).

Placeholder files for participants whose platform recording failed are named '*-duplicated*.xls'
and are skipped, so those participants have no platform data (they are excluded from the balance
analysis and handled in missing.py). Header fields with personal details are not exported.

Usage: python consolidate_hur_data.py [folder with the .xls exports]   (default ../data/hur_raw)
"""
import os
import re
import sys
from io import StringIO
from pathlib import Path
import pandas as pd
from common import DATA, HUR


def read_performance_table(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    start = next((i for i, l in enumerate(lines) if 'Performance Name' in l and 'Trace Length' in l), None)
    if start is None:
        print(f'warning: no data table in {path}')
        return None
    text = '\n'.join(l.rstrip().rstrip('\t') for l in lines[start:])
    df = pd.read_csv(StringIO(text), sep='\t')
    df.columns = df.columns.str.strip()
    return df


def main(folder):
    files = sorted(f for f in Path(folder).iterdir() if re.match(r'^E\d+.*\.xls$', f.name))
    tables = []
    for f in files:
        if 'duplicated' in f.name:
            print(f'skipping placeholder {f.name}')
            continue
        df = read_performance_table(f)
        if df is not None and not df.empty:
            df['subject_id'] = f.name.split('.')[0]
            tables.append(df)
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(HUR, index=False)
    print(f'{out["subject_id"].nunique()} participants, {len(out)} rows -> {HUR}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'hur_raw'))

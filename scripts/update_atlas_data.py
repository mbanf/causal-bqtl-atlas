#!/usr/bin/env python3
"""
Update the interactive gene atlas HTML with new analysis results:
- Motif creation vs disruption (created families, rewiring category)
- Sole copy verification within peaks
- Rewiring deep analysis (clean switches, sole families)
"""

from pathlib import Path
import pandas as pd
import json
import re

RESULTS = str(Path(__file__).resolve().parent.parent / 'results')
INTERACTIVE = str(Path(__file__).resolve().parent.parent / 'interactive')
HTML_FILE = f'{INTERACTIVE}/gene_atlas_728_interactive.html'

# Load new data
creation = pd.read_csv(f'{RESULTS}/motif_creation_vs_disruption_728.csv')
rewiring = pd.read_csv(f'{RESULTS}/rewiring_deep_analysis_728.csv')
sole = pd.read_csv(f'{RESULTS}/sole_motif_verification_728.csv')

print(f"Loaded: creation={len(creation)}, rewiring={len(rewiring)}, sole={len(sole)}")

# Build lookup by (gene_id, chr, pos)
def make_key(row):
    return (row['gene_id'], row['chr'], int(row['pos']))

creation_map = {}
for _, r in creation.iterrows():
    k = make_key(r)
    creation_map[k] = {
        'created_families': [f for f in str(r['created_families']).split(',') if f and f != 'nan'],
        'n_created': int(r['n_created']),
        'n_unchanged': int(r['n_unchanged']),
    }

rewiring_map = {}
for _, r in rewiring.iterrows():
    k = make_key(r)
    cat = str(r['category'])
    rewiring_map[k] = {
        'rewiringCat': cat,
        'cleanSwitch': bool(r.get('clean_switch', False)),
        'disruptedSoleFamilies': [f for f in str(r.get('disrupted_sole_families', '')).split(',') if f and f != 'nan'],
        'createdSoleFamilies': [f for f in str(r.get('created_sole_families', '')).split(',') if f and f != 'nan'],
    }

sole_map = {}
for _, r in sole.iterrows():
    k = make_key(r)
    sole_map[k] = {
        'inPeak': bool(r.get('in_peak', False)),
        'hasSoleCopy': bool(r.get('has_any_sole', False)),
        'soleFamilies': [f for f in str(r.get('sole_families', '')).split(',') if f and f != 'nan'],
        'peakWidth': int(r['peak_width']) if pd.notna(r.get('peak_width', None)) else 0,
    }

# Read HTML and extract GENES JSON
with open(HTML_FILE, 'r') as f:
    html = f.read()

# Find the GENES line
m = re.search(r'const GENES = (\[.*?\]);\s*\n', html, re.DOTALL)
if not m:
    raise ValueError("Could not find GENES array in HTML")

genes = json.loads(m.group(1))
print(f"Found {len(genes)} genes in atlas")

# Enrich each gene
matched = 0
for g in genes:
    k = (g['id'], g['chr'], int(g['pos']))

    # Creation data
    if k in creation_map:
        c = creation_map[k]
        g['createdFamilies'] = c['created_families']
        g['nCreated'] = c['n_created']
        g['nUnchanged'] = c['n_unchanged']
        matched += 1
    else:
        g['createdFamilies'] = []
        g['nCreated'] = 0
        g['nUnchanged'] = 0

    # Rewiring category
    if k in rewiring_map:
        r = rewiring_map[k]
        g['rewiringCat'] = r['rewiringCat']
        g['cleanSwitch'] = r['cleanSwitch']
    else:
        # Infer from creation data
        has_d = len(g.get('tfFamilies', [])) > 0
        has_c = g['nCreated'] > 0
        if has_d and has_c:
            g['rewiringCat'] = 'rewired'
        elif has_d:
            g['rewiringCat'] = 'only_disrupted'
        elif has_c:
            g['rewiringCat'] = 'only_created'
        else:
            g['rewiringCat'] = 'neither'
        g['cleanSwitch'] = False

    # Sole copy
    if k in sole_map:
        s = sole_map[k]
        g['inPeak'] = s['inPeak']
        g['hasSoleCopy'] = s['hasSoleCopy']
        g['soleFamilies'] = s['soleFamilies']
        g['peakWidth'] = s['peakWidth']
    else:
        g['inPeak'] = False
        g['hasSoleCopy'] = False
        g['soleFamilies'] = []
        g['peakWidth'] = 0

print(f"Matched {matched}/{len(genes)} genes with creation data")

# Count categories
from collections import Counter
cats = Counter(g['rewiringCat'] for g in genes)
print(f"Rewiring categories: {dict(cats)}")
print(f"Clean switches: {sum(1 for g in genes if g['cleanSwitch'])}")
print(f"Has sole copy: {sum(1 for g in genes if g['hasSoleCopy'])}")

# Generate new GENES JSON (compact)
new_json = json.dumps(genes, separators=(',', ': '))
new_line = f'const GENES = {new_json};'

# Replace in HTML
html_new = html[:m.start()] + new_line + '\n' + html[m.end():]

with open(HTML_FILE, 'w') as f:
    f.write(html_new)

print(f"Updated {HTML_FILE}")
print(f"New fields per gene: createdFamilies, nCreated, nUnchanged, rewiringCat, cleanSwitch, inPeak, hasSoleCopy, soleFamilies, peakWidth")

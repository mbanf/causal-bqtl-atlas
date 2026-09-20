#!/usr/bin/env python3
"""
67_build_interactive.py — v2

Generate interactive HTML visualization of 728 causal bQTL target genes.
Features: layered cell diagram, rich tooltips, filter summaries,
OpenAI API integration for AI-powered gene/group summaries.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import os

RESULTS = str(Path(__file__).resolve().parent.parent / 'results')

# Load and merge data
atlas = pd.read_csv(f'{RESULTS}/gene_atlas_728.csv')
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')

merged = atlas.merge(
    causal[['gene_id','variant_hybrids','ref_hybrids','fdr']],
    on='gene_id', how='left'
)

LAYER_MAP = {
    'Receptor/Sensor': 'Receptor & Sensor',
    'Membrane/Transport': 'Membrane & Transport',
    'Hormone': 'Hormone Signaling',
    'Signal Transduction': 'Signal Transduction',
    'Second Messenger': 'Signal Transduction',
    'Protein Processing': 'Protein Quality Control',
    'Transcription Factor': 'Transcription Factor',
    'Chromatin/Epigenetic': 'Chromatin & Epigenetic',
    'Translation': 'Translation & Ribosome',
    'Metabolism': 'Metabolism & Biosynthesis',
    'Biosynthesis': 'Metabolism & Biosynthesis',
    'Redox/Stress': 'Redox & Stress Response',
    'Redox/Metabolism': 'Redox & Stress Response',
    'Cell wall': 'Cell Wall & Structure',
    'Kinase/Phosphorylation': 'Signal Transduction',
    'Other': 'Other Function',
    'Unknown': 'Uncharacterized',
}
merged['layer'] = merged['signaling_layer'].map(LAYER_MAP).fillna('Uncharacterized')

genes = []
for _, row in merged.iterrows():
    desc = str(row['description']) if pd.notna(row['description']) else ''
    if desc.startswith('XP_') or desc.startswith('NP_'):
        parts = desc.split(' ', 1)
        desc = parts[1] if len(parts) > 1 else desc
    if '[' in desc:
        desc = desc[:desc.rfind('[')].strip()

    tf_fams = str(row['tf_families']) if pd.notna(row['tf_families']) and row['tf_families'] else ''
    go = str(row['go_biological']) if pd.notna(row['go_biological']) else ''
    kegg = str(row['kegg_pathway']) if pd.notna(row['kegg_pathway']) else ''

    genes.append({
        'id': row['gene_id'],
        'chr': row['chr'],
        'pos': int(row['pos']),
        'absD': round(row['abs_d'], 2),
        'cohensD': round(row['cohens_d'], 2),
        'desc': desc[:200],
        'cog': str(row['cog_description'])[:120] if pd.notna(row['cog_description']) else '',
        'func': row['func_category'] if pd.notna(row['func_category']) else 'Unknown',
        'layer': row['layer'],
        'tfFamilies': tf_fams.split(',') if tf_fams else [],
        'nFamilies': int(row['n_families']),
        'log2fcDrought': round(float(row['log2fc_drought']), 2) if pd.notna(row['log2fc_drought']) else None,
        'droughtCat': row['drought_cat'] if pd.notna(row['drought_cat']) else 'Unknown',
        'go': go[:300],
        'kegg': kegg[:300],
        'nVar': int(row['n_variant_hybrids']),
        'nRef': int(row['n_ref_hybrids']),
        'varHybrids': row['variant_hybrids'].split('|') if pd.notna(row['variant_hybrids']) else [],
        'refHybrids': row['ref_hybrids'].split('|') if pd.notna(row['ref_hybrids']) else [],
        'fdr': round(float(row['fdr']), 4) if pd.notna(row['fdr']) else None,
    })

genes_json = json.dumps(genes)

layer_order = [
    'Receptor & Sensor', 'Membrane & Transport', 'Hormone Signaling',
    'Signal Transduction', 'Protein Quality Control', 'Redox & Stress Response',
    'Metabolism & Biosynthesis', 'Cell Wall & Structure',
    'Transcription Factor', 'Chromatin & Epigenetic', 'Translation & Ribosome',
    'Other Function', 'Uncharacterized',
]

layer_colors = {
    'Receptor & Sensor': '#E53935', 'Membrane & Transport': '#00ACC1',
    'Hormone Signaling': '#FF9800', 'Signal Transduction': '#D81B60',
    'Protein Quality Control': '#8E24AA', 'Redox & Stress Response': '#F4511E',
    'Metabolism & Biosynthesis': '#43A047', 'Cell Wall & Structure': '#6D4C41',
    'Transcription Factor': '#1E88E5', 'Chromatin & Epigenetic': '#3949AB',
    'Translation & Ribosome': '#7CB342', 'Other Function': '#9E9E9E',
    'Uncharacterized': '#BDBDBD',
}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>728 Causal bQTL — Interactive Gene Atlas</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background:#0a0a1a; color:#e0e0e0; overflow:hidden; }}

#app {{ display:grid; grid-template-columns:260px 1fr 380px; grid-template-rows:auto 1fr; height:100vh; }}

/* Header */
#header {{
    grid-column: 1 / -1; background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 10px 20px; border-bottom: 1px solid #2a2a4a;
    display: flex; align-items: center; justify-content: space-between;
}}
#header h1 {{ font-size:15px; font-weight:600; color:#fff; }}
#header .sub {{ font-size:11px; color:#7a7a9a; margin-left:12px; }}
#stats {{ font-size:11px; color:#999; }}
#stats span {{ color:#4fc3f7; font-weight:600; }}

/* Left: Filters */
#filters {{
    background:#10102a; border-right:1px solid #2a2a4a; padding:10px; overflow-y:auto;
}}
.fsec {{ margin-bottom:12px; }}
.fsec h3 {{ font-size:10px; text-transform:uppercase; color:#7a7a9a; margin-bottom:5px; letter-spacing:0.5px; }}
.fbtn {{
    display:inline-block; padding:2px 7px; margin:2px; border-radius:10px;
    font-size:10px; cursor:pointer; border:1px solid #333358;
    background:#16163a; color:#bbb; transition:all 0.12s; user-select:none;
}}
.fbtn:hover {{ background:#22225a; }}
.fbtn.active {{ background:#1565C0; color:#fff; border-color:#1976D2; }}
.fbtn .ct {{ color:#777; margin-left:2px; font-size:9px; }}
input[type=text] {{
    width:100%; padding:5px 9px; border-radius:5px; border:1px solid #333358;
    background:#16163a; color:#ddd; font-size:11px; outline:none;
}}
input[type=text]:focus {{ border-color:#1976D2; }}
input[type=range] {{ width:100%; margin:3px 0; accent-color:#1976D2; }}
.rl {{ display:flex; justify-content:space-between; font-size:9px; color:#777; }}

/* Main SVG */
#main {{ position:relative; overflow:hidden; background:#0d0d1e; }}
svg {{ width:100%; height:100%; }}

.layer-bg {{ fill-opacity:0.05; pointer-events:none; }}
.layer-label {{ font-size:10px; fill:#444; font-weight:500; pointer-events:none; }}

.gene-node {{ cursor:pointer; transition:opacity 0.15s; }}
.gene-node:hover {{ filter:brightness(1.5); }}
.gene-node.dimmed {{ opacity:0.06; }}
.gene-node.highlighted {{ stroke:#fff !important; stroke-width:2.5 !important; filter:brightness(1.3) drop-shadow(0 0 6px rgba(255,255,255,0.4)); }}
.gene-node.selected {{ stroke:#FFD700 !important; stroke-width:3 !important; filter:drop-shadow(0 0 8px rgba(255,215,0,0.6)); }}

/* Tooltip */
#tip {{
    position:absolute; display:none; pointer-events:none;
    background:rgba(14,14,36,0.97); border:1px solid #3a3a6a; border-radius:8px;
    padding:14px; max-width:420px; font-size:11px; line-height:1.6; z-index:100;
    box-shadow:0 8px 32px rgba(0,0,0,0.7);
}}
#tip .tt {{ font-size:14px; font-weight:700; color:#4fc3f7; }}
#tip .td {{ color:#ccc; margin:4px 0 6px; font-style:italic; font-size:12px; }}
#tip .tm {{ display:inline-block; padding:2px 7px; margin:2px; border-radius:4px; font-size:10px; }}
.tft {{ display:inline-block; padding:1px 6px; margin:1px; border-radius:8px; font-size:10px; background:#1a1a4a; border:1px solid #3a3a6a; color:#8ec8f0; }}
.ht {{ display:inline-block; padding:1px 4px; margin:1px; border-radius:3px; font-size:9px; }}
.hv {{ background:#1B5E20; color:#A5D6A7; }}
.hr {{ background:#1A237E; color:#9FA8DA; }}
#tip hr {{ border:none; border-top:1px solid #2a2a4a; margin:5px 0; }}

/* Right: Detail + AI */
#detail {{
    background:#10102a; border-left:1px solid #2a2a4a; padding:0; overflow-y:auto;
    display:flex; flex-direction:column;
}}
#detail-header {{
    padding:12px 14px; border-bottom:1px solid #2a2a4a; flex-shrink:0;
}}
#detail-header .tabs {{ display:flex; gap:2px; margin-top:8px; }}
.tab {{
    padding:5px 12px; border-radius:6px 6px 0 0; font-size:11px; cursor:pointer;
    background:#16163a; color:#888; border:1px solid #2a2a4a; border-bottom:none;
}}
.tab.active {{ background:#1a1a3e; color:#4fc3f7; font-weight:600; }}
#tab-content {{ flex:1; overflow-y:auto; padding:12px 14px; }}

/* Gene detail styles */
.gd-title {{ font-size:15px; font-weight:700; color:#4fc3f7; margin-bottom:2px; }}
.gd-desc {{ color:#ccc; font-style:italic; margin-bottom:8px; font-size:12px; line-height:1.5; }}
.gd-section {{ margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #1e1e3a; }}
.gd-section h4 {{ font-size:10px; text-transform:uppercase; color:#6a6a8a; margin-bottom:4px; letter-spacing:0.4px; }}
.gd-row {{ margin:2px 0; }}
.gd-key {{ color:#777; display:inline-block; min-width:110px; font-size:11px; }}
.gd-val {{ color:#ddd; font-size:11px; }}
.gd-val.up {{ color:#EF5350; }}
.gd-val.dn {{ color:#42A5F5; }}

/* Summary view */
.sum-card {{ background:#16163a; border:1px solid #2a2a4a; border-radius:8px; padding:10px; margin-bottom:8px; }}
.sum-card h4 {{ font-size:11px; color:#4fc3f7; margin-bottom:4px; }}
.sum-list {{ font-size:10px; color:#aaa; line-height:1.7; max-height:200px; overflow-y:auto; }}
.sum-gene {{ padding:3px 6px; margin:1px 0; border-radius:4px; cursor:pointer; display:flex; align-items:center; gap:6px; }}
.sum-gene:hover {{ background:#1e1e4a; }}
.sum-gene .sg-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.sum-gene .sg-id {{ color:#8ec8f0; font-weight:600; min-width:140px; }}
.sum-gene .sg-desc {{ color:#999; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sum-gene .sg-d {{ color:#aaa; font-size:9px; min-width:50px; text-align:right; }}

/* AI panel */
#ai-panel {{ padding:0; }}
#ai-config {{ padding:10px 14px; border-bottom:1px solid #2a2a4a; }}
#ai-config input {{ margin-bottom:6px; }}
#ai-config select {{ width:100%; padding:5px; border-radius:5px; background:#16163a; color:#ddd; border:1px solid #333358; font-size:11px; }}
#ai-btn {{
    width:100%; padding:8px; border:none; border-radius:6px; cursor:pointer;
    background:linear-gradient(135deg, #1565C0, #0D47A1); color:#fff; font-size:12px; font-weight:600;
    margin-top:6px; transition:opacity 0.15s;
}}
#ai-btn:hover {{ opacity:0.85; }}
#ai-btn:disabled {{ opacity:0.4; cursor:not-allowed; }}
#ai-output {{
    padding:14px; font-size:11px; line-height:1.7; color:#ccc; white-space:pre-wrap;
    overflow-y:auto; flex:1;
}}
#ai-output .thinking {{ color:#666; font-style:italic; }}

/* Legend */
#legend {{
    position:absolute; bottom:10px; left:10px; background:rgba(14,14,36,0.92);
    border:1px solid #2a2a4a; border-radius:8px; padding:8px 10px;
    font-size:9px; z-index:50; max-height:50vh; overflow-y:auto;
}}
.li {{ display:flex; align-items:center; margin:1px 0; }}
.ld {{ width:9px; height:9px; border-radius:50%; margin-right:5px; flex-shrink:0; }}
</style>
</head>
<body>
<div id="app">
    <div id="header">
        <div style="display:flex;align-items:baseline">
            <h1>728 Causal bQTL Target Genes</h1>
            <span class="sub">Interactive Regulatory Atlas</span>
        </div>
        <div id="stats">
            Showing <span id="n-vis">728</span>/728 &nbsp;|&nbsp;
            <span id="n-layers">13</span> layers &nbsp;|&nbsp;
            <span id="n-tf">36</span> TF families
        </div>
    </div>

    <div id="filters">
        <div class="fsec">
            <h3>Search</h3>
            <input type="text" id="search" placeholder="Gene ID, protein name, function...">
        </div>
        <div class="fsec">
            <h3>Effect Size |d|</h3>
            <input type="range" id="effect-min" min="0" max="15" step="0.5" value="0">
            <div class="rl"><span id="eff-lbl">Min: 0</span><span>15</span></div>
        </div>
        <div class="fsec"><h3>Signaling Layer</h3><div id="f-layer"></div></div>
        <div class="fsec"><h3>Disrupted TF Family</h3><div id="f-tf"></div></div>
        <div class="fsec"><h3>Drought Response</h3><div id="f-drought"></div></div>
        <div class="fsec"><h3>Hybrid (variant carriers)</h3><div id="f-hybrid" style="max-height:160px;overflow-y:auto"></div></div>
    </div>

    <div id="main">
        <svg id="svg-container"></svg>
        <div id="tip"></div>
        <div id="legend"></div>
    </div>

    <div id="detail">
        <div id="detail-header">
            <div style="font-size:13px;font-weight:600;color:#fff;">Detail Panel</div>
            <div class="tabs">
                <div class="tab active" data-tab="summary" onclick="switchTab('summary')">Summary</div>
                <div class="tab" data-tab="gene" onclick="switchTab('gene')">Gene</div>
                <div class="tab" data-tab="ai" onclick="switchTab('ai')">AI Summary</div>
            </div>
        </div>
        <div id="tab-content">
            <div id="panel-summary"></div>
            <div id="panel-gene" style="display:none"></div>
            <div id="panel-ai" style="display:none">
                <div id="ai-config">
                    <input type="password" id="api-key" placeholder="OpenAI API key (sk-...)">
                    <select id="ai-model">
                        <option value="gpt-4o-mini">gpt-4o-mini (fast)</option>
                        <option value="gpt-4o">gpt-4o</option>
                        <option value="gpt-4.1-mini">gpt-4.1-mini</option>
                        <option value="gpt-4.1">gpt-4.1</option>
                    </select>
                    <button id="ai-btn" onclick="aiSummarize()">Summarize current selection</button>
                </div>
                <div id="ai-output"><span class="thinking">Enter your API key above, select genes via filters or clicking, then press Summarize.</span></div>
            </div>
        </div>
    </div>
</div>

<script>
const GENES = {genes_json};
const LAYER_ORDER = {json.dumps(layer_order)};
const LAYER_COLORS = {json.dumps(layer_colors)};
const ALL_HYBRIDS = ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
    'IL14H','Ki11','Ki3','Ky21','M162W','M37W','Mo18W','Ms71','NC358','Oh43','Oh7b','P39','Tx303'];

const allTF = [...new Set(GENES.flatMap(g => g.tfFamilies))].sort();

// State
let activeLayerF = new Set(), activeTFF = new Set(), activeDroughtF = new Set();
let activeHybridF = null, effectMin = 0, searchTerm = '', selectedGene = null;

// ── Tabs ──
function switchTab(tab) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
    ['summary','gene','ai'].forEach(p => {{
        document.getElementById('panel-' + p).style.display = (p === tab) ? 'block' : 'none';
    }});
}}

// ── Filters ──
function buildFilters() {{
    const layerDiv = document.getElementById('f-layer');
    LAYER_ORDER.forEach(layer => {{
        const c = GENES.filter(g => g.layer === layer).length;
        if (!c) return;
        const b = mkBtn(layer, c, LAYER_COLORS[layer]);
        b.onclick = () => {{ toggle(activeLayerF, layer, b); updateAll(); }};
        layerDiv.appendChild(b);
    }});

    const tfDiv = document.getElementById('f-tf');
    allTF.forEach(tf => {{
        const c = GENES.filter(g => g.tfFamilies.includes(tf)).length;
        const b = mkBtn(tf, c);
        b.onclick = () => {{ toggle(activeTFF, tf, b); updateAll(); }};
        tfDiv.appendChild(b);
    }});

    const drDiv = document.getElementById('f-drought');
    const drCats = ['Strong up','Mild up','Stable','Mild down','Strong down'];
    const drCol = {{'Strong up':'#EF5350','Mild up':'#FFAB91','Stable':'#9E9E9E','Mild down':'#90CAF9','Strong down':'#42A5F5'}};
    drCats.forEach(cat => {{
        const c = GENES.filter(g => g.droughtCat === cat).length;
        const b = mkBtn(cat, c); b.style.borderColor = drCol[cat] || '#333';
        b.onclick = () => {{ toggle(activeDroughtF, cat, b); updateAll(); }};
        drDiv.appendChild(b);
    }});

    const hyDiv = document.getElementById('f-hybrid');
    ALL_HYBRIDS.forEach(hyb => {{
        const c = GENES.filter(g => g.varHybrids.includes(hyb)).length;
        const b = mkBtn(hyb, c);
        b.onclick = () => {{
            if (activeHybridF === hyb) {{ activeHybridF = null; b.classList.remove('active'); }}
            else {{ hyDiv.querySelectorAll('.fbtn').forEach(x => x.classList.remove('active')); activeHybridF = hyb; b.classList.add('active'); }}
            updateAll();
        }};
        hyDiv.appendChild(b);
    }});
}}

function mkBtn(label, count, dotColor) {{
    const b = document.createElement('div');
    b.className = 'fbtn';
    const dot = dotColor ? `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${{dotColor}};margin-right:3px"></span>` : '';
    b.innerHTML = `${{dot}}${{label}} <span class="ct">${{count}}</span>`;
    return b;
}}

function toggle(set, val, btn) {{
    if (set.has(val)) {{ set.delete(val); btn.classList.remove('active'); }}
    else {{ set.add(val); btn.classList.add('active'); }}
}}

function isVisible(g) {{
    if (effectMin > 0 && g.absD < effectMin) return false;
    if (activeLayerF.size && !activeLayerF.has(g.layer)) return false;
    if (activeTFF.size && !g.tfFamilies.some(f => activeTFF.has(f))) return false;
    if (activeDroughtF.size && !activeDroughtF.has(g.droughtCat)) return false;
    if (activeHybridF && !g.varHybrids.includes(activeHybridF)) return false;
    if (searchTerm) {{
        const s = searchTerm.toLowerCase();
        return g.id.toLowerCase().includes(s) || g.desc.toLowerCase().includes(s) ||
               g.cog.toLowerCase().includes(s) || g.func.toLowerCase().includes(s) ||
               g.go.toLowerCase().includes(s);
    }}
    return true;
}}

// ── SVG ──
const svg = d3.select('#svg-container');
const mainG = svg.append('g');
let W, H;
function resize() {{
    const r = document.getElementById('main').getBoundingClientRect();
    W = r.width; H = r.height;
    svg.attr('viewBox', `0 0 ${{W}} ${{H}}`);
}}
resize();
window.addEventListener('resize', () => {{ resize(); layoutGenes(); }});

const nodeData = GENES.map((g, i) => ({{ ...g, index: i, x: 0, y: 0 }}));
let simulation, nodeSelection;
const rScale = d => Math.max(3, Math.min(13, 2 + d * 1.5));

function layoutGenes() {{
    const m = {{ t:25, b:15, l:15, r:15 }};
    const layerH = (H - m.t - m.b) / LAYER_ORDER.length;
    const layerY = {{}};
    LAYER_ORDER.forEach((l, i) => {{ layerY[l] = m.t + (i + 0.5) * layerH; }});

    mainG.selectAll('.layer-bg,.layer-label').remove();
    LAYER_ORDER.forEach((layer, i) => {{
        const y = m.t + i * layerH;
        const c = GENES.filter(g => g.layer === layer).length;
        if (!c) return;
        mainG.append('rect').attr('class','layer-bg')
            .attr('x', m.l).attr('y', y).attr('width', W - m.l - m.r).attr('height', layerH)
            .attr('fill', LAYER_COLORS[layer]).attr('rx', 3);
        mainG.append('text').attr('class','layer-label')
            .attr('x', m.l + 5).attr('y', y + 12)
            .text(`${{layer}} (${{c}})`);
    }});

    nodeData.forEach(d => {{
        const ly = layerY[d.layer] || H / 2;
        if (!d._init) {{
            d.x = m.l + 30 + Math.random() * (W - m.l - m.r - 60);
            d.y = ly + (Math.random() - 0.5) * layerH * 0.7;
            d._init = true;
        }}
        d.targetY = ly;
    }});

    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(nodeData)
        .force('x', d3.forceX(W / 2).strength(0.008))
        .force('y', d3.forceY(d => d.targetY).strength(0.3))
        .force('collide', d3.forceCollide(d => rScale(d.absD) + 1.5).iterations(2))
        .force('charge', d3.forceManyBody().strength(-1.5))
        .alphaDecay(0.03)
        .on('tick', () => {{
            if (nodeSelection) nodeSelection.attr('cx', d => d.x).attr('cy', d => d.y);
        }});
}}

function drawNodes() {{
    mainG.selectAll('.gene-node').remove();
    nodeSelection = mainG.selectAll('.gene-node')
        .data(nodeData, d => d.id).join('circle')
        .attr('class', 'gene-node')
        .attr('r', d => rScale(d.absD))
        .attr('fill', d => LAYER_COLORS[d.layer] || '#666')
        .attr('stroke', d => {{
            if (d.droughtCat.includes('up')) return '#EF5350';
            if (d.droughtCat.includes('down')) return '#42A5F5';
            return 'rgba(255,255,255,0.12)';
        }})
        .attr('stroke-width', d => d.droughtCat.includes('Strong') ? 2 : d.droughtCat.includes('Mild') ? 1.2 : 0.5)
        .on('mouseover', showTip).on('mousemove', moveTip).on('mouseout', hideTip)
        .on('click', (ev, d) => clickGene(d));
}}

function updateAll() {{
    const vis = new Set(nodeData.filter(isVisible).map(d => d.id));
    document.getElementById('n-vis').textContent = vis.size;
    if (nodeSelection) {{
        nodeSelection
            .classed('dimmed', d => !vis.has(d.id))
            .classed('highlighted', d => {{
                if (!selectedGene || selectedGene.id === d.id) return false;
                return selectedGene.tfFamilies.some(f => d.tfFamilies.includes(f));
            }})
            .classed('selected', d => selectedGene && selectedGene.id === d.id);
    }}
    updateSummary(vis);
}}

// ── Tooltip ──
const tip = document.getElementById('tip');

function showTip(ev, d) {{
    moveTip(ev);
    const tfs = d.tfFamilies.map(f => `<span class="tft">${{f}}</span>`).join('');
    const drC = d.log2fcDrought > 1 ? '#EF5350' : d.log2fcDrought < -1 ? '#42A5F5' : '#777';
    const drL = d.log2fcDrought !== null ? `${{d.log2fcDrought > 0 ? '+' : ''}}${{d.log2fcDrought.toFixed(2)}}` : '—';

    tip.innerHTML = `
        <div class="tt">${{d.id}}</div>
        <div class="td">${{d.desc || 'Uncharacterized protein'}}</div>
        <div style="margin-bottom:6px">
            <span class="tm" style="background:#1B5E2088;color:#A5D6A7"><b>|d| = ${{d.absD}}</b></span>
            <span class="tm" style="background:${{drC}}33;color:${{drC}}">Drought: ${{drL}}</span>
            <span class="tm" style="background:#1a1a4a">Layer: <b style="color:${{LAYER_COLORS[d.layer]}}">${{d.layer}}</b></span>
        </div>
        ${{d.cog && d.cog !== 'nan' ? `<div style="color:#999;font-size:10px;margin-bottom:4px">COG: ${{d.cog}}</div>` : ''}}
        <hr>
        <div style="color:#888;font-size:10px">Disrupted TF motifs (${{d.nFamilies}}):</div>
        <div style="margin:3px 0">${{tfs || '<span style="color:#555">None identified</span>'}}</div>
        <hr>
        <div style="font-size:10px">
            <span style="color:#A5D6A7">Variant (${{d.nVar}}):</span> ${{d.varHybrids.join(', ')}}
        </div>
        <div style="font-size:10px;margin-top:2px">
            <span style="color:#9FA8DA">Reference (${{d.nRef}}):</span> ${{d.refHybrids.join(', ')}}
        </div>
        <div style="font-size:10px;color:#555;margin-top:4px">${{d.chr}}:${{d.pos.toLocaleString()}} &nbsp; FDR=${{d.fdr || '—'}}</div>
    `;
    tip.style.display = 'block';
}}
function moveTip(ev) {{
    const rect = document.getElementById('main').getBoundingClientRect();
    const x = ev.clientX - rect.left + 15;
    const y = ev.clientY - rect.top - 10;
    tip.style.left = Math.min(x, W - 440) + 'px';
    tip.style.top = Math.max(Math.min(y, H - 350), 5) + 'px';
}}
function hideTip() {{ tip.style.display = 'none'; }}

// ── Gene click → detail ──
function clickGene(d) {{
    selectedGene = (selectedGene && selectedGene.id === d.id) ? null : d;
    updateAll();
    if (selectedGene) {{
        switchTab('gene');
        showGeneDetail(selectedGene);
    }}
}}

function showGeneDetail(g) {{
    const drC = g.log2fcDrought > 1 ? 'up' : g.log2fcDrought < -1 ? 'dn' : '';
    const tfs = g.tfFamilies.map(f => {{
        const n = nodeData.filter(o => o.id !== g.id && o.tfFamilies.includes(f)).length;
        return `<span class="tft">${{f}}</span> <span style="color:#666;font-size:9px">(${{n}} co-regulated)</span>`;
    }}).join('<br>');

    const varH = g.varHybrids.map(h => `<span class="ht hv">${{h}}</span>`).join(' ');
    const refH = g.refHybrids.map(h => `<span class="ht hr">${{h}}</span>`).join(' ');

    // Co-regulated genes grouped by shared TF
    let coRegHTML = '';
    g.tfFamilies.forEach(f => {{
        const coGenes = nodeData.filter(o => o.id !== g.id && o.tfFamilies.includes(f));
        if (!coGenes.length) return;
        coRegHTML += `<div style="margin-top:6px"><span class="tft">${{f}}</span> targets (${{coGenes.length}}):</div>`;
        coRegHTML += '<div style="max-height:120px;overflow-y:auto;margin:3px 0">';
        coGenes.sort((a,b) => b.absD - a.absD).slice(0, 20).forEach(cg => {{
            coRegHTML += `<div class="sum-gene" onclick="clickGene(nodeData[${{cg.index}}])">
                <span class="sg-dot" style="background:${{LAYER_COLORS[cg.layer]}}"></span>
                <span class="sg-id">${{cg.id}}</span>
                <span class="sg-desc">${{cg.desc.substring(0,50)}}</span>
                <span class="sg-d">|d|=${{cg.absD}}</span>
            </div>`;
        }});
        if (coGenes.length > 20) coRegHTML += `<div style="color:#555;font-size:9px;padding:2px 6px">+${{coGenes.length-20}} more</div>`;
        coRegHTML += '</div>';
    }});

    document.getElementById('panel-gene').innerHTML = `
        <div class="gd-section">
            <div class="gd-title">${{g.id}}</div>
            <div class="gd-desc">${{g.desc || 'Uncharacterized protein'}}</div>
            ${{g.cog && g.cog !== 'nan' ? `<div style="color:#888;font-size:10px">${{g.cog}}</div>` : ''}}
        </div>
        <div class="gd-section">
            <h4>Metrics</h4>
            <div class="gd-row"><span class="gd-key">Effect size</span> <span class="gd-val"><b>|d| = ${{g.absD}}</b> (d = ${{g.cohensD}})</span></div>
            <div class="gd-row"><span class="gd-key">FDR</span> <span class="gd-val">${{g.fdr || '—'}}</span></div>
            <div class="gd-row"><span class="gd-key">Drought</span> <span class="gd-val ${{drC}}">${{g.log2fcDrought !== null ? (g.log2fcDrought>0?'+':'') + g.log2fcDrought.toFixed(2) : '—'}} (${{g.droughtCat}})</span></div>
            <div class="gd-row"><span class="gd-key">Layer</span> <span class="gd-val" style="color:${{LAYER_COLORS[g.layer]}}">${{g.layer}}</span></div>
            <div class="gd-row"><span class="gd-key">Function</span> <span class="gd-val">${{g.func}}</span></div>
            <div class="gd-row"><span class="gd-key">Position</span> <span class="gd-val">${{g.chr}}:${{g.pos.toLocaleString()}}</span></div>
        </div>
        <div class="gd-section">
            <h4>Disrupted TF Motifs (${{g.nFamilies}})</h4>
            <div>${{tfs || '<span style="color:#555">None identified</span>'}}</div>
        </div>
        <div class="gd-section">
            <h4>Hybrids</h4>
            <div style="margin-bottom:4px"><span style="color:#A5D6A7;font-size:10px">Variant (${{g.nVar}}):</span> ${{varH}}</div>
            <div><span style="color:#9FA8DA;font-size:10px">Reference (${{g.nRef}}):</span> ${{refH}}</div>
        </div>
        ${{g.go ? `<div class="gd-section"><h4>GO Biological Process</h4><div style="font-size:10px;color:#999;line-height:1.6">${{g.go}}</div></div>` : ''}}
        ${{g.kegg ? `<div class="gd-section"><h4>KEGG Pathway</h4><div style="font-size:10px;color:#999">${{g.kegg}}</div></div>` : ''}}
        <div class="gd-section">
            <h4>Co-regulated Genes (shared disrupted TF)</h4>
            ${{coRegHTML || '<span style="color:#555">No co-regulated genes</span>'}}
        </div>
    `;
}}

// ── Summary panel (filter context) ──
function updateSummary(visibleSet) {{
    const panel = document.getElementById('panel-summary');
    const vis = nodeData.filter(d => visibleSet.has(d.id));

    if (vis.length === 728 && !selectedGene) {{
        panel.innerHTML = `
            <div class="sum-card"><h4 style="color:#fff;font-size:14px">728 Causal bQTL Target Genes</h4>
                <div style="color:#999;margin-top:6px;font-size:11px;line-height:1.6">
                    Each gene has a <b>single</b> bQTL variant in its promoter that causally shifts allele-specific expression across NAM hybrids (FDR&lt;0.05).
                    <br><br>Use the filters on the left to explore by signaling layer, TF family, drought response, or hybrid.
                    <br><br>Click any gene for full details. Use the <b>AI Summary</b> tab to get LLM-powered analysis of your selection.
                </div>
            </div>
            <div class="sum-card"><h4>Quick Stats</h4>
                <div style="font-size:11px;color:#aaa;line-height:2">
                    Median effect size: <b style="color:#4fc3f7">|d| = ${{median(GENES.map(g=>g.absD)).toFixed(2)}}</b><br>
                    Median variant hybrids: <b style="color:#4fc3f7">${{median(GENES.map(g=>g.nVar))}}</b> / 19<br>
                    Drought UP (|log2FC|>1): <b style="color:#EF5350">${{GENES.filter(g=>g.log2fcDrought>1).length}}</b><br>
                    Drought DOWN (|log2FC|<-1): <b style="color:#42A5F5">${{GENES.filter(g=>g.log2fcDrought<-1).length}}</b><br>
                    TF genes as targets: <b style="color:#4fc3f7">${{GENES.filter(g=>g.func==='Transcription factor').length}}</b>
                </div>
            </div>`;
        return;
    }}

    // Active filters description
    const filters = [];
    if (activeLayerF.size) filters.push(`Layer: ${{[...activeLayerF].join(', ')}}`);
    if (activeTFF.size) filters.push(`TF: ${{[...activeTFF].join(', ')}}`);
    if (activeDroughtF.size) filters.push(`Drought: ${{[...activeDroughtF].join(', ')}}`);
    if (activeHybridF) filters.push(`Hybrid: ${{activeHybridF}}`);
    if (effectMin > 0) filters.push(`|d| >= ${{effectMin}}`);
    if (searchTerm) filters.push(`Search: "${{searchTerm}}"`);

    // Layer breakdown
    const layerDist = {{}};
    vis.forEach(g => {{ layerDist[g.layer] = (layerDist[g.layer] || 0) + 1; }});

    // TF family breakdown
    const tfDist = {{}};
    vis.forEach(g => g.tfFamilies.forEach(f => {{ tfDist[f] = (tfDist[f] || 0) + 1; }}));
    const tfSorted = Object.entries(tfDist).sort((a,b) => b[1] - a[1]);

    // Top genes by effect
    const topGenes = [...vis].sort((a,b) => b.absD - a.absD).slice(0, 15);

    let layerHTML = Object.entries(layerDist).sort((a,b) => b[1] - a[1])
        .map(([l,c]) => `<span style="color:${{LAYER_COLORS[l]}}">${{l}}</span>: ${{c}}`).join('<br>');

    let tfHTML = tfSorted.slice(0, 10)
        .map(([f,c]) => `<span class="tft">${{f}}</span> ${{c}}`).join(' &nbsp; ');

    let geneListHTML = topGenes.map(g =>
        `<div class="sum-gene" onclick="clickGene(nodeData[${{g.index}}])">
            <span class="sg-dot" style="background:${{LAYER_COLORS[g.layer]}}"></span>
            <span class="sg-id">${{g.id}}</span>
            <span class="sg-desc">${{g.desc.substring(0,45)}}</span>
            <span class="sg-d">|d|=${{g.absD}}</span>
        </div>`
    ).join('');

    const meanD = (vis.reduce((s,g) => s + g.absD, 0) / vis.length).toFixed(2);

    panel.innerHTML = `
        <div class="sum-card">
            <h4>${{vis.length}} Genes Selected</h4>
            <div style="font-size:10px;color:#888;margin-top:4px">${{filters.join(' | ') || 'No filters'}}</div>
            <div style="font-size:11px;color:#aaa;margin-top:6px">Mean |d| = <b style="color:#4fc3f7">${{meanD}}</b></div>
        </div>
        <div class="sum-card"><h4>Signaling Layers</h4><div style="font-size:10px;line-height:1.8;margin-top:4px">${{layerHTML}}</div></div>
        <div class="sum-card"><h4>Top TF Families Disrupted</h4><div style="margin-top:4px">${{tfHTML}}</div></div>
        <div class="sum-card"><h4>Top Genes by Effect Size</h4><div class="sum-list">${{geneListHTML}}</div></div>
    `;
}}

function median(arr) {{
    const s = [...arr].sort((a,b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
}}

// ── AI Summary (OpenAI API) ──
async function aiSummarize() {{
    const key = document.getElementById('api-key').value.trim();
    if (!key) {{ alert('Please enter an OpenAI API key'); return; }}

    const model = document.getElementById('ai-model').value;
    const out = document.getElementById('ai-output');
    const btn = document.getElementById('ai-btn');
    btn.disabled = true;

    // Gather context: filtered genes or selected gene
    const vis = nodeData.filter(isVisible);
    let context;

    if (selectedGene) {{
        const g = selectedGene;
        const coReg = nodeData.filter(o => o.id !== g.id && g.tfFamilies.some(f => o.tfFamilies.includes(f)));
        context = `Selected gene: ${{g.id}}
Description: ${{g.desc}}
COG: ${{g.cog}}
Function category: ${{g.func}}
Signaling layer: ${{g.layer}}
Effect size: |d| = ${{g.absD}} (Cohen's d = ${{g.cohensD}})
Drought response: log2FC = ${{g.log2fcDrought}} (${{g.droughtCat}})
Disrupted TF motifs: ${{g.tfFamilies.join(', ') || 'none'}}
Variant hybrids (${{g.nVar}}): ${{g.varHybrids.join(', ')}}
GO: ${{g.go || 'none'}}
KEGG: ${{g.kegg || 'none'}}
Position: ${{g.chr}}:${{g.pos}}

Co-regulated genes (sharing disrupted TF family): ${{coReg.length}}
Top co-regulated: ${{coReg.sort((a,b) => b.absD - a.absD).slice(0,10).map(c => c.id + ' (' + c.desc.substring(0,40) + ', ' + c.layer + ')').join('; ')}}`;
    }} else {{
        // Summary of filtered set
        const sample = vis.sort((a,b) => b.absD - a.absD).slice(0, 30);
        const layerDist = {{}};
        vis.forEach(g => {{ layerDist[g.layer] = (layerDist[g.layer] || 0) + 1; }});
        const tfDist = {{}};
        vis.forEach(g => g.tfFamilies.forEach(f => {{ tfDist[f] = (tfDist[f] || 0) + 1; }}));

        context = `${{vis.length}} genes currently selected.
Active filters: ${{[...activeLayerF].join(',')||'none'}} | TF: ${{[...activeTFF].join(',')||'none'}} | Drought: ${{[...activeDroughtF].join(',')||'none'}} | Hybrid: ${{activeHybridF||'none'}}

Layer distribution: ${{Object.entries(layerDist).map(([l,c]) => l+':'+c).join(', ')}}
TF families disrupted: ${{Object.entries(tfDist).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([f,c]) => f+':'+c).join(', ')}}

Top 30 genes by effect size:
${{sample.map(g => `${{g.id}} | ${{g.desc.substring(0,60)}} | ${{g.layer}} | |d|=${{g.absD}} | TF:${{g.tfFamilies.join(',')}} | drought=${{g.log2fcDrought}}`).join('\\n')}}`;
    }}

    const systemPrompt = `You are a plant molecular biologist analyzing maize bQTL target genes. These 728 genes each have a single nucleotide variant in their promoter that disrupts a TF binding motif and causally shifts allele-specific expression across NAM F1 hybrids (B73 × diverse founders).

Your task: Provide a concise but insightful biological interpretation. Focus on:
1. What the gene/protein does (molecular function, known roles)
2. How it connects to other genes in the selection (signaling pathways, protein complexes)
3. Biological significance of the disrupted TF motif — what does it mean that this binding site is variable?
4. Drought/stress implications if relevant
5. For gene sets: identify coherent signaling modules, regulatory cascades, or functional themes

Be specific about maize biology when possible. Reference known pathways (BR signaling, ABA, MAPK cascades, etc). Note any genes that are particularly interesting targets for functional validation.`;

    out.innerHTML = '<span class="thinking">Thinking...</span>';

    try {{
        const resp = await fetch('https://api.openai.com/v1/chat/completions', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json', 'Authorization': `Bearer ${{key}}` }},
            body: JSON.stringify({{
                model: model,
                messages: [
                    {{ role: 'system', content: systemPrompt }},
                    {{ role: 'user', content: context }}
                ],
                max_tokens: 2000,
                temperature: 0.3
            }})
        }});

        if (!resp.ok) {{
            const err = await resp.json().catch(() => ({{}}));
            throw new Error(err.error?.message || `HTTP ${{resp.status}}`);
        }}

        const data = await resp.json();
        const text = data.choices[0].message.content;
        // Simple markdown rendering
        out.innerHTML = text
            .replace(/\\*\\*(.+?)\\*\\*/g, '<b>$1</b>')
            .replace(/\\*(.+?)\\*/g, '<i>$1</i>')
            .replace(/^### (.+)$/gm, '<h4 style="color:#4fc3f7;margin:8px 0 4px">$1</h4>')
            .replace(/^## (.+)$/gm, '<h3 style="color:#4fc3f7;margin:10px 0 4px">$1</h3>')
            .replace(/^- (.+)$/gm, '<div style="padding-left:12px">• $1</div>')
            .replace(/`(.+?)`/g, '<code style="background:#1a1a4a;padding:1px 4px;border-radius:3px;color:#8ec8f0">$1</code>')
            .replace(/\\n/g, '<br>');
    }} catch (e) {{
        out.innerHTML = `<span style="color:#EF5350">Error: ${{e.message}}</span>`;
    }}
    btn.disabled = false;
}}

// ── Legend ──
function buildLegend() {{
    const leg = document.getElementById('legend');
    let h = '<div style="font-weight:600;margin-bottom:4px;color:#999">Signaling Layer</div>';
    LAYER_ORDER.forEach(l => {{
        const c = GENES.filter(g => g.layer === l).length;
        if (!c) return;
        h += `<div class="li"><div class="ld" style="background:${{LAYER_COLORS[l]}}"></div>${{l}} (${{c}})</div>`;
    }});
    h += `<div style="margin-top:6px;font-weight:600;color:#999">Border = Drought</div>
        <div class="li"><div class="ld" style="background:transparent;border:2px solid #EF5350;width:7px;height:7px"></div>UP</div>
        <div class="li"><div class="ld" style="background:transparent;border:2px solid #42A5F5;width:7px;height:7px"></div>DOWN</div>
        <div style="margin-top:6px;font-weight:600;color:#999">Size = |Effect|</div>`;
    leg.innerHTML = h;
}}

// ── Event listeners ──
document.getElementById('search').addEventListener('input', e => {{ searchTerm = e.target.value; updateAll(); }});
document.getElementById('effect-min').addEventListener('input', e => {{
    effectMin = parseFloat(e.target.value);
    document.getElementById('eff-lbl').textContent = `Min: ${{effectMin}}`;
    updateAll();
}});

// ── Init ──
buildFilters();
buildLegend();
drawNodes();
layoutGenes();
updateAll();
</script>
</body>
</html>"""

INTERACTIVE = str(Path(__file__).resolve().parent.parent / 'interactive')
os.makedirs(INTERACTIVE, exist_ok=True)
out_path = f'{INTERACTIVE}/gene_atlas_728_interactive.html'
with open(out_path, 'w') as f:
    f.write(html)
print(f"Saved: {out_path} ({len(html)//1024}KB)")

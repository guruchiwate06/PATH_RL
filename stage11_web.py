"""
stage11_web.py
--------------
Stage 11 Browser Demo: Regulatory & Exit-Placement Constraint Validation.

A Flask-based web server that runs the ExitValidator against loaded scenarios
and serves the results as a rich interactive browser dashboard.

Usage:
    python stage11_web.py
    # Open http://localhost:5111 in your browser
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).parent.parent.resolve()
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from flask import Flask, jsonify, render_template_string

from evacuation_simulation.simulation.config import ExitConfiguration, SimulationConfig
from evacuation_simulation.simulation.generation import (
    generate_candidate_exits,
    generate_exit_configurations,
)
from evacuation_simulation.simulation.validator import make_validator

app = Flask(__name__)
validator = make_validator()
SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def build_scenario_data(scenario_name: str) -> dict:
    filepath = SCENARIOS_DIR / (scenario_name + ".json")
    if not filepath.exists():
        return {"error": f"{filepath.name} not found"}

    cfg = SimulationConfig.from_json(filepath)
    fp = cfg.floor_plan
    occ = cfg.occupants

    ext_doors = [d for d in fp.doors if d.exterior]
    int_doors = [d for d in fp.doors if not d.exterior]
    candidates = generate_candidate_exits(fp)

    # Best valid config (max separation)
    best_result = None
    best_cfg_info = None
    if len(candidates) >= 2:
        configs = list(generate_exit_configurations(candidates, exit_count=2))
        best_cfg = max(
            configs,
            key=lambda ec: math.sqrt(
                (ec.exits[0][0] - ec.exits[1][0]) ** 2 +
                (ec.exits[0][1] - ec.exits[1][1]) ** 2
            ),
        )
        result = validator.check(fp, best_cfg, occ)
        best_result = result.model_dump()
        best_cfg_info = {
            "configuration_id": best_cfg.configuration_id,
            "exits": [[int(e[0]), int(e[1])] for e in best_cfg.exits],
            "widths": list(best_cfg.widths),
        }

    # Invalid config: use first interior door if available
    bad_result = None
    bad_cfg_info = None
    if int_doors:
        d = int_doors[0]
        bad_cfg = ExitConfiguration(
            configuration_id="invalid_demo",
            exits=[d.position],
            widths=[d.width],
        )
        br = validator.check(fp, bad_cfg, occ)
        bad_result = br.model_dump()
        bad_cfg_info = {
            "configuration_id": "invalid_demo",
            "exits": [[int(d.position[0]), int(d.position[1])]],
            "widths": [d.width],
            "reason": "interior door selected",
        }
    elif candidates:
        # Fallback: duplicate + narrow width
        c = candidates[0]
        bad_cfg = ExitConfiguration(
            configuration_id="invalid_demo",
            exits=[c.position, c.position],
            widths=[0.3, 0.3],
        )
        br = validator.check(fp, bad_cfg, occ)
        bad_result = br.model_dump()
        bad_cfg_info = {
            "configuration_id": "invalid_demo",
            "exits": [[int(c.position[0]), int(c.position[1])]] * 2,
            "widths": [0.3, 0.3],
            "reason": "duplicate exit + narrow width",
        }

    # Mask demo
    mask_data = []
    if candidates:
        partial = ExitConfiguration(exits=[candidates[0].position], widths=[candidates[0].width])
        mask = validator.mask(fp, occ, partial, candidates)
        for c in candidates:
            pos = (c.position[0], c.position[1])
            mask_data.append({
                "position": list(pos),
                "door_id": c.door_id,
                "width": c.width,
                "allowed": mask.get(pos, True),
                "note": "already selected" if pos == (candidates[0].position[0], candidates[0].position[1]) else "",
            })

    return {
        "scenario_name": cfg.scenario_name,
        "grid": {"rows": fp.grid.rows, "cols": fp.grid.cols},
        "walls": [[int(r), int(c)] for r, c in fp.walls],
        "agents": [{"row": a.row, "col": a.col, "id": a.agent_id} for a in occ.agents],
        "ext_doors": [
            {"position": [int(d.position[0]), int(d.position[1])], "width": d.width, "door_id": d.door_id}
            for d in ext_doors
        ],
        "int_doors": [
            {"position": [int(d.position[0]), int(d.position[1])], "width": d.width, "door_id": d.door_id}
            for d in int_doors
        ],
        "candidates": [
            {"position": [int(c.position[0]), int(c.position[1])], "width": c.width, "door_id": c.door_id}
            for c in candidates
        ],
        "valid_config": best_cfg_info,
        "valid_result": best_result,
        "invalid_config": bad_cfg_info,
        "invalid_result": bad_result,
        "mask": mask_data,
    }


@app.route("/api/scenarios")
def api_scenarios():
    names = [p.stem for p in sorted(SCENARIOS_DIR.glob("*.json"))]
    return jsonify(names)


@app.route("/api/scenario/<name>")
def api_scenario(name):
    return jsonify(build_scenario_data(name))


@app.route("/api/rules")
def api_rules():
    summary = validator.regulatory_summary()
    return jsonify(summary)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stage 11 — Regulatory Constraint Validator | PATH_RL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2234;
    --surface3: #1e2a40;
    --border: #1f2d45;
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.18);
    --green: #10b981;
    --green-glow: rgba(16,185,129,0.15);
    --red: #ef4444;
    --red-glow: rgba(239,68,68,0.15);
    --amber: #f59e0b;
    --amber-glow: rgba(245,158,11,0.15);
    --purple: #8b5cf6;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --text-dim: #94a3b8;
    --mono: 'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Background mesh */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse 80% 50% at 20% 20%, rgba(59,130,246,0.06) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  .container { max-width: 1400px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }

  /* Header */
  header {
    border-bottom: 1px solid var(--border);
    padding: 20px 0;
    background: rgba(10,14,26,0.95);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .header-inner {
    display: flex; align-items: center; justify-content: space-between;
    max-width: 1400px; margin: 0 auto; padding: 0 24px;
  }
  .logo {
    display: flex; align-items: center; gap: 12px;
  }
  .logo-badge {
    background: linear-gradient(135deg, var(--accent), var(--purple));
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; letter-spacing: -0.5px;
    box-shadow: 0 0 20px var(--accent-glow);
  }
  .logo-text { font-size: 16px; font-weight: 600; }
  .logo-sub { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
  .disclaimer-badge {
    background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
    color: var(--amber); padding: 6px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 500; letter-spacing: 0.3px;
  }

  /* Scenario tabs */
  .tabs {
    display: flex; gap: 8px; padding: 28px 0 0;
  }
  .tab {
    padding: 10px 24px; border-radius: 10px 10px 0 0;
    border: 1px solid var(--border); border-bottom: none;
    background: var(--surface); cursor: pointer;
    font-size: 13px; font-weight: 500; color: var(--text-muted);
    transition: all 0.2s; user-select: none;
  }
  .tab:hover { color: var(--text); background: var(--surface2); }
  .tab.active { background: var(--surface2); color: var(--accent); border-color: var(--accent); }

  /* Main grid */
  .main { padding: 0 0 60px; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Top stats strip */
  .stats-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; padding: 20px 0 24px;
  }
  .stat-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
  .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
  .stat-value { font-size: 22px; font-weight: 700; }
  .stat-value.blue { color: var(--accent); }
  .stat-value.green { color: var(--green); }
  .stat-value.red { color: var(--red); }
  .stat-value.amber { color: var(--amber); }

  /* Two-column layout */
  .two-col {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;
  }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }

  .three-col {
    display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 16px; margin-bottom: 16px;
  }
  @media (max-width: 1100px) { .three-col { grid-template-columns: 1fr; } }

  /* Cards */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden;
    transition: box-shadow 0.2s;
  }
  .card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .card-header {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }
  .card-title { font-size: 13px; font-weight: 600; }
  .card-badge {
    margin-left: auto; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge-pass { background: var(--green-glow); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
  .badge-fail { background: var(--red-glow); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
  .badge-info { background: var(--accent-glow); color: var(--accent); border: 1px solid rgba(59,130,246,0.3); }
  .card-body { padding: 16px 18px; }

  /* Grid visualization */
  .grid-wrap { display: flex; justify-content: center; padding: 8px 0; }
  canvas.floor-grid {
    border-radius: 8px;
    image-rendering: pixelated;
  }

  /* Door list */
  .door-list { display: flex; flex-direction: column; gap: 8px; }
  .door-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 10px;
    background: var(--surface2); border: 1px solid var(--border);
    transition: all 0.15s;
  }
  .door-item:hover { border-color: var(--accent); }
  .door-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }
  .dot-ext { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot-int { background: var(--red); opacity: 0.6; }
  .dot-agent { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
  .door-info { flex: 1; }
  .door-pos { font-family: var(--mono); font-size: 12px; font-weight: 500; }
  .door-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
  .door-width {
    font-family: var(--mono); font-size: 12px; font-weight: 600;
    color: var(--accent); background: var(--accent-glow);
    padding: 2px 8px; border-radius: 6px;
  }
  .door-excluded {
    font-size: 10px; color: var(--red); background: var(--red-glow);
    padding: 2px 8px; border-radius: 6px; border: 1px solid rgba(239,68,68,0.2);
  }

  /* Compliance result */
  .compliance-header {
    display: flex; align-items: center; gap: 14px; padding: 16px 18px;
    border-bottom: 1px solid var(--border);
  }
  .compliance-icon { font-size: 28px; }
  .compliance-label { font-size: 18px; font-weight: 700; }
  .compliance-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .result-pass { background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(16,185,129,0.02)); border-color: rgba(16,185,129,0.25); }
  .result-fail { background: linear-gradient(135deg, rgba(239,68,68,0.06), rgba(239,68,68,0.02)); border-color: rgba(239,68,68,0.25); }

  /* Violations */
  .violation-list { display: flex; flex-direction: column; gap: 10px; padding: 14px 18px; }
  .violation-item {
    background: var(--surface2); border: 1px solid var(--border);
    border-left: 3px solid var(--red); border-radius: 0 10px 10px 0;
    padding: 12px 14px;
    animation: slideIn 0.3s ease;
  }
  @keyframes slideIn { from { opacity:0; transform:translateX(-8px); } to { opacity:1; transform:translateX(0); } }
  .viol-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .viol-rule {
    font-family: var(--mono); font-size: 11px; font-weight: 600;
    color: var(--accent); background: var(--accent-glow);
    padding: 2px 8px; border-radius: 5px;
  }
  .viol-sev {
    font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--red);
  }
  .viol-msg { font-size: 12px; color: var(--text-dim); line-height: 1.5; margin-bottom: 8px; }
  .viol-metrics {
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .metric-pill {
    display: flex; align-items: center; gap: 4px;
    background: var(--surface3); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 11px;
  }
  .metric-pill .k { color: var(--text-muted); }
  .metric-pill .v { font-family: var(--mono); font-weight: 600; }
  .metric-pill .v.measured { color: var(--red); }
  .metric-pill .v.required { color: var(--green); }
  .metric-pill .v.clause { color: var(--amber); font-size: 10px; max-width: 200px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Mask table */
  .mask-table { width: 100%; border-collapse: collapse; }
  .mask-table th {
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--text-muted); padding: 8px 12px; text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .mask-table td { padding: 10px 12px; border-bottom: 1px solid rgba(31,45,69,0.5); font-size: 12px; }
  .mask-table tr:last-child td { border-bottom: none; }
  .mask-table tr:hover td { background: var(--surface2); }
  .mask-allowed {
    display: inline-flex; align-items: center; gap: 5px;
    color: var(--green); font-weight: 600; font-size: 11px;
    background: var(--green-glow); padding: 3px 10px; border-radius: 20px;
    border: 1px solid rgba(16,185,129,0.3);
  }
  .mask-blocked {
    display: inline-flex; align-items: center; gap: 5px;
    color: var(--red); font-weight: 600; font-size: 11px;
    background: var(--red-glow); padding: 3px 10px; border-radius: 20px;
    border: 1px solid rgba(239,68,68,0.3);
  }
  .mask-note { color: var(--text-muted); font-style: italic; font-size: 11px; }

  /* Rules panel */
  .rules-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; padding: 24px 0; }
  .rule-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
    transition: all 0.2s;
  }
  .rule-card:hover { transform: translateY(-2px); border-color: var(--accent); box-shadow: 0 8px 24px rgba(59,130,246,0.1); }
  .rule-id {
    font-family: var(--mono); font-size: 12px; font-weight: 600;
    color: var(--accent); margin-bottom: 8px;
  }
  .rule-desc { font-size: 12px; color: var(--text-dim); line-height: 1.5; }
  .rule-verified {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; font-weight: 600; color: var(--green);
    margin-top: 10px;
  }

  /* Info note */
  .info-note {
    background: rgba(59,130,246,0.07); border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px; padding: 12px 16px; margin: 12px 0;
    font-size: 12px; color: var(--text-dim); line-height: 1.6;
  }
  .info-note strong { color: var(--accent); }

  /* Exit path visualization on canvas */
  .selected-exits {
    display: flex; flex-wrap: wrap; gap: 8px; padding: 4px 0;
  }
  .exit-pill {
    font-family: var(--mono); font-size: 12px; font-weight: 500;
    padding: 5px 12px; border-radius: 20px;
    background: var(--accent-glow); color: var(--accent);
    border: 1px solid rgba(59,130,246,0.3);
  }

  /* Loading */
  .loading { text-align: center; padding: 60px; color: var(--text-muted); }
  .spinner {
    width: 32px; height: 32px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .no-violations {
    padding: 20px; text-align: center; color: var(--green);
    font-size: 13px; font-weight: 500;
  }
  .no-violations::before { content: '✓  '; }
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="logo">
      <div class="logo-badge">R11</div>
      <div>
        <div class="logo-text">PATH_RL &mdash; Stage 11</div>
        <div class="logo-sub">Regulatory &amp; Exit-Placement Constraint Layer</div>
      </div>
    </div>
    <div class="disclaimer-badge">&#9888; Research Tool Only &mdash; Not Legal Certification</div>
  </div>
</header>

<div class="container">
  <div class="tabs" id="tabs"></div>
</div>

<div class="container main" id="panels-container"></div>

<!-- Rules Tab -->
<div class="container main" id="rules-panel" style="display:none;">
  <div class="rules-grid" id="rules-grid"></div>
  <div class="info-note">
    <strong>Disclaimer:</strong> All regulatory values are derived from NBC 2016 Part 4 (Fire and Life Safety)
    for research and design-assistance purposes only. This system does <strong>not</strong> constitute legal certification.
    Verified values are explicitly marked. Unverified/configurable values are disclosed transparently.
  </div>
</div>

<script>
const COLORS = {
  wall: '#1e2a40',
  open: '#111827',
  agent: '#f59e0b',
  exit_valid: '#10b981',
  exit_invalid: '#ef4444',
  candidate: '#3b82f6',
  int_door: '#ef444488',
  grid_line: '#1f2d45'
};

let scenarioData = {};
let activeTab = null;

async function init() {
  const [scenariosRes, rulesRes] = await Promise.all([
    fetch('/api/scenarios'),
    fetch('/api/rules')
  ]);
  const scenarios = await scenariosRes.json();
  const rules = await rulesRes.json();

  buildRulesPanel(rules);

  const tabsEl = document.getElementById('tabs');
  const panelsEl = document.getElementById('panels-container');

  // Add scenario tabs
  scenarios.forEach((name, i) => {
    const tab = document.createElement('div');
    tab.className = 'tab' + (i === 0 ? ' active' : '');
    tab.textContent = name.replace(/_/g, ' ');
    tab.dataset.name = name;
    tab.addEventListener('click', () => activateTab(name));
    tabsEl.appendChild(tab);
  });

  // Add Rules tab
  const rulesTab = document.createElement('div');
  rulesTab.className = 'tab';
  rulesTab.textContent = 'NBC 2016 Rules';
  rulesTab.dataset.name = '__rules__';
  rulesTab.addEventListener('click', () => activateTab('__rules__'));
  tabsEl.appendChild(rulesTab);

  // Create panels
  scenarios.forEach((name, i) => {
    const panel = document.createElement('div');
    panel.className = 'tab-panel' + (i === 0 ? ' active' : '');
    panel.id = 'panel-' + name;
    panel.innerHTML = '<div class="loading"><div class="spinner"></div>Loading scenario…</div>';
    panelsEl.appendChild(panel);
  });

  if (scenarios.length > 0) {
    activeTab = scenarios[0];
    await loadScenario(scenarios[0]);
  }
}

function activateTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('rules-panel').style.display = 'none';
  document.getElementById('panels-container').style.display = '';

  document.querySelector(`[data-name="${name}"]`).classList.add('active');
  activeTab = name;

  if (name === '__rules__') {
    document.getElementById('panels-container').style.display = 'none';
    document.getElementById('rules-panel').style.display = 'block';
    return;
  }

  const panel = document.getElementById('panel-' + name);
  if (panel) panel.classList.add('active');

  if (!scenarioData[name]) loadScenario(name);
}

async function loadScenario(name) {
  const res = await fetch('/api/scenario/' + name);
  const data = await res.json();
  scenarioData[name] = data;
  renderScenario(name, data);
}

function renderScenario(name, data) {
  const panel = document.getElementById('panel-' + name);
  if (!panel) return;

  const extCount = data.ext_doors.length;
  const intCount = data.int_doors.length;
  const agentCount = data.agents.length;
  const validPass = data.valid_result?.is_compliant;
  const invalidViolCount = data.invalid_result?.violations?.length || 0;

  panel.innerHTML = `
    <div class="stats-strip">
      <div class="stat-card">
        <div class="stat-label">Grid Size</div>
        <div class="stat-value blue">${data.grid.rows}&times;${data.grid.cols}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Exterior Exits</div>
        <div class="stat-value green">${extCount}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Interior Doors</div>
        <div class="stat-value amber">${intCount}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Candidates</div>
        <div class="stat-value blue">${data.candidates.length}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Occupants</div>
        <div class="stat-value">${agentCount}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Valid Config</div>
        <div class="stat-value ${validPass ? 'green' : 'red'}">${validPass ? 'PASS' : 'FAIL'}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Invalid Violations</div>
        <div class="stat-value red">${invalidViolCount}</div>
      </div>
    </div>

    <div class="three-col">
      <!-- Floor Plan -->
      <div class="card" style="grid-row: span 2;">
        <div class="card-header">
          <span>&#9632;</span>
          <span class="card-title">Floor Plan</span>
          <span class="card-badge badge-info">${data.grid.rows}&times;${data.grid.cols}</span>
        </div>
        <div class="card-body">
          <div class="grid-wrap"><canvas id="canvas-${name}" class="floor-grid"></canvas></div>
          <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; font-size:11px; color:var(--text-muted);">
            <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:10px;height:10px;background:#10b981;border-radius:2px;"></span> Exterior exit</span>
            <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;opacity:0.7;"></span> Interior door</span>
            <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border-radius:2px;"></span> Agent</span>
            <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:10px;height:10px;background:#1e2a40;border-radius:2px;"></span> Wall</span>
          </div>
        </div>
      </div>

      <!-- Exterior Doors -->
      <div class="card">
        <div class="card-header">
          <span>&#x1F6AA;</span>
          <span class="card-title">Exterior DoorOpenings</span>
          <span class="card-badge badge-pass">${extCount} eligible</span>
        </div>
        <div class="card-body">
          <div class="door-list">
            ${data.ext_doors.map(d => `
              <div class="door-item">
                <div class="door-dot dot-ext"></div>
                <div class="door-info">
                  <div class="door-pos">(${d.position[0]}, ${d.position[1]})</div>
                  <div class="door-meta">${d.door_id || 'unnamed'}</div>
                </div>
                <span class="door-width">${d.width.toFixed(1)}m</span>
              </div>
            `).join('')}
          </div>
        </div>
      </div>

      <!-- Interior Doors -->
      <div class="card">
        <div class="card-header">
          <span>&#128274;</span>
          <span class="card-title">Interior DoorOpenings</span>
          <span class="card-badge badge-fail">${intCount} ineligible</span>
        </div>
        <div class="card-body">
          ${intCount === 0
            ? '<div style="color:var(--text-muted);font-size:12px;padding:8px 0;">No interior doors defined.</div>'
            : `<div class="door-list">
              ${data.int_doors.map(d => `
                <div class="door-item" style="border-color:rgba(239,68,68,0.2);">
                  <div class="door-dot dot-int"></div>
                  <div class="door-info">
                    <div class="door-pos">(${d.position[0]}, ${d.position[1]})</div>
                    <div class="door-meta">${d.door_id || 'unnamed'}</div>
                  </div>
                  <span class="door-excluded">EXCLUDED</span>
                </div>
              `).join('')}
            </div>`
          }
          <div class="info-note" style="margin-top:10px;">
            <strong>R1:</strong> Interior doors cannot become emergency exits.
          </div>
        </div>
      </div>
    </div>

    <div class="two-col">
      <!-- Valid Config -->
      <div class="card ${validPass ? 'result-pass' : 'result-fail'}">
        <div class="compliance-header">
          <div class="compliance-icon">${validPass ? '✅' : '❌'}</div>
          <div>
            <div class="compliance-label" style="color:${validPass ? 'var(--green)' : 'var(--red)'}">
              ${validPass ? 'COMPLIANT' : 'NON-COMPLIANT'}
            </div>
            <div class="compliance-sub">Best separated 2-exit configuration</div>
          </div>
          <span class="card-badge ${validPass ? 'badge-pass' : 'badge-fail'}" style="margin-left:auto;">
            ${validPass ? 'PASS' : 'FAIL'}
          </span>
        </div>
        <div class="card-body">
          ${data.valid_config ? `
            <div style="margin-bottom:12px;">
              <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">Selected exits:</div>
              <div class="selected-exits">
                ${data.valid_config.exits.map((e,i) =>
                  `<span class="exit-pill">(${e[0]},${e[1]}) &mdash; ${data.valid_config.widths[i].toFixed(1)}m</span>`
                ).join('')}
              </div>
            </div>
          ` : ''}
          ${renderViolations(data.valid_result?.violations || [], true)}
        </div>
      </div>

      <!-- Invalid Config -->
      <div class="card result-fail">
        <div class="compliance-header">
          <div class="compliance-icon">❌</div>
          <div>
            <div class="compliance-label" style="color:var(--red)">NON-COMPLIANT</div>
            <div class="compliance-sub">${data.invalid_config?.reason || 'Invalid configuration'}</div>
          </div>
          <span class="card-badge badge-fail" style="margin-left:auto;">FAIL</span>
        </div>
        <div class="card-body">
          ${data.invalid_config ? `
            <div style="margin-bottom:12px;">
              <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">Selected exits:</div>
              <div class="selected-exits">
                ${data.invalid_config.exits.map((e,i) =>
                  `<span class="exit-pill" style="background:var(--red-glow);color:var(--red);border-color:rgba(239,68,68,0.3);">(${e[0]},${e[1]}) &mdash; ${data.invalid_config.widths[i].toFixed(1)}m</span>`
                ).join('')}
              </div>
            </div>
          ` : ''}
          ${renderViolations(data.invalid_result?.violations || [], false)}
        </div>
      </div>
    </div>

    <!-- Action Mask -->
    <div class="card" style="margin-bottom:16px;">
      <div class="card-header">
        <span>&#127899;</span>
        <span class="card-title">Action Mask &mdash; Partial Selection</span>
        <span class="card-badge badge-info">R1 only</span>
      </div>
      <div class="card-body">
        <div class="info-note">
          <strong>Partial mask</strong> blocks already-selected exits and is evaluated per-candidate (R1 only).
          R2/R3/R4/R5 require a <strong>complete configuration</strong> and are only evaluated by <code>check()</code>.
          A mask=ALLOWED value does <strong>not</strong> guarantee the final configuration will be compliant.
        </div>
        <table class="mask-table">
          <thead>
            <tr>
              <th>Position</th><th>Door ID</th><th>Width</th><th>Status</th><th>Note</th>
            </tr>
          </thead>
          <tbody>
            ${data.mask.map(m => `
              <tr>
                <td><span style="font-family:var(--mono)">(${m.position[0]},${m.position[1]})</span></td>
                <td style="color:var(--text-muted)">${m.door_id || '—'}</td>
                <td><span style="font-family:var(--mono);color:var(--accent)">${m.width.toFixed(1)}m</span></td>
                <td>
                  ${m.allowed
                    ? '<span class="mask-allowed">&#10003; ALLOWED</span>'
                    : '<span class="mask-blocked">&#10007; BLOCKED</span>'
                  }
                </td>
                <td><span class="mask-note">${m.note}</span></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;

  // Draw canvas
  requestAnimationFrame(() => drawFloorPlan(name, data));
}

function renderViolations(violations, isValid) {
  if (violations.length === 0) {
    return `<div class="no-violations">All encoded constraints satisfied</div>`;
  }
  return `<div class="violation-list">
    ${violations.map(v => `
      <div class="violation-item">
        <div class="viol-header">
          <span class="viol-rule">${v.rule_id}</span>
          <span class="viol-sev">${v.severity.toUpperCase()}</span>
        </div>
        <div class="viol-msg">${v.message}</div>
        <div class="viol-metrics">
          ${v.measured_value !== null && v.measured_value !== undefined ? `
            <div class="metric-pill"><span class="k">measured</span><span class="v measured">${v.measured_value} ${v.unit}</span></div>
            <div class="metric-pill"><span class="k">required</span><span class="v required">${v.required_value} ${v.unit}</span></div>
          ` : ''}
          ${v.clause ? `<div class="metric-pill"><span class="k">clause</span><span class="v clause" title="${v.clause}">${v.clause}</span></div>` : ''}
        </div>
      </div>
    `).join('')}
  </div>`;
}

function drawFloorPlan(name, data) {
  const canvas = document.getElementById('canvas-' + name);
  if (!canvas) return;
  const rows = data.grid.rows;
  const cols = data.grid.cols;
  const MAX_SIZE = 380;
  const cellSize = Math.min(Math.floor(MAX_SIZE / Math.max(rows, cols)), 32);
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = COLORS.open;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Grid lines
  ctx.strokeStyle = COLORS.grid_line;
  ctx.lineWidth = 0.5;
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath(); ctx.moveTo(0, r * cellSize); ctx.lineTo(cols * cellSize, r * cellSize); ctx.stroke();
  }
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath(); ctx.moveTo(c * cellSize, 0); ctx.lineTo(c * cellSize, rows * cellSize); ctx.stroke();
  }

  // Walls
  ctx.fillStyle = COLORS.wall;
  data.walls.forEach(([r, c]) => {
    ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
  });

  // Interior doors (semi-transparent red)
  ctx.fillStyle = 'rgba(239,68,68,0.35)';
  data.int_doors.forEach(d => {
    const [r, c] = d.position;
    ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
    ctx.strokeStyle = 'rgba(239,68,68,0.7)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(c * cellSize + 1.75, r * cellSize + 1.75, cellSize - 3.5, cellSize - 3.5);
  });

  // Exterior doors / candidates
  data.ext_doors.forEach(d => {
    const [r, c] = d.position;
    ctx.fillStyle = 'rgba(16,185,129,0.25)';
    ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.strokeRect(c * cellSize + 1.5, r * cellSize + 1.5, cellSize - 3, cellSize - 3);
  });

  // Highlight valid selected exits
  if (data.valid_config) {
    data.valid_config.exits.forEach(([r, c]) => {
      ctx.fillStyle = 'rgba(16,185,129,0.55)';
      ctx.fillRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);

      // White dot
      ctx.fillStyle = '#fff';
      const cx = c * cellSize + cellSize / 2;
      const cy = r * cellSize + cellSize / 2;
      ctx.beginPath(); ctx.arc(cx, cy, cellSize * 0.2, 0, Math.PI * 2); ctx.fill();
    });
  }

  // Agents
  data.agents.forEach(a => {
    const cx = a.col * cellSize + cellSize / 2;
    const cy = a.row * cellSize + cellSize / 2;
    ctx.fillStyle = COLORS.agent;
    ctx.beginPath(); ctx.arc(cx, cy, cellSize * 0.28, 0, Math.PI * 2); ctx.fill();
  });
}

function buildRulesPanel(rules) {
  const grid = document.getElementById('rules-grid');
  const ruleDetails = [
    { id: 'R1_VALID_EXIT_POSITION', clause: 'PATH_RL geometric domain rule', verified: true },
    { id: 'R2_MINIMUM_EXIT_COUNT', clause: 'NBC 2016 Part 4 §4.4.2.4.3', verified: false },
    { id: 'R3_EXIT_SEPARATION', clause: 'NBC 2016 Part 4 egress placement', verified: false },
    { id: 'R4_EXIT_WIDTH', clause: 'NBC 2016 Part 4 Table 4', verified: false },
    { id: 'R5_TRAVEL_DISTANCE', clause: 'NBC 2016 Part 4 Table 5', verified: true },
  ];
  const detailMap = {};
  ruleDetails.forEach(r => detailMap[r.id] = r);

  grid.innerHTML = rules.rules.map(r => {
    const detail = detailMap[r.rule_id] || {};
    return `
      <div class="rule-card">
        <div class="rule-id">${r.rule_id}</div>
        <div class="rule-desc">${r.description}</div>
        ${detail.clause ? `<div style="margin-top:8px;font-size:11px;color:var(--text-muted);">${detail.clause}</div>` : ''}
        ${detail.verified ? '<div class="rule-verified">✓ VERIFIED NBC CONSTRAINT</div>' : '<div class="rule-verified" style="color:var(--amber);">⚠ CONFIGURABLE / UNVERIFIED PARAMETER</div>'}
      </div>
    `;
  }).join('');
}

init();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("=" * 60)
    print("  PATH_RL Stage 11 — Web Validator")
    print("  http://localhost:5111")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5111, debug=False)

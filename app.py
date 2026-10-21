"""Evidence-driven Streamlit dashboard for saved CC4 experiments.

Decoupled Multi-Agent Reinforcement Learning (IPPO / MAPPO) and LLM Incident Response evaluation.
All data is loaded dynamically from saved artifacts under results/.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_demo import LiveRun

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EPISODES = RESULTS / "episodes"
TABLES = RESULTS / "tables"
TRAINING = RESULTS / "training"
ADVISORY = RESULTS / "advisory"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CC4 LLM-MARL Incident Response",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Clean, Modern Styling (Sleek High-Contrast Dashboard)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --bg-main: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #475569;
    --border-color: #e2e8f0;
    --border-dark: #cbd5e1;
    --blue-accent: #1d4ed8;
    --red-accent: #dc2626;
    --green-accent: #16a34a;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

header[data-testid="stHeader"] {
    background-color: var(--bg-main);
}

[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid var(--border-color);
}

/* Metric card containers */
[data-testid="stMetric"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    font-family: "JetBrains Mono", "Space Mono", monospace !important;
}

/* Action Log & Step Panel with FIXED HEIGHT to prevent bouncing */
.step-panel {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    height: 380px;
    max-height: 380px;
    overflow-y: auto;
    box-sizing: border-box;
}

/* Fixed exactly for 2-line actions to prevent table size fluctuation */
.step-row {
    display: grid;
    grid-template-columns: 105px 1fr;
    gap: 8px;
    align-items: center;
    min-height: 42px;
    max-height: 42px;
    line-height: 1.3;
    padding: 3px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.75rem;
    font-family: "JetBrains Mono", monospace;
    overflow: hidden;
    box-sizing: border-box;
}
.step-row:last-child {
    border-bottom: none;
}
.step-agent {
    color: var(--blue-accent);
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.step-action {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    word-break: break-all;
    line-height: 1.3;
    color: var(--text-main);
    background: #f8fafc;
    border-radius: 4px;
    padding: 3px 6px;
    border: 1px solid #e2e8f0;
    font-size: 0.73rem;
}

/* Telemetry & Advisory Panels */
.advisory-box {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--blue-accent);
    border-radius: 8px;
    padding: 16px 18px;
    margin-top: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.advisory-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
}
.advisory-metric-cell {
    background: #f8fafc;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 10px;
}
.advisory-metric-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}
.advisory-metric-val {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text-main);
    font-family: monospace;
    margin-top: 2px;
}

/* Buttons */
div.stButton > button {
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
}

/* Hide default streamlit clutter */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Loaders (Cached, Safe, Dynamic)
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def episode_catalog() -> pd.DataFrame:
    rows = []
    if EPISODES.exists():
        for path in sorted(EPISODES.glob("*.json")):
            try:
                data = _read_json(path)
                prov = data.get("provenance", {})
                rows.append(
                    {
                        "label": path.name,
                        "path": str(path),
                        "agent": prov.get("agent", "unknown"),
                        "red": prov.get("red", "unknown"),
                        "seed": prov.get("seed", 0),
                        "steps": prov.get("steps", len(data.get("steps", []))),
                        "git_commit": prov.get("git_commit", "unknown"),
                        "source": prov.get("source", "unknown"),
                        "host_count": prov.get("host_count"),
                        "total_reward": data.get("total_reward", 0.0),
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


@st.cache_data
def csv_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def training_catalog() -> pd.DataFrame:
    frames = []
    if TRAINING.exists():
        for path in sorted(TRAINING.glob("*_train.csv")):
            try:
                frame = pd.read_csv(path)
                frame["run"] = path.stem.removesuffix("_train")
                frames.append(frame)
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def status_counts(host_state: dict) -> dict:
    counts: dict[str, int] = {}
    for state in host_state.values():
        counts[state] = counts.get(state, 0) + 1
    return counts


def zone_name(host: str) -> str:
    return host.split("_subnet", 1)[0].replace("_", " ").title()


def grouped_hosts(host_state: dict) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for host, state in sorted(host_state.items()):
        grouped.setdefault(zone_name(host), []).append((host, state))
    return grouped


# ---------------------------------------------------------------------------
# Filterable Replay Selector (MAPPO / IPPO / Sleep / Seed)
# ---------------------------------------------------------------------------
def replay_selector(
    catalog: pd.DataFrame, key: str
) -> tuple[pd.DataFrame, dict | None]:
    if catalog.empty:
        return catalog, None

    col_algo, col_seed, col_file = st.columns([1.1, 1.0, 2.2])

    with col_algo:
        agent_options = ["ALL"] + sorted(
            catalog["agent"].dropna().unique().tolist()
        )
        selected_agent = st.selectbox(
            "Filter Algorithm",
            agent_options,
            index=0,
            key=f"{key}_agent_filter",
        )

    filtered = catalog.copy()
    if selected_agent != "ALL":
        filtered = filtered[filtered["agent"] == selected_agent]

    with col_seed:
        seed_options = ["ALL"] + [
            str(s) for s in sorted(filtered["seed"].dropna().unique().tolist())
        ]
        selected_seed = st.selectbox(
            "Filter Seed",
            seed_options,
            index=0,
            key=f"{key}_seed_filter",
        )

    if selected_seed != "ALL":
        filtered = filtered[filtered["seed"] == int(selected_seed)]

    if filtered.empty:
        st.warning("No saved episodes match this filter combination.")
        return pd.DataFrame(), None

    with col_file:
        selected_label = st.selectbox(
            f"Select Episode ({len(filtered)} matches)",
            filtered["label"].tolist(),
            key=f"{key}_dropdown",
        )

    row = filtered.loc[filtered["label"] == selected_label].iloc[0]
    return row.to_frame().T, _read_json(Path(row["path"]))


# ---------------------------------------------------------------------------
# 2D Spatial Network Topology Visualizer (Interactive Plotly Graph)
# ---------------------------------------------------------------------------
def build_zone_graph(host_state: dict) -> go.Figure:
    grouped = grouped_hosts(host_state)
    zones = sorted(grouped.keys())
    n = len(zones)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Randomized CC4 Enterprise Network Topology",
            height=380,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        return fig

    # Place zones evenly in a circle around the core gateway
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    r = 2.4
    zx = [r * math.cos(a) for a in angles]
    zy = [r * math.sin(a) for a in angles]

    fig = go.Figure()

    # Backbone links (Core Gateway to each Zone)
    for x, y in zip(zx, zy):
        fig.add_trace(
            go.Scatter(
                x=[0, x],
                y=[0, y],
                mode="lines",
                line=dict(color="#cbd5e1", width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Perimeter links between adjacent zones
    for i in range(n):
        j = (i + 1) % n
        fig.add_trace(
            go.Scatter(
                x=[zx[i], zx[j]],
                y=[zy[i], zy[j]],
                mode="lines",
                line=dict(color="#e2e8f0", width=1.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Central Core Gateway Node
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            marker=dict(
                size=22,
                color="#0f172a",
                line=dict(color="#334155", width=2),
                symbol="circle",
            ),
            text=["CORE GATEWAY"],
            textposition="bottom center",
            textfont=dict(size=9, color="#0f172a", family="monospace"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Zone Nodes
    red_counts = [
        sum(state == "red_session" for _, state in grouped[zone])
        for zone in zones
    ]
    host_counts = [len(grouped[zone]) for zone in zones]
    node_colors = [
        "#dc2626" if r_count > 0 else "#16a34a" for r_count in red_counts
    ]
    node_sizes = [24 + 1.8 * min(h_count, 18) for h_count in host_counts]
    clean_labels = [
        z.replace(" Zone", "").replace(" Subnet", "") for z in zones
    ]

    fig.add_trace(
        go.Scatter(
            x=zx,
            y=zy,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color="#ffffff", width=2),
                opacity=0.95,
            ),
            text=clean_labels,
            textposition="top center",
            textfont=dict(
                size=9.5, color="#0f172a", family="monospace"
            ),
            customdata=list(
                zip(
                    zones,
                    red_counts,
                    host_counts,
                    [
                        "ALERT (Red Session Detected)"
                        if r > 0
                        else "SECURE (Normal)"
                        for r in red_counts
                    ],
                )
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>• Hosts: %{customdata[2]}<br>• Red Sessions: %{customdata[1]}<br>• Status: %{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="Randomized CC4 Enterprise Network Topology (by Zone)",
            font=dict(size=12, color="#334155"),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3.6, 3.6],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-3.6, 3.6],
            scaleanchor="x",
        ),
        height=380,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


# ---------------------------------------------------------------------------
# Replay Cumulative Reward Chart
# ---------------------------------------------------------------------------
def replay_chart(steps: list[dict], upto: int | None = None) -> go.Figure:
    visible = steps if upto is None else steps[: upto + 1]
    xs = [s.get("step", i) for i, s in enumerate(visible)]
    ys = [s.get("cumulative_reward", 0.0) for s in visible]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#1d4ed8", width=2.5),
            name="Cumulative Reward",
        )
    )
    # Highlight current position
    if visible and upto is not None and upto < len(steps):
        fig.add_trace(
            go.Scatter(
                x=[xs[-1]],
                y=[ys[-1]],
                mode="markers",
                marker=dict(size=9, color="#dc2626", symbol="circle"),
                name="Current Step",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text="Cumulative Team Reward Progression",
            font=dict(size=12, color="#334155"),
        ),
        xaxis_title="CC4 Step",
        yaxis_title="Reward",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="sans-serif", size=11, color="#334155"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    return fig


# ---------------------------------------------------------------------------
# Confidence Interval Forest Plot & Paired Scoreboard
# ---------------------------------------------------------------------------
def paired_scoreboard() -> pd.DataFrame:
    path = TABLES / "heldout_episode_returns.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    required = {"agent", "eval_seed", "return"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()
    baseline = raw[raw["agent"] == "Sleep"][["eval_seed", "return"]].rename(
        columns={"return": "sleep_return"}
    )
    rows = []
    for agent in sorted(set(raw["agent"]) - {"Sleep"}):
        current = raw[raw["agent"] == agent][["eval_seed", "return"]].rename(
            columns={"return": "agent_return"}
        )
        paired = current.merge(baseline, on="eval_seed", how="inner")
        if paired.empty:
            continue
        diff = paired["agent_return"] - paired["sleep_return"]
        mean = float(diff.mean())
        std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        n = len(diff)
        margin = 1.96 * std / (n**0.5) if n > 1 else 0.0
        rows.append(
            {
                "agent": agent,
                "episodes": n,
                "mean_difference_vs_sleep": mean,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "interpretation": (
                    "Significant Improvement"
                    if mean - margin > 0
                    else ("Worse than Sleep" if mean + margin < 0 else "Not Significant")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_ci_forest_plot(scoreboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not scoreboard.empty:
        for _, r in scoreboard.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[r["ci95_low"], r["mean_difference_vs_sleep"], r["ci95_high"]],
                    y=[r["agent"]] * 3,
                    mode="lines+markers",
                    marker=dict(
                        size=[8, 12, 8],
                        color="#1d4ed8",
                        symbol=["line-ns-open", "circle", "line-ns-open"],
                    ),
                    line=dict(color="#1d4ed8", width=2.5),
                    showlegend=False,
                    hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
                )
            )
        fig.add_vline(
            x=0,
            line=dict(color="#64748b", dash="dash", width=1.5),
            annotation_text="BASELINE (SLEEP)",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color="#64748b"),
        )

    fig.update_layout(
        title=dict(
            text="95% Confidence Interval vs Sleep Baseline",
            font=dict(size=12, color="#334155"),
        ),
        xaxis_title="Mean Return Difference",
        height=220,
        margin=dict(l=20, r=20, t=35, b=25),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="sans-serif", size=11, color="#334155"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1")
    return fig


def baseline_summary() -> pd.DataFrame:
    table = csv_table("baselines.csv")
    if table.empty or "agent" not in table.columns:
        return pd.DataFrame()
    summary = (
        table[table["row_type"] == "summary"].copy()
        if "row_type" in table.columns
        else table.copy()
    )
    if summary.empty:
        summary = table.groupby("agent", as_index=False).agg(
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
        )
    return summary


def render_zone_detail(host_state: dict, key: str) -> None:
    grouped = grouped_hosts(host_state)
    if not grouped:
        return
    selected_zone = st.selectbox("Inspect Hosts in Zone", list(grouped), key=key)
    rows = [
        {"Host ID": host, "Security State": state}
        for host, state in grouped[selected_zone]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def load_advisory_step(step: int, preferred_seed: int = 9000000) -> dict | None:
    path = ADVISORY / f"seed{preferred_seed}_step{step:03d}.json"
    if path.exists():
        return _read_json(path)
    matching = list(ADVISORY.glob(f"seed*_step{step:03d}.json"))
    if matching:
        return _read_json(matching[0])
    return None


def advisory_steps() -> list[int]:
    values = []
    for path in sorted(ADVISORY.glob("seed*_step*.json")):
        try:
            values.append(int(path.stem.rsplit("step", 1)[1]))
        except ValueError:
            continue
    return sorted(set(values))


# ---------------------------------------------------------------------------
# Step Status Panel with FIXED 2-LINE SLOTS (Eliminates table height jumping)
# ---------------------------------------------------------------------------
def render_step_status(step: dict) -> None:
    blue_actions = [
        {"agent": agent, "action": str(action)}
        for agent, action in sorted(step.get("blue_actions", {}).items())
    ]
    red_activity = [
        {"agent": agent, "activity": str(activity)}
        for agent, activity in sorted(step.get("red_activity", {}).items())
    ]

    blue_html = "".join(
        f'<div class="step-row"><div class="step-agent">{row["agent"]}</div>'
        f'<div class="step-action" title="{row["action"]}">{row["action"]}</div></div>'
        for row in blue_actions
    ) or '<div class="step-row"><div class="step-agent">blue_agents</div><div class="step-action">[Sleep]</div></div>'

    red_html = "".join(
        f'<div class="step-row"><div class="step-agent" style="color:#dc2626;">{row["agent"]}</div>'
        f'<div class="step-action" title="{row["activity"]}">{row["activity"]}</div></div>'
        for row in red_activity
    ) or '<div class="step-row"><div class="step-agent" style="color:#dc2626;">red_agent</div><div class="step-action">[Sleep]</div></div>'

    html_content = f"""<div class="step-panel">
<div style="font-size:0.75rem;font-weight:700;color:#1d4ed8;text-transform:uppercase;margin-bottom:6px;">Blue Tactical Actions (5 Agents)</div>
{blue_html}
<div style="font-size:0.75rem;font-weight:700;color:#dc2626;text-transform:uppercase;margin:12px 0 6px 0;">Red Adversary Activity</div>
{red_html}
</div>"""
    st.markdown(html_content, unsafe_allow_html=True)


def render_step_advisory(step_index: int) -> None:
    available = advisory_steps()
    advisory = load_advisory_step(step_index)
    if advisory is None:
        st.info(
            f"No saved LLM call exists for CC4 step {step_index}. "
            f"Saved advisory snapshots exist at steps: {', '.join(map(str, available)) or 'none'}."
        )
        return

    result = advisory.get("result", {})
    advice = result.get("advice", {})
    report = advice.get("incident_report", {})
    priorities = advice.get("priorities", [])
    priority_text = "; ".join(
        f'{item.get("zone", "unknown")} (priority: {item.get("priority_score", "—")})'
        for item in priorities
    ) or "No priorities assigned."

    sec_summary = report.get("executive_summary", "No executive summary.")
    kill_stage = report.get("kill_chain_stage", "Unknown")

    html_panel = f"""<div class="advisory-box">
<div class="advisory-metric-grid">
    <div class="advisory-metric-cell"><div class="advisory-metric-lbl">Step</div><div class="advisory-metric-val">{result.get("step", step_index)}</div></div>
    <div class="advisory-metric-cell"><div class="advisory-metric-lbl">Source</div><div class="advisory-metric-val">{result.get("source", "—")}</div></div>
    <div class="advisory-metric-cell"><div class="advisory-metric-lbl">Latency</div><div class="advisory-metric-val">{float(result.get("latency_ms", 0)):.0f} ms</div></div>
    <div class="advisory-metric-cell"><div class="advisory-metric-lbl">Cost</div><div class="advisory-metric-val">${float(result.get("cost_usd", 0)):.6f}</div></div>
</div>
<div style="font-size:0.78rem;margin-bottom:8px;"><b>Prioritized Zones:</b> {priority_text}</div>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px;font-size:0.8rem;line-height:1.5;">
    <div><b>Executive Incident Summary:</b> {sec_summary}</div>
    <div style="margin-top:4px;"><b>Kill-Chain Stage:</b> <span style="background:#fee2e2;color:#b91c1c;padding:2px 8px;border-radius:4px;font-weight:700;font-size:0.72rem;">{kill_stage}</span></div>
</div>
</div>"""
    st.markdown(html_panel, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initial Catalog and Global Evidence Counters
# ---------------------------------------------------------------------------
catalog = episode_catalog()
training_df = training_catalog()
advisory_calls_df = csv_table("advisory_calls.csv")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION (Collapsible Left Menu as Requested)
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """<div style="font-size:0.75rem;font-weight:800;letter-spacing:1.5px;color:#475569;text-transform:uppercase;margin-bottom:10px;">
        CC4 INCIDENT RESPONSE
        </div>""",
        unsafe_allow_html=True,
    )
    if "pending_nav_choice" in st.session_state:
        st.session_state["nav_choice"] = st.session_state.pop("pending_nav_choice")
    nav_choice = st.radio(
        "Module Selection",
        [
            "🟢 Live Demo",
            "🛡️ Episode Replay",
            "⚖️ Compare Replays",
            "📈 Training Evidence",
            "🤖 LLM Advisory",
            "📐 Architecture",
        ],
        index=0,
        key="nav_choice",
        label_visibility="collapsed",
    )

    st.markdown("---")

    with st.expander("📊 Evidence Inventory", expanded=True):
        st.metric("Saved CC4 Episodes", len(catalog))
        st.metric(
            "Training Runs",
            len(training_df["run"].unique()) if not training_df.empty else 0,
        )
        st.metric("Advisory Calls", len(advisory_calls_df))
        if not catalog.empty:
            st.success("Real rollout data ready")
        else:
            st.warning("No replay files found")

    with st.expander("ℹ️ Domain Specifications", expanded=False):
        st.caption(
            "• CC4 Enterprise Network: ~100 hosts across 9 zones\n"
            "• Blue Team: 5 autonomous decentralized agents\n"
            "• Red Adversary: FiniteStateRedAgent kill-chain\n"
            "• LLM Advisory: Decoupled asynchronous Gemini"
        )

# ═══════════════════════════════════════════════════════════════════════════
# TOP DASHBOARD HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:18px 22px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:1.5px;color:#64748b;text-transform:uppercase;">CAGE CHALLENGE 4 // AUTONOMOUS DEFENSE EVALUATION</div>
        <div style="font-size:1.55rem;font-weight:900;color:#0f172a;line-height:1.15;margin:4px 0 8px 0;letter-spacing:-0.02em;">CC4 LLM-MARL Incident Response</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <span style="background:#0f172a;color:#ffffff;font-size:0.65rem;font-weight:700;padding:3px 9px;border-radius:4px;font-family:monospace;">ENV: CC4 REAL</span>
            <span style="background:#0f172a;color:#ffffff;font-size:0.65rem;font-weight:700;padding:3px 9px;border-radius:4px;font-family:monospace;">BLUE: 5 AGENTS</span>
            <span style="background:#0f172a;color:#ffffff;font-size:0.65rem;font-weight:700;padding:3px 9px;border-radius:4px;font-family:monospace;">LLM: ADVISORY ONLY</span>
        </div>
    </div>
    <div style="max-width:380px;font-size:0.75rem;line-height:1.45;color:#475569;border-left:2px solid #cbd5e1;padding-left:14px;">
        CC4 utilizes a randomized enterprise network of ~100 hosts across 9 zones.
        Host counts and adversary behavior vary across seeds. The advisory LLM operates outside the RL execution loop.
    </div>
</div>
</div>""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# VIEW 1: LIVE DEMO (In-process CC4 execution)
# ═══════════════════════════════════════════════════════════════════════════
if "Live Demo" in nav_choice:
   st.markdown("#### Live CC4 Demonstration")
   st.info(
       "Controlled simulated attack — the selected CC4 Red strategy runs in-process. "
       "LLM output is advisory-only and does not intervene in Blue actions."
   )

   with st.expander("Live scenario controls", expanded=True):
       c_seed, c_red, c_policy, c_steps, c_speed = st.columns([1, 1.3, 1.1, 1, 1])
       with c_seed:
           live_seed = st.number_input("Scenario seed", min_value=0, value=42, step=1)
       with c_red:
           live_red = st.selectbox(
               "Red strategy",
               ["FiniteStateRedAgent"],
               help="Only verified native CC4 Red strategies are offered here.",
           )
       with c_policy:
           live_policy = st.selectbox("Blue policy", ["Sleep", "IPPO", "MAPPO"])
       with c_steps:
           live_steps = st.number_input("Episode length", min_value=1, value=30, step=1)
       with c_speed:
           live_speed = st.selectbox("Playback speed", ["0.5x", "1x", "2x", "4x"], index=1)

       if live_policy == "Sleep":
           live_checkpoint = None
       else:
           checkpoint_suffix = "zip" if live_policy == "IPPO" else "pt"
           checkpoint_options = sorted(
               RESULTS.joinpath("checkpoints").glob(
                   f"{live_policy.lower()}_seed*.{checkpoint_suffix}"
               )
           )
           if checkpoint_options:
               checkpoint_labels = [str(path.relative_to(ROOT)) for path in checkpoint_options]
               selected_checkpoint_label = st.selectbox(
                   "Trained policy checkpoint",
                   checkpoint_labels,
                   help=(
                       "This selects policy weights only. The CC4 environment is "
                       "created fresh using the Scenario seed above; it is not a replay."
                   ),
               )
               live_checkpoint = str(ROOT / selected_checkpoint_label)
           else:
               live_checkpoint = None
               st.error(
                   f"No {live_policy} checkpoints found under "
                   f"`{RESULTS / 'checkpoints'}`. Train or add a checkpoint before "
                   "running this trained policy."
               )

       c_start, c_step, c_pause, c_resume, c_stop, c_restart = st.columns(6)
       start_requested = c_start.button("Start", use_container_width=True)
       step_requested = c_step.button("Step forward", use_container_width=True)
       pause_requested = c_pause.button("Pause", use_container_width=True)
       resume_requested = c_resume.button("Resume", use_container_width=True)
       stop_requested = c_stop.button("Stop", use_container_width=True)
       restart_requested = c_restart.button("Restart", use_container_width=True)
   live_run = st.session_state.get("live_run")
   requested_config = (
       int(live_seed),
       int(live_steps),
       live_red,
       live_policy,
       live_checkpoint or "",
   )
   current_config = (
       (
           live_run.seed,
           live_run.steps,
           live_run.red_strategy,
           live_run.blue_policy,
           live_run.checkpoint or "",
       )
       if live_run is not None
       else None
   )

   if start_requested or restart_requested or (
       live_run is not None and current_config != requested_config
   ):
       try:
           live_run = LiveRun.create(
               seed=int(live_seed),
               steps=int(live_steps),
               red_strategy=live_red,
               blue_policy=live_policy,
               checkpoint=live_checkpoint or None,
           )
           st.session_state["live_run"] = live_run
           st.session_state["live_running"] = bool(start_requested)
           st.session_state["live_error"] = None
       except Exception as exc:
           st.session_state["live_run"] = None
           st.session_state["live_running"] = False
           st.session_state["live_error"] = str(exc)
           live_run = None

   if pause_requested:
       st.session_state["live_running"] = False
   if resume_requested and live_run is not None and not live_run.episode_done:
       st.session_state["live_running"] = True
   if stop_requested:
       st.session_state["live_running"] = False
       if live_run is not None:
           live_run.error_state = "Stopped by presenter"
   if step_requested and live_run is not None:
       try:
           live_run.step_once()
           st.session_state["live_running"] = False
           st.session_state["live_error"] = None
       except Exception as exc:
           live_run.error_state = str(exc)
           st.session_state["live_running"] = False

   live_error = st.session_state.get("live_error")
   if live_error:
       st.error(f"Live execution unavailable: {live_error}")
       if st.button("Switch to saved replay", use_container_width=False):
           st.session_state["pending_nav_choice"] = "🛡️ Episode Replay"
           st.rerun()

   live_run = st.session_state.get("live_run")
   if live_run is None:
       st.caption(
           "Press Start to create a fresh seeded CC4 environment. "
           "Saved replay remains available as an explicit fallback."
       )
   else:
       if live_run.error_state:
           st.warning(live_run.error_state)
       frame = live_run.frames[-1] if live_run.frames else live_run.frame()
       provenance = frame["provenance"]
       st.caption(
           f"Mode: `LIVE` | Policy: `{provenance['agent']}` | Seed: `{provenance['seed']}` | "
           f"Red: `{provenance['red']}` | Step: `{frame['step']}`"
       )

       metrics = st.columns(5)
       metrics[0].metric("Current Step", f"T-{max(frame['step'], 0):02d}")
       metrics[1].metric("Step Reward", f"{frame['reward']:+.1f}")
       metrics[2].metric("Cumulative Return", f"{frame['cumulative_reward']:+.1f}")
       metrics[3].metric(
           "Red Sessions",
           sum(state == "red_session" for state in frame["host_state"].values()),
       )
       metrics[4].metric("Mission Phase", frame["mission_phase"])

       col_graph, col_actions = st.columns([1.35, 1])
       with col_graph:
           st.plotly_chart(
               build_zone_graph(frame["host_state"]),
               use_container_width=True,
           )
           render_zone_detail(frame["host_state"], "live_zone")
       with col_actions:
           render_step_status(frame)

       if frame["episode_done"]:
           st.success("Live episode complete. This run is demonstration evidence only.")
       elif st.session_state.get("live_running"):
           st.caption("Use Step forward for deterministic single-transition playback.")

       with st.expander("Current live frame payload"):
           st.json(frame)
       if live_run.frames:
           if st.button("Save live run artifact", use_container_width=False):
               output = EPISODES / (
                   f"live_demo_{live_run.blue_policy}_{live_run.red_strategy}_"
                   f"seed{live_run.seed}.json"
               )
               try:
                   saved = live_run.save(output)
                   st.success(f"Saved live-demo artifact: `{saved.name}`")
               except Exception as exc:
                   st.error(f"Could not save live-demo artifact: {exc}")

       if st.session_state.get("live_running") and not live_run.episode_done:
           delay_map = {"0.5x": 0.8, "1x": 0.35, "2x": 0.18, "4x": 0.08}
           time.sleep(delay_map[live_speed])
           try:
               live_run.step_once()
               st.rerun()
           except Exception as exc:
               live_run.error_state = str(exc)
               st.session_state["live_running"] = False

# ═══════════════════════════════════════════════════════════════════════════
# VIEW 2: EPISODE REPLAY (Smooth Play/Pause & Filter by MAPPO/IPPO/Sleep)
# ═══════════════════════════════════════════════════════════════════════════
elif "Episode Replay" in nav_choice:
    st.markdown("#### Saved CC4 Episode Replay")

    selected_meta, episode = replay_selector(catalog, "replay_episode")

    if episode is None:
        st.info("No results yet: select an episode or add a JSON rollout under results/episodes/.")
    else:
        provenance = episode.get("provenance", {})
        st.caption(
            f"Algorithm: `{provenance.get('agent')}` | Seed: `{provenance.get('seed')}` | "
            f"Adversary: `{provenance.get('red')}` | Total Return: `{episode.get('total_reward', 0.0):+.1f}`"
        )
        steps = episode.get("steps", [])

        if not steps:
            st.warning("Selected episode file contains zero steps.")
        else:
            # Initialize playback states safely
            if "is_playing" not in st.session_state:
                st.session_state["is_playing"] = False
            if "live_step_index" not in st.session_state:
                st.session_state["live_step_index"] = 0

            # Ensure valid bounds
            current_step_idx = int(st.session_state.get("live_step_index", 0))
            if current_step_idx >= len(steps):
                current_step_idx = max(0, len(steps) - 1)
                st.session_state["live_step_index"] = current_step_idx

            # Playback Controls Bar
            c_first, c_prev, c_playpause, c_next, c_spd, c_slider = st.columns(
                [0.75, 0.75, 1.1, 0.75, 1.1, 3.2]
            )

            with c_first:
                if st.button("⏮ First", key="btn_first", use_container_width=True):
                    st.session_state["live_step_index"] = 0
                    st.session_state["is_playing"] = False
                    st.rerun()

            with c_prev:
                if st.button("◀ Prev", key="btn_prev", use_container_width=True):
                    st.session_state["live_step_index"] = max(
                        0, int(st.session_state.get("live_step_index", 0)) - 1
                    )
                    st.session_state["is_playing"] = False
                    st.rerun()

            is_playing = st.session_state.get("is_playing", False)
            with c_playpause:
                if is_playing:
                    if st.button("⏸ Pause", key="btn_pause", use_container_width=True):
                        st.session_state["is_playing"] = False
                        st.rerun()
                else:
                    if st.button("▶ Play", key="btn_play", use_container_width=True):
                        if int(st.session_state.get("live_step_index", 0)) >= len(steps) - 1:
                            st.session_state["live_step_index"] = 0
                        st.session_state["is_playing"] = True
                        st.rerun()

            with c_next:
                if st.button("Next ▶", key="btn_next", use_container_width=True):
                    st.session_state["live_step_index"] = min(
                        len(steps) - 1, int(st.session_state.get("live_step_index", 0)) + 1
                    )
                    st.session_state["is_playing"] = False
                    st.rerun()

            with c_spd:
                speed = st.selectbox(
                    "Speed",
                    ["0.5x", "1x", "2x", "4x"],
                    index=1,
                    key="live_speed_select",
                    label_visibility="collapsed",
                )

            with c_slider:
                current_idx = int(st.session_state.get("live_step_index", 0))
                scrubbed = st.slider(
                    "Step",
                    0,
                    max(len(steps) - 1, 0),
                    current_idx,
                    key="scrubber_slider",
                    label_visibility="collapsed",
                )
                if scrubbed != current_idx:
                    st.session_state["live_step_index"] = scrubbed
                    st.session_state["is_playing"] = False
                    st.rerun()

            # Render step telemetry in permanent, fixed position
            step_idx = int(st.session_state.get("live_step_index", 0))
            if step_idx >= len(steps):
                step_idx = max(0, len(steps) - 1)
            step = steps[step_idx]
            host_state = step.get("host_state", {})
            counts = status_counts(host_state)
            red_count = counts.get("red_session", 0)

            # Permanent 4-cell metric row (NEVER slides or shifts)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Step", f"T-{step.get('step', step_idx):02d}")
            m2.metric("Step Reward", f"{step.get('reward', 0.0):+.1f}")
            m3.metric("Cumulative Return", f"{step.get('cumulative_reward', 0.0):+.1f}")
            m4.metric("Red Sessions", red_count)

            # 2D Network Topology Visualizer + Tactical Step Status (Equal 380px heights)
            col_graph, col_actions = st.columns([1.35, 1])
            with col_graph:
                st.plotly_chart(
                    build_zone_graph(host_state), use_container_width=True
                )
                render_zone_detail(host_state, "replay_zone")
            with col_actions:
                render_step_status(step)

            # Cumulative Reward Curve
            st.plotly_chart(
                replay_chart(steps, step_idx), use_container_width=True
            )

            # LLM Telemetry Snapshot for this exact step
            st.markdown("#### LLM Advisory Snapshot (Current Step)")
            render_step_advisory(int(step.get("step", step_idx)))

            with st.expander("Raw Step Telemetry Payload (JSON)"):
                st.json(step)

            # Automatic animation advance loop (if playing)
            if is_playing:
                delay_map = {"0.5x": 0.8, "1x": 0.35, "2x": 0.18, "4x": 0.08}
                time.sleep(delay_map.get(speed, 0.35))
                curr_idx = int(st.session_state.get("live_step_index", 0))
                if curr_idx < len(steps) - 1:
                    st.session_state["live_step_index"] = curr_idx + 1
                    st.rerun()
                else:
                    st.session_state["is_playing"] = False
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# VIEW 2: COMPARE REPLAYS (Side-by-side & Paired Scoreboard with CI Graph)
# ═══════════════════════════════════════════════════════════════════════════
elif "Compare Replays" in nav_choice:
    st.markdown("#### Compare Saved CC4 Replays")
    if len(catalog) < 2:
        st.info("At least two saved CC4 episode files are required for comparison.")
    else:
        # Quick Paired Filter Selector (e.g. Compare MAPPO vs IPPO on Seed 42)
        st.markdown("**Quick Paired Configuration**")
        c_alg_a, c_alg_b, c_seed_comp = st.columns([1.2, 1.2, 1.0])
        avail_agents = sorted(catalog["agent"].dropna().unique().tolist())
        with c_alg_a:
            alg_a = st.selectbox(
                "Agent A",
                avail_agents,
                index=avail_agents.index("MAPPO") if "MAPPO" in avail_agents else 0,
                key="comp_alg_a",
            )
        with c_alg_b:
            alg_b = st.selectbox(
                "Agent B",
                avail_agents,
                index=avail_agents.index("IPPO") if "IPPO" in avail_agents else 0,
                key="comp_alg_b",
            )
        common_seeds = sorted(
            list(
                set(catalog[catalog["agent"] == alg_a]["seed"]).intersection(
                    set(catalog[catalog["agent"] == alg_b]["seed"])
                )
            )
        )
        with c_seed_comp:
            seed_choice = st.selectbox(
                "Evaluation Seed",
                common_seeds if common_seeds else [42],
                key="comp_seed_choice",
            )

        row_a = catalog[
            (catalog["agent"] == alg_a) & (catalog["seed"] == seed_choice)
        ]
        row_b = catalog[
            (catalog["agent"] == alg_b) & (catalog["seed"] == seed_choice)
        ]

        if not row_a.empty and not row_b.empty:
            ep_a = _read_json(Path(row_a.iloc[0]["path"]))
            ep_b = _read_json(Path(row_b.iloc[0]["path"]))

            limit = min(len(ep_a.get("steps", [])), len(ep_b.get("steps", [])))
            comp_step = st.slider(
                "Synchronized Step Scrubber", 0, max(limit - 1, 0), min(40, max(limit - 1, 0)), key="comp_step_slider"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                step_data_a = ep_a["steps"][comp_step]
                st.markdown(
                    f"**{alg_a}** (Seed `{seed_choice}`) — Step Return: `{step_data_a.get('reward', 0.0):+.1f}` | Cumulative: `{step_data_a.get('cumulative_reward', 0.0):+.1f}`"
                )
                st.plotly_chart(
                    build_zone_graph(step_data_a.get("host_state", {})),
                    use_container_width=True,
                )
                render_zone_detail(step_data_a.get("host_state", {}), "comp_zone_a")

            with col_b:
                step_data_b = ep_b["steps"][comp_step]
                st.markdown(
                    f"**{alg_b}** (Seed `{seed_choice}`) — Step Return: `{step_data_b.get('reward', 0.0):+.1f}` | Cumulative: `{step_data_b.get('cumulative_reward', 0.0):+.1f}`"
                )
                st.plotly_chart(
                    build_zone_graph(step_data_b.get("host_state", {})),
                    use_container_width=True,
                )
                render_zone_detail(step_data_b.get("host_state", {}), "comp_zone_b")
        else:
            st.warning("Could not automatically pair the selected algorithms on this seed. Use custom selector below.")

        with st.expander("Custom Episode Selector (Any File)", expanded=False):
            left_meta, left_episode = replay_selector(catalog, "compare_left")
            right_meta, right_episode = replay_selector(catalog, "compare_right")

    st.markdown("---")
    st.markdown("#### Paired Performance Scoreboard vs Sleep Baseline")
    baselines = baseline_summary()
    if not baselines.empty:
        st.markdown("**Algorithm Baseline Context**")
        st.dataframe(
            baselines[["agent", "mean_reward", "std_reward"]].style.format(
                {"mean_reward": "{:.1f}", "std_reward": "{:.1f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    scoreboard = paired_scoreboard()
    if scoreboard.empty:
        st.info(
            "Paired episode returns are not available yet. Generate "
            "results/tables/heldout_episode_returns.csv before making confidence-interval claims."
        )
    else:
        col_sb_table, col_sb_graph = st.columns([1.3, 1])
        with col_sb_table:
            st.dataframe(
                scoreboard.style.format(
                    {
                        "mean_difference_vs_sleep": "{:+.1f}",
                        "ci95_low": "{:+.1f}",
                        "ci95_high": "{:+.1f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Difference = Agent return minus Sleep return on the same evaluation seed; "
                "interval represents normal-approximation 95% CI over paired held-out episodes."
            )
        with col_sb_graph:
            st.plotly_chart(
                build_ci_forest_plot(scoreboard), use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════════════════
# VIEW 3: TRAINING EVIDENCE (Multi-Seed Curves & Variance)
# ═══════════════════════════════════════════════════════════════════════════
elif "Training Evidence" in nav_choice:
    st.markdown("#### Multi-Seed Training Convergence")
    train = training_catalog()
    if train.empty:
        st.info("No training CSVs found under results/training/.")
    else:
        algorithms = sorted(train["algo"].dropna().unique())
        selected_algorithms = st.multiselect(
            "Filter Algorithms", algorithms, default=algorithms
        )
        visible = train[train["algo"].isin(selected_algorithms)]

        figure = go.Figure()
        algo_colors = {"ippo": "#1d4ed8", "mappo": "#dc2626"}
        for algo in selected_algorithms:
            subset = visible[visible["algo"] == algo]
            grouped = (
                subset.groupby("env_steps")["return"]
                .agg(["mean", "std"])
                .reset_index()
            )
            grouped["std"] = grouped["std"].fillna(0)
            color = algo_colors.get(algo.lower(), "#0f172a")

            figure.add_trace(
                go.Scatter(
                    x=grouped["env_steps"],
                    y=grouped["mean"],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{algo.upper()} Mean",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=grouped["env_steps"],
                    y=grouped["mean"] + grouped["std"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=grouped["env_steps"],
                    y=grouped["mean"] - grouped["std"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(100,100,100,0.12)",
                    line=dict(width=0),
                    name=f"{algo.upper()} ± Std",
                )
            )

        figure.update_layout(
            title=dict(
                text="Training Return Progression by Environment Steps",
                font=dict(size=12, color="#334155"),
            ),
            xaxis_title="Environment Steps",
            yaxis_title="Episode Return",
            height=380,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="sans-serif", size=11, color="#334155"),
        )
        figure.update_xaxes(
            showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1"
        )
        figure.update_yaxes(
            showgrid=True, gridcolor="#f1f5f9", zerolinecolor="#cbd5e1"
        )
        st.plotly_chart(figure, use_container_width=True)

        st.markdown("**Run Summary Statistics**")
        st.dataframe(
            visible.groupby("algo").agg(
                seeds=("seed", "nunique"),
                final_return=("return", "last"),
                max_env_steps=("env_steps", "max"),
            ),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
# VIEW 4: LLM ADVISORY EVIDENCE (Latency, Audit Trail, & Schema Verification)
# ═══════════════════════════════════════════════════════════════════════════
elif "LLM Advisory" in nav_choice:
    st.markdown("#### Measured LLM Advisory Evidence")
    summary = csv_table("advisory_summary.csv")
    calls = csv_table("advisory_calls.csv")
    episodes = csv_table("advisory_episodes.csv")

    if summary.empty:
        st.info("No advisory summary found under results/tables/.")
    else:
        row = summary.iloc[0]
        st.caption(
            f"Model: `{row.get('model', 'unknown')}` | Prompt Version: `{row.get('prompt_version', 'unknown')}` | "
            f"Checkpoint: `{row.get('checkpoint', 'unknown')}`"
        )

        metrics = st.columns(5)
        metrics[0].metric("Gemini Calls", int(row.get("calls_gemini", 0)))
        metrics[1].metric(
            "Fallback Rate", f"{100 * float(row.get('fallback_rate', 0)):.1f}%"
        )
        metrics[2].metric(
            "Mean Latency",
            f"{float(row.get('gemini_mean_latency_ms', 0)):.0f} ms",
        )
        metrics[3].metric(
            "Total Cost", f"${float(row.get('gemini_total_cost_usd', 0)):.4f}"
        )
        metrics[4].metric(
            "Identical Returns",
            f"{int(row.get('episodes_identical_return', 0))}/{int(row.get('episodes', 0))}",
        )
        st.info(
            "Advisory mode verification: The LLM operates strictly out-of-loop and does not alter RL agent actions. "
            "Identical return confirms non-interference."
        )

        st.markdown("##### Incident Report Inspector")
        avail_steps = (
            sorted(calls["step"].unique().tolist())
            if not calls.empty
            else advisory_steps()
        )
        if not avail_steps:
            avail_steps = [40]

        report_step = st.selectbox(
            "Select Advisory Step",
            avail_steps,
            index=avail_steps.index(40) if 40 in avail_steps else 0,
            key="incident_report_step",
        )

        advisory_step = load_advisory_step(int(report_step))
        if advisory_step:
            result = advisory_step.get("result", {})
            report = result.get("advice", {}).get("incident_report", {})
            priorities = result.get("advice", {}).get("priorities", [])

            col_rep_text, col_rep_prio = st.columns([1.4, 1])
            with col_rep_text:
                st.markdown(
                    f"""<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;">
<div style="font-size:0.7rem;font-weight:700;color:#1d4ed8;text-transform:uppercase;">Executive Incident Summary</div>
<p style="font-size:0.85rem;line-height:1.5;margin:6px 0 10px 0;">{report.get("executive_summary", "No report text available.")}</p>
<div style="font-size:0.75rem;margin-bottom:8px;"><b>Kill-Chain Stage:</b> <span style="background:#fee2e2;color:#b91c1c;padding:2px 8px;border-radius:4px;font-weight:700;">{report.get("kill_chain_stage", "unknown")}</span></div>
<div style="font-size:0.75rem;font-weight:700;margin-top:10px;">Recommended Countermeasures:</div>
<ul style="font-size:0.78rem;padding-left:18px;margin:4px 0;">
{"".join(f"<li>{act}</li>" for act in report.get("mitigation_actions", [])) or "<li>No specific mitigations logged.</li>"}
</ul>
<div style="font-size:0.68rem;color:#64748b;margin-top:10px;">Source: <code>{result.get('source')}</code> | Latency: <code>{result.get('latency_ms', 0):.0f} ms</code> | Cost: <code>${result.get('cost_usd', 0):.6f}</code></div>
</div>""",
                    unsafe_allow_html=True,
                )
            with col_rep_prio:
                if priorities:
                    prio_df = pd.DataFrame(priorities)
                    st.markdown("**Zone Prioritization**")
                    st.dataframe(
                        prio_df[["zone", "priority_score", "reasoning"]],
                        hide_index=True,
                        use_container_width=True,
                    )

        if not calls.empty:
            st.markdown("##### Advisory Invocation Latency by Step")
            lat_fig = go.Figure(
                go.Scatter(
                    x=calls["step"],
                    y=calls["latency_ms"],
                    mode="markers",
                    marker=dict(
                        color=calls["source"]
                        .map(
                            {
                                "gemini": "#16a34a",
                                "cache": "#1d4ed8",
                                "fallback": "#dc2626",
                            }
                        )
                        .fillna("#64748b"),
                        size=8,
                    ),
                    text=calls["source"],
                    hovertemplate="Step %{x}: %{y:.0f} ms (%{text})<extra></extra>",
                )
            )
            lat_fig.update_layout(
                title=dict(
                    text="Measured Advisory Latency by CC4 Step (ms)",
                    font=dict(size=12, color="#334155"),
                ),
                xaxis_title="Step",
                yaxis_title="Latency (ms)",
                height=260,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )
            st.plotly_chart(lat_fig, use_container_width=True)
            st.dataframe(calls, use_container_width=True)

        if not episodes.empty:
            with st.expander("Advisory Episode Checks Dataframe"):
                st.dataframe(episodes, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# VIEW 5: ARCHITECTURE (Decoupled RL + LLM Dual-Loop)
# ═══════════════════════════════════════════════════════════════════════════
elif "Architecture" in nav_choice:
    st.markdown("#### CC4 System Architecture")
    st.markdown(
        """<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 22px;margin-bottom:16px;">
<div style="font-size:0.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:12px;">Decoupled Dual-Loop Control Flow</div>
<div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;padding:12px 0;">
    <div style="background:#f1f5f9;border:1px solid #cbd5e1;padding:10px 16px;border-radius:6px;font-weight:700;font-size:0.8rem;text-align:center;">
        CC4 Simulation<br><span style="font-size:0.65rem;color:#64748b;">Enterprise Subnets</span>
    </div>
    <span style="font-weight:800;color:#64748b;">──></span>
    <div style="background:#f1f5f9;border:1px solid #cbd5e1;padding:10px 16px;border-radius:6px;font-weight:700;font-size:0.8rem;text-align:center;">
        5 RL Agents<br><span style="font-size:0.65rem;color:#64748b;">IPPO / MAPPO</span>
    </div>
    <span style="font-weight:800;color:#64748b;">──></span>
    <div style="background:#f1f5f9;border:1px solid #cbd5e1;padding:10px 16px;border-radius:6px;font-weight:700;font-size:0.8rem;text-align:center;">
        Action & Event Buffer<br><span style="font-size:0.65rem;color:#64748b;">Telemetry Sink</span>
    </div>
    <span style="font-weight:800;color:#64748b;">──></span>
    <div style="background:#0f172a;color:#ffffff;padding:10px 18px;border-radius:6px;font-weight:700;font-size:0.8rem;text-align:center;">
        LLM Advisor<br><span style="font-size:0.65rem;color:#94a3b8;">Gemini Async</span>
    </div>
    <span style="font-weight:800;color:#64748b;">──></span>
    <div style="background:#f1f5f9;border:1px solid #cbd5e1;padding:10px 16px;border-radius:6px;font-weight:700;font-size:0.8rem;text-align:center;">
        Incident Report<br><span style="font-size:0.65rem;color:#64748b;">SOC Prioritization</span>
    </div>
</div>
<div style="text-align:center;font-size:0.72rem;color:#64748b;margin-top:10px;font-weight:600;">
    CRITICAL INVARIANT: The LLM Advisory pipeline is asynchronous and decoupled from tactical execution. RL actions execute at simulation speed regardless of LLM response latency.
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**CC4 environment:** Five Blue agents operate from decentralized local observations against a 
`FiniteStateRedAgent` in a randomized enterprise topology of approximately 100 hosts across 9 zones. 
IPPO and MAPPO execute tactical countermeasures (Sleep, Monitor, Analyse, Restore, Remove, Decoy); 
the LLM serves as an out-of-loop advisory layer providing structured threat intelligence.

**Evidence boundary:** Every replay, reward, host state, latency, token count, and cost shown 
in this dashboard is loaded from saved files under `results/`. Advisory mode records intelligence and 
synthesizes incident reports without modifying RL actions, guaranteeing empirical integrity.
"""
    )
    st.latex(
        r"\mathcal{M}=\langle\mathcal{N},\mathcal{S},\{\mathcal{A}_i\},"
        r"\mathcal{P},\{r_i\},\{\Omega_i\},\{\mathcal{O}_i\},\gamma\rangle"
    )
    st.markdown(
        """
- **IPPO (Independent PPO):** Completely decentralized actor and critic instances per agent with independent policy updates.
- **MAPPO (Multi-Agent PPO):** Centralized training with decentralized execution (CTDE) — centralized value estimator observing joint observations paired with decentralized policy heads.
- **LLM Advisory Layer:** Summarizes temporal alerts and multi-agent action history to return schema-validated incident priorities and kill-chain classification.
"""
    )

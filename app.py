"""Evidence-driven Streamlit dashboard for saved CC4 experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EPISODES = RESULTS / "episodes"
TABLES = RESULTS / "tables"
TRAINING = RESULTS / "training"

st.set_page_config(
    page_title="CC4 LLM-MARL Incident Response",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .event-log { background: #1e1e1e; color: #d7ff8a; padding: 12px;
                 border-radius: 8px; height: 360px; overflow-y: auto;
                 font-family: monospace; white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def episode_catalog() -> pd.DataFrame:
    rows = []
    for path in sorted(EPISODES.glob("*.json")):
        try:
            data = _read_json(path)
            provenance = data.get("provenance", {})
            rows.append(
                {
                    "label": path.name,
                    "path": str(path),
                    "agent": provenance.get("agent", "unknown"),
                    "red": provenance.get("red", "unknown"),
                    "seed": provenance.get("seed"),
                    "steps": provenance.get("steps", len(data.get("steps", []))),
                    "git_commit": provenance.get("git_commit", "unknown"),
                    "source": provenance.get("source", "unknown"),
                    "host_count": provenance.get("host_count"),
                    "total_reward": data.get("total_reward"),
                }
            )
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return pd.DataFrame(rows)


@st.cache_data
def csv_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def training_catalog() -> pd.DataFrame:
    frames = []
    for path in sorted(TRAINING.glob("*_train.csv")):
        frame = pd.read_csv(path)
        frame["run"] = path.stem.removesuffix("_train")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def show_empty(message: str) -> None:
    st.info(message)


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


def build_zone_graph(host_state: dict) -> go.Figure:
    grouped = grouped_hosts(host_state)
    figure = go.Figure()
    zone_names = list(grouped)
    red_counts = [
        sum(state == "red_session" for _, state in grouped[zone])
        for zone in zone_names
    ]
    host_counts = [len(grouped[zone]) for zone in zone_names]
    figure.add_trace(
        go.Scatter(
            x=list(range(len(zone_names))),
            y=[0] * len(zone_names),
            mode="markers+text",
            marker={
                "size": [28 + 4 * count for count in red_counts],
                "color": red_counts,
                "colorscale": [[0, "#2e7d32"], [1, "#d32f2f"]],
                "cmin": 0,
                "cmax": max(max(red_counts, default=0), 1),
                "line": {"color": "#ffffff", "width": 2},
                "colorbar": {"title": "Red sessions"},
            },
            text=[
                f"{zone}<br>{red}/{total} Red sessions"
                for zone, red, total in zip(zone_names, red_counts, host_counts)
            ],
            textposition="top center",
            hovertemplate=[
                f"{zone}<br>{red} Red sessions / {total} hosts<extra></extra>"
                for zone, red, total in zip(zone_names, red_counts, host_counts)
            ],
            showlegend=False,
        )
    )
    figure.update_layout(
        title="Randomized CC4 enterprise network by zone",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(len(zone_names))),
            "ticktext": zone_names,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False, "range": [-1, 1]},
        height=300,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return figure


def replay_chart(steps: list[dict], upto: int | None = None) -> go.Figure:
    visible = steps if upto is None else steps[: upto + 1]
    return go.Figure(
        go.Scatter(
            x=[step["step"] for step in visible],
            y=[step["cumulative_reward"] for step in visible],
            mode="lines+markers",
            name="Cumulative team reward",
        )
    ).update_layout(
        title="Cumulative team reward",
        xaxis_title="CC4 step",
        yaxis_title="Reward",
        height=280,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )


def replay_selector(catalog: pd.DataFrame, key: str) -> tuple[pd.DataFrame, dict | None]:
    if catalog.empty:
        return catalog, None
    selected = st.selectbox("Saved CC4 episode", catalog["label"].tolist(), key=key)
    row = catalog.loc[catalog["label"] == selected].iloc[0]
    return row.to_frame().T, _read_json(Path(row["path"]))


def render_zone_detail(host_state: dict, key: str) -> None:
    grouped = grouped_hosts(host_state)
    if not grouped:
        return
    selected_zone = st.selectbox("Expand zone into hosts", list(grouped), key=key)
    rows = [
        {"host": host, "state": state}
        for host, state in grouped[selected_zone]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


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
        differences = paired["agent_return"] - paired["sleep_return"]
        mean = differences.mean()
        std = differences.std(ddof=1)
        n = len(differences)
        margin = 1.96 * std / (n**0.5) if n > 1 else float("nan")
        rows.append(
            {
                "agent": agent,
                "episodes": n,
                "mean_difference_vs_sleep": mean,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }
        )
    return pd.DataFrame(rows)


def load_advisory_step(step: int) -> dict | None:
    path = RESULTS / "advisory" / f"seed9000000_step{step:03d}.json"
    return _read_json(path) if path.exists() else None


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


catalog = episode_catalog()
st.title("🛡️ CC4 LLM-MARL Incident Response")
st.caption(
    "CC4 uses a randomized enterprise network of about 100 hosts across 9 zones. "
    "Exact host counts vary by episode and are read from the saved rollout."
)

with st.sidebar:
    st.header("Evidence inventory")
    st.metric("Saved CC4 episodes", len(catalog))
    st.metric("Training runs", len(training_catalog()["run"].unique()) if not training_catalog().empty else 0)
    st.metric("Advisory calls", len(csv_table("advisory_calls.csv")))
    if not catalog.empty:
        st.success("CC4 real rollout data available")
    else:
        st.warning("No CC4 replay files found")

tab_live, tab_compare, tab_train, tab_llm, tab_arch = st.tabs(
    ["Episode Replay", "Compare", "Training", "LLM Advisory", "Architecture"]
)

with tab_live:
    st.subheader("Saved CC4 episode replay")
    selected_meta, episode = replay_selector(catalog, "replay_episode")
    if episode is None:
        show_empty("No results yet: add a real JSON rollout under results/episodes/.")
    else:
        provenance = episode["provenance"]
        st.caption(
            f"Source: `{provenance.get('source')}` | agent `{provenance.get('agent')}` | "
            f"red `{provenance.get('red')}` | seed `{provenance.get('seed')}` | "
            f"commit `{provenance.get('git_commit')}`"
        )
        steps = episode.get("steps", [])
        step_index = st.slider("Replay step", 0, max(len(steps) - 1, 0), 0, key="replay_step")
        step = steps[step_index]
        host_state = step.get("host_state", {})
        counts = status_counts(host_state)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cumulative reward", f"{step.get('cumulative_reward', 0):.1f}")
        c2.metric("Hosts", len(host_state))
        c3.metric("Red sessions", counts.get("red_session", 0))
        c4.metric("Step reward", f"{step.get('reward', 0):.1f}")
        left, right = st.columns([1.35, 1])
        with left:
            st.plotly_chart(
                build_zone_graph(host_state),
                use_container_width=True,
            )
            render_zone_detail(host_state, "replay_zone")
        with right:
            st.markdown("**Event log**")
            lines = []
            for item in steps[: step_index + 1]:
                lines.append(
                    f"step {item.get('step')}: reward={item.get('reward')} | "
                    f"red={item.get('red_activity')} | blue={item.get('blue_actions')}"
                )
            st.markdown(
                f'<div class="event-log">{chr(10).join(reversed(lines))}</div>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(replay_chart(steps, step_index), use_container_width=True)
        with st.expander("Raw saved step"):
            st.json(step)

with tab_compare:
    st.subheader("Compare saved CC4 replays")
    if len(catalog) < 2:
        show_empty("At least two saved CC4 episode files are required for comparison.")
    else:
        left_meta, left_episode = replay_selector(catalog, "compare_left")
        right_meta, right_episode = replay_selector(catalog, "compare_right")
        if left_episode and right_episode:
            same_seed = (
                left_meta.iloc[0]["seed"] == right_meta.iloc[0]["seed"]
                and left_meta.iloc[0]["red"] == right_meta.iloc[0]["red"]
            )
            if not same_seed:
                st.warning("These files do not use the same seed and red policy.")
            limit = min(len(left_episode["steps"]), len(right_episode["steps"]))
            compare_step = st.slider("Comparison step", 0, max(limit - 1, 0), 0, key="compare_step")
            columns = st.columns(2)
            for side, (column, meta, data) in enumerate(
                zip(columns, [left_meta, right_meta], [left_episode, right_episode])
            ):
                item = data["steps"][compare_step]
                with column:
                    st.markdown(
                        f"**{meta.iloc[0]['agent']}**, seed `{meta.iloc[0]['seed']}`  \n"
                        f"Reward: `{item.get('cumulative_reward', 0):.1f}`"
                    )
                    st.plotly_chart(
                        build_zone_graph(item.get("host_state", {})),
                        use_container_width=True,
                    )
                    render_zone_detail(
                        item.get("host_state", {}),
                        f"compare_zone_{side}_{meta.iloc[0]['agent']}",
                    )

    st.markdown("---")
    st.subheader("Paired scoreboard versus Sleep")
    st.markdown("**Baseline context**")
    baselines = baseline_summary()
    if not baselines.empty:
        st.dataframe(
            baselines[["agent", "mean_reward", "std_reward"]].style.format(
                {"mean_reward": "{:.1f}", "std_reward": "{:.1f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    scoreboard = paired_scoreboard()
    if scoreboard.empty:
        show_empty(
            "Paired episode returns are not available yet. Generate "
            "results/tables/heldout_episode_returns.csv before making confidence-interval claims."
        )
    else:
        st.dataframe(
            scoreboard.style.format(
                {
                    "mean_difference_vs_sleep": "{:.1f}",
                    "ci95_low": "{:.1f}",
                    "ci95_high": "{:.1f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Difference is agent return minus Sleep return on the same evaluation seed; "
            "the interval is a normal-approximation 95% CI over paired episodes."
        )

with tab_train:
    st.subheader("Multi-seed training evidence")
    train = training_catalog()
    if train.empty:
        show_empty("No training CSVs found under results/training/.")
    else:
        algorithms = sorted(train["algo"].dropna().unique())
        selected_algorithms = st.multiselect("Algorithms", algorithms, default=algorithms)
        visible = train[train["algo"].isin(selected_algorithms)]
        figure = go.Figure()
        for algo in selected_algorithms:
            subset = visible[visible["algo"] == algo]
            grouped = subset.groupby("env_steps")["return"].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0)
            figure.add_trace(
                go.Scatter(
                    x=grouped["env_steps"],
                    y=grouped["mean"],
                    mode="lines",
                    name=f"{algo.upper()} mean",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=grouped["env_steps"],
                    y=grouped["mean"] + grouped["std"],
                    mode="lines",
                    line={"width": 0},
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
                    line={"width": 0},
                    name=f"{algo.upper()} ± std",
                )
            )
        figure.update_layout(
            title="Training return by environment steps",
            xaxis_title="Environment steps",
            yaxis_title="Episode return",
            height=430,
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(
            visible.groupby("algo").agg(
                seeds=("seed", "nunique"),
                final_return=("return", "last"),
                max_env_steps=("env_steps", "max"),
            ),
            use_container_width=True,
        )

with tab_llm:
    st.subheader("Measured LLM advisory evidence")
    summary = csv_table("advisory_summary.csv")
    calls = csv_table("advisory_calls.csv")
    episodes = csv_table("advisory_episodes.csv")
    if summary.empty:
        show_empty("No advisory summary found under results/tables/.")
    else:
        row = summary.iloc[0]
        st.caption(
            f"Model `{row.get('model', 'unknown')}` | prompt `{row.get('prompt_version', 'unknown')}` | "
            f"checkpoint `{row.get('checkpoint', 'unknown')}`"
        )
        metrics = st.columns(5)
        metrics[0].metric("Fresh Gemini calls", int(row.get("calls_gemini", 0)))
        metrics[1].metric("Fallback rate", f"{100 * float(row.get('fallback_rate', 0)):.1f}%")
        metrics[2].metric("Mean latency", f"{float(row.get('gemini_mean_latency_ms', 0)):.0f} ms")
        metrics[3].metric("Total cost", f"${float(row.get('gemini_total_cost_usd', 0)):.4f}")
        metrics[4].metric("Identical returns", f"{int(row.get('episodes_identical_return', 0))}/{int(row.get('episodes', 0))}")
        st.info("Advisory mode: it does not change actions.")
        report_step = st.selectbox(
            "Incident report step",
            sorted(calls["step"].unique().tolist()) if not calls.empty else [40],
            index=(
                sorted(calls["step"].unique().tolist()).index(40)
                if not calls.empty and 40 in calls["step"].unique()
                else 0
            ),
            key="incident_report_step",
        )
        advisory_step = load_advisory_step(int(report_step))
        if advisory_step:
            result = advisory_step["result"]
            report = result.get("advice", {}).get("incident_report", {})
            st.markdown("#### Incident report")
            st.markdown(f"**Executive summary:** {report.get('executive_summary', 'No report')}")
            st.markdown(f"**Kill-chain stage:** `{report.get('kill_chain_stage', 'unknown')}`")
            st.markdown("**Recommended remediation:**")
            for action in report.get("mitigation_actions", []):
                st.markdown(f"- {action}")
            st.caption(
                f"Source `{result.get('source')}` | measured latency "
                f"`{result.get('latency_ms', 0):.0f} ms` | measured cost "
                f"`${result.get('cost_usd', 0):.6f}`"
            )
        if not calls.empty:
            st.plotly_chart(
                go.Figure(
                    go.Scatter(
                        x=calls["step"],
                        y=calls["latency_ms"],
                        mode="markers",
                        marker={"color": calls["source"].map({"gemini": "#2e7d32", "cache": "#1565c0", "fallback": "#d32f2f"}).fillna("#78909c")},
                        text=calls["source"],
                    )
                ).update_layout(title="Measured advisory latency by CC4 step", xaxis_title="Step", yaxis_title="Latency (ms)"),
                use_container_width=True,
            )
            st.dataframe(calls, use_container_width=True)
        if not episodes.empty:
            with st.expander("Advisory episode checks"):
                st.dataframe(episodes, use_container_width=True)

with tab_arch:
    st.subheader("CC4 system architecture")
    st.markdown(
        """
        **CC4 environment:** five Blue agents act from local observations against a
        `FiniteStateRedAgent` in a randomized enterprise topology. IPPO and MAPPO
        execute tactical actions; the LLM is an out-of-loop advisory layer.

        **Evidence boundary:** every replay, reward, host state, latency, token
        count, and cost shown here comes from a saved file under `results/`.
        Advisory mode records guidance and validates it, but does not alter RL
        actions; equal RL-only and advisory returns are therefore an integrity check,
        not an improvement claim.
        """
    )
    st.latex(
        r"\mathcal{M}=\langle\mathcal{N},\mathcal{S},\{\mathcal{A}_i\},"
        r"\mathcal{P},\{r_i\},\{\Omega_i\},\{\mathcal{O}_i\},\gamma\rangle"
    )
    st.markdown(
        """
        - **IPPO:** independent policies for decentralized tactical execution.
        - **MAPPO:** decentralized actors with a centralized critic during training.
        - **LLM advisory:** summarizes telemetry and proposed actions, then returns
          schema-validated priorities and an incident report.
        """
    )

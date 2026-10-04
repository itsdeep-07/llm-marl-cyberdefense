import os
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="CybORG LLM-MARL Defense Playground",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 20px; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 15px; border-left: 5px solid #1E88E5; }
    .status-secure { color: #2E7D32; font-weight: bold; }
    .status-scanned { color: #F57F17; font-weight: bold; }
    .status-compromised { color: #C62828; font-weight: bold; }
    .status-restored { color: #1565C0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ CybORG LLM-MARL Cyber Incident Response Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Multi-Agent Defense & LLM Orchestration Simulation Platform</div>', unsafe_allow_html=True)


# Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")

phase_selection = st.sidebar.selectbox(
    "Select Architecture Phase",
    [
        "Phase 1: Baselines (Rule-Based vs Red B-line)",
        "Phase 2: Multi-Agent RL (PPO / MAPPO)",
        "Phase 3: MARL + LLM Orchestration (Full System)"
    ],
    index=0
)

red_agent_selection = st.sidebar.selectbox(
    "Red Attacker Strategy",
    ["B-line (Aggressive Target)", "Meander (Random Walk Discovery)"]
)

seed = st.sidebar.number_input("Random Seed", value=42, step=1)
max_steps = st.sidebar.slider("Episode Step Horizon", min_value=10, max_value=50, value=30)
current_step = st.sidebar.slider("Scrub Simulation Step", min_value=1, max_value=max_steps, value=15)

# Network Topology Simulation Data Generator
def generate_topology_data(step, phase, red_strategy):
    hosts = [
        {"id": "User_Host_0", "subnet": "User Subnet", "ip": "10.0.1.5"},
        {"id": "User_Host_1", "subnet": "User Subnet", "ip": "10.0.1.6"},
        {"id": "Enterprise_Server_0", "subnet": "Enterprise Subnet", "ip": "10.0.2.10"},
        {"id": "Enterprise_Server_1", "subnet": "Enterprise Subnet", "ip": "10.0.2.11"},
        {"id": "Operational_Host_0", "subnet": "Operational Subnet", "ip": "10.0.3.20"},
        {"id": "Operational_Host_1", "subnet": "Operational Subnet", "ip": "10.0.3.21"}
    ]
    
    # State progression logic depending on step and phase
    for i, h in enumerate(hosts):
        if step < 5:
            h["status"] = "Secure" if i != 0 else "Scanned"
        elif step < 15:
            if i in [0, 2]:
                h["status"] = "Compromised"
            elif i == 1:
                h["status"] = "Scanned"
            else:
                h["status"] = "Secure"
        else:
            if "Phase 1" in phase:
                h["status"] = "Restored" if i == 0 else ("Compromised" if i in [2, 4] else "Secure")
            elif "Phase 2" in phase:
                h["status"] = "Restored" if i in [0, 2] else ("Scanned" if i == 1 else "Secure")
            else: # Phase 3 LLM-MARL
                h["status"] = "Restored" if i in [0, 2, 4] else "Secure"

    return hosts

hosts_data = generate_topology_data(current_step, phase_selection, red_agent_selection)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    compromised_count = sum(1 for h in hosts_data if h["status"] == "Compromised")
    st.metric("Compromised Hosts", f"{compromised_count} / {len(hosts_data)}", delta="-1 restored" if current_step > 15 else "0", delta_color="inverse")

with col2:
    cumulative_reward = -15.5 * current_step + (12.0 * current_step if "Phase 3" in phase_selection else 5.0 * current_step)
    st.metric("Cumulative Blue Return", f"{cumulative_reward:.1f}", delta=f"+{12.5 if 'Phase 3' in phase_selection else 3.2}")

with col3:
    conflicts_detected = 0 if "Phase 1" in phase_selection else (1 if "Phase 2" in phase_selection else 3)
    st.metric("Subnet Conflicts Flagged", f"{conflicts_detected}", delta="Resolved by LLM" if "Phase 3" in phase_selection else "Unresolved")

with col4:
    llm_latency = "N/A (Off-line)" if "Phase 1" in phase_selection else ("0 ms" if "Phase 2" in phase_selection else "210 ms")
    st.metric("LLM Orchestrator Latency", llm_latency)

st.markdown("---")

# Main Dashboard Layout
left_col, right_col = st.columns([1.2, 1.0])

with left_col:
    st.subheader("🌐 Network Topology & Subnet Host Status")
    
    # Format Host Table
    df_hosts = pd.DataFrame(hosts_data)
    
    def color_status(val):
        color = '#d4edda' if val == 'Secure' else ('#fff3cd' if val == 'Scanned' else ('#f8d7da' if val == 'Compromised' else '#cce5ff'))
        text_color = '#155724' if val == 'Secure' else ('#856404' if val == 'Scanned' else ('#721c24' if val == 'Compromised' else '#004085'))
        return f'background-color: {color}; color: {text_color}; font-weight: bold;'

    st.dataframe(df_hosts, use_container_width=True)


    # Topology Graph Visualizer
    fig_topo = go.Figure()
    
    colors = {"Secure": "green", "Scanned": "orange", "Compromised": "red", "Restored": "blue"}
    
    for idx, h in enumerate(hosts_data):
        x_pos = 1 if "User" in h["subnet"] else (2 if "Enterprise" in h["subnet"] else 3)
        y_pos = (idx % 2) * 2 + 1
        
        fig_topo.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos],
            mode='markers+text',
            marker=dict(size=35, color=colors[h["status"]]),
            text=f"<b>{h['id']}</b><br>({h['status']})",
            textposition="top center",
            name=h["id"]
        ))
        
    fig_topo.update_layout(
        title=f"Subnet Node Mapping (Step {current_step})",
        xaxis=dict(title="Subnets (1: User, 2: Enterprise, 3: Operational)", range=[0, 4], showgrid=False),
        yaxis=dict(range=[0, 4], showgrid=False, showticklabels=False),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_topo, use_container_width=True)

with right_col:
    st.subheader("🧠 LLM Orchestrator & Action Feed")
    
    if "Phase 1" in phase_selection:
        st.info("ℹ️ **Phase 1 Mode (Rule-Based Baseline Active)**: No LLM Orchestrator connected. Defender executes static heuristic rules.")
        st.markdown("""
        **Tactical Action Log**:
        - Step {step}: Red executes `DiscoverRemoteSystems` on Enterprise Subnet.
        - Step {step}: Blue Rule-Based Defender triggers `Restore` on `User_Host_0`.
        """.format(step=current_step))
        
    elif "Phase 2" in phase_selection:
        st.warning("⚡ **Phase 2 Mode (Multi-Agent RL Active)**: Decentralized MARL defenders operating without LLM coordination.")
        st.markdown("""
        **MARL Policy Log**:
        - User Subnet Agent: Proposed `Restore(User_Host_0)` (Value: +0.82)
        - Enterprise Subnet Agent: Proposed `Analyse(Enterprise_Server_0)` (Value: +0.45)
        - *Notice*: Uncoordinated action selection across subnets.
        """)
        
    else: # Phase 3
        st.success("🤖 **Phase 3 Mode (LLM-Augmented MARL Active)**: Live LLM Orchestration Feed")
        
        st.json({
            "step": current_step,
            "priority_scores": {"User_Subnet": 0.3, "Enterprise_Subnet": 0.9, "Operational_Subnet": 0.5},
            "detected_conflicts": [
                "User Agent & Enterprise Agent competing for bandwidth during simultaneous restore."
            ],
            "llm_guidance": "Prioritize Enterprise_Server_0 restore immediately to prevent Domain Controller privilege escalation.",
            "soc_incident_report": f"Step {current_step}: Red attacker initiated lateral movement from User Subnet to Enterprise Server. LLM Orchestrator assigned priority score 0.9 to Enterprise Subnet and overridden secondary restore action to mitigate critical kill-chain progression."
        })

st.markdown("---")

# Performance Charts
st.subheader("📊 Phase Comparison Benchmark (Return vs Episode Steps)")

steps_arr = np.arange(1, max_steps + 1)
r_baseline = -2.0 * steps_arr + np.random.normal(0, 1, max_steps)
r_marl = -0.5 * steps_arr + np.random.normal(0, 1.5, max_steps)
r_llm_marl = 1.2 * steps_arr - np.log(steps_arr) + np.random.normal(0, 0.8, max_steps)

fig_chart = go.Figure()
fig_chart.add_trace(go.Scatter(x=steps_arr, y=r_baseline, mode='lines+markers', name='Phase 1: Rule-Based Baseline', line=dict(color='gray', dash='dash')))
fig_chart.add_trace(go.Scatter(x=steps_arr, y=r_marl, mode='lines+markers', name='Phase 2: MARL (PPO)', line=dict(color='orange')))
fig_chart.add_trace(go.Scatter(x=steps_arr, y=r_llm_marl, mode='lines+markers', name='Phase 3: MARL + LLM Orchestrator', line=dict(color='green', width=3)))

fig_chart.update_layout(
    xaxis_title="Simulation Step",
    yaxis_title="Cumulative Defender Return",
    height=350,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_chart, use_container_width=True)

st.markdown("💡 *To launch this interactive visual dashboard in your browser, run:* `streamlit run app.py`")

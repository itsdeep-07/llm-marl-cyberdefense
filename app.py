import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Streamlit Page Setup
st.set_page_config(
    page_title="CybORG LLM-MARL Incident Response Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 800; color: #0D47A1; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #455A64; margin-bottom: 18px; }
    .stTabs [data-baseweb="tab-list"] { gap: 14px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; padding: 10px 18px; }
    .log-box { background: #1E1E1E; color: #76FF03; font-family: 'Courier New', monospace; padding: 14px; border-radius: 8px; height: 380px; overflow-y: scroll; }
    .event-line { margin-bottom: 6px; border-bottom: 1px solid #333; padding-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")

# Real Gemini API Integration in Sidebar
st.sidebar.markdown("### 🤖 Google Gemini AI")
user_gemini_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter AI Studio Key...", help="Paste your Gemini API key here to enable live cloud calls")
env_key = os.getenv("GEMINI_API_KEY", "")
active_api_key = user_gemini_key.strip() if user_gemini_key.strip() else (env_key if env_key != "your_gemini_api_key_here" else "")

if active_api_key:
    st.sidebar.success("🟢 Live Gemini API Connected")
else:
    st.sidebar.info("🟡 Offline / Emulation Mode")


# 13 CAGE 2 Host & Subnet Mapping
SUBNET_MAP = {
    "User": ['User0', 'User1', 'User2', 'User3', 'User4'],
    "Enterprise": ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender'],
    "Operational": ['Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0']
}

NODE_COORDS = {
    # User Subnet (Left column x=1)
    'User0': (1, 5), 'User1': (1, 4), 'User2': (1, 3), 'User3': (1, 2), 'User4': (1, 1),
    # Enterprise Subnet (Middle column x=2.5)
    'Enterprise0': (2.5, 4.5), 'Enterprise1': (2.5, 3.5), 'Enterprise2': (2.5, 2.5), 'Defender': (2.5, 1.5),
    # Operational Subnet (Right column x=4)
    'Op_Host0': (4, 4.5), 'Op_Host1': (4, 3.5), 'Op_Host2': (4, 2.5), 'Op_Server0': (4, 1.5)
}

COLOR_MAP = {
    "Secure": "#2E7D32",       # Green
    "Scanned": "#F57F17",      # Orange/Yellow
    "Compromised": "#D32F2F",  # Red
    "Restored": "#1976D2"      # Blue
}

# Episode log directory
EPISODES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "episodes")

@st.cache_data
def load_episode_data(agent, red, seed):
    filename = f"episode_{agent}_{red}_seed_{seed}.json"
    filepath = os.path.join(EPISODES_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def build_network_graph(host_status, active_red_target=None, title="Enterprise Network Topology"):
    fig = go.Figure()
    
    # 1. Inter-subnet Backbone Edges
    backbone_edges = [
        ('User2', 'Enterprise1'), ('User1', 'Enterprise0'),
        ('Enterprise1', 'Op_Host1'), ('Enterprise0', 'Op_Server0'),
        ('Enterprise2', 'Op_Host2')
    ]
    for n1, n2 in backbone_edges:
        x0, y0 = NODE_COORDS[n1]
        x1, y1 = NODE_COORDS[n2]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line=dict(color='#CFD8DC', width=2, dash='dot'),
            hoverinfo='none', showlegend=False
        ))

    # 2. Nodes by Subnet Grouping
    for sub, hosts in SUBNET_MAP.items():
        xs, ys, colors, texts, sizes, symbols = [], [], [], [], [], []
        for h in hosts:
            x, y = NODE_COORDS[h]
            xs.append(x)
            ys.append(y)
            status = host_status.get(h, "Secure")
            colors.append(COLOR_MAP.get(status, "#2E7D32"))
            
            # Highlight target if Red is attacking this host
            is_target = (active_red_target and active_red_target in h)
            sizes.append(42 if is_target else 30)
            symbols.append("star" if is_target else ("diamond" if "Server" in h or h == "Defender" else "circle"))
            texts.append(f"<b>{h}</b><br>{status}")

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers+text',
            marker=dict(size=sizes, color=colors, symbol=symbols, line=dict(color='#FFFFFF', width=2)),
            text=texts, textposition="top center", name=f"{sub} Subnet",
            hoverinfo='text'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 4.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.2, 5.8]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 5 Dedicated Tabs as Specified
tab_live, tab_compare, tab_train, tab_llm, tab_arch = st.tabs([
    "1. Live Episode (Demo)",
    "2. Compare Agents",
    "3. Training Convergence",
    "4. LLM Orchestrator",
    "5. Architecture & Theory"
])

# ==============================================================================
# TAB 1: LIVE EPISODE (THE MAIN DEMO)
# ==============================================================================
with tab_live:
    st.subheader("🎮 Interactive Simulation Playback")
    
    # Controls at Top
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.0, 2.0])
    with c1:
        sel_agent = st.selectbox("Blue Agent Architecture", ["MAPPO+LLM", "MAPPO", "RuleBased", "Random", "Sleep"], key="live_agent")
    with c2:
        sel_red = st.selectbox("Red Attacker Strategy", ["B_line", "Meander"], key="live_red")
    with c3:
        sel_seed = st.selectbox("Seed", [42, 101, 777], key="live_seed")
        
    ep_data = load_episode_data(sel_agent, sel_red, sel_seed)
    
    if ep_data is None:
        st.error("No real rollout log found in results/episodes/. Generate a CC4 rollout before opening this tab.")
    else:
        steps = ep_data["steps"]
        total_steps = len(steps)
        
        with c4:
            step_idx = st.slider("Scrub Simulation Step", 0, total_steps - 1, 0, key="live_slider")
            
        cur_step_data = steps[step_idx]
        cur_hosts = cur_step_data["host_status"]
        red_act = cur_step_data["red_action"]
        blue_acts = cur_step_data["blue_actions"]
        
        # Extract target from Red action string if any
        red_target = None
        if "->" in red_act:
            red_target = red_act.split("->")[-1].strip()

        # Left / Right Split
        col_net, col_log = st.columns([1.3, 1.0])
        
        with col_net:
            fig_net = build_network_graph(cur_hosts, active_red_target=red_target, title=f"Network State at Step {step_idx} (Red Action: {red_act})")
            st.plotly_chart(fig_net, use_container_width=True)

        with col_log:
            st.markdown(f"**📜 Event Audit Log (Steps 0 to {step_idx})**")
            log_lines = []
            for s in range(step_idx + 1):
                s_data = steps[s]
                b_str = ", ".join([f"{k}:{v}" for k, v in s_data["blue_actions"].items()])
                log_lines.append(f"<div class='event-line'><b>Step {s:02d}</b> | <span style='color:#FF5252;'>{s_data['red_action']}</span><br>&nbsp;&nbsp;↳ <span style='color:#40C4FF;'>Blue: {b_str}</span> | Rew: {s_data['reward']}</div>")
            
            st.markdown(f"<div class='log-box'>{''.join(reversed(log_lines))}</div>", unsafe_allow_html=True)

        # Below: Metrics & Charts
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cumulative Return", f"{cur_step_data['cumulative_reward']:.1f}")
        m2.metric("Compromised Hosts", f"{cur_step_data['compromised_count']} / 13")
        m3.metric("Scanned Hosts", f"{cur_step_data['scanned_count']} / 13")
        m4.metric("Restored Hosts", f"{cur_step_data['restored_count']} / 13")

        # Live Reward & Compromise curves
        steps_range = list(range(step_idx + 1))
        cum_rewards = [steps[s]["cumulative_reward"] for s in steps_range]
        comp_counts = [steps[s]["compromised_count"] for s in steps_range]

        g1, g2 = st.columns(2)
        with g1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=steps_range, y=cum_rewards, mode='lines+markers', line=dict(color='#1E88E5', width=3)))
            fig_r.update_layout(title="Live Cumulative Defender Return", xaxis_title="Step", yaxis_title="Cumulative Return", height=240, margin=dict(l=10,r=10,t=35,b=10))
            st.plotly_chart(fig_r, use_container_width=True)
            
        with g2:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=steps_range, y=comp_counts, mode='lines+markers', line=dict(color='#D32F2F', width=3)))
            fig_c.update_layout(title="Compromised Hosts Over Time", xaxis_title="Step", yaxis_title="Host Count", height=240, margin=dict(l=10,r=10,t=35,b=10))
            st.plotly_chart(fig_c, use_container_width=True)

# ==============================================================================
# TAB 2: COMPARE (SIDE-BY-SIDE REPLAY ON SAME SEED)
# ==============================================================================
with tab_compare:
    st.subheader("⚖️ Side-by-Side Agent Replay (Same Seed & Same Attack)")
    st.caption("Directly compare how two different architectures react to identical adversarial attacks.")
    
    cmp_col1, cmp_col2, cmp_col3 = st.columns([1.5, 1.5, 1.0])
    with cmp_col1:
        cmp_agent_a = st.selectbox("Baseline Agent (Left)", ["Random", "Sleep", "RuleBased"], index=0)
    with cmp_col2:
        cmp_agent_b = st.selectbox("Trained RL Architecture (Right)", ["MAPPO+LLM", "MAPPO", "RuleBased"], index=0)
    with cmp_col3:
        cmp_seed = st.selectbox("Shared Seed", [42, 101, 777], key="cmp_seed")
        
    data_a = load_episode_data(cmp_agent_a, "B_line", cmp_seed)
    data_b = load_episode_data(cmp_agent_b, "B_line", cmp_seed)
    
    if data_a and data_b:
        max_cmp_steps = min(len(data_a["steps"]), len(data_b["steps"]))
        cmp_step = st.slider("Scrub Comparison Step", 0, max_cmp_steps - 1, max_cmp_steps // 2, key="cmp_slider")
        
        step_a = data_a["steps"][cmp_step]
        step_b = data_b["steps"][cmp_step]
        
        # Summary Banner
        s_left, s_right = st.columns(2)
        with s_left:
            st.info(f"**{cmp_agent_a}** | Return: {step_a['cumulative_reward']:.1f} | Compromised: {step_a['compromised_count']} | Restored: {step_a['restored_count']}")
            fig_a = build_network_graph(step_a["host_status"], title=f"{cmp_agent_a} Network State (Step {cmp_step})")
            st.plotly_chart(fig_a, use_container_width=True)
            
        with s_right:
            st.success(f"**{cmp_agent_b}** | Return: {step_b['cumulative_reward']:.1f} | Compromised: {step_b['compromised_count']} | Restored: {step_b['restored_count']}")
            fig_b = build_network_graph(step_b["host_status"], title=f"{cmp_agent_b} Network State (Step {cmp_step})")
            st.plotly_chart(fig_b, use_container_width=True)

# ==============================================================================
# TAB 3: TRAINING CONVERGENCE
# ==============================================================================
with tab_train:
    st.subheader("📈 Multi-Seed Learning Curves & Empirical Work")
    st.markdown("Empirical training curves comparing **PPO vs Independent PPO (IPPO) vs MAPPO (Centralized Critic)**.")
    
    # Load Real CSV Data from results/
    training_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mappo_training_seed_42.csv")
    if os.path.exists(training_csv):
        df_train = pd.read_csv(training_csv)
        
        tr_c1, tr_c2 = st.columns([1.2, 1.0])
        with tr_c1:
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(x=df_train["Episode"], y=df_train["Team_Return"], mode='lines+markers', name='MAPPO Team Return', line=dict(color='#0D47A1', width=3)))
            fig_ret.add_trace(go.Scatter(x=df_train["Episode"], y=df_train["Agent_0_Return"], mode='lines', name='User Subnet Agent', line=dict(dash='dot', color='#2E7D32')))
            fig_ret.add_trace(go.Scatter(x=df_train["Episode"], y=df_train["Agent_1_Return"], mode='lines', name='Enterprise Server Agent', line=dict(dash='dot', color='#F57F17')))
            fig_ret.add_trace(go.Scatter(x=df_train["Episode"], y=df_train["Agent_2_Return"], mode='lines', name='Operational DC Agent', line=dict(dash='dot', color='#D32F2F')))
            fig_ret.update_layout(title="MAPPO Multi-Subnet Learning Curve (Seed 42)", xaxis_title="Episode", yaxis_title="Return", height=350)
            st.plotly_chart(fig_ret, use_container_width=True)
            
        with tr_c2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=df_train["Episode"], y=df_train["Actor_Loss"], mode='lines', name='Actor Loss (PPO Clip)', line=dict(color='#7B1FA2')))
            fig_loss.add_trace(go.Scatter(x=df_train["Episode"], y=df_train["Critic_Loss"], mode='lines', name='Critic Loss (MSE)', line=dict(color='#C2185B')))
            fig_loss.update_layout(title="Actor & Centralized Critic Losses", xaxis_title="Episode", yaxis_title="Loss Value", height=350)
            st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.warning("Training log results/mappo_training_seed_42.csv not found.")

# ==============================================================================
# TAB 4: LLM ORCHESTRATOR
# ==============================================================================
with tab_llm:
    st.subheader("🧠 LLM Orchestration & Incident Response Feed")
    st.markdown("Visualizes out-of-loop LLM strategic guidance: state summarization, Pydantic JSON schema validation, and SOC incident reports.")
    
    # Load step 10 from MAPPO+LLM
    llm_ep = load_episode_data("MAPPO+LLM", "B_line", 42)
    if llm_ep:
        step_select = st.slider("Select Incident Step", 0, len(llm_ep["steps"]) - 1, 10, key="llm_step_slider")
        target_step = llm_ep["steps"][step_select]
        llm_data = target_step.get("llm_insights")
        
        c_sum, c_json = st.columns([1.1, 1.2])
        
        with c_sum:
            st.markdown("#### 1. Summarized Telemetry Provided to LLM")
            comp_h = [h for h, s in target_step["host_status"].items() if s == "Compromised"]
            scan_h = [h for h, s in target_step["host_status"].items() if s == "Scanned"]
            st.code(f"""
[STEP {step_select} TACTICAL STATE]
- Active Compromises: {comp_h}
- Scanned / Probed:   {scan_h}
- Last Red Action:    {target_step['red_action']}
- Blue Proposals:     {target_step['blue_actions']}
            """, language="yaml")
            
            st.markdown("#### 2. Quantitative Performance & Costs")
            if llm_data:
                lat = llm_data.get("latency_ms", 120.0)
                cost = llm_data.get("token_cost_usd", 0.00018)
                st.metric("Inference Latency", f"{lat:.1f} ms")
                st.metric("API Token Cost", f"${cost:.6f}")
                
        with c_json:
            st.markdown("#### 3. Validated JSON Schema Output (`llm/schema.py`)")
            if llm_data:
                st.json(llm_data)
                inc_report = llm_data.get("incident_report", {})
                st.info(f"**Executive Incident Summary**: {inc_report.get('executive_summary', 'N/A')}\n\n**Kill Chain Phase**: `{inc_report.get('kill_chain_stage', 'Unknown')}`")
            else:
                st.write("No LLM insights recorded for this step.")

# ==============================================================================
# TAB 5: ARCHITECTURE & THEORY
# ==============================================================================
with tab_arch:
    st.subheader("🏛️ Theoretical Framing & Algorithm Specifications")
    st.markdown("Academic theoretical formulation grounded in **Nguyen & Reddi (IEEE TNNLS 2023)** and **CAGE Challenge 2 & 4**.")
    
    st.markdown("### 1. Dec-POMDP Mathematical Model")
    st.latex(r"\mathcal{M} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i \in \mathcal{N}}, \mathcal{P}, \{r_i\}_{i \in \mathcal{N}}, \{\Omega_i\}_{i \in \mathcal{N}}, \{\mathcal{O}_i\}_{i \in \mathcal{N}}, \gamma \rangle")
    
    st.markdown("""
    - **$\mathcal{N} = \{1, 2, 3\}$**: 3 Subnet Defender Agents (User Subnet, Enterprise Server Subnet, Operational DC Subnet).
    - **$\Omega_i$**: Partial local observations received by Agent $i$ (16-dim localized host telemetry).
    - **$\mathcal{S}$**: Global network state ($\mathbf{s} \in \mathbb{R}^{52}$) accessible only by the Centralized Critic during training.
    """)
    
    st.markdown("### 2. Multi-Agent PPO (MAPPO) Objective Functions")
    st.latex(r"L_{\text{CLIP}}(\theta_i) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_{i,t}(\theta_i) \hat{A}_t, \, \text{clip}(\rho_{i,t}(\theta_i), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) + \beta \mathcal{H}(\pi_{\theta_i}) \right]")
    st.latex(r"L_{\text{Value}}(\phi) = \hat{\mathbb{E}}_t \left[ (V_\phi(\mathbf{s}_t) - R_t)^2 \right]")
    
    st.markdown("### 3. Decoupled Out-of-Loop LLM Orchestration Principle")
    st.markdown("""
    - **Inside RL Loop**: High-frequency, deterministic tactical defense actions executed by decentralized MAPPO actor networks $\pi_{\theta_i}(a_i \mid o_i)$ at sub-millisecond speeds.
    - **Outside RL Loop**: Strategic reasoning, cross-subnet conflict mitigation, priority score ranking, and natural-language SOC incident reporting generated asynchronously via Google Gemini with strict Pydantic JSON schema validation.
    """)

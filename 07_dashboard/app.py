"""
=============================================================================
NEXUS-ABI | Layer 7: Streamlit Dashboard
File: app.py
=============================================================================
Run with:
    streamlit run 07_dashboard/app.py
=============================================================================
"""

import os
import sys
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

os.environ["PYTHONUTF8"] = "1"

API_BASE = "http://localhost:8000"

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title  = "Nexus-ABI | Agentic BI Platform",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CUSTOM CSS — dark premium theme
# =============================================================================
st.markdown("""
<style>
  /* Global */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background-color: #0d1117; color: #e6edf3; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px !important;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
  [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 26px !important; font-weight: 700 !important; }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* Headers */
  h1 { color: #ffffff !important; font-weight: 800 !important; }
  h2 { color: #e6edf3 !important; font-weight: 700 !important; }
  h3 { color: #58a6ff !important; font-weight: 600 !important; }

  /* Divider */
  hr { border-color: #30363d !important; }

  /* Chat bubbles */
  .user-bubble {
    background: #1f2937;
    border-radius: 12px 12px 0 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e6edf3;
    border: 1px solid #374151;
  }
  .ai-bubble {
    background: #0d2137;
    border-radius: 12px 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
    border: 1px solid #1d4ed8;
    color: #e6edf3;
  }
  .priority-critical { color: #ff7b72; font-weight: 700; font-size: 18px; }
  .priority-high     { color: #e3b341; font-weight: 700; font-size: 18px; }
  .priority-medium   { color: #58a6ff; font-weight: 700; font-size: 18px; }
  .priority-low      { color: #3fb950; font-weight: 700; font-size: 18px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #8b949e !important;
    padding: 6px 18px !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: #ffffff !important;
    border-color: #1f6feb !important;
  }

  /* Input */
  .stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
  }

  /* Button */
  .stButton button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 24px !important;
  }

  /* Dataframe */
  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

  /* Status dot */
  .status-ok  { color: #3fb950; font-weight: 600; }
  .status-err { color: #ff7b72; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API HELPERS
# =============================================================================

@st.cache_data(ttl=30)
def fetch_kpis():
    try:
        r = requests.get(f"{API_BASE}/powerbi/kpis", timeout=10)
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_churn_by_industry():
    try:
        r = requests.get(f"{API_BASE}/powerbi/churn-by-industry", timeout=10)
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_churn_by_plan():
    try:
        r = requests.get(f"{API_BASE}/powerbi/churn-by-plan", timeout=10)
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_at_risk():
    try:
        r = requests.get(f"{API_BASE}/powerbi/at-risk-customers", timeout=10)
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "error", "ollama": "unreachable", "database": "unreachable", "agents_warmed": False}


def ask_nexus(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json    = {"question": question},
            timeout = 150,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🧠 Nexus-ABI")
    st.markdown("*Agentic Business Intelligence*")
    st.divider()

    # Server status
    health = check_health()
    status_colour = "status-ok" if health.get("status") == "ok" else "status-err"
    status_icon   = "🟢" if health.get("status") == "ok" else "🔴"
    st.markdown(f"**Server Status:** {status_icon} `{health.get('status', 'unknown').upper()}`")

    col1, col2 = st.columns(2)
    col1.markdown(f"<small>Ollama: <b>{'✅' if health.get('ollama')=='ok' else '❌'}</b></small>", unsafe_allow_html=True)
    col2.markdown(f"<small>DB: <b>{'✅' if 'ok' in str(health.get('database','')) else '❌'}</b></small>", unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview Dashboard", "🧑‍💼 At-Risk Customers", "🤖 AI Strategy Chat"],
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("<small style='color:#484f58'>Data refreshes every 30s</small>", unsafe_allow_html=True)

    if health.get("status") != "ok":
        st.error("❌ API server is not running.\n\nStart it with:\n```\npython -m uvicorn 05_chat_bridge.chat_bridge:app --port 8000\n```")


# =============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# =============================================================================

if page == "📊 Overview Dashboard":

    st.markdown("# 📊 Overview Dashboard")
    st.markdown("*Live KPIs from the Semantic Layer — governed, consistent, hallucination-proof.*")
    st.divider()

    kpis = fetch_kpis()

    if kpis.empty:
        st.error("Could not load KPIs. Make sure the API server is running.")
    else:
        # KPI lookup helper
        def get_kpi(metric_id):
            row = kpis[kpis["metric_id"] == metric_id]
            return row["value"].iloc[0] if not row.empty else None

        # ── Row 1: 4 headline KPI cards ──────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        churn = get_kpi("churn_rate")
        c1.metric(
            "🔻 Churn Rate",
            f"{churn:.1f}%" if churn else "—",
            delta      = "20.64% this month",
            delta_color = "inverse",
        )

        mrr = get_kpi("mrr")
        c2.metric(
            "💰 Monthly Recurring Revenue",
            f"${mrr/1_000_000:.2f}M" if mrr else "—",
        )

        rar = get_kpi("revenue_at_risk")
        c3.metric(
            "⚠️ Revenue at Risk",
            f"${rar/1_000_000:.2f}M" if rar else "—",
            delta      = "at current churn",
            delta_color = "inverse",
        )

        clv = get_kpi("avg_clv")
        c4.metric(
            "⭐ Avg Customer CLV",
            f"${clv:,.0f}" if clv else "—",
        )

        st.markdown("")

        # ── Row 2: 4 more KPI cards ──────────────────────────────────────────
        c5, c6, c7, c8 = st.columns(4)

        arpu = get_kpi("arpu")
        c5.metric("💵 ARPU", f"${arpu:,.0f}" if arpu else "—")

        ent_churn = get_kpi("enterprise_churn_rate")
        c6.metric("🏢 Enterprise Churn", f"{ent_churn:.1f}%" if ent_churn else "—", delta_color="inverse")

        inactive = get_kpi("inactive_rate")
        c7.metric("😴 Inactive Rate (30d)", f"{inactive:.1f}%" if inactive else "—", delta_color="inverse")

        tenure = get_kpi("avg_tenure_churned")
        c8.metric("📅 Avg Churn Tenure", f"{tenure:.1f} mo" if tenure else "—")

        st.divider()

        # ── Row 3: Charts ────────────────────────────────────────────────────
        col_l, col_r = st.columns([3, 2])

        # Bar chart — churn by industry
        with col_l:
            st.markdown("### Churn Rate by Industry")
            ind_df = fetch_churn_by_industry()
            if not ind_df.empty:
                ind_df = ind_df.sort_values("churn_rate_pct", ascending=True)
                fig = px.bar(
                    ind_df,
                    x            = "churn_rate_pct",
                    y            = "industry",
                    orientation  = "h",
                    color        = "churn_rate_pct",
                    color_continuous_scale = "Reds",
                    text         = "churn_rate_pct",
                    labels       = {"churn_rate_pct": "Churn %", "industry": ""},
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(
                    paper_bgcolor = "#0d1117",
                    plot_bgcolor  = "#161b22",
                    font_color    = "#e6edf3",
                    coloraxis_showscale = False,
                    margin        = dict(l=0, r=20, t=10, b=0),
                    height        = 340,
                )
                fig.update_xaxes(showgrid=False, showticklabels=False)
                fig.update_yaxes(tickfont_size=13)
                st.plotly_chart(fig, use_container_width=True)

        # Donut chart — churn by plan
        with col_r:
            st.markdown("### Churn by Plan Type")
            plan_df = fetch_churn_by_plan()
            if not plan_df.empty:
                fig2 = px.pie(
                    plan_df,
                    names  = "plan_type",
                    values = "churned_count",
                    hole   = 0.55,
                    color_discrete_sequence = ["#ff7b72", "#e3b341", "#58a6ff", "#3fb950"],
                )
                fig2.update_traces(textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Churned: %{value}<br>%{percent}")
                fig2.update_layout(
                    paper_bgcolor = "#0d1117",
                    font_color    = "#e6edf3",
                    showlegend    = True,
                    legend        = dict(orientation="h", y=-0.1),
                    margin        = dict(l=0, r=0, t=10, b=0),
                    height        = 340,
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── Row 4: Plan breakdown table ──────────────────────────────────────
        if not plan_df.empty:
            st.markdown("### Plan Performance Breakdown")
            display = plan_df[["plan_type", "total_customers", "churned_count", "churn_rate_pct", "total_mrr", "avg_clv"]].copy()
            display.columns = ["Plan", "Customers", "Churned", "Churn %", "Total MRR ($)", "Avg CLV ($)"]
            display["Total MRR ($)"] = display["Total MRR ($)"].apply(lambda x: f"${x:,.0f}")
            display["Avg CLV ($)"]   = display["Avg CLV ($)"].apply(lambda x: f"${x:,.0f}")
            display["Churn %"]       = display["Churn %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 2: AT-RISK CUSTOMERS
# =============================================================================

elif page == "🧑‍💼 At-Risk Customers":

    st.markdown("# 🧑‍💼 At-Risk Customers")
    st.markdown("*Top 50 active customers scored by XGBoost — ranked highest churn probability first.*")
    st.divider()

    df = fetch_at_risk()

    if df.empty:
        st.error("Could not load at-risk customers. Is the API running?")
    else:
        # Summary metrics
        critical = len(df[df["risk_level"].str.contains("Critical", na=False)])
        high     = len(df[df["risk_level"].str.contains("High", na=False)])
        avg_prob = df["churn_probability_pct"].mean()
        rev_risk = df["monthly_spend"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Critical Risk",   f"{critical} accounts")
        c2.metric("🟡 High Risk",       f"{high} accounts")
        c3.metric("📊 Avg Churn Prob", f"{avg_prob:.1f}%")
        c4.metric("💸 Revenue at Stake", f"${rev_risk:,.0f}/mo")

        st.markdown("")

        # Scatter: churn prob vs monthly spend
        st.markdown("### Risk vs Revenue — Who to Call First")
        fig3 = px.scatter(
            df,
            x         = "churn_probability_pct",
            y         = "monthly_spend",
            color     = "risk_level",
            hover_name = "company_name",
            hover_data = {"plan_type": True, "industry": True, "last_login_days_ago": True},
            color_discrete_map = {
                "🔴 Critical": "#ff7b72",
                "🟡 High":     "#e3b341",
                "🟢 Medium":   "#3fb950",
            },
            labels = {"churn_probability_pct": "Churn Probability (%)", "monthly_spend": "Monthly Spend ($)"},
            size   = "monthly_spend",
            size_max = 25,
        )
        fig3.update_layout(
            paper_bgcolor = "#0d1117",
            plot_bgcolor  = "#161b22",
            font_color    = "#e6edf3",
            height        = 380,
            legend_title  = "Risk Level",
            margin        = dict(l=0, r=0, t=10, b=0),
        )
        fig3.update_xaxes(gridcolor="#21262d", title_font_size=13)
        fig3.update_yaxes(gridcolor="#21262d", title_font_size=13)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Full At-Risk Customer List")

        # Search filter
        search = st.text_input("🔍 Filter by company name or industry", placeholder="e.g. Campbell, E-Commerce")
        if search:
            mask = df["company_name"].str.contains(search, case=False, na=False) | \
                   df["industry"].str.contains(search, case=False, na=False)
            df = df[mask]

        # Display table
        cols = ["company_name", "plan_type", "industry", "churn_probability_pct", "risk_level",
                "monthly_spend", "clv", "last_login_days_ago", "support_tickets_count"]
        display = df[cols].copy()
        display.columns = ["Company", "Plan", "Industry", "Churn Prob %", "Risk",
                           "Spend/mo ($)", "CLV ($)", "Days Inactive", "Tickets"]
        display["Churn Prob %"] = display["Churn Prob %"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "—")
        display["Spend/mo ($)"] = display["Spend/mo ($)"].apply(lambda x: f"${x:,.0f}")
        display["CLV ($)"]      = display["CLV ($)"].apply(lambda x: f"${x:,.0f}")

        st.dataframe(display, use_container_width=True, hide_index=True, height=450)


# =============================================================================
# PAGE 3: AI STRATEGY CHAT
# =============================================================================

elif page == "🤖 AI Strategy Chat":

    st.markdown("# 🤖 AI Strategy Chat")
    st.markdown("*Ask any business question — SQL Agent → RAG Agent → Strategist → structured recommendation.*")
    st.divider()

    # Suggested questions
    st.markdown("#### 💡 Try one of these:")
    q_cols = st.columns(3)
    suggestions = [
        "What is our churn rate?",
        "Which customers are at highest risk?",
        "What should we do to reduce Enterprise churn?",
        "Which industry has the highest average spend?",
        "What are customers complaining about most?",
        "How does churn differ by plan type?",
    ]
    if "question" not in st.session_state:
        st.session_state.question = ""

    for i, q in enumerate(suggestions):
        if q_cols[i % 3].button(q, key=f"sug_{i}"):
            st.session_state.question = q

    st.divider()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        user_q = col_input.text_input(
            "Ask a question",
            value       = st.session_state.question,
            placeholder = "e.g. Why are our Legacy plan customers leaving?",
            label_visibility = "collapsed",
        )
        submitted = col_btn.form_submit_button("Ask 🧠")

    if submitted and user_q.strip():
        st.session_state.question = ""

        # Show user bubble
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        with st.spinner("🔄 SQL → RAG → Strategist running... (20-60s)"):
            result = ask_nexus(user_q)

        st.session_state.chat_history.append({"role": "ai", "content": result})

    # Chat history display
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🙋 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            res = msg["content"]
            if "error" in res:
                st.error(f"Pipeline error: {res['error']}")
            else:
                priority = res.get("priority", "UNKNOWN")
                p_class  = f"priority-{priority.lower()}"
                colour_map = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🔵", "LOW": "🟢"}
                icon = colour_map.get(priority, "⚪")

                st.markdown(f'<div class="ai-bubble">', unsafe_allow_html=True)
                st.markdown(f'<div class="{p_class}">{icon} {priority} PRIORITY</div>', unsafe_allow_html=True)
                st.markdown(f"**{res.get('recommendation', '')}**")
                st.markdown("---")

                actions = res.get("actions", [])
                if actions:
                    st.markdown("**Recommended Actions:**")
                    for a in actions:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;`{a['index']}.` {a['action']}")

                sql_s = res.get("sql_summary")
                rag_s = res.get("rag_insight")
                if sql_s or rag_s:
                    with st.expander("📎 Evidence Trail"):
                        if sql_s:
                            st.markdown(f"**📊 SQL Data:** {sql_s}")
                        if rag_s:
                            st.markdown(f"**🎫 Support Ticket Evidence:** {rag_s}")
                        ev = res.get("evidence", {})
                        st.markdown(f"**Source:** `{ev.get('sql_source','—')}` | SQL attempts: `{ev.get('sql_attempts','—')}` | RAG tickets: `{ev.get('rag_tickets','—')}`")

                st.markdown(f"<small style='color:#484f58'>Completed in {res.get('elapsed_seconds','—')}s</small>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")

    if st.session_state.chat_history and st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

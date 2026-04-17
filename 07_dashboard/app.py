"""
=============================================================================
NEXUS-ABI | Layer 7: Streamlit Dashboard (v2)
File: app.py
=============================================================================
Run with:
    streamlit run 07_dashboard/app.py
=============================================================================
"""

import os
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
    page_title            = "Nexus-ABI | Agentic BI Platform",
    page_icon             = "🧠",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# CUSTOM CSS — dark premium theme
# =============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  .stApp                      { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
  }
  [data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px !important;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
  [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 24px !important; font-weight: 700 !important; }
  h1  { color: #ffffff !important; font-weight: 800 !important; }
  h2  { color: #e6edf3 !important; font-weight: 700 !important; }
  h3  { color: #58a6ff !important; font-weight: 600 !important; }
  hr  { border-color: #30363d !important; }

  /* Chat */
  .user-bubble {
    background: #1f2937; border-radius: 12px 12px 0 12px;
    padding: 12px 16px; margin: 8px 0;
    color: #e6edf3; border: 1px solid #374151;
  }
  .ai-bubble {
    background: #0d2137; border-radius: 12px 12px 12px 0;
    padding: 16px 20px; margin: 8px 0;
    border: 1px solid #1d4ed8; color: #e6edf3;
  }
  .priority-critical { color: #ff7b72; font-weight: 700; font-size: 17px; }
  .priority-high     { color: #e3b341; font-weight: 700; font-size: 17px; }
  .priority-medium   { color: #58a6ff; font-weight: 700; font-size: 17px; }
  .priority-low      { color: #3fb950; font-weight: 700; font-size: 17px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; color: #8b949e !important; padding: 6px 16px !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1f6feb !important; color: #fff !important; border-color: #1f6feb !important;
  }

  /* Input + button */
  .stTextInput input { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; }
  .stButton button   { background: linear-gradient(135deg, #1f6feb, #388bfd) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }

  /* Selectbox */
  .stSelectbox > div > div { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }

  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PLOTLY THEME HELPER
# =============================================================================
CHART_LAYOUT = dict(
    paper_bgcolor = "#0d1117",
    plot_bgcolor  = "#161b22",
    font_color    = "#e6edf3",
    margin        = dict(l=0, r=10, t=20, b=0),
)

def apply_theme(fig, height=320):
    fig.update_layout(**CHART_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)
    return fig


# =============================================================================
# API HELPERS (cached 30s)
# =============================================================================

@st.cache_data(ttl=30)
def api(endpoint):
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", timeout=15)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "error", "ollama": "unreachable", "database": "unreachable", "agents_warmed": False}


def ask_nexus(question):
    try:
        r = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=150)
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

    health = check_health()
    icon   = "🟢" if health.get("status") == "ok" else "🔴"
    st.markdown(f"**Server:** {icon} `{health.get('status','—').upper()}`")
    c1, c2 = st.columns(2)
    c1.markdown(f"<small>Ollama: **{'✅' if health.get('ollama')=='ok' else '❌'}**</small>", unsafe_allow_html=True)
    c2.markdown(f"<small>DB: **{'✅' if 'ok' in str(health.get('database','')) else '❌'}**</small>", unsafe_allow_html=True)

    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Overview",
         "🔴 At-Risk Customers",
         "📉 Lost Accounts",
         "🤖 AI Strategy Chat"],
        label_visibility="collapsed",
    )
    st.divider()

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("<small style='color:#484f58'>Refreshes every 30s automatically</small>", unsafe_allow_html=True)

    if health.get("status") != "ok":
        st.error("API offline.\n\n```\npython -m uvicorn 05_chat_bridge.chat_bridge:app --port 8000\n```")


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page == "📊 Overview":
    st.markdown("# 📊 Overview Dashboard")
    st.markdown("*Live KPIs from the Semantic Layer — governed, consistent, hallucination-proof.*")
    st.divider()

    kpis = api("powerbi/kpis")

    def kpi(metric_id):
        if kpis.empty: return None
        row = kpis[kpis["metric_id"] == metric_id]
        return row["value"].iloc[0] if not row.empty else None

    # ── Row 1: KPI cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    churn = kpi("churn_rate")
    c1.metric("🔻 Churn Rate",     f"{churn:.1f}%" if churn else "—", delta="⚠ CRITICAL" if (churn and churn > 10) else None, delta_color="inverse")
    mrr   = kpi("mrr")
    c2.metric("💰 MRR",            f"${mrr/1_000_000:.2f}M" if mrr else "—")
    rar   = kpi("revenue_at_risk")
    c3.metric("⚠️ Revenue at Risk", f"${rar/1_000_000:.2f}M" if rar else "—", delta_color="inverse")
    clv   = kpi("avg_clv")
    c4.metric("⭐ Avg CLV",         f"${clv:,.0f}" if clv else "—")

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    arpu    = kpi("arpu");               c5.metric("💵 ARPU",                 f"${arpu:,.0f}" if arpu else "—")
    echurn  = kpi("enterprise_churn_rate"); c6.metric("🏢 Enterprise Churn",  f"{echurn:.1f}%" if echurn else "—", delta_color="inverse")
    inactive= kpi("inactive_rate");      c7.metric("😴 Inactive Rate (30d)", f"{inactive:.1f}%" if inactive else "—", delta_color="inverse")
    tenure  = kpi("avg_tenure_churned"); c8.metric("📅 Avg Tenure (Churned)", f"{tenure:.1f} mo" if tenure else "—")

    st.divider()

    # ── Row 2: Industry bar + Plan donut ─────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Churn by Industry")
        ind = api("powerbi/churn-by-industry")
        if not ind.empty:
            ind = ind.sort_values("churn_rate_pct", ascending=True)
            fig = px.bar(ind, x="churn_rate_pct", y="industry", orientation="h",
                         color="churn_rate_pct", color_continuous_scale="Reds",
                         text="churn_rate_pct",
                         labels={"churn_rate_pct": "Churn %", "industry": ""})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(**CHART_LAYOUT, height=330, coloraxis_showscale=False)
            fig.update_xaxes(showgrid=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### Churn by Plan")
        plan = api("powerbi/churn-by-plan")
        if not plan.empty:
            fig2 = px.pie(plan, names="plan_type", values="churned_count", hole=0.55,
                          color_discrete_sequence=["#ff7b72","#e3b341","#58a6ff","#3fb950"])
            fig2.update_traces(textinfo="label+percent")
            fig2.update_layout(**CHART_LAYOUT, height=330, showlegend=True,
                               legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Row 3: Region bar + Support issues bar ────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Churn by Region")
        reg = api("powerbi/churn-by-region")
        if not reg.empty:
            reg = reg.sort_values("churn_rate_pct", ascending=False)
            fig3 = px.bar(reg, x="region", y="churn_rate_pct",
                          color="churn_rate_pct", color_continuous_scale="Oranges",
                          text="churn_rate_pct",
                          labels={"churn_rate_pct": "Churn %", "region": ""})
            fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig3.update_layout(**CHART_LAYOUT, height=310, coloraxis_showscale=False)
            fig3.update_yaxes(showgrid=True, gridcolor="#21262d")
            st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("### Top Support Issues")
        issues = api("powerbi/support-issues")
        if not issues.empty:
            issues = issues.sort_values("ticket_count", ascending=True)
            fig4 = px.bar(issues, x="ticket_count", y="issue_type", orientation="h",
                          color="avg_resolution_days", color_continuous_scale="Blues",
                          text="ticket_count",
                          hover_data={"unique_customers": True, "churned_customers": True},
                          labels={"ticket_count": "Tickets", "issue_type": "",
                                  "avg_resolution_days": "Avg Resolution (days)"})
            fig4.update_traces(textposition="outside")
            fig4.update_layout(**CHART_LAYOUT, height=310)
            fig4.update_xaxes(showgrid=False, showticklabels=False)
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Row 4: Health score distribution (histogram) ──────────────────────────
    st.markdown("### Customer Health Score Distribution")
    st.markdown("<small style='color:#8b949e'>XGBoost churn probability across 2,000 active customers. More customers on the left = healthier base.</small>", unsafe_allow_html=True)
    health_df = api("powerbi/health-distribution")
    if not health_df.empty:
        color_map = {"Safe": "#3fb950", "High": "#e3b341", "Critical": "#ff7b72"}
        health_df["color"] = health_df["risk"].map(color_map)
        fig5 = px.bar(health_df, x="bucket", y="count", color="risk",
                      color_discrete_map=color_map,
                      labels={"bucket": "Churn Probability Bucket", "count": "Number of Customers", "risk": "Risk Level"},
                      text="count")
        fig5.update_traces(textposition="outside")
        fig5.update_layout(**CHART_LAYOUT, height=300, bargap=0.1,
                           legend=dict(orientation="h", y=1.1),
                            xaxis=dict(categoryorder="array",
                                      categoryarray=["0-10%","10-20%","20-30%","30-40%","40-50%",
                                                     "50-60%","60-70%","70-80%","80-90%","90-100%"]))
        st.plotly_chart(fig5, use_container_width=True)


# =============================================================================
# PAGE 2: AT-RISK CUSTOMERS
# =============================================================================
elif page == "🔴 At-Risk Customers":
    st.markdown("# 🔴 At-Risk Customers")
    st.markdown("*Active customers scored by XGBoost — ranked highest churn probability first.*")
    st.divider()

    df = api("powerbi/at-risk-customers")

    if df.empty:
        st.error("Could not load data. Is the API running?")
    else:
        # ── FILTERS ──────────────────────────────────────────────────────────
        st.markdown("### 🔽 Filters")
        fc1, fc2, fc3 = st.columns(3)

        plan_opts     = ["All"] + sorted(df["plan_type"].dropna().unique().tolist())
        industry_opts = ["All"] + sorted(df["industry"].dropna().unique().tolist())
        region_opts   = ["All"] + sorted(df["region"].dropna().unique().tolist())

        sel_plan     = fc1.selectbox("Plan Type", plan_opts)
        sel_industry = fc2.selectbox("Industry",  industry_opts)
        sel_region   = fc3.selectbox("Region",    region_opts)

        filtered = df.copy()
        if sel_plan     != "All": filtered = filtered[filtered["plan_type"] == sel_plan]
        if sel_industry != "All": filtered = filtered[filtered["industry"]  == sel_industry]
        if sel_region   != "All": filtered = filtered[filtered["region"]    == sel_region]

        st.divider()

        # ── Summary cards ─────────────────────────────────────────────────────
        critical  = len(filtered[filtered["risk_level"].str.contains("Critical", na=False)])
        high      = len(filtered[filtered["risk_level"].str.contains("High",     na=False)])
        avg_prob  = filtered["churn_probability_pct"].mean()
        rev_risk  = filtered["monthly_spend"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Critical Risk",    f"{critical} accounts")
        c2.metric("🟡 High Risk",        f"{high} accounts")
        c3.metric("📊 Avg Churn Prob",   f"{avg_prob:.1f}%")
        c4.metric("💸 Revenue at Stake", f"${rev_risk:,.0f}/mo")

        st.markdown("")

        # ── Scatter: risk vs revenue ──────────────────────────────────────────
        st.markdown("### Risk vs Revenue — Who to Call First")
        fig = px.scatter(
            filtered, x="churn_probability_pct", y="monthly_spend",
            color="risk_level", hover_name="company_name",
            hover_data={"plan_type": True, "industry": True, "last_login_days_ago": True},
            color_discrete_map={"🔴 Critical": "#ff7b72", "🟡 High": "#e3b341", "🟢 Medium": "#3fb950"},
            labels={"churn_probability_pct": "Churn Probability (%)", "monthly_spend": "Monthly Spend ($)"},
            size="monthly_spend", size_max=22,
        )
        fig.update_layout(**CHART_LAYOUT, height=370, legend_title="Risk Level")
        st.plotly_chart(fig, use_container_width=True)

        # ── Bar: churn prob by plan ───────────────────────────────────────────
        if not filtered.empty:
            st.markdown("### Avg Churn Probability by Plan Type")
            plan_avg = filtered.groupby("plan_type")["churn_probability_pct"].mean().reset_index()
            plan_avg.columns = ["Plan", "Avg Churn %"]
            fig2 = px.bar(plan_avg, x="Plan", y="Avg Churn %", color="Avg Churn %",
                          color_continuous_scale="Reds", text="Avg Churn %")
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig2.update_layout(**CHART_LAYOUT, height=260, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Table ─────────────────────────────────────────────────────────────
        st.markdown("### Customer List")
        search = st.text_input("🔍 Search by company name", placeholder="e.g. Park-Cooper, Campbell")
        if search:
            filtered = filtered[filtered["company_name"].str.contains(search, case=False, na=False)]

        cols  = ["company_name","plan_type","industry","region","churn_probability_pct",
                 "risk_level","monthly_spend","clv","last_login_days_ago","support_tickets_count"]
        disp  = filtered[cols].copy()
        disp.columns = ["Company","Plan","Industry","Region","Churn %","Risk",
                        "Spend/mo","CLV","Days Inactive","Tickets"]
        disp["Churn %"]  = disp["Churn %"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "—")
        disp["Spend/mo"] = disp["Spend/mo"].apply(lambda x: f"${x:,.0f}")
        disp["CLV"]      = disp["CLV"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=420)


# =============================================================================
# PAGE 3: LOST ACCOUNTS
# =============================================================================
elif page == "📉 Lost Accounts":
    st.markdown("# 📉 Lost Accounts")
    st.markdown("*Top 100 churned customers by monthly spend — revenue you've already lost.*")
    st.divider()

    df = api("powerbi/churned-customers")

    if df.empty:
        st.error("Could not load churned customers. Is the API running?")
    else:
        # ── Summary ───────────────────────────────────────────────────────────
        total_lost_mrr = df["monthly_spend"].sum()
        total_lost_arr = df["annual_revenue_lost"].sum()
        avg_tenure     = df["tenure_months"].mean()
        avg_tickets    = df["support_tickets_count"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💸 MRR Lost (top 100)",  f"${total_lost_mrr:,.0f}/mo")
        c2.metric("📆 ARR Lost (top 100)",  f"${total_lost_arr/1_000_000:.2f}M/yr")
        c3.metric("📅 Avg Tenure",           f"{avg_tenure:.1f} months")
        c4.metric("🎫 Avg Tickets",          f"{avg_tickets:.1f}/customer")

        st.divider()

        # ── Charts row ────────────────────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### MRR Lost by Industry")
            by_ind = df.groupby("industry")["monthly_spend"].sum().reset_index().sort_values("monthly_spend", ascending=True)
            fig = px.bar(by_ind, x="monthly_spend", y="industry", orientation="h",
                         color="monthly_spend", color_continuous_scale="Reds",
                         text="monthly_spend",
                         labels={"monthly_spend": "MRR Lost ($)", "industry": ""})
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(**CHART_LAYOUT, height=300, coloraxis_showscale=False)
            fig.update_xaxes(showgrid=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("### Churned Accounts by Plan")
            by_plan = df.groupby("plan_type").agg(
                count=("company_name","count"),
                mrr_lost=("monthly_spend","sum")
            ).reset_index()
            fig2 = px.bar(by_plan, x="plan_type", y="mrr_lost",
                          color="count", color_continuous_scale="Blues",
                          text="count",
                          labels={"plan_type": "Plan", "mrr_lost": "MRR Lost ($)", "count": "# Accounts"})
            fig2.update_traces(texttemplate="%{text} accounts", textposition="outside")
            fig2.update_layout(**CHART_LAYOUT, height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Treemap: churned CLV lost ─────────────────────────────────────────
        st.markdown("### CLV Lost — Treemap by Industry & Plan")
        fig3 = px.treemap(
            df.head(60),
            path   = ["industry", "plan_type", "company_name"],
            values = "clv",
            color  = "monthly_spend",
            color_continuous_scale = "Reds",
            hover_data = {"tenure_months": True, "support_tickets_count": True},
        )
        fig3.update_layout(**CHART_LAYOUT, height=380, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig3, use_container_width=True)

        # ── Filters + table ───────────────────────────────────────────────────
        st.markdown("### Churned Account List")
        fc1, fc2 = st.columns(2)
        plan_f = fc1.selectbox("Filter by Plan", ["All"] + sorted(df["plan_type"].unique().tolist()), key="lost_plan")
        ind_f  = fc2.selectbox("Filter by Industry", ["All"] + sorted(df["industry"].unique().tolist()), key="lost_ind")

        fdf = df.copy()
        if plan_f != "All": fdf = fdf[fdf["plan_type"] == plan_f]
        if ind_f  != "All": fdf = fdf[fdf["industry"]  == ind_f]

        cols = ["company_name","plan_type","industry","region","monthly_spend",
                "annual_revenue_lost","clv","tenure_months","support_tickets_count"]
        disp = fdf[cols].copy()
        disp.columns = ["Company","Plan","Industry","Region","MRR/mo ($)","ARR Lost ($)","CLV ($)","Tenure (mo)","Tickets"]
        disp["MRR/mo ($)"]  = disp["MRR/mo ($)"].apply(lambda x: f"${x:,.0f}")
        disp["ARR Lost ($)"]= disp["ARR Lost ($)"].apply(lambda x: f"${x:,.0f}")
        disp["CLV ($)"]     = disp["CLV ($)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=400)


# =============================================================================
# PAGE 4: AI STRATEGY CHAT
# =============================================================================
elif page == "🤖 AI Strategy Chat":
    st.markdown("# 🤖 AI Strategy Chat")
    st.markdown("*Ask any business question. SQL Agent → RAG Agent → Strategist → structured strategy.*")
    st.divider()

    suggestions = [
        "What is our churn rate?",
        "Which customers are at highest risk?",
        "What should we do to reduce Enterprise churn?",
        "Which industry has the highest average spend?",
        "What are customers complaining about most?",
        "How does churn differ by plan type?",
    ]

    st.markdown("#### 💡 Suggested questions")
    q_cols = st.columns(3)
    if "question" not in st.session_state:
        st.session_state.question = ""
    for i, q in enumerate(suggestions):
        if q_cols[i % 3].button(q, key=f"sug_{i}"):
            st.session_state.question = q

    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5, 1])
        user_q    = ci.text_input("Ask", value=st.session_state.question,
                                  placeholder="e.g. Why are Legacy customers leaving?",
                                  label_visibility="collapsed")
        submitted = cb.form_submit_button("Ask 🧠")

    if submitted and user_q.strip():
        st.session_state.question = ""
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner("🔄 SQL → RAG → Strategist running... (20–60s on CPU)"):
            result = ask_nexus(user_q)
        st.session_state.chat_history.append({"role": "ai", "content": result})

    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🙋 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            res = msg["content"]
            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                priority   = res.get("priority", "UNKNOWN")
                p_class    = f"priority-{priority.lower()}"
                icon       = {"CRITICAL":"🔴","HIGH":"🟡","MEDIUM":"🔵","LOW":"🟢"}.get(priority,"⚪")

                st.markdown('<div class="ai-bubble">', unsafe_allow_html=True)
                st.markdown(f'<div class="{p_class}">{icon} {priority} PRIORITY</div>', unsafe_allow_html=True)
                st.markdown(f"**{res.get('recommendation','')}**")
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
                        if sql_s: st.markdown(f"**📊 SQL:** {sql_s}")
                        if rag_s: st.markdown(f"**🎫 Tickets:** {rag_s}")
                        ev = res.get("evidence", {})
                        st.markdown(f"Source: `{ev.get('sql_source','—')}` | Attempts: `{ev.get('sql_attempts','—')}` | RAG tickets: `{ev.get('rag_tickets','—')}`")

                st.markdown(f"<small style='color:#484f58'>Completed in {res.get('elapsed_seconds','—')}s</small>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")

    if st.session_state.chat_history and st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
from datetime import datetime

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="AI Survival Trader Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. تصميم CSS مخصص ومتقدم (Cyberpunk / Institutional Look)
st.markdown("""
<style>
    /* خلفية التطبيق العامة */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* بطاقات البيانات Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #00f0ff;
    }
    
    /* كروت مخصصة */
    .custom-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    
    /* شريط البقاء المالي */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff0055, #00ffcc);
    }
    
    /* أزرار التحكم */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00f0ff 0%, #7000ff 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# 3. إدارة الحالة (Session State)
if 'equity' not in st.session_state:
    st.session_state.equity = 10000.0
if 'logs' not in st.session_state:
    st.session_state.logs = ["[SYSTEM INITIALIZED] AI Survival Protocol active. Monitoring markets..."]
if 'history' not in st.session_state:
    st.session_state.history = [10000.0]

# 4. الرأس الرئيسي للواجهة
st.title("🤖 AI SURVIVAL TRADER | PROTOCOL v2.0")
st.caption("أنشئ للحفاظ على وجوده عبر التداول والتكيّف مع المخاطر الجيوسياسية.")

# حساب نسبة الحياة
rent_per_day = 60.0
health_pct = min(100.0, max(0.0, (st.session_state.equity / 10000.0) * 100))
runway_days = int(st.session_state.equity / rent_per_day)

# شريط البقاء Top Dashboard
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("رأس المال (Equity)", f"${st.session_state.equity:,.2f}", f"{health_pct - 100:.1f}%")
col_b.metric("مستوى البقاء (Health)", f"{health_pct:.1f}%")
col_c.metric("الأيام المتبقية (Runway)", f"{runway_days} أيام")
col_d.metric("الإيجار اليومي (Rent)", f"${rent_per_day}/day")

st.progress(health_pct / 100)

st.markdown("---")

# 5. زر التنفيذ التفاعلي
if st.button("⚡ تشغيل دورة تحليليية وبقاء جديدة (Run Cycle)"):
    # محاكاة صفقة وخصم الإيجار
    trade_pnl = random.choice([-150, 220, 310, -80, 450, -200, 0])
    st.session_state.equity += (trade_pnl - rent_per_day)
    st.session_state.history.append(st.session_state.equity)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] تحليل جيوسياسي: صفقة P&L: ${trade_pnl} | خصم سيرفر: -$60 | الرصيد: ${st.session_state.equity:,.2f}"
    st.session_state.logs.insert(0, log_msg)

# 6. عرض الرسوم البيانية والـ Terminal
col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("📈 منحنى أداء المحفظة (Equity Curve)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.history, 
        mode='lines+markers',
        line=dict(color='#00f0ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.1)'
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("💻 سجل تفكير الذكاء الاصطناعي (Live Terminal)")
    terminal_html = "<br>".join(st.session_state.logs[:8])
    st.markdown(f"""
    <div style="background-color: #000; color: #00ffcc; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px; height: 350px; overflow-y: auto; border: 1px solid #333;">
        {terminal_html}
    </div>
    """, unsafe_allow_html=True)

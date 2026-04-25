import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Zid ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Arabic:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Arabic', sans-serif;
    }
    .main { background-color: #0a0f1e; }
    .stApp { background-color: #0a0f1e; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00b4d8, #0077b6, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: #90e0ef;
        font-size: 1rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
    }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a, #1a2a3a);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,180,216,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00b4d8;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-label {
        color: #90e0ef;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
    }
    .section-header {
        color: #00b4d8;
        font-size: 1.3rem;
        font-weight: 600;
        border-left: 4px solid #00b4d8;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #0d1b2a, #0a1628);
        border: 1px solid #00b4d8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #caf0f8;
        font-size: 0.9rem;
    }
    .tag {
        display: inline-block;
        background: #023e8a;
        color: #90e0ef;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px;
    }
    .stSelectbox label, .stSlider label, .stFileUploader label {
        color: #90e0ef !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stSidebarContent"] {
        background: #060d1a;
        border-right: 1px solid #1e3a5f;
    }
    .result-high   { color: #2a9d8f; font-weight: 700; font-size: 1.4rem; }
    .result-medium { color: #f4a261; font-weight: 700; font-size: 1.4rem; }
    .result-low    { color: #e63946; font-weight: 700; font-size: 1.4rem; }
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

COLORS = ['#00b4d8', '#0077b6', '#48cae4', '#90e0ef', '#023e8a']
plt.rcParams.update({'figure.facecolor': '#0d1b2a', 'axes.facecolor': '#0d1b2a',
                     'axes.edgecolor': '#1e3a5f', 'text.color': '#caf0f8',
                     'axes.labelcolor': '#90e0ef', 'xtick.color': '#90e0ef',
                     'ytick.color': '#90e0ef', 'grid.color': '#1e3a5f',
                     'grid.alpha': 0.5})

# ── Data loading & model training ────────────────────────────
@st.cache_data
def load_and_train(kpi_file, ship_file, cancel_file):
    # Load
    kpi      = pd.read_excel(kpi_file, sheet_name=0)
    shipping = pd.read_excel(ship_file, sheet_name=0)
    cancel   = pd.read_excel(cancel_file, sheet_name=0)

    # Clean KPI
    pct_cols = [c for c in kpi.columns if '%' in c]
    for col in pct_cols:
        kpi[col] = kpi[col].astype(str).str.replace('%','', regex=False).astype(float)

    kpi.columns = ['store_id','total_orders','fulfillment_rate',
                   'cancellation_rate','retention_rate','avg_order_value','avg_daily_orders']
    shipping.columns = ['shipping_method','total_orders_processed','total_canceled','cancellation_rate_pct']
    cancel.columns   = ['shipping_name','total_orders','total_cancellations',
                        'total_delays','cancellation_rate','delay_rate','classification']

    # Feature engineering
    df = kpi.copy()
    q99 = df['avg_daily_orders'].quantile(0.99)
    df  = df[df['avg_daily_orders'] <= q99].copy()
    df['performance_score'] = (df['fulfillment_rate']*0.35 + df['retention_rate']*0.30 + (100-df['cancellation_rate'])*0.35)
    df['est_daily_revenue']  = df['avg_daily_orders'] * df['avg_order_value']
    df['is_high_value']      = (df['avg_order_value'] > df['avg_order_value'].median()).astype(int)

    feature_cols = ['total_orders','fulfillment_rate','cancellation_rate',
                    'retention_rate','avg_order_value','performance_score','is_high_value']

    # Regression
    X = df[feature_cols]; y = df['avg_daily_orders']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reg.fit(Xtr, ytr)
    reg_preds = rf_reg.predict(Xte)
    reg_metrics = dict(
        r2  = r2_score(yte, reg_preds),
        mae = mean_absolute_error(yte, reg_preds),
        rmse= np.sqrt(mean_squared_error(yte, reg_preds))
    )

    # Classification
    q33 = df['avg_daily_orders'].quantile(0.33)
    q66 = df['avg_daily_orders'].quantile(0.66)
    df['tier'] = df['avg_daily_orders'].apply(lambda x: 'Low' if x<=q33 else ('Medium' if x<=q66 else 'High'))
    le = LabelEncoder()
    y_cls = le.fit_transform(df['tier'])
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(df[feature_cols], y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cls.fit(Xc_tr, yc_tr)
    cls_acc = accuracy_score(yc_te, rf_cls.predict(Xc_te))
    cls_report = classification_report(yc_te, rf_cls.predict(Xc_te), target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(yc_te, rf_cls.predict(Xc_te))

    return df, shipping, cancel, rf_reg, rf_cls, le, feature_cols, reg_metrics, cls_acc, cls_report, cm, Xte, yte, reg_preds, q33, q66

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛒 Zid ML Dashboard")
    st.markdown("<span class='tag'>Data Science Project</span> <span class='tag'>by Mousa</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 📂 Upload Your Data")
    kpi_file    = st.file_uploader("Store Performance KPIs",        type=['xlsx'])
    ship_file   = st.file_uploader("Shipping Method Performance",   type=['xlsx'])
    cancel_file = st.file_uploader("Cancellation & Delay Data",     type=['xlsx'])

    st.markdown("---")
    st.markdown("#### 🔗 Links")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-mousa--dsc-0077b6?logo=github)](https://github.com/mousa-dsc)")
    st.markdown("---")
    st.caption("Built with Python · Scikit-learn · Streamlit")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🛒 Zid Store Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine Learning Dashboard · Real Data · 3,851 Stores</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not (kpi_file and ship_file and cancel_file):
    st.info("⬅️ ارفع الملفات الثلاثة من الـ Sidebar عشان يشتغل الداشبورد")
    st.markdown("""
    **الملفات المطلوبة:**
    - `Store_Performance_KPIs.xlsx`
    - `Shipping_Method_Performance.xlsx`
    - `Cancellation___Delay_for_Shipping.xlsx`
    """)
    st.stop()

with st.spinner("🔄 جاري تحميل البيانات وتدريب النماذج..."):
    df, shipping, cancel, rf_reg, rf_cls, le, feature_cols, reg_metrics, cls_acc, cls_report, cm, Xte, yte, reg_preds, q33, q66 = load_and_train(kpi_file, ship_file, cancel_file)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🤖 ML Models", "🚚 Shipping", "🎯 Predict", "💡 Insights"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Stores</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["avg_daily_orders"].median():.1f}</div><div class="metric-label">Median Daily Orders</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["avg_order_value"].mean():.0f} $</div><div class="metric-label">Avg Order Value</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["cancellation_rate"].mean():.1f}%</div><div class="metric-label">Avg Cancellation Rate</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Data Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#0d1b2a')
    cols = ['total_orders','fulfillment_rate','cancellation_rate','retention_rate','avg_order_value','avg_daily_orders']
    titles = ['Total Orders','Fulfillment Rate (%)','Cancellation Rate (%)','Retention Rate (%)','Avg Order Value ($)','Avg Daily Orders']
    for ax, col, title in zip(axes.flat, cols, titles):
        data = df[col].clip(upper=df[col].quantile(0.99))
        ax.hist(data, bins=35, color=COLORS[0], edgecolor='#0a0f1e', alpha=0.9)
        ax.axvline(df[col].median(), color='#e63946', linestyle='--', linewidth=1.5)
        ax.set_title(title, fontsize=10, fontweight='bold', color='#caf0f8')
        ax.set_xlabel(''); ax.set_ylabel('')
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    fig2.patch.set_facecolor('#0d1b2a')
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
                linewidths=0.5, ax=ax2, cbar_kws={'shrink':0.8})
    ax2.set_facecolor('#0d1b2a')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — ML MODELS
# ══════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">🔵 Regression Model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{reg_metrics["r2"]*100:.1f}%</div><div class="metric-label">R² Score — Variance Explained</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-value">±{reg_metrics["mae"]:.2f}</div><div class="metric-label">MAE — Orders/Day Error</div></div>', unsafe_allow_html=True)

        # Actual vs Predicted
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#0d1b2a')
        ax3.scatter(yte, reg_preds, alpha=0.3, color=COLORS[0], s=8)
        mn, mx = yte.min(), yte.max()
        ax3.plot([mn,mx],[mn,mx],'r--', linewidth=2, label='Perfect')
        ax3.set_title('Actual vs Predicted Daily Orders', color='#caf0f8', fontweight='bold')
        ax3.set_xlabel('Actual'); ax3.set_ylabel('Predicted')
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_r:
        st.markdown('<div class="section-header">🟢 Classification Model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{cls_acc*100:.1f}%</div><div class="metric-label">Accuracy — Store Tier Prediction</div></div>', unsafe_allow_html=True)

        # Confusion matrix
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        fig4.patch.set_facecolor('#0d1b2a')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax4.set_title('Confusion Matrix', color='#caf0f8', fontweight='bold')
        ax4.set_ylabel('Actual'); ax4.set_xlabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # Feature Importance
    st.markdown('<div class="section-header">🔑 Feature Importance</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_reg.feature_importances_}).sort_values('Importance')
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    fig5.patch.set_facecolor('#0d1b2a')
    bars = ax5.barh(imp_df['Feature'], imp_df['Importance'], color=COLORS[0])
    for bar, val in zip(bars, imp_df['Importance']):
        ax5.text(val+0.002, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9, color='#caf0f8')
    ax5.set_title('Random Forest Feature Importance', color='#caf0f8', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — SHIPPING
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🚚 Shipping Methods Analysis</div>', unsafe_allow_html=True)
    top10 = shipping.nlargest(10, 'total_orders_processed')

    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))
    fig6.patch.set_facecolor('#0d1b2a')

    axes6[0].barh(top10['shipping_method'], top10['total_orders_processed'], color=COLORS[0])
    axes6[0].set_title('Total Orders Processed (Top 10)', color='#caf0f8', fontweight='bold')
    axes6[0].invert_yaxis()

    bar_colors = ['#e63946' if x>7 else '#2a9d8f' for x in top10['cancellation_rate_pct']]
    axes6[1].barh(top10['shipping_method'], top10['cancellation_rate_pct'], color=bar_colors)
    axes6[1].axvline(7, color='#f4a261', linestyle='--', linewidth=1.5, label='7% threshold')
    axes6[1].set_title('Cancellation Rate % — 🟢 Safe / 🔴 High Risk', color='#caf0f8', fontweight='bold')
    axes6[1].legend()
    axes6[1].invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    st.markdown('<div class="section-header">⚠️ Risk Classification</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 Highest Cancellation Rate**")
        high_c = cancel[cancel['classification']=='Highest Cancellations'][['shipping_name','total_orders','cancellation_rate']].head(8)
        st.dataframe(high_c.style.background_gradient(cmap='Reds', subset=['cancellation_rate']), use_container_width=True)
    with c2:
        st.markdown("**🟢 Most Reliable (Lowest Delays)**")
        low_d = cancel[cancel['classification']=='Lowest Delays'][['shipping_name','total_orders','delay_rate']].head(8)
        st.dataframe(low_d.style.background_gradient(cmap='Greens_r', subset=['delay_rate']), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🎯 Predict Store Performance</div>', unsafe_allow_html=True)
    st.markdown("أدخل بيانات متجر جديد وشوف النموذج يتنبأ بأدائه")

    c1, c2 = st.columns(2)
    with c1:
        total_orders    = st.slider("📦 Total Orders",          1, 500, 150)
        fulfillment     = st.slider("✅ Fulfillment Rate (%)",  0.0, 100.0, 88.0)
        cancellation    = st.slider("❌ Cancellation Rate (%)", 0.0, 100.0, 4.5)
    with c2:
        retention       = st.slider("🔄 Retention Rate (%)",   0.0, 100.0, 35.0)
        order_value     = st.slider("💰 Avg Order Value ($)",   10, 1000, 210)
        is_high_value   = st.selectbox("🏷️ Above Median Order Value?", [1, 0], format_func=lambda x: "Yes ✅" if x==1 else "No ❌")

    perf_score = fulfillment*0.35 + retention*0.30 + (100-cancellation)*0.35

    new_store = pd.DataFrame([{
        'total_orders':      total_orders,
        'fulfillment_rate':  fulfillment,
        'cancellation_rate': cancellation,
        'retention_rate':    retention,
        'avg_order_value':   order_value,
        'performance_score': perf_score,
        'is_high_value':     is_high_value,
    }])

    pred_orders   = rf_reg.predict(new_store)[0]
    pred_tier_enc = rf_cls.predict(new_store)[0]
    pred_tier     = le.inverse_transform([pred_tier_enc])[0]
    tier_color    = {'High':'result-high','Medium':'result-medium','Low':'result-low'}.get(pred_tier,'result-medium')

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pred_orders:.1f}</div><div class="metric-label">Predicted Daily Orders</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pred_orders*30:.0f}</div><div class="metric-label">Estimated Monthly Orders</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pred_orders*30*order_value:,.0f}</div><div class="metric-label">Est. Monthly Revenue ($)</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="metric-card"><div class="{tier_color}">{pred_tier}</div><div class="metric-label">Performance Tier</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="insight-box">📊 Performance Score: <strong>{perf_score:.1f}/100</strong> | هذا المتجر يقع في الـ <strong>{pred_tier}</strong> tier</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">💡 Key Business Insights</div>', unsafe_allow_html=True)

    high = df[df['avg_daily_orders'] > q66]
    low  = df[df['avg_daily_orders'] <= q33]

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**📈 High-Tier Stores vs Low-Tier Stores**")
        compare = pd.DataFrame({
            'Metric': ['Avg Daily Orders','Retention Rate (%)','Fulfillment Rate (%)','Cancellation Rate (%)','Avg Order Value ($)'],
            'High Tier': [high['avg_daily_orders'].mean(), high['retention_rate'].mean(),
                          high['fulfillment_rate'].mean(), high['cancellation_rate'].mean(), high['avg_order_value'].mean()],
            'Low Tier':  [low['avg_daily_orders'].mean(),  low['retention_rate'].mean(),
                          low['fulfillment_rate'].mean(),  low['cancellation_rate'].mean(),  low['avg_order_value'].mean()],
        })
        compare['High Tier'] = compare['High Tier'].round(2)
        compare['Low Tier']  = compare['Low Tier'].round(2)
        st.dataframe(compare.set_index('Metric'), use_container_width=True)

    with i2:
        st.markdown("**🎯 Model Summary**")
        st.markdown(f"""
        <div class="insight-box">🔵 <strong>Regression (Random Forest)</strong><br>
        R² = {reg_metrics['r2']*100:.1f}% | MAE = ±{reg_metrics['mae']:.2f} orders/day</div>

        <div class="insight-box">🟢 <strong>Classification (Random Forest)</strong><br>
        Accuracy = {cls_acc*100:.1f}% على 3 فئات (High/Medium/Low)</div>

        <div class="insight-box">📌 <strong>أهم Feature</strong><br>
        {pd.DataFrame({'f':feature_cols,'i':rf_reg.feature_importances_}).nlargest(1,'i').iloc[0]['f']} هو الأكثر تأثيراً على عدد الطلبات</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("🔄 تحسين الـ Retention Rate", "المتاجر High-tier عندها retention أعلى بكثير — ابنِ loyalty programs"),
        ("❌ تقليل نسبة الإلغاء", "كل 1% انخفاض في cancellation rate يرفع الطلبات اليومية"),
        ("🚚 اختيار شركة الشحن الصح", "Aramex تتصدر الحجم — المتاجر ذات الإلغاء العالي تستفيد من التحول لها"),
        ("💰 تحسين قيمة الطلب", "المتاجر بـ avg_order_value أعلى تنتمي للـ High tier بنسبة أكبر"),
    ]
    for title, desc in recs:
        st.markdown(f'<div class="insight-box"><strong>{title}</strong><br>{desc}</div>', unsafe_allow_html=True)

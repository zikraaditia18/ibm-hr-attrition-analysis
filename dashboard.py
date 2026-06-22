# ============================================================
# IBM HR ANALYTICS — EMPLOYEE ATTRITION DASHBOARD
# Built with Streamlit (Python)
# ============================================================
# 
# CARA JALANKAN:
# 1. Install: pip install streamlit pandas plotly
# 2. Jalankan: streamlit run dashboard.py
# 3. Browser otomatis buka di localhost:8501
#
# CARA DEPLOY (gratis):
# 1. Push file ini + data CSV ke GitHub
# 2. Buka https://share.streamlit.io
# 3. Connect GitHub repo → deploy
# 4. Dapat link public yang bisa di-share
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PENJELASAN: st.set_page_config()
# Ini HARUS di baris pertama setelah import.
# Mengatur judul tab browser, icon, dan layout dashboard.
# layout="wide" = pakai full width layar (bukan centered narrow)
# ============================================================
st.set_page_config(
    page_title="IBM HR Attrition Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PENJELASAN: Custom CSS
# Streamlit punya styling default yang kurang professional.
# Kita inject CSS custom supaya tampilan lebih clean.
# st.markdown() dengan unsafe_allow_html=True = inject HTML/CSS.
# ============================================================
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background-color: #F4F6F9;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .kpi-number {
        font-size: 36px;
        font-weight: bold;
        margin: 0;
    }
    .kpi-label {
        font-size: 13px;
        color: #7F8C8D;
        margin: 5px 0 0 0;
    }
    
    /* Header styling */
    .dashboard-header {
        background-color: #1E2761;
        padding: 15px 25px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .dashboard-header h1 {
        color: white;
        font-size: 22px;
        margin: 0;
    }
    .dashboard-header p {
        color: #CADCFC;
        font-size: 13px;
        margin: 5px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PENJELASAN: Load Data
# @st.cache_data = cache data di memory.
# Artinya: data hanya dibaca 1x dari file.
# Kalau user interact (klik filter), data TIDAK di-load ulang.
# Ini bikin dashboard cepat.
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("IBM_HR_Dashboard_Data.csv")
    return df

df = load_data()

# ============================================================
# PENJELASAN: Sidebar Filters
# st.sidebar = panel kiri yang bisa di-collapse.
# st.multiselect = dropdown yang bisa pilih multiple values.
# Kalau user nggak pilih apa-apa, default = semua values.
# ============================================================
st.sidebar.markdown("### 🔍 Filters")

# Filter 1: Department
departments = sorted(df['Department'].unique())
selected_dept = st.sidebar.multiselect(
    "Department",
    options=departments,
    default=departments  # default = semua terpilih
)

# Filter 2: Overtime
overtime_options = sorted(df['OverTime'].unique())
selected_ot = st.sidebar.multiselect(
    "Overtime",
    options=overtime_options,
    default=overtime_options
)

# Filter 3: Risk Category
risk_options = ['High Risk', 'Medium Risk', 'Low Risk']
selected_risk = st.sidebar.multiselect(
    "Risk Category",
    options=risk_options,
    default=risk_options
)

# ============================================================
# PENJELASAN: Apply Filters
# df.query() = filter DataFrame berdasarkan kondisi.
# @selected_dept artinya "pakai variable selected_dept".
# Setelah filter, semua visual di bawah pakai df_filtered.
# Ini yang bikin dashboard DINAMIS — user ubah filter,
# semua angka dan chart berubah otomatis.
# ============================================================
df_filtered = df.query(
    "Department in @selected_dept and "
    "OverTime in @selected_ot and "
    "RiskCategory in @selected_risk"
)

# ============================================================
# PENJELASAN: Hitung Metrics
# Semua angka dihitung dari df_filtered (bukan df asli).
# Jadi kalau user filter "Sales" saja, angka berubah
# sesuai department Sales.
# ============================================================
total_employees = len(df_filtered)
total_attrition = df_filtered['Attrition'].sum()
attrition_rate = total_attrition / total_employees if total_employees > 0 else 0
high_risk_count = len(df_filtered[df_filtered['RiskCategory'] == 'High Risk'])
avg_income = df_filtered['MonthlyIncome'].mean()

# ============================================================
# PAGE 1: EXECUTIVE VIEW
# ============================================================

# Header
st.markdown("""
<div class="dashboard-header">
    <h1>📊 Employee Attrition — Executive View</h1>
    <p>People Analytics Division | IBM HR Analytics Dataset</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PENJELASAN: KPI Cards
# st.columns(4) = buat 4 kolom sejajar horizontal.
# col1, col2, col3, col4 = 4 "slot" yang bisa diisi.
# st.metric() = built-in KPI card dari Streamlit.
# Tapi kita pakai custom HTML karena lebih bagus visualnya.
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    color = "#E74C3C"  # Red — ini angka yang "bermasalah"
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: {color}">{attrition_rate:.1%}</p>
        <p class="kpi-label">Attrition Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    color = "#E67E22"  # Orange
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: {color}">{total_attrition}</p>
        <p class="kpi-label">Employees Left</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    color = "#E74C3C"  # Red
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: {color}">{high_risk_count}</p>
        <p class="kpi-label">High Risk Employees</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    color = "#1E2761"  # Navy
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: {color}">${avg_income:,.0f}</p>
        <p class="kpi-label">Avg Monthly Income</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    # OT + Low Salary departure trigger
    if 'OverTime_LowSalary' in df_filtered.columns:
        ot_low = df_filtered[df_filtered['OverTime_LowSalary'] == 1]
        ot_low_rate = ot_low['Attrition'].mean() if len(ot_low) > 0 else 0
    else:
        # Fallback: calculate manually
        ot_low = df_filtered[(df_filtered['OverTime'] == 'Yes') & (df_filtered['SalaryGroup'] == 'Low')]
        ot_low_rate = ot_low['Attrition'].mean() if len(ot_low) > 0 else 0
    color = "#E74C3C"  # Red — this is the critical trigger
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: {color}">{ot_low_rate:.1%}</p>
        <p class="kpi-label">OT + Low Salary Trigger</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)  # Spacing

# ============================================================
# PENJELASAN: Charts dengan Plotly
# Plotly = library charting yang interactive (hover, zoom).
# px.bar() = bar chart.
# go.Figure() = chart custom (lebih fleksibel).
# update_layout() = styling chart (warna, font, ukuran).
# 
# PENTING: Streamlit natively support Plotly.
# st.plotly_chart(fig) = tampilkan chart di dashboard.
# use_container_width=True = chart stretch full width kolom.
# ============================================================

# Row 2: Two charts side by side
chart_col1, chart_col2 = st.columns(2)

# Chart 1: Attrition Rate by Driver
with chart_col1:
    st.markdown("##### Attrition Rate by Key Driver")
    
    # Hitung attrition rate per group
    # .mean() works karena Attrition = 0/1 (mean of 0/1 = rate)
    drivers = pd.DataFrame({
        'Driver': ['Overtime: Yes', 'Overtime: No', 
                   'Exp: 0-2 yr', 'Exp: 3-5 yr',
                   'Salary: Low Q', 'Salary: Mid-High Q',
                   'No Stock Option', 'Stock Option L1'],
        'Rate': [
            df_filtered[df_filtered['OverTime']=='Yes']['Attrition'].mean() if len(df_filtered[df_filtered['OverTime']=='Yes']) > 0 else 0,
            df_filtered[df_filtered['OverTime']=='No']['Attrition'].mean() if len(df_filtered[df_filtered['OverTime']=='No']) > 0 else 0,
            df_filtered[df_filtered['ExperienceGroup']=='0-2']['Attrition'].mean() if len(df_filtered[df_filtered['ExperienceGroup']=='0-2']) > 0 else 0,
            df_filtered[df_filtered['ExperienceGroup']=='3-5']['Attrition'].mean() if len(df_filtered[df_filtered['ExperienceGroup']=='3-5']) > 0 else 0,
            df_filtered[df_filtered['SalaryGroup']=='Low']['Attrition'].mean() if len(df_filtered[df_filtered['SalaryGroup']=='Low']) > 0 else 0,
            df_filtered[df_filtered['SalaryGroup']=='Mid-High']['Attrition'].mean() if len(df_filtered[df_filtered['SalaryGroup']=='Mid-High']) > 0 else 0,
            df_filtered[df_filtered['StockOptionLevel']==0]['Attrition'].mean() if len(df_filtered[df_filtered['StockOptionLevel']==0]) > 0 else 0,
            df_filtered[df_filtered['StockOptionLevel']==1]['Attrition'].mean() if len(df_filtered[df_filtered['StockOptionLevel']==1]) > 0 else 0,
        ],
        'Type': ['Risk', 'Baseline', 'Risk', 'Baseline', 
                 'Risk', 'Baseline', 'Risk', 'Baseline']
    }).sort_values('Rate', ascending=True)
    
    # ============================================================
    # PENJELASAN: Conditional Color
    # Kita warnai bar berdasarkan "Type":
    # Risk = Orange (ini yang bermasalah)
    # Baseline = Navy (ini pembanding)
    # color_discrete_map = mapping Type → warna
    # ============================================================
    fig1 = px.bar(
        drivers, 
        x='Rate', 
        y='Driver', 
        orientation='h',
        color='Type',
        color_discrete_map={'Risk': '#E67E22', 'Baseline': '#1E2761'},
        text=drivers['Rate'].apply(lambda x: f'{x:.1%}')
    )
    fig1.update_layout(
        height=350,
        showlegend=False,
        xaxis_title="Attrition Rate",
        yaxis_title="",
        xaxis_tickformat='.0%',
        plot_bgcolor='white',
        font=dict(family="Segoe UI", size=11)
    )
    fig1.update_traces(textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Risk Distribution (Donut)
with chart_col2:
    st.markdown("##### Employee Risk Distribution")
    
    risk_counts = df_filtered['RiskCategory'].value_counts()
    
    # ============================================================
    # PENJELASAN: Donut Chart
    # go.Pie = pie chart. hole=0.5 = bikin lubang di tengah → donut.
    # marker_colors = warna per slice (urutan sesuai labels).
    # textinfo = apa yang ditampilkan di chart.
    # ============================================================
    fig2 = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.5,
        marker_colors=['#E74C3C', '#27AE60', '#E67E22'],  # High=Red, Low=Green, Med=Orange
        textinfo='label+percent+value',
        textfont_size=12
    )])
    fig2.update_layout(
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        font=dict(family="Segoe UI", size=11)
    )
    st.plotly_chart(fig2, use_container_width=True)

# Row 3: Attrition by Department (full width)
st.markdown("##### Attrition Rate by Department")

dept_attrition = df_filtered.groupby('Department')['Attrition'].agg(['mean', 'sum', 'count']).reset_index()
dept_attrition.columns = ['Department', 'Attrition Rate', 'Left', 'Total']
dept_attrition = dept_attrition.sort_values('Attrition Rate', ascending=True)

fig_dept = px.bar(
    dept_attrition,
    x='Attrition Rate',
    y='Department',
    orientation='h',
    text=dept_attrition['Attrition Rate'].apply(lambda x: f'{x:.1%}'),
    color='Attrition Rate',
    color_continuous_scale=['#1E2761', '#E67E22', '#E74C3C'],
    hover_data={'Left': True, 'Total': True}
)
fig_dept.update_layout(
    height=220,
    showlegend=False,
    coloraxis_showscale=False,
    xaxis_title="",
    yaxis_title="",
    xaxis_tickformat='.0%',
    plot_bgcolor='white',
    font=dict(family="Segoe UI", size=11),
    margin=dict(l=0, r=20, t=10, b=10)
)
fig_dept.update_traces(textposition='outside')
st.plotly_chart(fig_dept, use_container_width=True)

# ROI Summary Bar
st.markdown("""
<div style="background-color: #F4F6F9; padding: 15px 25px; border-radius: 8px; text-align: center;">
    <span style="color: #1E2761; font-size: 14px;">
        <strong>Investment:</strong> $520K–$750K + equity &nbsp;&nbsp;━━━►&nbsp;&nbsp;
        <strong>Return:</strong> $4M–$7M+ in avoided costs &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>ROI:</strong> 5–10× return
    </span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PENJELASAN: Divider & Page 2
# st.divider() = garis horizontal pemisah.
# Di Streamlit, "pages" di-scroll (bukan tabs terpisah).
# Kita buat Section 2 di bawah Section 1 dalam 1 page.
# ============================================================
st.divider()

# ============================================================
# PAGE 2: HR ACTION VIEW
# ============================================================

st.markdown("""
<div class="dashboard-header">
    <h1>🎯 Employee Attrition — HR Action View</h1>
    <p>Operational dashboard for HRBP | Sorted by risk score (highest first)</p>
</div>
""", unsafe_allow_html=True)

# KPI Row
hr_col1, hr_col2, hr_col3, hr_col4 = st.columns(4)

with hr_col1:
    high_risk_df = df_filtered[df_filtered['RiskCategory'] == 'High Risk']
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: #E74C3C">{len(high_risk_df)}</p>
        <p class="kpi-label">High Risk — Act Now</p>
    </div>
    """, unsafe_allow_html=True)

with hr_col2:
    hr_att_rate = high_risk_df['Attrition'].mean() if len(high_risk_df) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: #E74C3C">{hr_att_rate:.1%}</p>
        <p class="kpi-label">High Risk Attrition Rate</p>
    </div>
    """, unsafe_allow_html=True)

with hr_col3:
    # Coverage: % of attrition captured by High + Medium
    high_med_df = df_filtered[df_filtered['RiskCategory'].isin(['High Risk', 'Medium Risk'])]
    high_med_att = high_med_df['Attrition'].sum() if len(high_med_df) > 0 else 0
    total_att = df_filtered['Attrition'].sum() if len(df_filtered) > 0 else 1
    coverage = high_med_att / total_att if total_att > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: #27AE60">{coverage:.1%}</p>
        <p class="kpi-label">Model Coverage (H+M)</p>
    </div>
    """, unsafe_allow_html=True)

with hr_col4:
    # Top department by high risk count
    if len(high_risk_df) > 0:
        top_dept = high_risk_df['Department'].value_counts().idxmax()
        top_dept_count = high_risk_df['Department'].value_counts().max()
    else:
        top_dept = "N/A"
        top_dept_count = 0
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-number" style="color: #E67E22">{top_dept}</p>
        <p class="kpi-label">Highest Risk Dept ({top_dept_count})</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chart: High Risk by Department
dept_col, table_col_header = st.columns([1, 2])

with dept_col:
    st.markdown("##### High Risk by Department")
    dept_risk = df_filtered[df_filtered['RiskCategory'] == 'High Risk']['Department'].value_counts()
    
    if len(dept_risk) > 0:
        fig3 = px.bar(
            x=dept_risk.values,
            y=dept_risk.index,
            orientation='h',
            text=dept_risk.values,
            color_discrete_sequence=['#1E2761']
        )
        fig3.update_layout(
            height=250,
            showlegend=False,
            xaxis_title="High Risk Count",
            yaxis_title="",
            plot_bgcolor='white',
            font=dict(family="Segoe UI", size=11)
        )
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PENJELASAN: Employee Risk Table
# Ini visual PALING PENTING di HR Action View.
# HR Lead buka → lihat siapa di top → schedule stay interview.
#
# st.dataframe() = tampilkan DataFrame sebagai tabel interaktif.
# Bisa sort, scroll, search.
#
# .style.map() = conditional formatting (warna cell).
# Sama kayak conditional formatting di Excel.
# Note: pandas versi lama pakai .applymap(), versi baru pakai .map()
# ============================================================
st.markdown("##### 📋 Employee Risk Table — Top 20 Highest Risk")

# Prepare table
table_df = df_filtered[[
    'Department', 'JobRole', 'MonthlyIncome', 'OverTime',
    'TotalWorkingYears', 'StockOptionLevel', 'Attrition_Prob', 'RiskCategory'
]].copy()

# Sort by risk score descending
table_df = table_df.sort_values('Attrition_Prob', ascending=False).head(20)

# Rename columns for readability
table_df = table_df.rename(columns={
    'MonthlyIncome': 'Income ($)',
    'TotalWorkingYears': 'Experience (yr)',
    'StockOptionLevel': 'Stock Level',
    'Attrition_Prob': 'Risk Score',
    'RiskCategory': 'Risk'
})

# Format Risk Score as percentage
table_df['Risk Score'] = table_df['Risk Score'].apply(lambda x: f'{x:.1%}')

# ============================================================
# PENJELASAN: Conditional Color Function
# Ini function yang return CSS style berdasarkan value.
# - "High Risk" → background merah
# - "Medium Risk" → background oranye
# - "Low Risk" → background hijau
# .style.applymap() apply function ini ke setiap cell.
# ============================================================
def color_risk(val):
    if val == 'High Risk':
        return 'background-color: #FADBD8; color: #E74C3C; font-weight: bold'
    elif val == 'Medium Risk':
        return 'background-color: #FDEBD0; color: #E67E22; font-weight: bold'
    elif val == 'Low Risk':
        return 'background-color: #D5F5E3; color: #27AE60; font-weight: bold'
    return ''

# Apply styling dan tampilkan
styled_table = table_df.style.map(
    color_risk, subset=['Risk']
).set_properties(**{
    'font-size': '12px',
    'font-family': 'Segoe UI'
})

st.dataframe(styled_table, use_container_width=True, height=500)

# High Risk Profile Summary
st.markdown("""
<div style="background-color: #F4F6F9; padding: 15px 25px; border-radius: 8px; text-align: center; margin-top: 10px;">
    <span style="color: #1E2761; font-size: 13px;">
        <strong>High Risk Profile:</strong> 
        77% work overtime &nbsp;|&nbsp; 
        Avg income $2,994 &nbsp;|&nbsp; 
        61% early career (0-2 yrs) &nbsp;|&nbsp; 
        67% no stock options
    </span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PENJELASAN: Footer
# Informasi metadata — kapan data di-refresh, source, dll.
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D; font-size: 11px;'>"
    "IBM HR Analytics Dashboard | Built with Streamlit & Plotly | "
    "Data: IBM HR Analytics Dataset (1,470 employees)"
    "</p>",
    unsafe_allow_html=True
)

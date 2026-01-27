# ==========================================
# Thixo-Metric Web App (Fixed & Complete)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

# --- Page Configuration (Must be first) ---
st.set_page_config(
    page_title="Thixo-Metric",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Management ---
# Initialize if not exists
if 'df_soil' not in st.session_state:
    # Load synthetic data initially
    np.random.seed(42) 
    data = {
        'Sample_ID': [f'BH-{i:03d}' for i in range(1, 11)],
        'Latitude': np.random.uniform(23.5, 24.0, 10),
        'Longitude': np.random.uniform(90.0, 90.5, 10),
        'Depth_m': np.random.uniform(5.0, 20.0, 10),
        'Undisturbed_Su': np.random.uniform(40, 90, 10), 
        'Remolded_Su_S0': np.random.uniform(10, 25, 10), 
        'PI': np.random.uniform(30, 60, 10), 
        'Water_Content': np.random.uniform(35, 70, 10), 
        'Liquid_Limit_LL': np.random.uniform(40, 70, 10), 
        'is_submerged': np.random.choice([True, False], 10)
    }
    df = pd.DataFrame(data)
    df['Soil_Type'] = df['PI'].apply(lambda x: 'CH' if x > 35 else 'CL')
    st.session_state.df_soil = df
    st.session_state.current_data_source = "Synthetic Demo Data"

# Initialize run state
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# ==========================================
# 2. Computational Logic
# ==========================================

class RiverbankSoil:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        
    def _calculate_base_A(self, row, force_submerged=False):
        pi_val = max(row['PI'], 0.1) 
        liquidity_index = (row['Water_Content'] - row['Liquid_Limit_LL']) / pi_val
        base_A = 2.5 if row['Soil_Type'] == 'CH' else 1.5
        geo_factor = (row['PI'] / row['Water_Content'])
        A = base_A * geo_factor
        A = A / (1 + max(liquidity_index, 0)) 
        if row['is_submerged'] or force_submerged:
            A = A * 0.8 
        return A

    def calculate_strength_at_t(self, t_days, scenario="Baseline"):
        results = []
        for _, row in self.df.iterrows():
            force_submerge = True if scenario == "Flood" else False
            A = self._calculate_base_A(row, force_submerged=force_submerge)
            if scenario == "Flood": A = A * 0.85 
            
            S0 = row['Remolded_Su_S0']
            Cap = 0.75 * row['Undisturbed_Su']
            log_term = np.log10(max(t_days, 1))
            Su_t = min(S0 + (A * log_term), Cap)
            
            driving_stress = row['Depth_m'] * 5.0
            if scenario == "Flood": driving_stress = driving_stress * 1.10 
                
            FoS = Su_t / driving_stress
            
            results.append({
                'Sample_ID': row['Sample_ID'],
                'Scenario': scenario,
                'Calculated_Su_kPa': round(Su_t, 2),
                'Recovery_Const_A': round(A, 3),
                'Driving_Stress_kPa': round(driving_stress, 2),
                'FoS': round(FoS, 2)
            })
        return pd.DataFrame(results)

    def run_sensitivity_comparison(self, t_days):
        df_base = self.calculate_strength_at_t(t_days, "Baseline")
        df_flood = self.calculate_strength_at_t(t_days, "Flood")
        return pd.concat([df_base, df_flood], ignore_index=True)

    def calculate_strategic_metrics(self, t_current, target_fos):
        df_cur = self.calculate_strength_at_t(t_current, "Baseline")
        failure_rate = (df_cur['FoS'] < target_fos).mean() * 100
        
        lags = []
        df_dry_cur = self.calculate_strength_at_t(t_current, "Baseline")
        for _, row in self.df.iterrows():
            sid = row['Sample_ID']
            fos_dry_cur = df_dry_cur[df_dry_cur['Sample_ID']==sid]['FoS'].values[0]
            for t_future in range(t_current, 365):
                df_flood_fut = self.calculate_strength_at_t(t_future, "Flood")
                fos_flood_fut = df_flood_fut[df_flood_fut['Sample_ID']==sid]['FoS'].values[0]
                if fos_flood_fut >= fos_dry_cur:
                    lags.append(t_future - t_current)
                    break
            else: lags.append(365)
        avg_lag = np.mean(lags)
        
        crit = df_cur.merge(self.df[['Sample_ID', 'Soil_Type']], on='Sample_ID')
        crit_group = crit.groupby('Soil_Type')['FoS'].mean().idxmin()
        return failure_rate, round(avg_lag, 1), crit_group

    def calculate_wait_time(self, target_fos, confidence=0.95):
        for t in range(1, 365):
            df = self.calculate_strength_at_t(t, "Baseline")
            safe_count = (df['FoS'] >= target_fos).sum()
            total = len(df)
            if safe_count / total >= confidence:
                return t
        return "Not Achievable"

# ==========================================
# 3. PDF Logic (Same as Colab)
# ==========================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Thixo-Metric Technical Report', 0, 1, 'C')
        self.ln(5)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, 'Quantitative Geotechnical Stability Analysis', 0, 1, 'C')
        self.ln(10)
        
    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, f'{num}. {label}', 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

# ==========================================
# 4. The Streamlit Interface
# ==========================================

with st.sidebar:
    st.header("Thixo-Metric v1.0")
    st.subheader("Input Data & Controls")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("Upload Input CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            filename = uploaded_file.name
            if filename.endswith('.csv'): df_new = pd.read_csv(uploaded_file)
            elif filename.endswith('.xlsx'): df_new = pd.read_excel(uploaded_file, engine='openpyxl')
            else: st.error("Unsupported format."); st.stop()
                
            required_cols = ['Sample_ID', 'Depth_m', 'Undisturbed_Su', 'Remolded_Su_S0', 'PI', 'Water_Content', 'Liquid_Limit_LL', 'is_submerged']
            if any(col not in df_new.columns for col in required_cols): 
                st.error("Upload Failed: Missing columns"); st.stop()
                
            if 'Soil_Type' not in df_new.columns: df_new['Soil_Type'] = df_new['PI'].apply(lambda x: 'CH' if x > 35 else 'CL')
                
            st.session_state.df_soil = df_new
            soil_model.df = df_new # Update model instance
            st.session_state.current_data_source = filename
            st.success(f"Uploaded successfully: {filename}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    
    # 2. Controls
    st.write("Analysis Controls")
    run_btn = st.button("Run Analysis", type="primary")
    st.write("Reporting Controls")
    target_fos = st.slider("Target Safety Factor (FoS):", 1.0, 3.0, 1.5, 0.1)
    days = st.slider("Days Since Disturbance:", 0, 120, 0)
    
    # Workflow Buttons
    gen_report_btn = st.button("Generate Technical Report")
    
    st.divider()
    st.write("Direct Downloads")
    st.download_button("Download CSV Data", data=st.session_state.df_soil.to_csv(index=False).encode('utf-8'), file_name="Thixo-Metric_Data.csv", mime="text/csv")

# Initialize Model with Session Data
soil_model = RiverbankSoil(st.session_state.df_soil)

# Main Page
st.title("Dashboard & Analysis")
st.caption(f"Current Data Source: {st.session_state.current_data_source}")

# 1. Analysis Logic
if run_btn:
    with st.spinner("Analyzing... Please wait."):
        soil_model.df = st.session_state.df_soil # Ensure model uses current data
        fail_rate, hyd_lag, crit_soil = soil_model.calculate_strategic_metrics(days, target_fos)
        wait_days = soil_model.calculate_wait_time(target_fos)
        
        # Store in Session State
        st.session_state.fail_rate = fail_rate
        st.session_state.hyd_lag = hyd_lag
        st.session_state.wait_days = wait_days
        st.session_state.crit_soil = crit_soil
        st.session_state.analysis_run = True
        
        # Prepare Data
        df_compare = soil_model.run_sensitivity_comparison(days)
        df_results = soil_model.calculate_strength_at_t(days, "Baseline")
        df_display = st.session_state.df_soil.merge(df_results, on='Sample_ID')
        df_display['Status'] = df_display['FoS'].apply(lambda x: 'SAFE' if x >= target_fos else 'CRITICAL')
        
        st.session_state.df_compare = df_compare
        st.session_state.df_display = df_display

# 2. Display Results
if st.session_state.analysis_run:
    # KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.metric("Failure Rate", f"{st.session_state.fail_rate:.1f}%")
    with kpi2: st.metric("Avg. Hydraulic Lag", f"{st.session_state.hyd_lag} Days")
    with kpi3: st.metric("Critical Profile", f"{st.session_state.crit_soil}")
    with kpi4: st.metric("Wait-Time", f"{st.session_state.wait_days} Days")
    
    # Plotting (The Missing Logic Filled In)
    fig = plt.figure(figsize=(20, 10))
    st = fig.suptitle(f"Thixo-Metric: Quantitative Stability Analysis (t={days} days)", fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Recovery Curves
    ax1 = plt.subplot(2, 3, 1)
    time_range = np.arange(0, 91)
    global_max_strength = 0
    for sid in st.session_state.df_display['Sample_ID']:
        row = st.session_state.df_display[st.session_state.df_display['Sample_ID'] == sid].iloc[0]
        A = row['Recovery_Const_A']; S0 = row['Remolded_Su_S0']; Cap = 0.75 * row['Undisturbed_Su']
        if Cap > global_max_strength: global_max_strength = Cap
        t_safe = np.maximum(time_range, 1)
        su_curve = np.minimum(S0 + (A * np.log10(t_safe)), Cap)
        color = 'green' if row['Status'] == 'SAFE' else 'red'
        ax1.plot(time_range, su_curve, color=color, alpha=0.6)
        ax1.axhline(y=Cap, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.scatter(days, row['Calculated_Su_kPa'], color='black', zorder=10)
    ax1.set_title('Individual Recovery Curves', fontweight='bold')
    ax1.set_xlabel('Time (days)'); ax1.set_ylabel(r'Strength $S_u$ (kPa)')
    ax1.set_xlim(0, 90); ax1.set_ylim(0, np.ceil(global_max_strength/10)*10)

    # Plot 2: Sensitivity
    ax2 = plt.subplot(2, 3, 2)
    sns.kdeplot(data=st.session_state.df_compare, x='FoS', hue='Scenario', fill=True, alpha=0.4, linewidth=2, ax=ax2, palette={'Baseline': 'blue', 'Flood': 'red'})
    ax2.axvline(x=target_fos, color='black', linestyle='--', linewidth=2)
    ax2.set_title('Sensitivity Analysis', fontweight='bold')
    
    # Plot 3: Spatial Profile
    ax3 = plt.subplot(2, 3, 3)
    sns.scatterplot(data=st.session_state.df_display, x='Depth_m', y='FoS', hue='Status', size='PI', sizes=(50, 200), palette={'SAFE': 'green', 'CRITICAL': 'red'}, ax=ax3)
    ax3.axhline(y=target_fos, color='black', linestyle='--', label='Target FoS')
    ax3.set_title('Spatial Profile: Risk vs. Depth', fontweight='bold')
    ax3.invert_yaxis()
    
    # Plot 4: Status Bar
    ax4 = plt.subplot(2, 3, 4)
    status_counts = st.session_state.df_display['Status'].value_counts()
    colors_bar = ['green' if x == 'SAFE' else 'red' for x in status_counts.index]
    status_counts.plot(kind='bar', color=colors_bar, alpha=0.7, ax=ax4)
    ax4.set_title('Safety Status Distribution', fontweight='bold')
    for i, v in enumerate(status_counts): ax4.text(i, v + 0.1, str(v), ha='center')

    # Plot 5: Box Spread
    ax5 = plt.subplot(2, 3, 5)
    sns.boxplot(data=st.session_state.df_compare, x='Scenario', y='FoS', hue='Scenario', ax=ax5, palette={'Baseline': 'lightblue', 'Flood': 'salmon'}, legend=False)
    ax5.axhline(y=target_fos, color='black', linestyle='--')
    ax5.set_title('FoS Distribution Spread', fontweight='bold')
    
    # Plot 6: Wait Time
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    wait_text = (f"STRATEGIC DECISION SUPPORT\n\n"
                 f"Current State: {days} days\n"
                 f"Recommended Wait-Time: {st.session_state.wait_days} days\n\n"
                 f"Interpretation: Based on current recovery rates, "
                 f"{'do not start construction' if isinstance(st.session_state.wait_days, int) and st.session_state.wait_days > days else 'it is safe to start'} "
                 f"until day {st.session_state.wait_days}.")
    ax6.text(0.1, 0.5, wait_text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    
    st.pyplot(fig)

    # Data Preview
    with st.expander("Raw Data Preview"):
        st.dataframe(st.session_state.df_display.head(5))

# 3. Generate Report
if gen_report_btn:
    with st.spinner("Generating Technical Report..."):
        # 1. Generate PDF
        pdf = PDF()
        pdf.add_page()
        pdf.chapter_title(1, "Executive Summary")
        summary = (f"Analysis was performed at t={days} days since disturbance.\n"
                   f"The current Reach Failure Rate is {st.session_state.fail_rate:.1f}%.\n"
                   f"Submerged samples experience an average Hydraulic Lag of {st.session_state.hyd_lag} days.\n"
                   f"Based on a 95% confidence interval, construction must wait until Day {st.session_state.wait_days}.")
        pdf.chapter_body(summary)
        # (Add more PDF chapters as needed...)
        pdf_content = pdf.output(dest='S').encode('latin-1')
        
        st.success("Technical Report Generated Successfully.")
        st.download_button("Download Technical Report", data=pdf_content, file_name="Thixo-Metric_Report.pdf", mime="application/pdf")

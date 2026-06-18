import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import warnings

# Mute the LangChain Community sunset warning
warnings.filterwarnings("ignore", message=".*langchain-community.*")

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Life Stage Financial Leverage Strategist", layout="wide", page_icon="📊")
st.title("Life Stage Financial Leverage Strategist")

# --- Expanded 3-Paragraph Socio-Economic Lead Text ---
st.markdown("""
    <div style='font-size: 1.15rem; line-height: 1.7; border-left: 5px solid #1f77b4; padding: 20px; background-color: #f8faff; border-radius: 6px; margin-bottom: 25px; text-align: justify;'>
        <p style='margin-top: 0; margin-bottom: 15px;'>
            <strong>Global & Developed Economy Paradigms:</strong> In an increasingly interconnected and volatile global economy, a firm's capital structure functions as a primary driver of its macroeconomic resilience and competitive agility. While traditional corporate finance frameworks—pioneered in developed Western markets—largely treat capital structure as a static optimization problem balancing tax shields against bankruptcy overheads, modern market realities demand a more fluid perspective. In highly mature corporate ecosystems, capital structure decisions are intrinsically bound to secular shifts, supply chain reorganizations, and shifting monetary policies, proving that static debt-to-equity formulas fail to safeguard long-term shareholder value.
        </p>
        <p style='margin-bottom: 15px;'>
            <strong>The Asian Corporate Landscape:</strong> Translating these theories into the dynamic Asian business context reveals a unique matrix of institutional constraints and corporate behaviors. Across emerging Asian economies, corporations face distinct capital allocation challenges, characterized by localized credit friction, concentrated ownership structures, and a historical reliance on relationship-based banking systems. In these environments, business growth and entrepreneurial survival are fundamentally dictated by a firm's operational cash flow volatility, turning capital structure management into an active, strategic defense mechanism rather than a passive accounting exercise.
        </p>
        <p style='margin-bottom: 0;'>
            <strong>The Indian Scenario & The IBC Framework:</strong> Within this regional context, the Indian corporate ecosystem stands out as a highly compelling testing ground for financial theory. The formal operationalization of rigorous regulatory frameworks, most notably the Insolvency and Bankruptcy Code (IBC), has fundamentally transformed the default landscape and rewritten the rules of corporate risk-taking in India. By disaggregating capital structure dynamics into sequential corporate life stages, this research engine bridges the gap between traditional accounting abstractions and the real-world operational strains of Indian enterprises, providing a systemic toolkit for corporate survival and credit risk forecasting.
        </p>
    </div>
""", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. SIDEBAR NAVIGATION & CONFIGURATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942269.png", width=100)

# --- Component A: Life Stage Definition ---
st.sidebar.header("A. Definition of Corporate Life Stages")
lifestage_def = st.sidebar.radio(
    "Select Classification Model:",
    (
        "Standard Dickinson (corplifestage)", 
        "Distress Focus (lifestageNdecline)"
    )
)
# Determine which column to use based on selection
target_stage_col = 'lifestageNdecline' if "Distress" in lifestage_def else 'corplifestage'

st.sidebar.divider()

# --- Component B: Data Source Configuration ---
st.sidebar.header("B. Data Source Configuration")
data_source_label = st.sidebar.radio(
    "Select Dataset:",
    (
        "Original Thesis Data (24 Years)", 
        "Appended Thesis Data (25 Years)", 
        "Fresh Download Data (25 Years)"
    )
)

file_map = {
    "Original Thesis Data (24 Years)": "sp401nf24y_furtherEd_oldCLS.dta",
    "Appended Thesis Data (25 Years)": "sp401nf25yrByAppend.dta",
    "Fresh Download Data (25 Years)": "nf400withMktRet25yrs.dta"
}
selected_file = file_map[data_source_label]

st.sidebar.divider()

# --- Component C: Navigational Controls ---
st.sidebar.header("C. Navigational Controls")
analysis_type = st.sidebar.radio(
    "Select Analytical Module:",
    (
        "Aggregate Market View", 
        "Company Drill-Down", 
        "Life Stage Distribution", 
        "CFO Predictive Benchmark",
        "Credit Risk Screener (EWS)",
        "Econometric Research Engine",
        "Automated White Paper",
        "AI Research Assistant (RAG)"
    )
)

st.sidebar.divider()
st.sidebar.info(f"**Active Dataset:**\n{selected_file}\n\n**Active Stage Model:**\n{target_stage_col}")

# ==========================================
# 3. BULLETPROOF DATA LOADER
# ==========================================
@st.cache_data
def load_data(file_name, stage_col_name):
    df = pd.read_stata(file_name)
    df = df.copy() # FIX: De-fragment the massive Stata dataframe
    
    # Fallback safety check: If user selects the new definition but an old dataset
    if stage_col_name not in df.columns:
        stage_col_name = 'corplifestage' # Force fallback to standard
        
    if stage_col_name in df.columns:
        # Standardize the target column to a generic 'active_stage' for the app engine
        df['active_stage'] = df[stage_col_name].astype(str)
        numeric_map = {
            '1': 'Startup', '1.0': 'Startup', '2': 'Growth', '2.0': 'Growth',
            '3': 'Maturity', '3.0': 'Maturity', '4': 'Shakeout1', '4.0': 'Shakeout1',
            '5': 'Shakeout2', '5.0': 'Shakeout2', '6': 'Shakeout3', '6.0': 'Shakeout3',
            '7': 'Decline', '7.0': 'Decline', '8': 'Decay', '8.0': 'Decay'
        }
        df['active_stage'] = df['active_stage'].apply(lambda x: numeric_map.get(x, str(x).capitalize()))
    
    stage_order = ["Startup", "Growth", "Maturity", "Shakeout1", "Shakeout2", "Shakeout3", "Decline", "Decay"]
    
    if 'active_stage' in df.columns:
        df['active_stage'] = pd.Categorical(df['active_stage'], categories=stage_order, ordered=True)
    
    numeric_cols = ['year', 'leverage', 'size', 'prof', 'tang']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df, stage_order, stage_col_name

try:
    data, stage_order, actual_col_used = load_data(selected_file, target_stage_col)
    if actual_col_used != target_stage_col:
        st.warning(f"⚠️ Column '{target_stage_col}' not found in {selected_file}. Fell back to 'corplifestage'.")
    st.markdown(f"**Current Context:** Analyzing {len(data):,} observations from *{data_source_label}* using *{actual_col_used}*.")
except FileNotFoundError:
    st.error(f"⚠️ Data file missing! Please ensure '{selected_file}' is in the same folder as this script (Case Sensitive on Cloud!).")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()

# ==========================================
# VIEW 1: AGGREGATE MARKET VIEW 
# ==========================================
if analysis_type == "Aggregate Market View":
    st.header("Aggregate Market Analysis")
    st.write("Evaluating average financial leverage across the sequential progression of corporate life stages.")
    
    st.subheader("1. Average Leverage by Life Stage Sequence")
    agg_data = data.groupby('active_stage', observed=False)['leverage'].mean().reset_index()
    fig_bar = px.bar(agg_data, x='active_stage', y='leverage', color='active_stage', category_orders={"active_stage": stage_order}, title="Leverage Follows the Corporate Lifecycle")
    st.plotly_chart(fig_bar, width='stretch')

    st.divider()
    st.subheader("2. Time Trends: Aggregate vs. Individual Stages")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        overall_time_data = data.groupby('year')['leverage'].mean().reset_index()
        fig_line_overall = px.line(overall_time_data, x='year', y='leverage', title="Overall Market Trend", markers=True)
        fig_line_overall.update_traces(line_color='black', line_width=3)
        st.plotly_chart(fig_line_overall, width='stretch')

    with col2:
        time_data = data.groupby(['year', 'active_stage'], observed=False)['leverage'].mean().reset_index()
        fig_line_stages = px.line(time_data, x='year', y='leverage', color='active_stage', facet_col='active_stage', facet_col_wrap=4, category_orders={"active_stage": stage_order}, title="Trends Separated by Individual Life Stage")
        fig_line_stages.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_line_stages.update_layout(showlegend=False, height=500) 
        st.plotly_chart(fig_line_stages, width='stretch')

# ==========================================
# VIEW 2: COMPANY DRILL-DOWN
# ==========================================
elif analysis_type == "Company Drill-Down":
    st.header("Company-Specific Transitions")
    companies = sorted(data['companyname'].dropna().unique())
    selected_company = st.selectbox("Search and Select a Company:", companies)
    company_data = data[data['companyname'] == selected_company].sort_values('year')
    
    if not company_data.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Years of Data", len(company_data))
        c2.metric("Latest Year", int(company_data.iloc[-1]['year']))
        c3.metric("Latest Life Stage", company_data.iloc[-1]['active_stage'])
        c4.metric("Latest Leverage", f"{company_data.iloc[-1]['leverage']:.2f}%")
        
        fig_company = px.scatter(company_data, x='year', y='leverage', color='active_stage', size='size', hover_data=['prof'], title=f"Timeline for {selected_company}", category_orders={"active_stage": stage_order})
        fig_company.update_traces(mode='lines+markers') 
        st.plotly_chart(fig_company, width='stretch')
        
        display_cols = [c for c in ['year', 'active_stage', 'leverage', 'size', 'prof', 'tang'] if c in company_data.columns]
        st.dataframe(company_data[display_cols].set_index('year'), width='stretch')

# ==========================================
# VIEW 3: LIFE STAGE DISTRIBUTION
# ==========================================
elif analysis_type == "Life Stage Distribution":
    st.header("Life Stage Deep Dive (Cross-Sectional)")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_stage = st.selectbox("Select a Life Stage to Analyze:", stage_order, index=2)
    with col_filter2:
        years = sorted(data['year'].dropna().unique(), reverse=True)
        selected_year = st.selectbox("Select Year:", years)
    
    stage_data = data[(data['active_stage'] == selected_stage) & (data['year'] == selected_year)]
    all_stages_year_data = data[data['year'] == selected_year]
    
    if not stage_data.empty:
        st.subheader(f"Snapshot of '{selected_stage}' Stage in {int(selected_year)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Number of Firms", len(stage_data))
        m2.metric("Avg Leverage in Stage", f"{stage_data['leverage'].mean():.2f}%")
        m3.metric("Avg Profitability", f"{stage_data['prof'].mean():.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(all_stages_year_data, x="active_stage", y="leverage", color="active_stage", category_orders={"active_stage": stage_order})
            st.plotly_chart(fig_box, width='stretch')
        with col2:
            fig_scatter = px.scatter(stage_data, x="size", y="leverage", color="prof", hover_name="companyname", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_scatter, width='stretch')
    else:
        st.warning(f"No firms found in the '{selected_stage}' stage for the year {selected_year}.")

# ==========================================
# VIEW 4: CFO PREDICTIVE BENCHMARK
# ==========================================
elif analysis_type == "CFO Predictive Benchmark":
    st.header("CFO Predictive Benchmark: Optimal Capital Structure")

    with st.form("cfo_inputs"):
        col1, col2, col3 = st.columns(3)
        with col1:
            current_stage = st.selectbox("Current Corporate Life Stage:", stage_order)
            current_leverage = st.number_input("Current Book Leverage (%)", min_value=0.0, value=25.0)
        with col2:
            prof_input = st.slider("Operating Profitability (%)", min_value=-50.0, max_value=50.0, value=15.0) 
            tang_input = st.slider("Asset Tangibility (%)", min_value=0.0, max_value=100.0, value=30.0) 
        with col3:
            st.info("The model benchmarks your firm against the 50th (Optimal) and 90th (Distressed) percentiles.")
            submit_button = st.form_submit_button(label="Generate Benchmark")

    if submit_button:
        prof = prof_input / 100.0
        tang = tang_input / 100.0
        median_base, tail_base = 2.10, 31.18
        median_prof_coef, tail_prof_coef = -27.44, -73.22
        median_tang_coef, tail_tang_coef = 54.10, 53.90
        
        stage_penalties = {
            "Maturity": {"median": 0.00, "tail": 0.00}, "Growth": {"median": 11.77, "tail": 9.29},
            "Startup": {"median": 19.33, "tail": 25.11}, "Decline": {"median": 11.33, "tail": 25.06},
            "Decay": {"median": 3.87, "tail": 10.04}, "Shakeout1": {"median": 0.90, "tail": 1.07},
            "Shakeout2": {"median": 6.44, "tail": 15.64}, "Shakeout3": {"median": 1.68, "tail": 2.70}
        }
        
        optimal_leverage = max(0, median_base + (prof * median_prof_coef) + (tang * median_tang_coef) + stage_penalties[current_stage]["median"])
        distress_ceiling = max(0, tail_base + (prof * tail_prof_coef) + (tang * tail_tang_coef) + stage_penalties[current_stage]["tail"])
        
        st.divider()
        st.subheader("Capital Structure Diagnosis")
        if current_leverage < optimal_leverage:
            status_color, status_msg = "green", "Under-levered: Excess Debt Capacity Available."
        elif current_leverage > distress_ceiling:
            status_color, status_msg = "red", "Critical Alert: Leverage exceeds distress ceiling."
        else:
            status_color, status_msg = "orange", "Over-levered: Approaching distress ceiling."
            
        st.markdown(f"**Diagnosis:** :{status_color}[{status_msg}]")
        
        fig = go.Figure(go.Indicator(
            mode = "number+gauge+delta", value = current_leverage, domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': optimal_leverage, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'shape': "bullet", 'axis': {'range': [None, max(100, distress_ceiling + 20)]},
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': current_leverage},
                'steps': [{'range': [0, optimal_leverage], 'color': "lightgreen"}, {'range': [optimal_leverage, distress_ceiling], 'color': "navajowhite"}, {'range': [distress_ceiling, max(100, distress_ceiling + 20)], 'color': "lightcoral"}],
                'bar': {'color': "black", 'thickness': 0.1}
            }
        ))
        fig.update_layout(height=200, margin={'t': 20, 'b': 20, 'l': 50, 'r': 50})
        st.plotly_chart(fig, width='stretch')
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Your Current Leverage", f"{current_leverage:.1f}%")
        m2.metric("Optimal Peer Benchmark", f"{optimal_leverage:.1f}%")
        m3.metric("Distress Ceiling (Tail Risk)", f"{distress_ceiling:.1f}%")

# ==========================================
# VIEW 5: CREDIT RISK SCREENER (EWS)
# ==========================================
elif analysis_type == "Credit Risk Screener (EWS)":
    st.header("Credit Risk Screener: Early Warning System")
    max_year = data['year'].max()
    selected_year = st.selectbox("Select Year to Scan:", sorted(data['year'].dropna().unique(), reverse=True), index=0)
    scan_data = data[data['year'] == selected_year].dropna(subset=['leverage', 'prof', 'tang', 'active_stage']).copy()
    
    if not scan_data.empty:
        penalties = {"Maturity": 0.00, "Growth": 9.29, "Startup": 25.11, "Decline": 25.06, "Decay": 10.04, "Shakeout1": 1.07, "Shakeout2": 15.64, "Shakeout3": 2.70}
        def calc_distress(row): return max(0, 31.18 + (row['prof'] * -73.22) + (row['tang'] * 53.90) + penalties.get(row['active_stage'], 0))
        scan_data['Distress_Ceiling'] = scan_data.apply(calc_distress, axis=1)
        scan_data['Risk_Delta'] = scan_data['leverage'] - scan_data['Distress_Ceiling']
        flagged_firms = scan_data[scan_data['Risk_Delta'] > 0].sort_values(by='Risk_Delta', ascending=False)
        
        st.error(f"🚨 Found {len(flagged_firms)} high-risk firms in {int(selected_year)} operating above their statistical distress ceiling.")
        
        display_df = flagged_firms[['companyname', 'industrygroup', 'active_stage', 'prof', 'Distress_Ceiling', 'leverage']].copy()
        display_df['prof'] = (display_df['prof'] * 100).round(2).astype(str) + '%'
        display_df['Distress_Ceiling'] = display_df['Distress_Ceiling'].round(2)
        display_df['leverage'] = display_df['leverage'].round(2)
        st.dataframe(display_df, width='stretch')

# ==========================================
# VIEW 6: ECONOMETRIC RESEARCH ENGINE
# ==========================================
elif analysis_type == "Econometric Research Engine":
    st.header("Econometric Research Engine")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from linearmodels.panel import PanelOLS, RandomEffects, compare
    except ImportError:
        st.error("⚠️ Missing econometric libraries! Please run: `pip install statsmodels linearmodels` in your terminal.")
        st.stop()

    reg_cols = ['companyname', 'year', 'leverage', 'active_stage', 'prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
    missing_cols = [c for c in reg_cols if c not in data.columns]
    
    if missing_cols:
        st.warning(f"The active dataset is missing the following control variables: {missing_cols}")
    else:
        with st.spinner("Preparing Panel Data..."):
            df_reg = data[reg_cols].dropna().copy()
            df_reg = df_reg.sort_values(by=['companyname', 'year'])
            df_reg['L_leverage'] = df_reg.groupby('companyname')['leverage'].shift(1)
            df_panel = df_reg.set_index(['companyname', 'year'])
            formula_smf = "leverage ~ prof + tang + dvnd + taxShield + GFC + ibc2016 + dcovid20less + interest + returnIndexClosing + C(active_stage) + C(year)"
            exog_vars = ['prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']

        st.markdown("""<style>table.simpletable { width: 100%; border-collapse: collapse; margin-bottom: 20px; } table.simpletable td, table.simpletable th { border: 1px solid #ddd; padding: 8px; text-align: center; } table.simpletable th { background-color: #f2f2f2; font-weight: bold; }</style>""", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Simple OLS", "2. FE & RE", "3. Quantile", "4. Margins Analysis", "5. Dynamic Panel & Diagnostics"])

        with tab1:
            if st.button("Run Pooled OLS"):
                ols_model = smf.ols(formula_smf, data=df_reg).fit()
                st.markdown(ols_model.summary().as_html(), unsafe_allow_html=True)

        with tab2:
            if st.button("Run FE/RE Models"):
                exog = sm.add_constant(df_panel[exog_vars])
                stage_dummies = pd.get_dummies(df_panel['active_stage'], drop_first=True, dtype=float)
                exog = pd.concat([exog, stage_dummies], axis=1)
                fe_res = PanelOLS(df_panel['leverage'], exog, entity_effects=True, time_effects=False, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
                re_res = RandomEffects(df_panel['leverage'], exog).fit()
                st.markdown(compare({"Fixed Effects": fe_res, "Random Effects": re_res}).summary.as_html(), unsafe_allow_html=True)

        with tab3:
            if st.button("Run Quantile Regressions"):
                results = []
                for q in [0.10, 0.50, 0.90]:
                    res = smf.quantreg(formula_smf, data=df_reg).fit(q=q, max_iter=2000)
                    df_res = pd.DataFrame({'Coefficient': res.params, 'P-Value': res.pvalues})
                    df_res.columns = [f'Q{int(q*100)} Coef', f'Q{int(q*100)} P-Val']
                    results.append(df_res)
                final_q_table = pd.concat(results, axis=1)
                st.dataframe(final_q_table.style.format("{:.4f}").background_gradient(cmap='Blues', subset=['Q10 Coef', 'Q50 Coef', 'Q90 Coef']), width='stretch')

        with tab4:
            st.subheader("Asymmetric Risk: Marginal Impact of Life Stages")
            if st.button("Generate Grouped Margins Chart"):
                q50 = smf.quantreg(formula_smf, data=df_reg).fit(q=0.50)
                q90 = smf.quantreg(formula_smf, data=df_reg).fit(q=0.90)
                stage_cols = [c for c in q50.params.index if 'active_stage' in c]
                clean_labels = [c.replace("C(active_stage)[T.", "").replace("]", "") for c in stage_cols]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=clean_labels, y=q50.params[stage_cols], name='Normal Firm (50th Percentile)', marker_color='royalblue'))
                fig.add_trace(go.Bar(x=clean_labels, y=q90.params[stage_cols], name='Distressed Firm (90th Percentile)', marker_color='firebrick'))
                fig.update_layout(title='Marginal Increase in Leverage by Life Stage', xaxis_title='Life Stage', yaxis_title='Marginal Effect (%)', barmode='group')
                st.plotly_chart(fig, width='stretch')

        with tab5:
            st.subheader("5. Dynamic Panel (Fixed Effects & IV-GMM)")
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            run_fe = col_btn1.button("Run Dynamic Fixed Effects")
            run_gmm = col_btn2.button("Run Simplified IV-GMM")
            run_ar_test = col_btn3.button("Run AR(1) & AR(2) Diagnostics")
            
            if run_fe:
                try:
                    df_dyn = df_reg.dropna(subset=['L_leverage']).copy()
                    exog_vars_dyn = ['L_leverage', 'prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
                    exog_dyn = sm.add_constant(df_dyn[exog_vars_dyn])
                    stage_dummies_dyn = pd.get_dummies(df_dyn['active_stage'], drop_first=True, dtype=float)
                    exog_dyn = pd.concat([exog_dyn, stage_dummies_dyn], axis=1)
                    df_dyn_panel = df_dyn.set_index(['companyname', 'year'])
                    
                    dyn_res = PanelOLS(df_dyn_panel['leverage'], exog_dyn, entity_effects=True, time_effects=False, drop_absorbed=True).fit(cov_type='robust')
                    st.markdown("### Dynamic Fixed Effects Results")
                    st.markdown(dyn_res.summary.as_html(), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Dynamic FE Calculation Failed: {e}")

            if run_gmm:
                try:
                    try:
                        from linearmodels.iv import IVGMM
                    except ImportError:
                        st.error("Missing linearmodels.iv")
                        st.stop()
                    df_gmm = df_reg.copy()
                    df_gmm['L2_leverage'] = df_gmm.groupby('companyname')['leverage'].shift(2)
                    df_gmm = df_gmm.dropna(subset=['L_leverage', 'L2_leverage']).copy()
                    exog_iv = ['prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
                    exog_df = sm.add_constant(df_gmm[exog_iv])
                    stage_dummies_gmm = pd.get_dummies(df_gmm['active_stage'], drop_first=True, dtype=float)
                    exog_df = pd.concat([exog_df, stage_dummies_gmm], axis=1)
                    endog_df = df_gmm[['L_leverage']]
                    instr_df = df_gmm[['L2_leverage']]
                    dep_df = df_gmm[['leverage']]
                    gmm_res = IVGMM(dependent=dep_df, exog=exog_df, endog=endog_df, instruments=instr_df).fit(cov_type='clustered', clusters=df_gmm['companyname'])
                    st.markdown("### Instrumental Variable GMM")
                    st.markdown(gmm_res.summary.as_html(), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"IV-GMM Calculation Failed: {e}")

            if run_ar_test:
                try:
                    df_dyn = df_reg.dropna(subset=['L_leverage']).copy()
                    exog_vars_dyn = ['L_leverage', 'prof', 'tang', 'dvnd', 'taxShield', 'GFC', 'ibc2016', 'dcovid20less', 'interest', 'returnIndexClosing']
                    exog_dyn = sm.add_constant(df_dyn[exog_vars_dyn])
                    stage_dummies_dyn = pd.get_dummies(df_dyn['active_stage'], drop_first=True, dtype=float)
                    exog_dyn = pd.concat([exog_dyn, stage_dummies_dyn], axis=1)
                    df_dyn_panel = df_dyn.set_index(['companyname', 'year'])
                    dyn_res = PanelOLS(df_dyn_panel['leverage'], exog_dyn, entity_effects=True, time_effects=False, drop_absorbed=True).fit(cov_type='robust')
                    
                    resid_df = pd.DataFrame({'resid': dyn_res.resids})
                    resid_df['L1_resid'] = resid_df.groupby(level='companyname')['resid'].shift(1)
                    resid_df['L2_resid'] = resid_df.groupby(level='companyname')['resid'].shift(2)
                    
                    resid_test_df = resid_df.dropna().reset_index()
                    
                    ar1_model = smf.ols('resid ~ L1_resid', data=resid_test_df).fit(cov_type='HC1')
                    ar2_model = smf.ols('resid ~ L2_resid', data=resid_test_df).fit(cov_type='HC1')
                    ar_results = pd.DataFrame({
                        'Test': ['AR(1) Test (Lag 1)', 'AR(2) Test (Lag 2)'],
                        'Coefficient': [ar1_model.params['L1_resid'], ar2_model.params['L2_resid']],
                        'P-Value': [ar1_model.pvalues['L1_resid'], ar2_model.pvalues['L2_resid']]
                    })
                    st.table(ar_results.style.format({'Coefficient': '{:.4f}', 'P-Value': '{:.4f}'}))
                except Exception as e:
                    st.error(f"AR Diagnostics Failed: {e}")

# ==========================================
# VIEW 7: AUTOMATED WHITE PAPER
# ==========================================
elif analysis_type == "Automated White Paper":
    
    st.markdown("""
        <style>
        .wp-text { font-size: 16px; line-height: 1.8; text-align: justify; margin-bottom: 20px; font-family: 'Georgia', serif;}
        .wp-h1 { text-align: center; font-size: 42px; font-weight: bold; margin-bottom: 10px; font-family: 'Arial', sans-serif; }
        .wp-h2 { text-align: center; font-size: 24px; color: #555; margin-bottom: 40px; font-family: 'Arial', sans-serif; }
        .wp-h3 { font-size: 26px; font-weight: bold; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 5px;}
        .wp-quote { border-left: 5px solid #0056b3; padding-left: 15px; font-style: italic; background-color: #f9f9f9; padding: 10px; margin-bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.header("📄 Generate Exhaustive Academic White Paper")
    audience = st.radio("Select Target Audience Context:", ["Academic / Peer Review", "Practitioner / CFO Advisory"], horizontal=True)
    
    if st.button("Compile and Generate Comprehensive Document"):
        with st.spinner("Synthesizing multi-dimensional data, calculating advanced econometrics, and drafting comprehensive manuscript..."):
            
            try:
                import statsmodels.formula.api as smf
            except ImportError:
                st.error("Missing `statsmodels`. Please install to generate the full white paper.")
                st.stop()

            total_obs = len(data)
            unique_firms = data['companyname'].nunique() if 'companyname' in data.columns else "N/A"
            start_year = int(data['year'].min())
            end_year = int(data['year'].max())
            panel_length = end_year - start_year + 1
            
            extreme_df = data[data['leverage'] > 100]
            normal_df = data[data['leverage'] <= 100]
            ext_count = len(extreme_df)
            
            if ext_count > 0:
                ext_prof = (extreme_df['prof'].mean() * 100) if pd.notnull(extreme_df['prof'].mean()) else 0
                norm_prof = (normal_df['prof'].mean() * 100) if pd.notnull(normal_df['prof'].mean()) else 0
                top_ext_stage = extreme_df['active_stage'].mode()[0] if not extreme_df.empty else "N/A"
            else:
                ext_prof, norm_prof, top_ext_stage = 0, 0, "N/A"

            yearly_avg = data.groupby('year')['leverage'].mean()
            peak_year = int(yearly_avg.idxmax())
            peak_lev = yearly_avg.max()
            trough_year = int(yearly_avg.idxmin())
            trough_lev = yearly_avg.min()
            start_lev = yearly_avg.iloc[0]
            end_lev = yearly_avg.iloc[-1]
            
            startup_lev = data[data['active_stage'] == 'Startup']['leverage'].mean()
            growth_lev = data[data['active_stage'] == 'Growth']['leverage'].mean()
            maturity_lev = data[data['active_stage'] == 'Maturity']['leverage'].mean()
            decline_lev = data[data['active_stage'] == 'Decline']['leverage'].mean()
            decay_lev = data[data['active_stage'] == 'Decay']['leverage'].mean()
            
            latest_year_data = data[data['year'] == end_year].dropna(subset=['leverage', 'prof', 'tang', 'active_stage']).copy()
            penalties = {"Maturity": 0.00, "Growth": 9.29, "Startup": 25.11, "Decline": 25.06, "Decay": 10.04, "Shakeout1": 1.07, "Shakeout2": 15.64, "Shakeout3": 2.70}
            def calc_distress(row): return max(0, 31.18 + (row['prof'] * -73.22) + (row['tang'] * 53.90) + penalties.get(row['active_stage'], 0))
            
            if not latest_year_data.empty:
                latest_year_data['Distress_Ceiling'] = latest_year_data.apply(calc_distress, axis=1)
                latest_year_data['Risk_Delta'] = latest_year_data['leverage'] - latest_year_data['Distress_Ceiling']
                flagged_count = len(latest_year_data[latest_year_data['Risk_Delta'] > 0])
                distress_pct = (flagged_count / len(latest_year_data)) * 100
            else:
                flagged_count, distress_pct = 0, 0

            df_reg = data[['leverage', 'prof', 'tang', 'active_stage']].dropna()
            ols_base = smf.ols("leverage ~ prof + tang + C(active_stage)", data=df_reg).fit()
            q50 = smf.quantreg("leverage ~ prof + tang + C(active_stage)", data=df_reg).fit(q=0.50)
            q90 = smf.quantreg("leverage ~ prof + tang + C(active_stage)", data=df_reg).fit(q=0.90)
            prof_coef = ols_base.params['prof']
            tang_coef = ols_base.params['tang']
            stage_cols = [c for c in q50.params.index if 'active_stage' in c]
            clean_labels = [c.replace("C(active_stage)[T.", "").replace("]", "") for c in stage_cols]

            st.divider()
            st.markdown(f"<div class='wp-h1'>Financial Leverage Decisions: Responsible Finance during Distress</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-h2'>Empirical Evidence from Top Indian Non-Financial Firms ({start_year} - {end_year})</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><em>Active Classification Matrix: {target_stage_col} | N = {total_obs:,}</em></p>", unsafe_allow_html=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            
            st.markdown("<div class='wp-h3'>Chapter 1: Executive Summary & Abstract</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>This comprehensive white paper synthesizes an exhaustive empirical analysis of the determinants of capital structure among non-financial firms listed in the S&P BSE 500 index. Spanning an extensive {panel_length}-year longitudinal timeline from {start_year} to {end_year}, the dataset under examination encompasses **{total_obs:,} firm-year observations** derived from **{unique_firms} unique corporate entities**. </div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>Moving decisively beyond conventional, static industry-average analyses, this paper pioneers the operationalization of cash flow-based life stage methodologies. By categorizing firms based on the net combinations of their operating, investing, and financing cash flows, we establish that optimal financial leverage is a dynamic target. Furthermore, utilizing advanced econometrics including Dynamic Fixed Effects (FE) panel models and Quantile Regression, this paper establishes mathematically rigorous predictive benchmarks and systemic Early Warning Systems (EWS) for credit risk, with a distinct focus on survival and revival strategies for distressed firms.</div>", unsafe_allow_html=True)
            
            if audience == "Practitioner / CFO Advisory":
                st.markdown(f"<div class='wp-quote'><strong>Executive Strategic Imperative:</strong> The reliance on static, sector-wide debt targets actively destroys shareholder value. As of the end of {end_year}, the live EWS algorithm reveals that <strong>{distress_pct:.1f}%</strong> of analyzed firms are operating dangerously above their mathematically derived 'Distress Ceilings.' CFOs must pivot to dynamic, life-stage-adjusted capital allocation strategies to optimize WACC and execute timely interventions before decline transitions into irreversible decay.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='wp-quote'><strong>Primary Academic Contribution:</strong> This research fundamentally extends the Pecking Order and Trade-Off theories. By utilizing Quantile Regression, this paper mathematically proves that the penalizing effects of negative profitability and operational decline are highly asymmetric, exacerbating tail-risk. It critically distinguishes between 'Decline' and 'Decay' phenomena within the Indian macroeconomic context.</div>", unsafe_allow_html=True)

            st.markdown("<div class='wp-h3'>Chapter 2: Theoretical Framework & Literature Context</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>Historically, capital structure theory has been dominated by the <strong>Trade-Off Theory</strong> (balancing tax shields against bankruptcy costs) and the <strong>Pecking Order Theory</strong> (preferring internal financing over debt due to information asymmetry). However, both theories traditionally assume firm homogeneity over time. A firm's capacity to absorb debt and its ability to generate internal funds are not static; they change radically as the underlying business matures amidst intensifying global competition.</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>To bridge this gap, this research introduces a vital third dimension: <strong>The Corporate Life Stage</strong>. By mapping the algebraic signs (+/-) of a firm's Operating, Investing, and Financing cash flows, we classify firms into dynamic stages. This paper hypothesizes that a firm's optimal capital structure is inextricably tethered to its current placement within this framework, shifting the perspective of finance from mere accounting to an active survival tool.</div>", unsafe_allow_html=True)

            st.markdown("<div class='wp-h3'>Chapter 3: The Mechanics of Distress — Decline vs. Decay</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>A cornerstone of this research is the critical disaggregation of financial distress into two distinct phases: <strong>Decline</strong> and <strong>Decay</strong>. While often conflated in traditional literature, their risk profiles and financial behaviors are entirely divergent.</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'><strong>The Decline Phase:</strong> Firms in the Decline stage typically exhibit insufficient operational cash flows as product viability wanes. Crucially, however, they display a <em>positive</em> financing cash flow. This indicates that the firm still possesses market credibility and is actively seeking external capital to orchestrate a turnaround or pivot strategy. Because they are investing in recovery, firms in early Decline exhibit a tangibly higher probability of survival.</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'><strong>The Decay Phase:</strong> Conversely, Decay represents a structural point of no return. Firms in Decay exhibit <em>negative</em> financing cash flows coupled with net asset sales. This signature indicates that the firm is actively being wound down; it is liquidating assets not to reinvest, but merely to repay outstanding liabilities to trapped creditors. The probability of survival drops precipitously as the firm inches toward formal bankruptcy procedures.</div>", unsafe_allow_html=True)

            st.markdown("<div class='wp-h3'>Chapter 4: Macroeconomic Dynamics & Secular Deleveraging</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>The overarching narrative of the Indian corporate sector from {start_year} to {end_year} is one of structural deleveraging, falling from <strong>{start_lev:.1f}%</strong> to <strong>{end_lev:.1f}%</strong>. Yet, this masks immense cyclical volatility. As Figure 1 demonstrates, corporate leverage spiked to <strong>{peak_lev:.1f}% in {peak_year}</strong>, reflecting aggressive debt-fueled capital expenditure cycles. Conversely, rigorous deleveraging mandates (e.g., the Insolvency and Bankruptcy Code of 2016) fundamentally altered the consequence of default in India, driving the market to a trough of <strong>{trough_lev:.1f}% in {trough_year}</strong>.</div>", unsafe_allow_html=True)
            
            overall_time_data = data.groupby('year')['leverage'].mean().reset_index()
            fig1 = px.line(overall_time_data, x='year', y='leverage', markers=True, title="Figure 1: Aggregate Market Leverage Over Time")
            fig1.update_traces(line_color='#2E4053', line_width=3)
            fig1.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(240,240,240,0.5)')
            st.plotly_chart(fig1, width='stretch')

            st.markdown("<div class='wp-h3'>Chapter 5: The Empirical U-Shaped Leverage Curve</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>Disaggregating the data by life stage validates the U-shaped leverage curve. Highly capital-intensive <strong>Startup</strong> and <strong>Growth</strong> firms rely heavily on external financing (averaging <strong>{startup_lev:.1f}%</strong> and <strong>{growth_lev:.1f}%</strong> respectively). As firms reach <strong>Maturity</strong>, they utilize stable internal accruals to systematically retire debt, dragging average leverage down to <strong>{maturity_lev:.1f}%</strong> in strict adherence to the Pecking Order Theory.</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>However, as firms enter the <strong>Decline</strong> stage, operating cash flows evaporate, and accumulated losses erode the equity base, causing leverage to structurally spike back up to <strong>{decline_lev:.1f}%</strong>. For firms that fail to execute a turnaround and cascade into <strong>Decay</strong>, the forced liquidation of assets to repay creditors is reflected in a shifting leverage dynamic (averaging <strong>{decay_lev:.1f}%</strong>).</div>", unsafe_allow_html=True)
            
            agg_data = data.groupby('active_stage', observed=False)['leverage'].mean().reset_index()
            fig2 = px.bar(agg_data, x='active_stage', y='leverage', color='active_stage', category_orders={"active_stage": stage_order}, title="Figure 2: The U-Shaped Leverage Curve by Corporate Life Stage")
            fig2.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), showlegend=False, plot_bgcolor='rgba(240,240,240,0.5)')
            st.plotly_chart(fig2, width='stretch')

            st.markdown("<div class='wp-h3'>Chapter 6: Econometric Methodology & The Dynamic Panel Defense</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>Baseline Pooled OLS confirms foundational theories: Operating Profitability exhibits a highly significant negative coefficient (<strong>" + str(round(prof_coef, 2)) + "</strong>), while Asset Tangibility yields a positive coefficient (<strong>" + str(round(tang_coef, 2)) + "</strong>). To account for capital structure adjustment costs, a dynamic panel model incorporating a lagged dependent variable ($L.leverage$) is strictly necessary.</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>While Generalized Method of Moments (System GMM) is common, it suffers from fatal 'instrument proliferation' in long panels. Because this dataset spans an extended panel of <strong>$T = {panel_length}$ years</strong>, the Nickell bias mathematically converges toward zero. Deploying a <strong>Dynamic Fixed Effects (FE) model</strong> is statistically robust, stripping out unobserved firm-level heterogeneity without artificially overfitting endogenous variables.</div>", unsafe_allow_html=True)

            st.markdown("<div class='wp-h3'>Chapter 7: Asymmetric Risk & Quantile Margin Analysis</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>Utilizing <strong>Quantile Regression (50th vs 90th percentiles)</strong>, this paper proves that structural credit risk is highly asymmetric. Figure 3 illustrates this clarity. If a healthy, median-levered firm (Blue Bar) enters the 'Decline' stage, its leverage barely shifts due to robust equity buffers. However, if a heavily indebted firm at the 90th percentile (Red Bar) enters 'Decline', it experiences a massive, uncontrollable marginal spike in leverage. Credit risk compounds exponentially at the tail.</div>", unsafe_allow_html=True)
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=clean_labels, y=q50.params[stage_cols], name='Normal Firm (50th Percentile)', marker_color='#2E86C1'))
            fig3.add_trace(go.Bar(x=clean_labels, y=q90.params[stage_cols], name='Highly Levered Firm (90th Percentile)', marker_color='#C0392B'))
            fig3.update_layout(title='Figure 3: The Asymmetric Marginal Increase in Leverage (Baseline = Maturity)', barmode='group', height=450, plot_bgcolor='rgba(240,240,240,0.5)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig3, width='stretch')

            st.markdown("<div class='wp-h3'>Chapter 8: The Tail-Risk — Zombie Firms & Extreme Leverage</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='wp-text'>Standard averages obscure the extremities of financial distress. Within the {data_source_label} dataset, <strong>{ext_count} observations</strong> recorded book leverage exceeding 100%—indicating negative equity. Unlike healthy participants operating with <strong>{norm_prof:.1f}%</strong> profitability, these extreme outliers bleed capital at <strong>{ext_prof:.1f}%</strong>. Fascinatingly, the most common life stage for these distressed entities is <strong>{top_ext_stage}</strong>. This empirically validates the 'Zombie Firm' phenomenon, where structurally declining entities are kept artificially alive on life support via continuous debt rollovers from trapped creditors.</div>", unsafe_allow_html=True)

            st.markdown("<div class='wp-h3'>Chapter 9: Systemic Risk & Responsible Finance</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>The ultimate utility of this research lies in its predictive capacity. By mapping the 90th percentile of the leverage distribution, this model establishes a definitive 'Distress Ceiling'. When a firm breaches this bespoke, stage-adjusted ceiling, its probability of default accelerates exponentially. This necessitates a framework of <strong>Responsible Finance</strong>—timely interventions and customized financial tools tailored by life stage. Asset-backed lending and tax shields, for example, become highly relevant in later stages.</div>", unsafe_allow_html=True)
            
            wp_err_msg = f"🚨 **EWS LIVE MARKET SCAN ({end_year}):** Scanning {len(latest_year_data)} active firms reveals that **{flagged_count} entities** are operating with leverage ratios critically exceeding their Distress Ceilings. These firms require immediate deleveraging interventions."
            st.error(wp_err_msg)

            st.markdown("<div class='wp-h3'>Chapter 10: Conclusion & Actionable Takeaways</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'>This research fundamentally disrupts the assumption that capital structure optimization is a static exercise. Financing decisions are organically tethered to the operational lifecycle.</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'><strong>1. Timely Intervention Boosts Revival Odds:</strong> Firms in initial Decline have tangibly higher chances of recovery if proactive financing and investment/divestment choices are made early.</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'><strong>2. Tailored Strategies for Distress:</strong> Decline and Decay require fundamentally unique financial approaches. A strategy built for Decline will fail in Decay.</div>", unsafe_allow_html=True)
            st.markdown("<div class='wp-text'><strong>3. The Macro-Policy Imperative:</strong> The persistence of Zombie Firms highlights the absolute necessity for the unhindered application of bankruptcy frameworks. Capital trapped in decaying firms stifles macroeconomic growth; it must be efficiently liquidated and reallocated.</div>", unsafe_allow_html=True)

            st.divider()
            st.caption("🖨️ **Export Instructions:** This expanded comprehensive white paper has been dynamically generated based on your dataset selection. To export, press `Ctrl + P` (or `Cmd + P`) and select 'Save as PDF'.")

# ==========================================
# VIEW 8: AI RESEARCH ASSISTANT (REAL RAG - GEMINI) - LCEL COMPLIANT
# ==========================================
elif analysis_type == "AI Research Assistant (RAG)":
    st.header("🤖 AI Literature & Methodology Assistant")
    st.write("Ask theoretical questions. The AI retrieves perfectly cited answers directly from the PDFs in your `literature_pdfs/` folder.")

    # --- 1. API KEY INPUT & DATABASE CONTROLS ---
    st.sidebar.divider()
    st.sidebar.subheader("🔑 AI Configuration")
    api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password", help="Required to read PDFs and generate answers via Google Generative AI.")
    
    if st.sidebar.button("🔄 Reload Literature Database", help="Click this if you have added new PDFs to the folder."):
        st.cache_resource.clear()
        st.rerun()

    if not api_key:
        st.warning("⚠️ **Waiting for API Key:** Please enter your Google Gemini API key in the sidebar to initialize the AI Researcher.")
    else:
        # --- 2. INITIALIZE REAL RAG BACKEND (CACHED VIA LCEL) ---
        @st.cache_resource(show_spinner=False)
        def initialize_rag_system(key):
            try:
                os.environ["GOOGLE_API_KEY"] = key
                from pypdf import PdfReader
                from langchain_core.documents import Document
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
                from langchain_community.vectorstores import FAISS  
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.runnables import RunnablePassthrough
                from langchain_core.output_parsers import StrOutputParser
                
                # Check if folder exists
                if not os.path.exists("literature_pdfs"):
                    return None, "The folder `literature_pdfs/` does not exist. Please create it and add your PDFs."
                
                # Native Future-Proof PDF Loader
                docs = []
                directory = "literature_pdfs/"
                for filename in os.listdir(directory):
                    if filename.endswith(".pdf"):
                        filepath = os.path.join(directory, filename)
                        reader = PdfReader(filepath)
                        text = ""
                        for page in reader.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted + "\n"
                        docs.append(Document(page_content=text, metadata={"source": filename}))
                
                if not docs:
                    return None, "No PDFs found in the `literature_pdfs/` folder. Please add your academic papers."
                
                # Split text into searchable paragraphs
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Create the Vector Database using FAISS (Streamlit safe)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                
                # --- ANTI-CRASH BATCHING LOGIC ---
                batch_size = 80 
                vectorstore = FAISS.from_documents(documents=splits[:batch_size], embedding=embeddings)
                
                for i in range(batch_size, len(splits), batch_size):
                    time.sleep(60) 
                    batch = splits[i : i + batch_size]
                    vectorstore.add_documents(batch)
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # Create the LLM Model Brain
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                
                # Setup Academic Prompt Engineering
                system_prompt = (
                    "You are an elite academic research assistant for a doctoral student in corporate finance.\n"
                    "Use the following retrieved pieces of literature to comprehensively answer the question.\n"
                    "If you don't know the answer based on the context, state that clearly.\n"
                    "Always explicitly cite the authors and year (e.g., Dickinson, 2011) if the information is available in the context.\n\n"
                    "Context: {context}\n\n"
                    "Question: {input}\n\n"
                    "Answer:"
                )
                prompt = ChatPromptTemplate.from_template(system_prompt)
                
                # Helper function to join retrieved source docs
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # Build LCEL chain
                lcel_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                return (lcel_chain, retriever), "Success"
            except Exception as e:
                return None, f"An error occurred during initialization: {e}"

        with st.spinner("Initializing Vector Database from `literature_pdfs/` (This only happens once, but may take a few minutes to respect rate limits)..."):
            rag_output, status_msg = initialize_rag_system(api_key)
            
        if rag_output is None:
            st.error(f"🚨 **Initialization Failed:** {status_msg}")
        else:
            st.success("✅ **AI Researcher is online and has indexed your literature!**")
            lcel_chain, retriever = rag_output

            # --- 3. STREAMLIT CHAT UI ---
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I am your AI Literature Assistant. Click a suggested question below, or type your own specific query in the chat box to search your PDFs."}
                ]

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # --- 4. EXPANDED PRE-DEFINED QUESTIONS GRID ---
            st.write("") 
            st.caption("💡 **Suggested Research Queries:**")
            
            active_prompt = None
            
            st.markdown("**1. Theoretical Foundations**")
            col1, col2, col3 = st.columns(3)
            if col1.button("Pecking Order vs Trade-Off?", width='stretch'):
                active_prompt = "Compare the Pecking Order and Trade-Off theories based on the literature."
            if col2.button("Role of asymmetric information?", width='stretch'):
                active_prompt = "How does asymmetric information drive capital structure decisions?"
            if col3.button("Agency costs of debt?", width='stretch'):
                active_prompt = "Explain the agency costs associated with high debt levels."

            st.markdown("**2. Corporate Life Stages**")
            col4, col5, col6 = st.columns(3)
            if col4.button("Define a 'Shakeout' firm?", width='stretch'):
                active_prompt = "How does Dickinson (2011) classify a 'Shakeout' firm using cash flows?"
            if col5.button("Why do 'Maturity' firms deleverage?", width='stretch'):
                active_prompt = "Based on the literature, why do 'Maturity' firms deleverage so aggressively?"
            if col6.button("Startup financing constraints?", width='stretch'):
                active_prompt = "What are the unique external financing constraints faced by 'Startup' stage firms?"

            st.markdown("**3. Methodology & Macro Policy**")
            col7, col8, col9 = st.columns(3)
            if col7.button("What is Nickell bias?", width='stretch'):
                active_prompt = "Explain the Nickell bias in dynamic panel models and when it diminishes."
            if col8.button("Macro risks of Zombie Firms?", width='stretch'):
                active_prompt = "What are the macroeconomic and systemic risks of Zombie Firms?"
            if col9.button("Impact of the IBC 2016?", width='stretch'):
                active_prompt = "How does the implementation of the Insolvency and Bankruptcy Code (IBC) 2016 alter deleveraging behavior?"

            # --- 5. CUSTOM CHAT INPUT ---
            user_input = st.chat_input("Or type your own theoretical question here to search your PDFs...")
            if user_input:
                active_prompt = user_input

            # --- 6. EXECUTE THE LCEL PIPELINE ---
            if active_prompt:
                st.session_state.messages.append({"role": "user", "content": active_prompt})
                with st.chat_message("user"):
                    st.markdown(active_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Scanning PDFs and synthesizing answer using Gemini..."):
                        
                        # Invoke the modern unified expression pipeline 
                        answer = lcel_chain.invoke(active_prompt)
                        
                        # Explicitly invoke retriever separately to populate source expansions inside the UI
                        source_documents = retriever.invoke(active_prompt)

                        st.markdown(answer)

                        # Show the exact source context pulled from the PDFs
                        if source_documents:
                            with st.expander("🔍 View Source References from your PDFs"):
                                for i, doc in enumerate(source_documents):
                                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                                    st.info(f"**Source {i+1}:** {source_file}\n\n**Excerpt:** {doc.page_content}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
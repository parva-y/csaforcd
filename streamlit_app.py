import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime

st.title("CoinDCX Branding Campaign Impact Dashboard")

# Upload section
search_file = st.file_uploader("Upload Search Volume CSV", type=["csv"])
spends_file = st.file_uploader("Upload Campaign Spends CSV", type=["csv"])
campaign_file = st.file_uploader("Upload Campaign Details CSV", type=["csv"])

if search_file and spends_file:
    # Load and preprocess
    search_df = pd.read_csv(search_file)
    spends_df = pd.read_csv(spends_file)

    search_df['Date'] = pd.to_datetime(search_df['Date'], dayfirst=True, errors='coerce')
    spends_df['Date'] = pd.to_datetime(spends_df['Date'], dayfirst=True, errors='coerce')

    merged = search_df[['Date', 'CoinDCX', 'Installs - DCX']].merge(
        spends_df[['Date', 'Spends']], on='Date', how='left')

    merged['Spends'] = pd.to_numeric(merged['Spends'], errors='coerce').fillna(0)
    merged['CoinDCX'] = pd.to_numeric(merged['CoinDCX'], errors='coerce')
    merged['Installs - DCX'] = pd.to_numeric(merged['Installs - DCX'], errors='coerce')

    # Date boundaries
    campaign_start = datetime(2024, 11, 11)
    campaign_end = datetime(2025, 3, 23)

    merged['during_campaign'] = ((merged['Date'] >= campaign_start) & (merged['Date'] <= campaign_end)).astype(int)
    merged['time_index'] = np.arange(len(merged))
    merged['time_post_campaign'] = merged['time_index'] * merged['during_campaign']

    st.subheader("Interrupted Time Series (ITS) Analysis")
    if st.button("Run ITS Model"):
        its_df = merged[['CoinDCX', 'time_index', 'during_campaign', 'time_post_campaign']].dropna()
        if len(its_df) < 30:
            st.warning("⚠️ Too few data points for robust ITS analysis. Results may be unreliable.")
        X = sm.add_constant(its_df[['time_index', 'during_campaign', 'time_post_campaign']])
        y = its_df['CoinDCX']
        model = sm.OLS(y, X).fit()
        st.text(model.summary())
        st.caption("Campaign effect is indicated by the coefficient on 'during_campaign'. Positive and significant suggests a lift in searches.")

    st.subheader("Difference-in-Differences (DiD) Setup")
    competitors = ['Angel One', 'Binance', 'Coinswitch', 'Delta Exchange', 'Dhan',
                   'Groww', 'Lemonn', 'Upstox', 'WazirX', 'Zerodha', 'Mudrex']

    if st.button("Run DiD Model"):
        long_df = pd.melt(search_df, id_vars=['Date'], value_vars=['CoinDCX'] + competitors,
                          var_name='brand', value_name='search_volume')
        long_df['is_CoinDCX'] = (long_df['brand'] == 'CoinDCX').astype(int)
        long_df['during_campaign'] = ((long_df['Date'] >= campaign_start) & (long_df['Date'] <= campaign_end)).astype(int)
        long_df['DiD_interaction'] = long_df['is_CoinDCX'] * long_df['during_campaign']
        long_df['search_volume'] = pd.to_numeric(long_df['search_volume'], errors='coerce')
        did_df = long_df.dropna(subset=['search_volume'])

        if did_df.groupby(['brand']).size().min() < 20:
            st.warning("⚠️ Not enough observations per brand for stable DiD estimation.")

        X_did = sm.add_constant(did_df[['is_CoinDCX', 'during_campaign', 'DiD_interaction']])
        y_did = did_df['search_volume']
        did_model = sm.OLS(y_did, X_did).fit()
        st.text(did_model.summary())
        st.caption("Key coefficient is on 'DiD_interaction': positive & significant means CoinDCX had a unique lift.")

    st.subheader("ROI Calculator")
    if st.checkbox("Show ROI Estimates"):
        before_avg = merged.loc[merged['during_campaign'] == 0, 'Installs - DCX'].mean()
        during_avg = merged.loc[merged['during_campaign'] == 1, 'Installs - DCX'].mean()
        incremental_installs = during_avg - before_avg
        total_spends = merged.loc[merged['during_campaign'] == 1, 'Spends'].sum()

        if np.isnan(incremental_installs) or np.isnan(total_spends):
            st.warning("⚠️ Insufficient data to compute ROI.")
        else:
            st.metric("Incremental Installs per Day", f"{incremental_installs:.0f}")
            st.metric("Total Spends During Campaign", f"₹{total_spends:,.0f}")
            if total_spends > 0:
                cost_per = total_spends / (incremental_installs * len(merged[merged['during_campaign'] == 1]))
                st.metric("Cost per Incremental Install", f"₹{cost_per:.2f}")
            st.caption("This assumes installs are influenced primarily by branding during campaign days.")

    st.subheader("Regression: Predict Searches")
    if st.button("Run Regression Model"):
        reg_df = merged.dropna(subset=['CoinDCX', 'Spends'])
        if len(reg_df) < 30:
            st.warning("⚠️ Too few data points for reliable regression. Results may be overfit.")
        X_reg = sm.add_constant(reg_df[['Spends', 'during_campaign']])
        y_reg = reg_df['CoinDCX']
        reg_model = sm.OLS(y_reg, X_reg).fit()
        st.text(reg_model.summary())
        st.caption("Interpret coefficients with caution, especially if 'Spends' have many zeros or are auto-filled.")

    st.subheader("Granger Causality: Does Spend Predict Searches?")
    if st.button("Run Granger Test"):
        from statsmodels.tsa.stattools import grangercausalitytests
        granger_df = merged[['CoinDCX', 'Spends']].dropna()
        if len(granger_df) < 40:
            st.warning("⚠️ Granger test needs more observations (ideally >40) for stability.")
        else:
            granger_result = grangercausalitytests(granger_df, maxlag=5, verbose=False)
            for lag in granger_result:
                f_test = granger_result[lag][0]['ssr_ftest']
                st.write(f"Lag {lag} - F-stat: {f_test[0]:.2f}, p-value: {f_test[1]:.4f}")
            st.caption("Low p-values suggest spends Granger-cause search spikes. Results depend on stationarity & lag choice.")

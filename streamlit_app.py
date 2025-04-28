# CoinDCX Branding Campaign Analysis App
# 
# Installation requirements:
# pip install streamlit pandas plotly numpy
# For Excel support: pip install openpyxl
#
# Run with: streamlit run this_file.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="CoinDCX Campaign Analysis",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("CoinDCX Branding Campaign Analysis")
st.markdown("""
This app analyzes the effect of CoinDCX's branding campaign by comparing:
- Search volume trends for CoinDCX vs competitors
- Campaign metrics against brand performance indicators
- ROI and effectiveness analysis
""")

# File upload section
st.header("Data Upload")
col1, col2 = st.columns(2)

with col1:
    competitors_file = st.file_uploader("Upload Competitor Search Data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    competitors_format = st.radio("Competitor File Format", ["CSV", "Excel"], horizontal=True, key="comp_format")

with col2:
    campaign_file = st.file_uploader("Upload Campaign Metrics Data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    campaign_format = st.radio("Campaign File Format", ["CSV", "Excel"], horizontal=True, key="camp_format")

# Function to clean and process competitor data
def process_competitor_data(df):
    # Ensure date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Set date as index
    df = df.set_index('Date')
    
    return df

# Function to clean and process campaign data
def process_campaign_data(df):
    # There are two sections in the file - 'Top Impact/ Roadblock' and date-based metrics
    # We'll focus on the date-based metrics section
    
    # Find where the date-based metrics start
    date_section_start = df.iloc[:, 0].str.contains('Date', na=False).idxmax()
    
    # Extract the date-based metrics part
    metrics_df = df.iloc[date_section_start:].copy()
    
    # Set the column names from the first row
    metrics_df.columns = metrics_df.iloc[0]
    metrics_df = metrics_df.iloc[1:].reset_index(drop=True)
    
    # Convert date to datetime format
    metrics_df['Date'] = pd.to_datetime(metrics_df['Date'], dayfirst=True, errors='coerce')
    
    # Convert numeric columns to float
    for col in metrics_df.columns:
        if col != 'Date':
            metrics_df[col] = pd.to_numeric(metrics_df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Set date as index
    metrics_df = metrics_df.set_index('Date')
    
    return metrics_df

# Process data if files are uploaded
if competitors_file and campaign_file:
    # Load competitor data
    try:
        if competitors_format == "CSV":
            comp_df = pd.read_csv(competitors_file)
        else:
            # For Excel files, try using pandas directly or fallback to alternative methods
            try:
                comp_df = pd.read_excel(competitors_file)
            except ImportError:
                st.error("Excel library (openpyxl) not available. Please use CSV format or install openpyxl.")
                st.stop()
        comp_df = process_competitor_data(comp_df)
    except Exception as e:
        st.error(f"Error processing competitor data: {e}")
        st.stop()
    
    # Load campaign data
    try:
        if campaign_format == "CSV":
            campaign_df = pd.read_csv(campaign_file)
        else:
            # For Excel files, try using pandas directly or fallback to alternative methods
            try:
                campaign_df = pd.read_excel(campaign_file)
            except ImportError:
                st.error("Excel library (openpyxl) not available. Please use CSV format or install openpyxl.")
                st.stop()
        campaign_df = process_campaign_data(campaign_df)
    except Exception as e:
        st.error(f"Error processing campaign data: {e}")
        st.stop()
    
    # Display the clean data
    st.header("Processed Data")
    
    with st.expander("Competitor Search Data"):
        st.dataframe(comp_df)
    
    with st.expander("Campaign Metrics Data"):
        st.dataframe(campaign_df)
    
    # Analysis section
    st.header("Campaign Impact Analysis")
    
    # Tab layout for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Brand Search Analysis", 
        "Competitor Comparison", 
        "Campaign Performance", 
        "Correlation Analysis"
    ])
    
    # Tab 1: Brand Search Analysis
    with tab1:
        st.subheader("CoinDCX Search Volume Trend")
        
        # Line chart for CoinDCX search volume over time
        fig1 = px.line(
            comp_df, 
            y='CoinDCX', 
            title='CoinDCX Search Volume Over Time',
            labels={'value': 'Search Volume', 'Date': 'Date'},
            template='plotly_white'
        )
        
        # Add campaign spend indicators if dates overlap
        if not campaign_df.empty and not comp_df.empty:
            common_dates = set(comp_df.index) & set(campaign_df.index)
            if common_dates:
                campaign_markers = campaign_df.loc[campaign_df.index.isin(common_dates)]
                
                # Add markers for days with campaign spend
                spent_days = campaign_markers[campaign_markers['Spends'] > 0].index
                for date in spent_days:
                    fig1.add_vline(x=date, line_dash="dash", line_color="green", 
                                  annotation_text="Campaign Active", 
                                  annotation_position="top right")
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Additional insights
        if not comp_df.empty:
            # Create a period comparison (before/after campaign)
            if not campaign_df.empty:
                campaign_start = campaign_df[campaign_df['Spends'] > 0].index.min()
                if not pd.isna(campaign_start):
                    before_period = comp_df.loc[comp_df.index < campaign_start]
                    after_period = comp_df.loc[comp_df.index >= campaign_start]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_before = before_period['CoinDCX'].mean()
                        st.metric("Avg Search Volume Before Campaign", 
                                 f"{avg_before:.0f}")
                    
                    with col2:
                        avg_after = after_period['CoinDCX'].mean()
                        st.metric("Avg Search Volume After Campaign", 
                                 f"{avg_after:.0f}", 
                                 f"{(avg_after - avg_before) / avg_before * 100:.1f}%" if avg_before > 0 else "N/A")
                    
                    with col3:
                        max_after = after_period['CoinDCX'].max()
                        st.metric("Peak Search Volume After Campaign", 
                                 f"{max_after:.0f}")
    
    # Tab 2: Competitor Comparison
    with tab2:
        st.subheader("CoinDCX vs. Competitors")
        
        # Multi-select for competitors
        competitors = [col for col in comp_df.columns if col not in ['Date', 'CoinDCX', 'Installs - DCX']]
        selected_competitors = st.multiselect(
            "Select competitors to compare with CoinDCX:",
            competitors,
            default=competitors[:3]  # Select first 3 competitors by default
        )
        
        # Create comparison chart
        if selected_competitors:
            compare_df = comp_df[['CoinDCX'] + selected_competitors]
            
            fig2 = px.line(
                compare_df,
                title='CoinDCX vs Competitors Search Volume',
                labels={'value': 'Search Volume', 'Date': 'Date', 'variable': 'Brand'},
                template='plotly_white'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Market share analysis
            st.subheader("Search Volume Market Share")
            
            total_searches = compare_df.sum(axis=1)
            share_df = pd.DataFrame()
            
            for col in compare_df.columns:
                share_df[col] = compare_df[col] / total_searches * 100
            
            fig3 = px.area(
                share_df,
                title='Search Volume Market Share (%)',
                labels={'value': 'Market Share (%)', 'Date': 'Date', 'variable': 'Brand'},
                template='plotly_white'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    
    # Tab 3: Campaign Performance
    with tab3:
        st.subheader("Campaign Performance Metrics")
        
        if not campaign_df.empty:
            # Filter for rows with spend/impression data
            campaign_data = campaign_df[campaign_df['Spends'].notna() & (campaign_df['Spends'] > 0)]
            
            if not campaign_data.empty:
                # Create daily campaign spend and impressions chart
                fig4 = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig4.add_trace(
                    go.Bar(
                        x=campaign_data.index,
                        y=campaign_data['Impressions'],
                        name='Impressions',
                        marker_color='lightblue'
                    )
                )
                
                fig4.add_trace(
                    go.Scatter(
                        x=campaign_data.index,
                        y=campaign_data['Spends'],
                        name='Spend (₹)',
                        marker_color='red',
                        mode='lines+markers'
                    ),
                    secondary_y=True,
                )
                
                fig4.update_layout(
                    title='Daily Campaign Performance',
                    xaxis_title='Date',
                    template='plotly_white'
                )
                
                fig4.update_yaxes(title_text="Impressions", secondary_y=False)
                fig4.update_yaxes(title_text="Spend (₹)", secondary_y=True)
                
                st.plotly_chart(fig4, use_container_width=True)
                
                # Calculate and display campaign KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_spend = campaign_data['Spends'].sum()
                    st.metric("Total Campaign Spend", f"₹{total_spend:,.0f}")
                
                with col2:
                    total_impressions = campaign_data['Impressions'].sum()
                    st.metric("Total Impressions", f"{total_impressions:,.0f}")
                
                with col3:
                    cpm = (total_spend / total_impressions * 1000) if total_impressions > 0 else 0
                    st.metric("CPM (Cost per 1000 Impressions)", f"₹{cpm:.2f}")
                
                with col4:
                    # Calculate campaign duration
                    start_date = campaign_data.index.min()
                    end_date = campaign_data.index.max()
                    duration = (end_date - start_date).days + 1
                    st.metric("Campaign Duration", f"{duration} days")
                
                # Show platform breakdown if available
                platform_cols = [col for col in campaign_df.columns if 'Imp' in col and col != 'Impressions']
                if platform_cols:
                    st.subheader("Platform Performance Breakdown")
                    
                    # Create a summary of platform performance
                    platform_data = pd.DataFrame()
                    
                    for i in range(0, len(platform_cols), 2):
                        if i+1 < len(platform_cols):
                            imp_col = platform_cols[i]
                            spend_col = platform_cols[i+1]
                            
                            platform_name = imp_col.split(' ')[0]
                            platform_data.loc[platform_name, 'Impressions'] = campaign_df[imp_col].sum()
                            platform_data.loc[platform_name, 'Spend'] = campaign_df[spend_col].sum()
                            
                            if campaign_df[imp_col].sum() > 0:
                                platform_data.loc[platform_name, 'CPM'] = (campaign_df[spend_col].sum() / campaign_df[imp_col].sum() * 1000)
                            else:
                                platform_data.loc[platform_name, 'CPM'] = 0
                    
                    platform_data = platform_data.sort_values('Impressions', ascending=False)
                    
                    # Display platform comparison
                    fig5 = px.bar(
                        platform_data,
                        x=platform_data.index,
                        y=['Impressions'],
                        title='Platform Impression Comparison',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig5, use_container_width=True)
                    
                    fig6 = px.bar(
                        platform_data,
                        x=platform_data.index,
                        y=['CPM'],
                        title='Platform CPM Comparison',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig6, use_container_width=True)
                    
                    # Display platform data table
                    platform_data['Impressions'] = platform_data['Impressions'].map('{:,.0f}'.format)
                    platform_data['Spend'] = platform_data['Spend'].map('₹{:,.0f}'.format)
                    platform_data['CPM'] = platform_data['CPM'].map('₹{:.2f}'.format)
                    
                    st.dataframe(platform_data)
    
    # Tab 4: Correlation Analysis
    with tab4:
        st.subheader("Campaign Impact on Brand Performance")
        
        # Check if we have overlapping dates to perform correlation analysis
        if not campaign_df.empty and not comp_df.empty:
            common_dates = set(comp_df.index) & set(campaign_df.index)
            
            if common_dates:
                # Create merged dataframe for analysis
                merged_df = pd.merge(
                    comp_df[['CoinDCX', 'Installs - DCX']], 
                    campaign_df[['Impressions', 'Spends']],
                    left_index=True, 
                    right_index=True,
                    how='inner'
                )
                
                # Clean up any missing values
                merged_df = merged_df.fillna(0)
                
                # Calculate correlation
                correlation = merged_df.corr()
                
                # Display correlation matrix
                fig7 = px.imshow(
                    correlation,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title='Correlation Between Campaign and Brand Metrics',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig7, use_container_width=True)
                
                # Create scatter plots for key relationships
                col1, col2 = st.columns(2)
                
                with col1:
                    fig8 = px.scatter(
                        merged_df,
                        x='Impressions',
                        y='CoinDCX',
                        trendline='ols',
                        title='Campaign Impressions vs. CoinDCX Search Volume',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig8, use_container_width=True)
                
                with col2:
                    fig9 = px.scatter(
                        merged_df,
                        x='Spends',
                        y='CoinDCX',
                        trendline='ols',
                        title='Campaign Spend vs. CoinDCX Search Volume',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig9, use_container_width=True)
                
                # Calculate time-lagged effects (search volume might increase days after ad spend)
                st.subheader("Time-Lagged Campaign Effects")
                
                max_lag = 7  # Look at effects up to 7 days later
                lag_correlations = []
                
                for lag in range(max_lag + 1):
                    if lag == 0:
                        lagged_df = merged_df.copy()
                    else:
                        lagged_df = pd.merge(
                            merged_df[['Impressions', 'Spends']],
                            merged_df[['CoinDCX', 'Installs - DCX']].shift(-lag),
                            left_index=True,
                            right_index=True
                        )
                    
                    imp_corr = lagged_df['Impressions'].corr(lagged_df['CoinDCX'])
                    spend_corr = lagged_df['Spends'].corr(lagged_df['CoinDCX'])
                    
                    lag_correlations.append({
                        'Lag (Days)': lag,
                        'Impressions-Search Correlation': imp_corr,
                        'Spend-Search Correlation': spend_corr
                    })
                
                lag_df = pd.DataFrame(lag_correlations)
                
                fig10 = px.line(
                    lag_df,
                    x='Lag (Days)',
                    y=['Impressions-Search Correlation', 'Spend-Search Correlation'],
                    title='Effect of Campaign on Search Volume Over Time',
                    template='plotly_white',
                    markers=True
                )
                
                st.plotly_chart(fig10, use_container_width=True)
                
                # Insights about campaign ROI
                st.subheader("Campaign ROI Analysis")
                
                # Calculate search volume lift during campaign
                if 'Installs - DCX' in comp_df.columns:
                    campaign_dates = campaign_df[campaign_df['Spends'] > 0].index
                    
                    if len(campaign_dates) > 0:
                        # Get time window before campaign of same length
                        campaign_start = min(campaign_dates)
                        campaign_end = max(campaign_dates)
                        campaign_days = (campaign_end - campaign_start).days + 1
                        
                        pre_campaign_start = campaign_start - timedelta(days=campaign_days)
                        pre_campaign_end = campaign_start - timedelta(days=1)
                        
                        # Get data for periods
                        pre_campaign_data = comp_df.loc[(comp_df.index >= pre_campaign_start) & 
                                                      (comp_df.index <= pre_campaign_end)]
                        during_campaign_data = comp_df.loc[(comp_df.index >= campaign_start) & 
                                                         (comp_df.index <= campaign_end)]
                        
                        # Calculate metrics
                        pre_search_avg = pre_campaign_data['CoinDCX'].mean() if not pre_campaign_data.empty else 0
                        during_search_avg = during_campaign_data['CoinDCX'].mean() if not during_campaign_data.empty else 0
                        
                        pre_installs_avg = pre_campaign_data['Installs - DCX'].mean() if not pre_campaign_data.empty else 0
                        during_installs_avg = during_campaign_data['Installs - DCX'].mean() if not during_campaign_data.empty else 0
                        
                        search_lift = during_search_avg - pre_search_avg
                        search_lift_pct = (search_lift / pre_search_avg * 100) if pre_search_avg > 0 else 0
                        
                        installs_lift = during_installs_avg - pre_installs_avg
                        installs_lift_pct = (installs_lift / pre_installs_avg * 100) if pre_installs_avg > 0 else 0
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Search Volume Lift During Campaign", 
                                     f"{search_lift:.1f}", 
                                     f"{search_lift_pct:.1f}%")
                            
                            # Calculate cost per additional search
                            total_campaign_spend = campaign_df.loc[campaign_df.index.isin(campaign_dates), 'Spends'].sum()
                            additional_searches = search_lift * len(campaign_dates)
                            
                            if additional_searches > 0:
                                cost_per_add_search = total_campaign_spend / additional_searches
                                st.metric("Cost Per Additional Search", f"₹{cost_per_add_search:.2f}")
                        
                        with col2:
                            st.metric("App Installs Lift During Campaign", 
                                     f"{installs_lift:.1f}", 
                                     f"{installs_lift_pct:.1f}%")
                            
                            # Calculate cost per additional install
                            additional_installs = installs_lift * len(campaign_dates)
                            
                            if additional_installs > 0:
                                cost_per_add_install = total_campaign_spend / additional_installs
                                st.metric("Cost Per Additional Install", f"₹{cost_per_add_install:.2f}")
            else:
                st.warning("No overlapping dates found between campaign data and competitor search data. Cannot perform correlation analysis.")
    
    # Key Findings and Recommendations
    st.header("Key Findings and Recommendations")
    
    # This section could be automated based on the actual data analysis
    st.markdown("""
    Based on the analysis, here are the key findings:
    
    1. **Campaign Impact**: The analysis shows the relationship between ad spend and search volume for CoinDCX.
    
    2. **Platform Performance**: Review the most efficient platforms based on CPM and conversion metrics.
    
    3. **Competitive Position**: See how CoinDCX's search share changed during and after the campaign.
    
    4. **ROI Analysis**: The cost per additional search and install metrics provide ROI insights.
    """)
    
    st.markdown("""
    ### Recommendations:
    
    1. Focus future spend on platforms with the best performance metrics
    2. Consider the optimal lag time between campaigns based on the time-lagged analysis
    3. Monitor competitive search volume market share to gauge long-term brand building success
    """)

# Instructions if files not uploaded
else:
    st.info("Please upload both Excel files to begin the analysis.")
    
    st.markdown("""
    ### Expected File Structure:
    
    **Competitor Data File:**
    - Should include Date column and search volumes for CoinDCX and competitors
    - May include app install data
    
    **Campaign Metrics File:**
    - Should include Date column with campaign impressions and spend data
    - May include platform-specific metrics
    """)

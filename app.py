"""
NovaRetail Customer Intelligence Interactive Dashboard
A comprehensive Streamlit application for analyzing customer behavior,
revenue patterns, and segment performance.
Uses Altair for visualizations (lighter weight, more reliable).
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NovaRetail Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

@st.cache_data
def load_data():
    """Load dataset from Excel file."""
    try:
        df = pd.read_excel("NR_dataset.xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """
    Data preprocessing pipeline:
    1. Drop index column if present
    2. Handle duplicates
    3. Handle missing values
    4. Apply Tukey's IQR outlier detection with winsorization
    """

    df_processed = df.copy()

    # Drop unnecessary index columns
    if 'idx' in df_processed.columns:
        df_processed = df_processed.drop('idx', axis=1)

    # Record original count
    original_count = len(df_processed)

    # Handle duplicates
    duplicates = df_processed.duplicated().sum()
    df_processed = df_processed.drop_duplicates().reset_index(drop=True)

    # Handle missing values in categorical columns
    df_processed['label'] = df_processed['label'].fillna('Unknown')

    # Create preprocessing metadata
    metadata = {
        'original_count': original_count,
        'duplicates_removed': duplicates,
        'final_count': len(df_processed),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return df_processed, metadata

@st.cache_data
def detect_outliers_tukey(series, multiplier=1.5):
    """Detect outliers using Tukey's IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = outlier_mask.sum()

    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': outlier_count,
        'outlier_mask': outlier_mask
    }

@st.cache_data
def apply_winsorization(df):
    """Apply Tukey's IQR winsorization to numerical columns."""
    df_winsorized = df.copy()
    outlier_report = {}

    numerical_cols = ['PurchaseAmount', 'CustomerSatisfaction']

    for col in numerical_cols:
        outlier_info = detect_outliers_tukey(df_winsorized[col])

        # Apply winsorization
        lower = outlier_info['lower_bound']
        upper = outlier_info['upper_bound']
        df_winsorized[col] = df_winsorized[col].clip(lower=lower, upper=upper)

        outlier_report[col] = outlier_info

    return df_winsorized, outlier_report

def calculate_statistics(series, name=""):
    """Calculate comprehensive descriptive statistics for a numerical series."""

    valid_series = series.dropna()

    if len(valid_series) == 0:
        return None

    stats_dict = {
        'Name': name,
        'Count': len(valid_series),
        'Mean': valid_series.mean(),
        'Median': valid_series.median(),
        'Mode': valid_series.mode()[0] if len(valid_series.mode()) > 0 else np.nan,
        'Std Dev': valid_series.std(),
        'Variance': valid_series.var(),
        'Min': valid_series.min(),
        'Max': valid_series.max(),
        'Range': valid_series.max() - valid_series.min(),
        'Q1 (25%)': valid_series.quantile(0.25),
        'Q3 (75%)': valid_series.quantile(0.75),
        'IQR': valid_series.quantile(0.75) - valid_series.quantile(0.25),
        'Skewness': stats.skew(valid_series),
        'Kurtosis': stats.kurtosis(valid_series),
        'Coeff. of Variation': (valid_series.std() / valid_series.mean() * 100) if valid_series.mean() != 0 else np.nan
    }

    return stats_dict

# ============================================================================
# SEGMENT COLOR MAPPING
# ============================================================================

SEGMENT_COLORS = {
    'Promising': '#2ecc71',  # Green
    'Growth': '#3498db',     # Blue
    'Stable': '#95a5a6',     # Gray
    'Decline': '#e74c3c',    # Red
    'Unknown': '#34495e'     # Dark gray
}

def get_segment_color(segment):
    """Get color for a specific segment."""
    return SEGMENT_COLORS.get(segment, '#34495e')

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY DASHBOARD
# ============================================================================

def page_executive_summary(df, metadata):
    """Executive Summary Dashboard - High-level KPIs and overview metrics."""

    st.title("📊 Executive Summary Dashboard")

    # Display data freshness
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"Data Last Updated: {metadata['timestamp']}")
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_exec"):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # ========== KPI CARDS ==========
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = df['PurchaseAmount'].sum()
        st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")

    with col2:
        total_customers = df['CustomerID'].nunique()
        st.metric("👥 Total Customers", int(total_customers))

    with col3:
        avg_purchase = df['PurchaseAmount'].mean()
        st.metric("🛒 Avg Purchase Value", f"${avg_purchase:,.2f}")

    with col4:
        avg_satisfaction = df['CustomerSatisfaction'].mean()
        st.metric("⭐ Avg Satisfaction", f"{avg_satisfaction:.2f}/5")

    st.divider()

    # ========== SEGMENT BREAKDOWN ==========
    st.subheader("Customer Segment Breakdown")

    segment_counts = df['label'].value_counts()
    segment_pcts = df['label'].value_counts(normalize=True) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Segment Distribution:**")
        segment_table = pd.DataFrame({
            'Segment': segment_counts.index,
            'Count': segment_counts.values,
            'Percentage': [f"{pct:.1f}%" for pct in segment_pcts.values]
        })
        st.dataframe(segment_table, use_container_width=True, hide_index=True)

    with col2:
        # Pie chart for segment distribution
        seg_data = pd.DataFrame({
            'Segment': segment_counts.index,
            'Count': segment_counts.values
        })

        pie_chart = alt.Chart(seg_data).mark_arc(innerRadius=0).encode(
            theta='Count:Q',
            color=alt.Color('Segment:N', scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                          range=list(SEGMENT_COLORS.values()))),
            tooltip=['Segment:N', 'Count:Q']
        ).properties(height=350)

        st.altair_chart(pie_chart, use_container_width=True)

    st.divider()

    # ========== VISUALIZATIONS ==========
    col1, col2 = st.columns(2)

    # Revenue by Segment
    with col1:
        st.subheader("Revenue by Customer Segment")
        revenue_by_segment = df.groupby('label')['PurchaseAmount'].sum().reset_index()
        revenue_by_segment = revenue_by_segment.sort_values('PurchaseAmount')

        chart = alt.Chart(revenue_by_segment).mark_bar().encode(
            y=alt.Y('label:N', sort='-x', title='Segment'),
            x=alt.X('PurchaseAmount:Q', title='Total Revenue ($)'),
            color=alt.Color('label:N', scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                        range=list(SEGMENT_COLORS.values())),
                           legend=None),
            tooltip=['label:N', alt.Tooltip('PurchaseAmount:Q', format='$,.0f')]
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    # Satisfaction Distribution
    with col2:
        st.subheader("Customer Satisfaction Distribution")

        satisfaction_counts = df['CustomerSatisfaction'].value_counts().sort_index().reset_index()
        satisfaction_counts.columns = ['Satisfaction', 'Count']

        chart = alt.Chart(satisfaction_counts).mark_bar(color='#3498db').encode(
            x=alt.X('Satisfaction:Q', scale=alt.Scale(domain=[1, 5]), title='Satisfaction Rating (1-5)'),
            y=alt.Y('Count:Q', title='Number of Customers'),
            tooltip=['Satisfaction:Q', 'Count:Q']
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    # Revenue Trend Over Time
    st.subheader("Revenue Trend Over Time")
    df_sorted = df.sort_values('TransactionDate').copy()
    df_sorted['CumulativeRevenue'] = df_sorted.groupby('label')['PurchaseAmount'].cumsum()

    chart = alt.Chart(df_sorted).mark_line(point=True).encode(
        x=alt.X('TransactionDate:T', title='Transaction Date'),
        y=alt.Y('CumulativeRevenue:Q', title='Cumulative Revenue ($)'),
        color=alt.Color('label:N', scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                    range=list(SEGMENT_COLORS.values()))),
        tooltip=['label:N', 'TransactionDate:T', alt.Tooltip('CumulativeRevenue:Q', format='$,.0f')]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

# ============================================================================
# PAGE 2: DETAILED SEGMENT ANALYSIS
# ============================================================================

def page_segment_analysis(df):
    """Detailed Customer Segment Analysis with comparative metrics."""

    st.title("🎯 Customer Segment Analysis")

    # ========== SIDEBAR FILTERS ==========
    st.sidebar.subheader("Segment Analysis Filters")

    selected_segments = st.sidebar.multiselect(
        "Select Segments to Analyze",
        options=sorted(df['label'].unique()),
        default=sorted(df['label'].unique())
    )

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['TransactionDate'].min().date(), df['TransactionDate'].max().date()),
        min_value=df['TransactionDate'].min().date(),
        max_value=df['TransactionDate'].max().date()
    )

    # Filter data
    df_filtered = df[df['label'].isin(selected_segments)]
    df_filtered = df_filtered[
        (df_filtered['TransactionDate'].dt.date >= date_range[0]) &
        (df_filtered['TransactionDate'].dt.date <= date_range[1])
    ]

    st.divider()

    # ========== STATISTICAL SUMMARY TABLE ==========
    st.subheader("Statistical Summary by Segment")

    summary_stats = []
    for segment in selected_segments:
        segment_data = df_filtered[df_filtered['label'] == segment]
        if len(segment_data) > 0:
            summary_stats.append({
                'Segment': segment,
                'Count': len(segment_data),
                'Total Revenue': f"${segment_data['PurchaseAmount'].sum():,.2f}",
                'Mean Purchase': f"${segment_data['PurchaseAmount'].mean():,.2f}",
                'Median Purchase': f"${segment_data['PurchaseAmount'].median():,.2f}",
                'Std Dev': f"${segment_data['PurchaseAmount'].std():,.2f}",
                'Skewness': f"{stats.skew(segment_data['PurchaseAmount']):.3f}",
                'Min': f"${segment_data['PurchaseAmount'].min():,.2f}",
                'Max': f"${segment_data['PurchaseAmount'].max():,.2f}",
                'Avg Satisfaction': f"{segment_data['CustomerSatisfaction'].mean():.2f}"
            })

    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # ========== VISUALIZATIONS ==========
    col1, col2 = st.columns(2)

    # Box Plot Alternative - using strip plot
    with col1:
        st.subheader("Purchase Amount Distribution by Segment")

        chart = alt.Chart(df_filtered).mark_point(opacity=0.6).encode(
            x=alt.X('label:N', title='Segment'),
            y=alt.Y('PurchaseAmount:Q', title='Purchase Amount ($)'),
            color=alt.Color('label:N', scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                        range=list(SEGMENT_COLORS.values())),
                           legend=None),
            tooltip=['label:N', alt.Tooltip('PurchaseAmount:Q', format='$,.2f')]
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    # Satisfaction vs Purchase Amount
    with col2:
        st.subheader("Satisfaction vs Purchase Amount")

        chart = alt.Chart(df_filtered).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X('PurchaseAmount:Q', title='Purchase Amount ($)'),
            y=alt.Y('CustomerSatisfaction:Q', title='Satisfaction (1-5)'),
            color=alt.Color('label:N', scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                                        range=list(SEGMENT_COLORS.values()))),
            tooltip=['label:N', alt.Tooltip('PurchaseAmount:Q', format='$,.2f'),
                    'CustomerSatisfaction:Q']
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    # Comparative Bar Chart
    st.subheader("Comparative Metrics by Segment")

    metrics_to_show = st.multiselect(
        "Select Metrics to Compare",
        options=['Mean Purchase Amount', 'Median Purchase Amount', 'Avg Satisfaction'],
        default=['Mean Purchase Amount', 'Avg Satisfaction']
    )

    if metrics_to_show:
        comparative_data = []
        for segment in selected_segments:
            segment_data = df_filtered[df_filtered['label'] == segment]
            if len(segment_data) > 0:
                row = {'Segment': segment}
                if 'Mean Purchase Amount' in metrics_to_show:
                    row['Mean Purchase Amount'] = segment_data['PurchaseAmount'].mean()
                if 'Median Purchase Amount' in metrics_to_show:
                    row['Median Purchase Amount'] = segment_data['PurchaseAmount'].median()
                if 'Avg Satisfaction' in metrics_to_show:
                    row['Avg Satisfaction'] = segment_data['CustomerSatisfaction'].mean()
                comparative_data.append(row)

        comp_df = pd.DataFrame(comparative_data)
        comp_df_melted = comp_df.melt(id_vars=['Segment'], var_name='Metric', value_name='Value')

        chart = alt.Chart(comp_df_melted).mark_bar().encode(
            x=alt.X('Segment:N', title='Segment'),
            y=alt.Y('Value:Q', title='Value'),
            color=alt.Color('Metric:N', title='Metric'),
            xOffset='Metric:N'
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

# ============================================================================
# PAGE 3: GEOGRAPHIC & CHANNEL PERFORMANCE
# ============================================================================

def page_geographic_performance(df):
    """Geographic and Channel Performance Analysis."""

    st.title("🌍 Geographic & Channel Performance")

    # ========== SIDEBAR FILTERS ==========
    st.sidebar.subheader("Geographic Filters")

    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=sorted(df['CustomerRegion'].unique()),
        default=sorted(df['CustomerRegion'].unique())
    )

    selected_channels = st.sidebar.multiselect(
        "Select Retail Channels",
        options=sorted(df['RetailChannel'].unique()),
        default=sorted(df['RetailChannel'].unique())
    )

    # Filter data
    df_filtered = df[
        (df['CustomerRegion'].isin(selected_regions)) &
        (df['RetailChannel'].isin(selected_channels))
    ]

    st.divider()

    col1, col2 = st.columns(2)

    # Heatmap: Revenue by Region × Channel
    with col1:
        st.subheader("Revenue by Region × Channel")

        heatmap_data = df_filtered.pivot_table(
            values='PurchaseAmount',
            index='CustomerRegion',
            columns='RetailChannel',
            aggfunc='sum',
            fill_value=0
        ).reset_index().melt(id_vars='CustomerRegion', var_name='RetailChannel', value_name='Revenue')

        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('RetailChannel:N', title='Retail Channel'),
            y=alt.Y('CustomerRegion:N', title='Region'),
            color=alt.Color('Revenue:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['CustomerRegion:N', 'RetailChannel:N', alt.Tooltip('Revenue:Q', format='$,.0f')]
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    # Regional Performance
    with col2:
        st.subheader("Regional Performance Overview")

        regional_stats = df_filtered.groupby('CustomerRegion').agg({
            'PurchaseAmount': 'sum',
            'CustomerSatisfaction': 'mean',
            'CustomerID': 'count'
        }).reset_index()
        regional_stats.columns = ['Region', 'Total Revenue', 'Avg Satisfaction', 'Customer Count']
        regional_stats = regional_stats.sort_values('Total Revenue', ascending=False)

        st.dataframe(regional_stats, use_container_width=True, hide_index=True)

    st.divider()

    # Regional Bar Chart with Satisfaction
    st.subheader("Revenue by Region")

    regional_data = df_filtered.groupby('CustomerRegion').agg({
        'PurchaseAmount': 'sum'
    }).reset_index()

    chart = alt.Chart(regional_data).mark_bar(color='#3498db').encode(
        x=alt.X('CustomerRegion:N', title='Region'),
        y=alt.Y('PurchaseAmount:Q', title='Total Revenue ($)'),
        tooltip=['CustomerRegion:N', alt.Tooltip('PurchaseAmount:Q', format='$,.0f')]
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    st.divider()

    # Channel Comparison
    st.subheader("Channel Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        channel_revenue = df_filtered.groupby('RetailChannel')['PurchaseAmount'].sum().reset_index()

        chart = alt.Chart(channel_revenue).mark_bar(color='#3498db').encode(
            x=alt.X('RetailChannel:N', title='Retail Channel'),
            y=alt.Y('PurchaseAmount:Q', title='Total Revenue ($)'),
            tooltip=['RetailChannel:N', alt.Tooltip('PurchaseAmount:Q', format='$,.0f')]
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    with col2:
        channel_satisfaction = df_filtered.groupby('RetailChannel')['CustomerSatisfaction'].mean().reset_index()

        chart = alt.Chart(channel_satisfaction).mark_bar(color='#2ecc71').encode(
            x=alt.X('RetailChannel:N', title='Retail Channel'),
            y=alt.Y('CustomerSatisfaction:Q', title='Avg Satisfaction (1-5)'),
            tooltip=['RetailChannel:N', alt.Tooltip('CustomerSatisfaction:Q', format='.2f')]
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

# ============================================================================
# PAGE 4: PRODUCT CATEGORY & DEMOGRAPHIC INSIGHTS
# ============================================================================

def page_product_demographics(df):
    """Product Category and Demographic Insights."""

    st.title("🛍️ Product Category & Demographic Insights")

    # ========== SIDEBAR FILTERS ==========
    st.sidebar.subheader("Product & Demographics Filters")

    selected_age_groups = st.sidebar.multiselect(
        "Select Age Groups",
        options=sorted(df['CustomerAgeGroup'].unique()),
        default=sorted(df['CustomerAgeGroup'].unique())
    )

    selected_genders = st.sidebar.multiselect(
        "Select Gender",
        options=sorted(df['CustomerGender'].unique()),
        default=sorted(df['CustomerGender'].unique())
    )

    selected_segments = st.sidebar.multiselect(
        "Select Segments",
        options=sorted(df['label'].unique()),
        default=sorted(df['label'].unique())
    )

    # Filter data
    df_filtered = df[
        (df['CustomerAgeGroup'].isin(selected_age_groups)) &
        (df['CustomerGender'].isin(selected_genders)) &
        (df['label'].isin(selected_segments))
    ]

    st.divider()

    col1, col2 = st.columns(2)

    # Revenue by Category
    with col1:
        st.subheader("Revenue by Product Category")

        category_revenue = df_filtered.groupby('ProductCategory')['PurchaseAmount'].sum().reset_index()
        category_revenue = category_revenue.sort_values('PurchaseAmount').tail(15)

        chart = alt.Chart(category_revenue).mark_bar().encode(
            y=alt.Y('ProductCategory:N', sort='-x', title=''),
            x=alt.X('PurchaseAmount:Q', title='Total Revenue ($)'),
            color=alt.Color('PurchaseAmount:Q', scale=alt.Scale(scheme='viridis'), legend=None),
            tooltip=['ProductCategory:N', alt.Tooltip('PurchaseAmount:Q', format='$,.0f')]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

    # Age Group Analysis
    with col2:
        st.subheader("Age Group Analysis")

        age_groups_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '55+']
        age_groups_order = [ag for ag in age_groups_order if ag in df_filtered['CustomerAgeGroup'].unique()]

        age_data = []
        for age_group in age_groups_order:
            ag_data = df_filtered[df_filtered['CustomerAgeGroup'] == age_group]
            if len(ag_data) > 0:
                age_data.append({
                    'AgeGroup': age_group,
                    'AvgSatisfaction': ag_data['CustomerSatisfaction'].mean(),
                    'Count': len(ag_data)
                })

        age_df = pd.DataFrame(age_data)

        chart = alt.Chart(age_df).mark_line(point=True).encode(
            x=alt.X('AgeGroup:N', title='Age Group', sort=age_groups_order),
            y=alt.Y('AvgSatisfaction:Q', title='Avg Satisfaction (1-5)', scale=alt.Scale(domain=[0, 5])),
            size=alt.Size('Count:Q', title='Customer Count'),
            tooltip=['AgeGroup:N', alt.Tooltip('AvgSatisfaction:Q', format='.2f'), 'Count:Q']
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    st.divider()

    # Demographics Matrix
    st.subheader("Demographics Matrix (Age × Gender)")

    age_groups_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '55+']
    age_groups_order = [ag for ag in age_groups_order if ag in df_filtered['CustomerAgeGroup'].unique()]

    demo_matrix = []
    for age_group in age_groups_order:
        row_data = {'Age Group': age_group}
        for gender in selected_genders:
            subset = df_filtered[(df_filtered['CustomerAgeGroup'] == age_group) &
                                (df_filtered['CustomerGender'] == gender)]
            if len(subset) > 0:
                row_data[f'{gender} - Count'] = len(subset)
                row_data[f'{gender} - Avg Purchase'] = f"${subset['PurchaseAmount'].mean():.2f}"
                row_data[f'{gender} - Avg Satisfaction'] = f"{subset['CustomerSatisfaction'].mean():.2f}"
            else:
                row_data[f'{gender} - Count'] = 0
                row_data[f'{gender} - Avg Purchase'] = "-"
                row_data[f'{gender} - Avg Satisfaction'] = "-"
        demo_matrix.append(row_data)

    demo_df = pd.DataFrame(demo_matrix)
    st.dataframe(demo_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 5: STATISTICAL DIAGNOSTICS & DATA QUALITY
# ============================================================================

def page_data_quality(df, df_winsorized, metadata, outlier_report):
    """Statistical Diagnostics and Data Quality Report."""

    st.title("📈 Statistical Diagnostics & Data Quality")

    # ========== DATA INGESTION & CLEANING REPORT ==========
    st.subheader("Data Ingestion & Cleaning Report")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Row Count", metadata['original_count'])

    with col2:
        st.metric("Duplicates Removed", metadata['duplicates_removed'])

    with col3:
        st.metric("Final Row Count", metadata['final_count'])

    with col4:
        st.metric("Data Last Updated", metadata['timestamp'][:10])

    st.divider()

    # ========== OUTLIER WINSORIZATION SUMMARY ==========
    st.subheader("Outlier Detection & Winsorization Summary (Tukey's IQR)")

    outlier_summary = []
    for col_name, outlier_info in outlier_report.items():
        outlier_summary.append({
            'Variable': col_name,
            'Q1': f"{outlier_info['Q1']:.2f}",
            'Q3': f"{outlier_info['Q3']:.2f}",
            'IQR': f"{outlier_info['IQR']:.2f}",
            'Lower Bound': f"{outlier_info['lower_bound']:.2f}",
            'Upper Bound': f"{outlier_info['upper_bound']:.2f}",
            'Outliers Detected': outlier_info['outlier_count'],
            'Action': 'Winsorized' if outlier_info['outlier_count'] > 0 else 'None'
        })

    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)

    st.info(
        "ℹ️ **Tukey's IQR Method:** Outliers detected outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]. "
        "Values were winsorized (capped at boundaries) rather than removed to preserve sample size."
    )

    st.divider()

    # ========== DESCRIPTIVE STATISTICS PANEL ==========
    st.subheader("Descriptive Statistics")

    selected_variable = st.selectbox(
        "Select Variable for Detailed Statistics",
        options=['PurchaseAmount', 'CustomerSatisfaction']
    )

    stats_dict = calculate_statistics(df_winsorized[selected_variable], selected_variable)

    if stats_dict:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Count", int(stats_dict['Count']))
            st.metric("Mean", f"{stats_dict['Mean']:.2f}")
            st.metric("Median", f"{stats_dict['Median']:.2f}")
            st.metric("Mode", f"{stats_dict['Mode']:.2f}")

        with col2:
            st.metric("Std Dev", f"{stats_dict['Std Dev']:.2f}")
            st.metric("Variance", f"{stats_dict['Variance']:.2f}")
            st.metric("Coeff. of Variation", f"{stats_dict['Coeff. of Variation']:.2f}%")
            st.metric("Range", f"{stats_dict['Range']:.2f}")

        with col3:
            st.metric("Min", f"{stats_dict['Min']:.2f}")
            st.metric("Q1 (25%)", f"{stats_dict['Q1 (25%)']:.2f}")
            st.metric("IQR", f"{stats_dict['IQR']:.2f}")
            st.metric("Max", f"{stats_dict['Max']:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            skewness_val = stats_dict['Skewness']
            if -0.5 <= skewness_val <= 0.5:
                skew_interpretation = "Approximately Symmetric"
            elif skewness_val > 0.5:
                skew_interpretation = "Right Skewed"
            else:
                skew_interpretation = "Left Skewed"

            st.info(f"**Skewness:** {skewness_val:.3f} - {skew_interpretation}")

        with col2:
            st.info(f"**Kurtosis:** {stats_dict['Kurtosis']:.3f}")

    st.divider()

    # ========== DISTRIBUTION DIAGNOSTICS ==========
    st.subheader("Distribution Comparisons")

    col1, col2 = st.columns(2)

    # Pre vs Post Winsorization
    with col1:
        st.write("**Purchase Amount: Pre vs Post-Winsorization**")

        pre_data = pd.DataFrame({'Value': df['PurchaseAmount'], 'Type': 'Pre-Winsorization'})
        post_data = pd.DataFrame({'Value': df_winsorized['PurchaseAmount'], 'Type': 'Post-Winsorization'})
        comparison_data = pd.concat([pre_data, post_data])

        chart = alt.Chart(comparison_data).mark_histogram(opacity=0.6, binned=True).encode(
            x=alt.X('Value:Q', bin=alt.Bin(maxbins=20), title='Purchase Amount ($)'),
            y=alt.Y('count()', title='Frequency'),
            color=alt.Color('Type:N', title='Data Type')
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    # Satisfaction Distribution
    with col2:
        st.write("**Customer Satisfaction Distribution**")

        sat_data = pd.DataFrame({
            'Satisfaction': df_winsorized['CustomerSatisfaction'],
            'Count': 1
        }).groupby('Satisfaction')['Count'].sum().reset_index()

        chart = alt.Chart(sat_data).mark_bar(color='#3498db').encode(
            x=alt.X('Satisfaction:Q', scale=alt.Scale(domain=[1, 5]), title='Satisfaction Rating (1-5)'),
            y=alt.Y('Count:Q', title='Frequency'),
            tooltip=['Satisfaction:Q', 'Count:Q']
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    st.divider()

    # Percentiles
    st.subheader("Percentiles Distribution")

    percentiles_dict = {
        'Percentile': ['10th', '25th', '50th (Median)', '75th', '90th'],
        'Purchase Amount': [
            f"${df_winsorized['PurchaseAmount'].quantile(0.10):.2f}",
            f"${df_winsorized['PurchaseAmount'].quantile(0.25):.2f}",
            f"${df_winsorized['PurchaseAmount'].quantile(0.50):.2f}",
            f"${df_winsorized['PurchaseAmount'].quantile(0.75):.2f}",
            f"${df_winsorized['PurchaseAmount'].quantile(0.90):.2f}"
        ],
        'Satisfaction': [
            f"{df_winsorized['CustomerSatisfaction'].quantile(0.10):.2f}",
            f"{df_winsorized['CustomerSatisfaction'].quantile(0.25):.2f}",
            f"{df_winsorized['CustomerSatisfaction'].quantile(0.50):.2f}",
            f"{df_winsorized['CustomerSatisfaction'].quantile(0.75):.2f}",
            f"{df_winsorized['CustomerSatisfaction'].quantile(0.90):.2f}"
        ]
    }

    percentiles_df = pd.DataFrame(percentiles_dict)
    st.dataframe(percentiles_df, use_container_width=True, hide_index=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""

    # Load and process data
    df = load_data()

    if df is None:
        st.error("Failed to load dataset. Please check that NR_dataset.xlsx exists in the application directory.")
        return

    # Preprocess data
    df_processed, metadata = preprocess_data(df)
    df_winsorized, outlier_report = apply_winsorization(df_processed)

    # Navigation
    st.sidebar.title("📊 NovaRetail Dashboard")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Select Page",
        options=[
            "Executive Summary",
            "Segment Analysis",
            "Geographic Performance",
            "Product & Demographics",
            "Data Quality"
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption("NovaRetail Customer Intelligence Dashboard | v2.0 (Altair)")

    # Route to selected page
    if page == "Executive Summary":
        page_executive_summary(df_processed, metadata)

    elif page == "Segment Analysis":
        page_segment_analysis(df_processed)

    elif page == "Geographic Performance":
        page_geographic_performance(df_processed)

    elif page == "Product & Demographics":
        page_product_demographics(df_processed)

    elif page == "Data Quality":
        page_data_quality(df_processed, df_winsorized, metadata, outlier_report)

if __name__ == "__main__":
    main()

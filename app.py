"""
NovaRetail Customer Intelligence Interactive Dashboard
A comprehensive Streamlit application for analyzing customer behavior,
revenue patterns, and segment performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
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

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .kpi-value {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }
        .kpi-label {
            font-size: 14px;
            opacity: 0.9;
        }
    </style>
""", unsafe_allow_html=True)

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
    5. Create derived features
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
    """
    Detect outliers using Tukey's IQR method.
    Returns: Q1, Q3, IQR, lower_bound, upper_bound, outlier_indices, outlier_count
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_indices = series[outlier_mask].index.tolist()
    outlier_count = outlier_mask.sum()

    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_indices': outlier_indices,
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

def get_color_palette(segments=None):
    """Get color palette for segments."""
    if segments is None:
        return SEGMENT_COLORS
    return [SEGMENT_COLORS.get(seg, '#34495e') for seg in segments]

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
        st.metric("💰 Total Revenue", f"${total_revenue:,.2f}", delta=None)

    with col2:
        total_customers = df['CustomerID'].nunique()
        st.metric("👥 Total Customers", int(total_customers), delta=None)

    with col3:
        avg_purchase = df['PurchaseAmount'].mean()
        st.metric("🛒 Avg Purchase Value", f"${avg_purchase:,.2f}", delta=None)

    with col4:
        avg_satisfaction = df['CustomerSatisfaction'].mean()
        st.metric("⭐ Avg Satisfaction", f"{avg_satisfaction:.2f}/5", delta=None)

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
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            marker=dict(colors=[get_color_palette().get(seg, '#34495e') for seg in segment_counts.index]),
            textposition='inside',
            textinfo='label+percent'
        )])
        fig.update_layout(
            height=350,
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ========== VISUALIZATIONS ==========
    col1, col2 = st.columns(2)

    # Revenue by Segment
    with col1:
        st.subheader("Revenue by Customer Segment")
        revenue_by_segment = df.groupby('label')['PurchaseAmount'].sum().sort_values(ascending=True)

        fig = go.Figure(data=[go.Bar(
            y=revenue_by_segment.index,
            x=revenue_by_segment.values,
            orientation='h',
            marker=dict(color=[get_color_palette().get(seg, '#34495e') for seg in revenue_by_segment.index]),
            text=[f'${x:,.0f}' for x in revenue_by_segment.values],
            textposition='outside'
        )])
        fig.update_layout(
            height=350,
            xaxis_title="Total Revenue ($)",
            yaxis_title="Segment",
            showlegend=False,
            margin=dict(l=0, r=100, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Satisfaction Distribution
    with col2:
        st.subheader("Customer Satisfaction Distribution")

        satisfaction_counts = df['CustomerSatisfaction'].value_counts().sort_index()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['CustomerSatisfaction'],
            nbinsx=5,
            name='Count',
            marker=dict(color='#3498db'),
            opacity=0.7
        ))

        # Add mean line
        mean_sat = df['CustomerSatisfaction'].mean()
        fig.add_vline(x=mean_sat, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_sat:.2f}")

        fig.update_layout(
            height=350,
            xaxis_title="Satisfaction Rating (1-5)",
            yaxis_title="Number of Customers",
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Revenue Trend Over Time
    st.subheader("Revenue Trend Over Time")
    df_sorted = df.sort_values('TransactionDate')
    df_sorted['CumulativeRevenue'] = df_sorted.groupby('label')['PurchaseAmount'].cumsum()

    fig = go.Figure()
    for segment in df_sorted['label'].unique():
        segment_data = df_sorted[df_sorted['label'] == segment].sort_values('TransactionDate')
        segment_data_cum = segment_data.copy()
        segment_data_cum['CumulativeRevenue'] = segment_data['PurchaseAmount'].cumsum()

        fig.add_trace(go.Scatter(
            x=segment_data_cum['TransactionDate'],
            y=segment_data_cum['CumulativeRevenue'],
            mode='lines',
            name=segment,
            line=dict(color=get_color_palette().get(segment, '#34495e'), width=2)
        ))

    fig.update_layout(
        height=400,
        title_text="Cumulative Revenue by Segment Over Time",
        xaxis_title="Transaction Date",
        yaxis_title="Cumulative Revenue ($)",
        hovermode='x unified',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

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
        options=df['label'].unique(),
        default=df['label'].unique()
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
                'IQR': f"${segment_data['PurchaseAmount'].quantile(0.75) - segment_data['PurchaseAmount'].quantile(0.25):,.2f}",
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

    # Box Plot
    with col1:
        st.subheader("Purchase Amount Distribution by Segment")

        fig = go.Figure()
        for segment in selected_segments:
            segment_data = df_filtered[df_filtered['label'] == segment]['PurchaseAmount']
            fig.add_trace(go.Box(
                y=segment_data,
                name=segment,
                marker=dict(color=get_color_palette().get(segment, '#34495e'))
            ))

        fig.update_layout(
            height=400,
            yaxis_title="Purchase Amount ($)",
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Satisfaction vs Purchase Amount
    with col2:
        st.subheader("Satisfaction vs Purchase Amount")

        fig = px.scatter(
            df_filtered,
            x='PurchaseAmount',
            y='CustomerSatisfaction',
            color='label',
            color_discrete_map=get_color_palette(),
            size='CustomerID',
            hover_data=['CustomerID', 'ProductCategory'],
            title=""
        )
        fig.update_layout(
            height=400,
            xaxis_title="Purchase Amount ($)",
            yaxis_title="Customer Satisfaction (1-5)",
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Histograms by Segment
    st.subheader("Purchase Amount Distribution per Segment")

    num_segments = len(selected_segments)
    cols = st.columns(min(num_segments, 3))

    for idx, segment in enumerate(selected_segments):
        segment_data = df_filtered[df_filtered['label'] == segment]['PurchaseAmount']

        with cols[idx % 3]:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=segment_data,
                nbinsx=10,
                name=segment,
                marker=dict(color=get_color_palette().get(segment, '#34495e')),
                opacity=0.7
            ))

            mean_val = segment_data.mean()
            median_val = segment_data.median()
            skewness = stats.skew(segment_data)

            fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_val:.0f}")
            fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                         annotation_text=f"Median: {median_val:.0f}")

            fig.update_layout(
                height=300,
                title=f"{segment} (Skew: {skewness:.3f})",
                xaxis_title="Purchase Amount ($)",
                yaxis_title="Count",
                showlegend=False,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

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

        fig = go.Figure()
        for metric in metrics_to_show:
            fig.add_trace(go.Bar(
                x=comp_df['Segment'],
                y=comp_df[metric],
                name=metric
            ))

        fig.update_layout(
            height=400,
            barmode='group',
            xaxis_title="Segment",
            yaxis_title="Value",
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

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
        options=df['CustomerRegion'].unique(),
        default=df['CustomerRegion'].unique()
    )

    selected_channels = st.sidebar.multiselect(
        "Select Retail Channels",
        options=df['RetailChannel'].unique(),
        default=df['RetailChannel'].unique()
    )

    selected_categories = st.sidebar.multiselect(
        "Select Product Categories",
        options=df['ProductCategory'].unique(),
        default=df['ProductCategory'].unique()
    )

    # Filter data
    df_filtered = df[
        (df['CustomerRegion'].isin(selected_regions)) &
        (df['RetailChannel'].isin(selected_channels)) &
        (df['ProductCategory'].isin(selected_categories))
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
        )

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            text=heatmap_data.values,
            texttemplate='$%{text:,.0f}',
            textfont={"size": 12}
        ))
        fig.update_layout(
            height=350,
            xaxis_title="Retail Channel",
            yaxis_title="Region",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

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

    # Regional Bar Chart with Satisfaction Overlay
    st.subheader("Revenue by Region with Satisfaction Overlay")

    regional_data = df_filtered.groupby('CustomerRegion').agg({
        'PurchaseAmount': 'sum',
        'CustomerSatisfaction': 'mean'
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=regional_data['CustomerRegion'],
        y=regional_data['PurchaseAmount'],
        name='Total Revenue',
        marker=dict(color='#3498db'),
        yaxis='y'
    ))

    fig.add_trace(go.Scatter(
        x=regional_data['CustomerRegion'],
        y=regional_data['CustomerSatisfaction'],
        name='Avg Satisfaction',
        mode='lines+markers',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Region",
        yaxis=dict(title="Total Revenue ($)", side='left'),
        yaxis2=dict(title="Avg Satisfaction (1-5)", overlaying='y', side='right'),
        hovermode='x unified',
        margin=dict(l=0, r=50, t=50, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Channel Comparison
    st.subheader("Channel Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        channel_revenue = df_filtered.groupby('RetailChannel')['PurchaseAmount'].sum()

        fig = go.Figure(data=[go.Bar(
            x=channel_revenue.index,
            y=channel_revenue.values,
            marker=dict(color=['#3498db', '#2ecc71']),
            text=[f'${x:,.0f}' for x in channel_revenue.values],
            textposition='outside'
        )])
        fig.update_layout(
            height=350,
            title="Revenue by Channel",
            xaxis_title="Retail Channel",
            yaxis_title="Total Revenue ($)",
            showlegend=False,
            margin=dict(l=0, r=50, t=50, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        channel_satisfaction = df_filtered.groupby('RetailChannel')['CustomerSatisfaction'].mean()

        fig = go.Figure(data=[go.Bar(
            x=channel_satisfaction.index,
            y=channel_satisfaction.values,
            marker=dict(color=['#3498db', '#2ecc71']),
            text=[f'{x:.2f}' for x in channel_satisfaction.values],
            textposition='outside'
        )])
        fig.update_layout(
            height=350,
            title="Avg Satisfaction by Channel",
            xaxis_title="Retail Channel",
            yaxis_title="Avg Satisfaction (1-5)",
            showlegend=False,
            margin=dict(l=0, r=50, t=50, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Top Categories by Region
    st.subheader("Top 10 Product Categories by Region")

    region_cols = st.columns(len(selected_regions))

    for idx, region in enumerate(selected_regions):
        with region_cols[idx]:
            region_data = df_filtered[df_filtered['CustomerRegion'] == region]
            top_categories = region_data.groupby('ProductCategory')['PurchaseAmount'].sum().nlargest(10)

            fig = go.Figure(data=[go.Bar(
                y=top_categories.index,
                x=top_categories.values,
                orientation='h',
                marker=dict(color='#3498db'),
                text=[f'${x:,.0f}' for x in top_categories.values],
                textposition='outside'
            )])
            fig.update_layout(
                height=400,
                title=f"{region} Region",
                xaxis_title="Revenue ($)",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=150, r=50, t=50, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: PRODUCT CATEGORY & DEMOGRAPHIC INSIGHTS
# ============================================================================

def page_product_demographics(df):
    """Product Category and Demographic Insights."""

    st.title("🛍️ Product Category & Demographic Insights")

    # ========== SIDEBAR FILTERS ==========
    st.sidebar.subheader("Product & Demographics Filters")

    selected_categories = st.sidebar.multiselect(
        "Select Product Categories",
        options=df['ProductCategory'].unique(),
        default=df['ProductCategory'].unique()
    )

    selected_age_groups = st.sidebar.multiselect(
        "Select Age Groups",
        options=df['CustomerAgeGroup'].unique(),
        default=df['CustomerAgeGroup'].unique()
    )

    selected_genders = st.sidebar.multiselect(
        "Select Gender",
        options=df['CustomerGender'].unique(),
        default=df['CustomerGender'].unique()
    )

    selected_segments = st.sidebar.multiselect(
        "Select Segments",
        options=df['label'].unique(),
        default=df['label'].unique()
    )

    # Filter data
    df_filtered = df[
        (df['ProductCategory'].isin(selected_categories)) &
        (df['CustomerAgeGroup'].isin(selected_age_groups)) &
        (df['CustomerGender'].isin(selected_genders)) &
        (df['label'].isin(selected_segments))
    ]

    st.divider()

    col1, col2 = st.columns(2)

    # Revenue by Category
    with col1:
        st.subheader("Revenue by Product Category")

        category_revenue = df_filtered.groupby('ProductCategory')['PurchaseAmount'].sum().sort_values(ascending=True).tail(15)

        fig = go.Figure(data=[go.Bar(
            y=category_revenue.index,
            x=category_revenue.values,
            orientation='h',
            marker=dict(color=category_revenue.values, colorscale='Viridis'),
            text=[f'${x:,.0f}' for x in category_revenue.values],
            textposition='outside'
        )])
        fig.update_layout(
            height=450,
            xaxis_title="Total Revenue ($)",
            yaxis_title="",
            showlegend=False,
            margin=dict(l=150, r=50, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Category by Segment
    with col2:
        st.subheader("Revenue by Category & Segment")

        category_segment = df_filtered.pivot_table(
            values='PurchaseAmount',
            index='ProductCategory',
            columns='label',
            aggfunc='sum',
            fill_value=0
        ).nlargest(10, df_filtered['label'].unique()[0] if len(df_filtered['label'].unique()) > 0 else 'Promising')

        view_type = st.radio("Display as:", ["Stacked", "Grouped"], horizontal=True, key="category_view")

        fig = go.Figure()
        for segment in category_segment.columns:
            fig.add_trace(go.Bar(
                x=category_segment.index,
                y=category_segment[segment],
                name=segment,
                marker=dict(color=get_color_palette().get(segment, '#34495e'))
            ))

        fig.update_layout(
            height=400,
            barmode='stack' if view_type == 'Stacked' else 'group',
            xaxis_title="Product Category",
            yaxis_title="Revenue ($)",
            xaxis={'tickangle': -45},
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=100)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Age Group Analysis
    st.subheader("Age Group Analysis")

    age_groups_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '55+']
    age_groups_order = [ag for ag in age_groups_order if ag in df_filtered['CustomerAgeGroup'].unique()]

    fig = go.Figure()

    for gender in selected_genders:
        gender_data = df_filtered[df_filtered['CustomerGender'] == gender]
        age_group_data = []
        age_group_labels = []
        age_group_counts = []

        for age_group in age_groups_order:
            ag_data = gender_data[gender_data['CustomerAgeGroup'] == age_group]
            if len(ag_data) > 0:
                avg_satisfaction = ag_data['CustomerSatisfaction'].mean()
                age_group_data.append(avg_satisfaction)
                age_group_labels.append(age_group)
                age_group_counts.append(len(ag_data))

        fig.add_trace(go.Scatter(
            x=age_group_labels,
            y=age_group_data,
            mode='lines+markers',
            name=gender,
            line=dict(width=2),
            marker=dict(size=10)
        ))

    fig.update_layout(
        height=400,
        title="Average Satisfaction by Age Group & Gender",
        xaxis_title="Age Group",
        yaxis_title="Avg Satisfaction (1-5)",
        hovermode='x unified',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Demographics Matrix
    st.subheader("Demographics Matrix (Age × Gender)")

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

    # ========== DISTRIBUTION SHAPE DIAGNOSTICS ==========
    st.subheader("Distribution Shape Diagnostics")

    col1, col2 = st.columns(2)

    # Histogram + KDE for PurchaseAmount
    with col1:
        st.write("**Purchase Amount Distribution (Pre & Post-Winsorization)**")

        fig = go.Figure()

        # Pre-winsorization
        fig.add_trace(go.Histogram(
            x=df['PurchaseAmount'],
            name='Pre-Winsorization',
            opacity=0.6,
            marker=dict(color='#e74c3c'),
            nbinsx=20
        ))

        # Post-winsorization
        fig.add_trace(go.Histogram(
            x=df_winsorized['PurchaseAmount'],
            name='Post-Winsorization',
            opacity=0.6,
            marker=dict(color='#2ecc71'),
            nbinsx=20
        ))

        fig.update_layout(
            height=350,
            barmode='overlay',
            xaxis_title="Purchase Amount ($)",
            yaxis_title="Frequency",
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Histogram for Satisfaction
    with col2:
        st.write("**Customer Satisfaction Distribution**")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df_winsorized['CustomerSatisfaction'],
            name='Satisfaction',
            marker=dict(color='#3498db'),
            nbinsx=5
        ))

        mean_satisfaction = df_winsorized['CustomerSatisfaction'].mean()
        fig.add_vline(x=mean_satisfaction, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_satisfaction:.2f}")

        fig.update_layout(
            height=350,
            xaxis_title="Satisfaction Rating (1-5)",
            yaxis_title="Frequency",
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

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
    st.sidebar.caption("NovaRetail Customer Intelligence Dashboard | v1.0")

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

# update: 29/01/2026

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from urllib.parse import quote
import time
import html

# Page configuration
st.set_page_config(
    page_title="UK Energy Market Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cascading style sheet
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0097a9 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #0097a9 0%, #005f6b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .dashboard-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: white !important;
    }
    
    .dashboard-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0097a9 0%, #006670 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 151, 169, 0.3);
        margin-bottom: 1rem;
    }
    
    .metric-card .label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .section-header {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .nomination-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin: 1rem 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .nomination-table th {
        background-color: #2C3E50;
        color: white;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
    }
    
    .nomination-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .nomination-table .demand { background-color: #F5D6B3; color: #333; }
    .nomination-table .demand-total { background-color: #E69F00; font-weight: 600; color: #333; }
    .nomination-table .supply { background-color: #B3D9F2; color: #333; }
    .nomination-table .supply-total { background-color: #0072B2; color: white; font-weight: 600; }
    .nomination-table .balance { background-color: #CC79A7; color: white; font-weight: 600; }
    
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #0097a9;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #333;
    }
    
    .no-data {
        text-align: center;
        padding: 3rem;
        background: #f8f9fa;
        border-radius: 12px;
        border: 2px dashed #dee2e6;
        color: #6c757d;
    }
    
    .no-data h3 {
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        padding: 1rem 1.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #333;
    }
    
    .legend-box {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .vessel-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin: 1rem 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .vessel-table th {
        background-color: #0097a9;
        color: white;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
    }
    
    .vessel-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #e0e0e0;
        background-color: white;
    }
    
    .vessel-table tr:hover td {
        background-color: #f0f9ff;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton > button {
        background: linear-gradient(135deg, #0097a9 0%, #006670 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .loading-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0097a9;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .loading-container h3 {
        color: #0097a9;
        margin-bottom: 1rem;
    }
    
    .loading-status {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .loading-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: #333;
    }
    
    .loading-item.complete {
        color: #059669;
    }
    
    .loading-item.pending {
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ELECTRICITY DEMAND FUNCTIONS (ELEXON API)
# ============================================================================

@st.cache_data(ttl=300)
def fetch_actual_demand_elexon(from_date, to_date):
    """
    Fetch actual electricity demand from Elexon API.
    API LIMIT: Maximum 7 days between from_date and to_date.
    
    Args:
        from_date: Start date for data retrieval
        to_date: End date for data retrieval
        
    Returns:
        DataFrame with timestamp and demand_mw columns
    """
    # Ensure date range doesn't exceed 7 days
    if (to_date - from_date).days > 7:
        to_date = from_date + timedelta(days=7)
    
    url = f"https://data.elexon.co.uk/bmrs/api/v1/demand/outturn/summary?from={from_date.strftime('%Y-%m-%d')}&to={to_date.strftime('%Y-%m-%d')}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['startTime'], utc=True)
        df['demand_mw'] = pd.to_numeric(df['demand'], errors='coerce')
        
        result = df[['timestamp', 'demand_mw']].dropna().sort_values('timestamp').reset_index(drop=True)
        
        return result
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            # Silently handle 400 errors (likely date range issues)
            return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        # Silently handle other errors to avoid cluttering the UI
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_forecast_demand_elexon(from_datetime, to_datetime):
    """
    Fetch day-ahead forecast demand from Elexon API.
    
    Args:
        from_datetime: Start datetime for forecast
        to_datetime: End datetime for forecast
        
    Returns:
        DataFrame with timestamp and demand_mw columns
    """
    from_str = from_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    to_str = to_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    url = f"https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/latest?format=json&from={from_str}&to={to_str}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['startTime'], utc=True)
        df['demand_mw'] = pd.to_numeric(df['transmissionSystemDemand'], errors='coerce')
        
        result = df[['timestamp', 'demand_mw']].dropna().sort_values('timestamp').reset_index(drop=True)
        
        return result
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_historical_demand_elexon(start_date, end_date, chunk_days=7):
    """
    Fetch historical electricity demand data in chunks.
    API LIMIT: Maximum 7 days per request.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        chunk_days: Number of days per API request (max 7 due to API limit)
        
    Returns:
        DataFrame with timestamp and demand_mw columns
    """
    # Ensure chunk_days doesn't exceed 7 (API limit)
    chunk_days = min(chunk_days, 7)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{chunk_days}D')
    if date_range[-1] < pd.Timestamp(end_date):
        date_range = date_range.append(pd.DatetimeIndex([end_date]))
    
    all_data = []
    total_chunks = len(date_range) - 1
    
    for i in range(total_chunks):
        chunk_start = date_range[i].date()
        chunk_end = date_range[i + 1].date()
        
        # Ensure chunk doesn't exceed 7 days
        if (chunk_end - chunk_start).days > 7:
            chunk_end = chunk_start + timedelta(days=7)
        
        chunk_data = fetch_actual_demand_elexon(chunk_start, chunk_end)
        if len(chunk_data) > 0:
            all_data.append(chunk_data)
        
        # Minimal rate limiting
        time.sleep(0.2)
    
    if len(all_data) == 0:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    return result


def calculate_seasonal_baseline_electricity(historical_data, target_month, min_observations=5):
    """
    Calculate seasonal baseline statistics for electricity demand.
    OPTIMIZED: Uses vectorized operations for speed.
    
    Args:
        historical_data: DataFrame with timestamp and demand_mw columns
        target_month: Month number (1-12) to calculate baseline for
        min_observations: Minimum number of observations required per hour/day-type bin
        
    Returns:
        DataFrame with baseline statistics by hour and day type
    """
    if len(historical_data) == 0:
        return pd.DataFrame()
    
    # Vectorized operations for speed
    df = historical_data.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract date and hour
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
    # Gas day calculation (vectorized)
    df['gas_day'] = pd.to_datetime(df['date']) - pd.to_timedelta((df['hour'] < 5).astype(int), unit='D')
    
    # Month and day type
    df['month'] = df['gas_day'].dt.month
    df['day_name'] = df['gas_day'].dt.day_name()
    df['day_type'] = df['day_name'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
    df['hour_bin'] = df['hour']
    
    # Filter to target month
    month_data = df[df['month'] == target_month].copy()
    
    if len(month_data) == 0:
        return pd.DataFrame()
    
    # Calculate statistics using groupby (vectorized)
    baseline = month_data.groupby(['day_type', 'hour_bin'])['demand_mw'].agg([
        ('mean_demand', 'mean'),
        ('q05', lambda x: x.quantile(0.05)),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('q95', lambda x: x.quantile(0.95)),
        ('n_obs', 'count')
    ]).reset_index()
    
    # Filter out bins with too few observations
    baseline = baseline[baseline['n_obs'] >= min_observations].copy()
    
    return baseline


def expand_baseline_to_timeline_electricity(baseline, start_time, end_time):
    """
    Expand baseline statistics to a full timeline with smoothing.
    OPTIMIZED: Uses vectorized operations for speed.
    
    Args:
        baseline: DataFrame with baseline statistics
        start_time: Start of timeline
        end_time: End of timeline
        
    Returns:
        DataFrame with baseline expanded to 30-minute intervals
    """
    if len(baseline) == 0:
        return pd.DataFrame()
    
    # Create 30-minute grid
    time_grid = pd.date_range(start=start_time, end=end_time, freq='30T')
    
    expanded = pd.DataFrame({'timestamp': time_grid})
    
    # Vectorized operations
    expanded['hour_val'] = expanded['timestamp'].dt.hour
    expanded['date'] = expanded['timestamp'].dt.date
    
    # Gas day calculation (vectorized)
    expanded['gas_day'] = pd.to_datetime(expanded['date']) - pd.to_timedelta((expanded['hour_val'] < 5).astype(int), unit='D')
    expanded['day_name'] = expanded['gas_day'].dt.day_name()
    expanded['day_type'] = expanded['day_name'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
    expanded['hour_bin'] = expanded['hour_val']
    
    # Merge with baseline
    expanded = expanded.merge(baseline, on=['day_type', 'hour_bin'], how='left')
    expanded = expanded.dropna(subset=['mean_demand'])
    expanded = expanded.sort_values('timestamp').reset_index(drop=True)
    
    if len(expanded) == 0:
        return pd.DataFrame()
    
    # Apply rolling smoothing (vectorized)
    smooth_window = 5
    for col in ['mean_demand', 'q05', 'q25', 'q75', 'q95']:
        if col in expanded.columns:
            expanded[col] = expanded[col].rolling(window=smooth_window, center=True, min_periods=1).mean()
    
    # Forward/backward fill any remaining NaNs
    expanded = expanded.ffill().bfill()
    
    return expanded


def create_electricity_demand_plot(yesterday_actual, today_actual, forecast_data, baseline_expanded):
    """
    Create the 48-hour electricity demand plot.
    
    Args:
        yesterday_actual: DataFrame with yesterday's actual demand
        today_actual: DataFrame with today's actual demand
        forecast_data: DataFrame with forecast demand
        baseline_expanded: DataFrame with seasonal baseline
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Layer 1: Seasonal uncertainty bands (5-95%)
    if len(baseline_expanded) > 0:
        fig.add_trace(go.Scatter(
            x=baseline_expanded['timestamp'],
            y=baseline_expanded['q95'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=baseline_expanded['timestamp'],
            y=baseline_expanded['q05'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(67, 147, 195, 0.15)',
            name='5-95% Range',
            legendgroup='baseline',
            hovertemplate='<b>5-95%% Range</b><extra></extra>'
        ))
        
        # Layer 2: Seasonal uncertainty bands (25-75%)
        fig.add_trace(go.Scatter(
            x=baseline_expanded['timestamp'],
            y=baseline_expanded['q75'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=baseline_expanded['timestamp'],
            y=baseline_expanded['q25'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(67, 147, 195, 0.25)',
            name='25-75% Range',
            legendgroup='baseline',
            hovertemplate='<b>25-75%% Range</b><extra></extra>'
        ))
        
        # Layer 3: Seasonal mean line
        fig.add_trace(go.Scatter(
            x=baseline_expanded['timestamp'],
            y=baseline_expanded['mean_demand'],
            mode='lines',
            line=dict(color='#2166AC', width=3, dash='dash'),
            name='Seasonal Mean',
            legendgroup='baseline',
            hovertemplate='<b>Seasonal Mean:</b> %{y:,.0f} MW<extra></extra>'
        ))
    
    # Layer 4: Yesterday's actual (grey, thinner)
    if len(yesterday_actual) > 0:
        fig.add_trace(go.Scatter(
            x=yesterday_actual['timestamp'],
            y=yesterday_actual['demand_mw'],
            mode='lines',
            line=dict(color='#7F7F7F', width=2.5),
            name='Yesterday Actual',
            hovertemplate='<b>Yesterday:</b> %{y:,.0f} MW<extra></extra>'
        ))
    
    # Layer 5: Today's actual (red, thick)
    if len(today_actual) > 0:
        fig.add_trace(go.Scatter(
            x=today_actual['timestamp'],
            y=today_actual['demand_mw'],
            mode='lines',
            line=dict(color='#D6604D', width=4),
            name='Today Actual',
            hovertemplate='<b>Actual Today:</b> %{y:,.0f} MW<extra></extra>'
        ))
    
    # Layer 6: Forecast (green, thick)
    if len(forecast_data) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_data['timestamp'],
            y=forecast_data['demand_mw'],
            mode='lines',
            line=dict(color='#4DAF4A', width=4),
            name='Forecast',
            hovertemplate='<b>Forecast:</b> %{y:,.0f} MW<extra></extra>'
        ))
    
    # Current time marker - FIX: Convert datetime to timestamp
    now = datetime.utcnow()
    all_demand = pd.concat([yesterday_actual, today_actual, forecast_data], ignore_index=True)
    if len(all_demand) > 0:
        y_max_for_label = all_demand['demand_mw'].max()
        
        fig.add_vline(
            x=now.timestamp() * 1000,  # Convert to milliseconds for Plotly
            line_dash='dot',
            line_color='#1e293b',
            line_width=2,
            annotation_text='Now',
            annotation_position='top',
            annotation=dict(
                font=dict(size=11, color='#1e293b', family='Arial Black'),
                bgcolor='white',
                bordercolor='#1e293b',
                borderwidth=2,
                borderpad=4
            )
        )
    
    # Layout
    current_month = datetime.now().month
    month_name = datetime.now().strftime('%B')
    year = datetime.now().year
    
    fig.update_layout(
        title=dict(
            text=f'<b>UK Electricity Demand: 48-Hour Outlook</b><br><sub>{month_name} {year} seasonal baseline | Updated: {datetime.utcnow().strftime("%d %b %H:%M UTC")}</sub>',
            font=dict(size=16, color='#1e293b')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1e293b', size=11),
        hovermode='x unified',
        height=550,
        margin=dict(l=60, r=60, t=100, b=60),
        template='plotly_white',
        xaxis=dict(
            gridcolor='#e2e8f0',
            linecolor='#1e293b',
            linewidth=2,
            tickfont=dict(color='#1e293b', size=11),
            title_font=dict(color='#1e293b'),
            showline=True,
            tickformat='%a %d<br>%H:%M'
        ),
        yaxis=dict(
            title='Demand (MW)',
            gridcolor='#e2e8f0',
            linecolor='#1e293b',
            linewidth=2,
            tickfont=dict(color='#1e293b', size=11),
            title_font=dict(color='#1e293b'),
            showline=True
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#1e293b',
            borderwidth=2,
            font=dict(size=12, color='#1e293b')
        )
    )
    
    return fig


# ============================================================================
# LNG VESSEL TRACKING FUNCTIONS
# ============================================================================

def get_vessel_info(ship_name: str) -> dict:
    """Fetch vessel details from VesselFinder website."""
    search_name = quote(ship_name, safe='')
    search_url = f"https://www.vesselfinder.com/vessels?name={search_name}"
    
    default_result = {
        'Ship': ship_name,
        'IMO': None,
        'MMSI': None,
        'Flag': None,
        'Deadweight': None,
        'GrossTonnage': None,
        'VesselFinderURL': None,
        'Status': 'Not Found'
    }
    
    try:
        time.sleep(1)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code != 200:
            default_result['Status'] = f'HTTP Error: {response.status_code}'
            return default_result
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        vessel_links = soup.find_all('a', href=True)
        vessel_link = None
        for link in vessel_links:
            if '/vessels/' in link['href'] and link['href'] != '/vessels/':
                vessel_link = link['href']
                break
        
        if not vessel_link:
            return default_result
        
        vessel_url = f"https://www.vesselfinder.com{vessel_link}"
        time.sleep(0.5)
        vessel_response = requests.get(vessel_url, headers=headers, timeout=10)
        
        if vessel_response.status_code != 200:
            default_result['Status'] = f'Vessel page error: {vessel_response.status_code}'
            return default_result
        
        vessel_soup = BeautifulSoup(vessel_response.content, 'html.parser')
        
        def extract_from_page(soup, label):
            for elem in soup.find_all(['td', 'span', 'div']):
                text = elem.get_text(strip=True)
                if text.lower() == label.lower():
                    next_elem = elem.find_next_sibling()
                    if next_elem:
                        return next_elem.get_text(strip=True)
            
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for i, cell in enumerate(cells):
                        if label.lower() in cell.get_text(strip=True).lower():
                            if i + 1 < len(cells):
                                return cells[i + 1].get_text(strip=True)
            return None
        
        imo = extract_from_page(vessel_soup, 'IMO')
        mmsi = extract_from_page(vessel_soup, 'MMSI')
        flag = extract_from_page(vessel_soup, 'Flag')
        deadweight = extract_from_page(vessel_soup, 'Deadweight')
        gross_tonnage = extract_from_page(vessel_soup, 'Gross tonnage')
        
        return {
            'Ship': ship_name,
            'IMO': imo,
            'MMSI': mmsi,
            'Flag': flag,
            'Deadweight': deadweight,
            'GrossTonnage': gross_tonnage,
            'VesselFinderURL': vessel_url,
            'Status': 'Found'
        }
        
    except Exception as e:
        default_result['Status'] = f'Error: {str(e)}'
        return default_result


@st.cache_data(ttl=300)
def get_milford_haven_vessels() -> pd.DataFrame:
    """Scrape Milford Haven Port Authority website for arriving vessels."""
    url = "https://www.mhpa.co.uk/live-information/vessels-arriving/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', class_='timetable-table')
        if not table:
            table = soup.find('table')
        
        if not table:
            return None
        
        headers_row = table.find('thead')
        if headers_row:
            headers = [th.get_text(strip=True) for th in headers_row.find_all(['th', 'td'])]
        else:
            first_row = table.find('tr')
            headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
        
        rows = table.find_all('tr')
        data = []
        
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                if len(row_data) >= len(headers):
                    data.append(row_data[:len(headers)])
                elif len(row_data) > 0:
                    row_data.extend([''] * (len(headers) - len(row_data)))
                    data.append(row_data)
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=headers)
        df.columns = [col.strip() for col in df.columns]
        
        return df
        
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def get_lng_vessels_with_details() -> pd.DataFrame:
    """Get LNG vessels with details from VesselFinder."""
    vessels_df = get_milford_haven_vessels()
    
    if vessels_df is None or len(vessels_df) == 0:
        return None
    
    ship_type_col = None
    for col in vessels_df.columns:
        if 'type' in col.lower() or 'ship type' in col.lower():
            ship_type_col = col
            break
    
    if ship_type_col is None:
        return vessels_df
    
    lng_df = vessels_df[
        vessels_df[ship_type_col].str.lower().str.contains('lng', na=False)
    ].copy()
    
    if len(lng_df) == 0:
        return None
    
    ship_col = None
    for col in vessels_df.columns:
        if col.lower() in ['ship', 'vessel', 'name', 'vessel name', 'ship name']:
            ship_col = col
            break
    
    if ship_col is None:
        ship_col = vessels_df.columns[0]
    
    unique_ships = lng_df[ship_col].unique()
    
    vessel_details = []
    for ship in unique_ships:
        details = get_vessel_info(ship)
        vessel_details.append(details)
    
    details_df = pd.DataFrame(vessel_details)
    final_df = lng_df.merge(details_df, left_on=ship_col, right_on='Ship', how='left')
    
    return final_df


def render_lng_vessel_table(df: pd.DataFrame):
    """Render LNG vessel table."""
    if df is None or len(df) == 0:
        st.markdown('''
        <div class="no-data">
            <h3>No LNG Vessels Found</h3>
            <p>No LNG tankers are currently scheduled.</p>
        </div>
        ''', unsafe_allow_html=True)
        return

    ship_col = None
    for col in df.columns:
        if col.lower() in ['ship', 'vessel', 'name']:
            ship_col = col
            break
    if not ship_col:
        ship_col = df.columns[0]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

    to_col = None
    for col in df.columns:
        if col.lower().strip() == 'to':
            to_col = col
            break

    display_cols = [ship_col]
    if to_col and to_col not in display_cols:
        display_cols.append(to_col)

    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['date', 'time', 'eta', 'arrival', 'from', 'agent']):
            if col not in display_cols:
                display_cols.append(col)

    detail_cols = ['IMO', 'Flag', 'Deadweight', 'GrossTonnage']
    for col in detail_cols:
        if col in df.columns and col not in display_cols:
            display_cols.append(col)

    display_cols = [col for col in display_cols if col in df.columns]
    display_df = df[display_cols].copy()

    rename_map = {'GrossTonnage': 'Gross Tonnage'}
    if to_col and to_col.lower() == 'to':
        rename_map[to_col] = 'Destination'

    display_df = display_df.rename(columns=rename_map)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# GAS MARKET FUNCTIONS (EXISTING CODE)
# ============================================================================

@st.cache_data(ttl=120)
def scrape_gassco_data():
    """Scrape GASSCO REMIT data."""
    try:
        session = requests.Session()
        session.get("https://umm.gassco.no/", timeout=10)
        session.get("https://umm.gassco.no/disclaimer/acceptDisclaimer", timeout=10)
        response = session.get("https://umm.gassco.no/", timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        msg_tables = soup.find_all('table', class_='msgTable')
        
        fields_df = parse_gassco_table(msg_tables[0]) if len(msg_tables) > 0 else None
        terminal_df = parse_gassco_table(msg_tables[1]) if len(msg_tables) > 1 else None
        
        return fields_df, terminal_df
    except Exception:
        return None, None


def parse_gassco_table(table):
    """Parse GASSCO table."""
    rows = table.find_all('tr', id=True)
    data = []
    
    for row in rows:
        cells = row.find_all('td')
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        if len(cell_texts) >= 19:
            data.append({
                'Affected Asset or Unit': cell_texts[1],
                'Event Status': cell_texts[2],
                'Type of Unavailability': cell_texts[3],
                'Publication date/time': cell_texts[5],
                'Event Start': cell_texts[6],
                'Event Stop': cell_texts[7],
                'Technical Capacity': cell_texts[9],
                'Available Capacity': cell_texts[10],
                'Unavailable Capacity': cell_texts[11],
                'Reason for the unavailability': cell_texts[12],
            })
    
    return pd.DataFrame(data) if data else None


def process_remit_data(df):
    """Process REMIT data."""
    if df is None or len(df) == 0:
        return None
    
    df = df[df['Event Status'] == 'Active'].copy()
    if len(df) == 0:
        return None
    
    df['Publication date/time'] = pd.to_datetime(df['Publication date/time'], format='ISO8601', utc=True)
    df['Event Start'] = pd.to_datetime(df['Event Start'], format='ISO8601', utc=True)
    df['Event Stop'] = pd.to_datetime(df['Event Stop'], format='ISO8601', utc=True)
    
    for col in ['Technical Capacity', 'Available Capacity', 'Unavailable Capacity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    cutoff = datetime.now(df['Event Start'].dt.tz) + timedelta(days=14)
    df = df[(df['Event Start'] <= cutoff) | (df['Event Stop'] <= cutoff)]
    
    if len(df) == 0:
        return None
    
    df = df.drop_duplicates()
    df['Duration'] = (df['Event Stop'] - df['Event Start']).dt.total_seconds() / (24 * 3600)
    df['midpoint'] = df['Event Start'] + (df['Event Stop'] - df['Event Start']) / 2
    
    return df.sort_values('Unavailable Capacity')


@st.cache_data(ttl=120)
def get_gas_data(request_type, max_retries=3):
    """Fetch gas data from National Gas API."""
    url = "https://data.nationalgas.com/api/gas-system-status-graph"
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, 
                json={"request": request_type}, 
                headers=headers, 
                timeout=30
            )
            response.raise_for_status()
            return pd.DataFrame(response.json()["data"])
        except:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    
    return None


def get_chart_layout(title="", height=500):
    """Get chart layout."""
    return dict(
        title=dict(text=title, font=dict(size=18, color='#1e293b')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1e293b', size=12),
        hovermode='x unified',
        height=height,
        margin=dict(l=60, r=60, t=100, b=60),
        template='plotly_white',
        xaxis=dict(
            gridcolor='#e2e8f0', 
            linecolor='#1e293b',
            linewidth=2,
            tickfont=dict(color='#1e293b', size=12),
            title_font=dict(color='#1e293b', size=14),
            showline=True
        ),
        yaxis=dict(
            gridcolor='#e2e8f0', 
            linecolor='#1e293b',
            linewidth=2,
            tickfont=dict(color='#1e293b', size=12),
            title_font=dict(color='#1e293b', size=14),
            showline=True
        )
    )


def create_gassco_timeline_plot(df, title_prefix):
    """Create GASSCO timeline plot."""
    colors = {'Planned': '#7fcdcd', 'Unplanned': '#f8b4b4'}
    fig = go.Figure()
    shown = set()
    
    for _, row in df.iterrows():
        color = colors.get(row['Type of Unavailability'], '#94a3b8')
        show_legend = row['Type of Unavailability'] not in shown
        if show_legend:
            shown.add(row['Type of Unavailability'])
        
        fig.add_trace(go.Scatter(
            x=[row['Event Start'], row['Event Stop']],
            y=[row['Affected Asset or Unit'], row['Affected Asset or Unit']],
            mode='lines', line=dict(color=color, width=20),
            name=row['Type of Unavailability'],
            legendgroup=row['Type of Unavailability'],
            showlegend=show_legend,
            hovertemplate=f"<b>{row['Affected Asset or Unit']}</b><br>Type: {row['Type of Unavailability']}<br>Unavailable: {row['Unavailable Capacity']:.1f} MSm³/d<extra></extra>"
        ))
        
        fig.add_annotation(
            x=row['midpoint'], y=row['Affected Asset or Unit'],
            text=f"<b>{row['Unavailable Capacity']:.1f}</b>",
            showarrow=False, font=dict(size=11, color='#1e293b'),
            yshift=22, bgcolor='rgba(255,255,255,0.9)', bordercolor='#cbd5e1', borderwidth=1, borderpad=4
        )
    
    today = datetime.now(df['Event Start'].dt.tz)
    layout = get_chart_layout(f"<b>{title_prefix} Outages Timeline</b>", max(450, len(df) * 65))
    layout['xaxis']['type'] = 'date'
    layout['xaxis']['tickformat'] = '%d %b'
    layout['yaxis']['categoryorder'] = 'array'
    layout['yaxis']['categoryarray'] = df['Affected Asset or Unit'].tolist()
    layout['shapes'] = [dict(type='line', x0=today, x1=today, y0=0, y1=1, yref='paper', line=dict(color='#ef4444', width=2, dash='dash'))]
    
    fig.update_layout(**layout)
    fig.add_annotation(x=today, y=1.02, yref='paper', text='<b>Today</b>', showarrow=False, font=dict(size=12, color='#ef4444'), bgcolor='white', bordercolor='#ef4444', borderwidth=1, borderpad=4)
    
    return fig


def create_gassco_cumulative_plot(df, title_prefix):
    """Create GASSCO cumulative plot."""
    events = []
    for _, row in df.iterrows():
        events.append({'time': row['Event Start'], 'delta': -row['Unavailable Capacity']})
        events.append({'time': row['Event Stop'], 'delta': row['Unavailable Capacity']})
    
    events_df = pd.DataFrame(events).groupby('time')['delta'].sum().reset_index().sort_values('time')
    events_df['cumulative'] = events_df['delta'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=events_df['time'], y=events_df['cumulative'],
        mode='lines+markers', line=dict(shape='hv', color='#dc2626', width=3),
        marker=dict(size=8), fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.1)',
        hovertemplate="<b>Time:</b> %{x|%d %b %Y %H:%M}<br><b>Cumulative:</b> %{y:.1f} MSm³/d<extra></extra>"
    ))
    
    today = datetime.now(df['Event Start'].dt.tz)
    layout = get_chart_layout(f"<b>{title_prefix} Cumulative Unavailable</b>", 450)
    layout['xaxis']['type'] = 'date'
    layout['yaxis']['title'] = 'Unavailable Capacity (MSm³/d)'
    layout['shapes'] = [dict(type='line', x0=today, x1=today, y0=0, y1=1, yref='paper', line=dict(color='#1e293b', width=2, dash='dash'))]
    layout['showlegend'] = False
    
    fig.update_layout(**layout)
    return fig


def create_flow_chart(df, column_name, chart_title, color='#0097a9'):
    """Create flow chart."""
    if column_name not in df.columns:
        return None, 0, 0, 0
    
    avg = np.average(df[column_name], weights=df['interval_seconds'])
    
    today = datetime.now().date()
    start = datetime.combine(today, datetime.min.time().replace(hour=5))
    end = start + timedelta(days=1)
    now = datetime.now()
    
    elapsed_pct = max(0, min(1, (now - start).total_seconds() / 86400))
    total = avg * elapsed_pct
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df[column_name],
        mode='lines', line=dict(color=color, width=3),
        fill='tozeroy', fillcolor='rgba(0, 151, 169, 0.15)',
        hovertemplate='<b>Time:</b> %{x|%H:%M}<br><b>Flow:</b> %{y:.2f} mcm<extra></extra>'
    ))
    
    fig.add_hline(y=avg, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text=f"<b>Avg: {avg:.2f}</b>", annotation_position="right",
                  annotation=dict(font=dict(size=12, color="#ef4444"), bgcolor="white", bordercolor="#ef4444", borderwidth=1))
    
    fig.add_vline(x=int(now.timestamp() * 1000), line_color="#1e293b", line_width=2,
                  annotation_text=f"<b>Now: {total:.2f}</b>", annotation_position="top",
                  annotation=dict(font=dict(size=12, color='#1e293b'), bgcolor="white", bordercolor="#1e293b", borderwidth=1))
    
    y_max = max(df[column_name].max(), 1)
    layout = get_chart_layout(f"<b>{chart_title}</b>", 450)
    layout['xaxis']['range'] = [start, end]
    layout['xaxis']['tickformat'] = '%H:%M'
    layout['yaxis']['range'] = [0, y_max * 1.25]
    layout['yaxis']['title'] = 'Flow Rate (mcm)'
    layout['showlegend'] = False
    
    fig.update_layout(**layout)
    
    return fig, avg, total, df[column_name].iloc[-1] if len(df) > 0 else 0


def render_metric_cards(metrics):
    """Render metric cards."""
    cols = st.columns(len(metrics))
    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value:.2f} {unit}</div>
            </div>
            """, unsafe_allow_html=True)


def render_nomination_table(demand_df, supply_df):
    """Render nomination table."""
    demand_cols = ["LDZ Offtake", "Power Station", "Industrial", "Storage Injection", "Bacton BBL Export", "Bacton INT Export", "Moffat Export"]
    supply_cols = ["Storage Withdrawal", "LNG", "Bacton BBL Import", "Bacton INT Import", "Beach (UKCS/Norway)"]
    
    def summarise(df, cols):
        n = len(df) if df is not None else 0
        pct = (n * 2) / 1440 if n > 0 else 0
        results = []
        for i, col in enumerate(cols):
            if df is not None and col in df.columns:
                avg = df[col].mean() if not df[col].isna().all() else 0
                comp = avg * pct
                inst = df[col].iloc[-1] if len(df) > 0 and not df[col].isna().all() else 0
            else:
                avg, comp, inst = 0, 0, 0
            results.append({"Category": col, "Avg": round(avg, 2), "Comp": round(comp, 2), "Inst": round(inst, 2)})
        return pd.DataFrame(results)
    
    demand_sum = summarise(demand_df, demand_cols)
    supply_sum = summarise(supply_df, supply_cols)
    
    d_tot = demand_sum[["Avg", "Comp", "Inst"]].sum()
    s_tot = supply_sum[["Avg", "Comp", "Inst"]].sum()
    bal = s_tot - d_tot
    
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item"><div class="legend-box" style="background-color: #F5D6B3;"></div> Demand</div>
        <div class="legend-item"><div class="legend-box" style="background-color: #E69F00;"></div> Demand Total</div>
        <div class="legend-item"><div class="legend-box" style="background-color: #B3D9F2;"></div> Supply</div>
        <div class="legend-item"><div class="legend-box" style="background-color: #0072B2;"></div> Supply Total</div>
        <div class="legend-item"><div class="legend-box" style="background-color: #CC79A7;"></div> Balance</div>
    </div>
    """, unsafe_allow_html=True)
    
    rows = []
    for _, r in demand_sum.iterrows():
        rows.append(f'<tr class="demand"><td>{r["Category"]}</td><td style="text-align:right;">{r["Avg"]:.1f}</td><td style="text-align:right;">{r["Comp"]:.1f}</td><td style="text-align:right;">{r["Inst"]:.1f}</td></tr>')
    rows.append(f'<tr class="demand-total"><td><strong>DEMAND TOTAL</strong></td><td style="text-align:right;"><strong>{d_tot["Avg"]:.1f}</strong></td><td style="text-align:right;"><strong>{d_tot["Comp"]:.1f}</strong></td><td style="text-align:right;"><strong>{d_tot["Inst"]:.1f}</strong></td></tr>')
    
    for _, r in supply_sum.iterrows():
        rows.append(f'<tr class="supply"><td>{r["Category"]}</td><td style="text-align:right;">{r["Avg"]:.1f}</td><td style="text-align:right;">{r["Comp"]:.1f}</td><td style="text-align:right;">{r["Inst"]:.1f}</td></tr>')
    rows.append(f'<tr class="supply-total"><td><strong>SUPPLY TOTAL</strong></td><td style="text-align:right;"><strong>{s_tot["Avg"]:.1f}</strong></td><td style="text-align:right;"><strong>{s_tot["Comp"]:.1f}</strong></td><td style="text-align:right;"><strong>{s_tot["Inst"]:.1f}</strong></td></tr>')
    rows.append(f'<tr class="balance"><td><strong>BALANCE</strong></td><td style="text-align:right;"><strong>{bal["Avg"]:.1f}</strong></td><td style="text-align:right;"><strong>{bal["Comp"]:.1f}</strong></td><td style="text-align:right;"><strong>{bal["Inst"]:.1f}</strong></td></tr>')
    
    st.markdown(f"""
    <table class="nomination-table">
        <thead><tr><th>Category</th><th style="text-align:right;">Avg Rate (mcm)</th><th style="text-align:right;">Completed (mcm)</th><th style="text-align:right;">Instant (mcm)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """, unsafe_allow_html=True)
    
    return bal


def render_gassco_table(df):
    """Render GASSCO table."""
    display_df = df.copy()
    for col in ['Publication date/time', 'Event Start', 'Event Stop']:
        if col in display_df.columns:
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M')
    
    cols = ['Affected Asset or Unit', 'Type of Unavailability', 'Event Start', 'Event Stop', 'Unavailable Capacity', 'Duration', 'Reason for the unavailability']
    cols = [c for c in cols if c in display_df.columns]
    
    st.dataframe(display_df[cols], use_container_width=True, hide_index=True)


# ============================================================================
# PRELOADING FUNCTIONS - EAGER LOADING ON APP START
# ============================================================================

@st.cache_data(ttl=120, show_spinner=False)
def preload_national_gas_data():
    """Preload National Gas demand and supply data."""
    demand_df = get_gas_data("demandCategoryGraph")
    supply_df = get_gas_data("supplyCategoryGraph")
    return demand_df, supply_df


@st.cache_data(ttl=120, show_spinner=False)
def preload_gassco_data():
    """Preload GASSCO REMIT data."""
    fields_df, terminal_df = scrape_gassco_data()
    fields_proc = process_remit_data(fields_df)
    terminal_proc = process_remit_data(terminal_df)
    return fields_proc, terminal_proc


@st.cache_data(ttl=300, show_spinner=False)
def preload_lng_data():
    """Preload LNG vessel data (basic - without VesselFinder details to avoid slow scraping)."""
    return get_milford_haven_vessels()


@st.cache_data(ttl=300, show_spinner=False)
def preload_elexon_data():
    """Preload Elexon electricity demand data."""
    today = datetime.utcnow().date()
    current_hour = datetime.utcnow().hour
    
    # Gas day logic
    if current_hour < 5:
        gas_day_today = today - timedelta(days=1)
    else:
        gas_day_today = today
    
    gas_day_yesterday = gas_day_today - timedelta(days=1)
    
    # Plot window
    plot_start = datetime.combine(gas_day_yesterday, datetime.min.time().replace(hour=5, tzinfo=None))
    plot_end = datetime.combine(gas_day_today + timedelta(days=2), datetime.min.time().replace(hour=5, tzinfo=None))
    today_gas_day_start = datetime.combine(gas_day_today, datetime.min.time().replace(hour=5, tzinfo=None))
    
    # Fetch actual demand
    actual_demand = fetch_actual_demand_elexon(gas_day_yesterday, today + timedelta(days=1))
    
    # Fetch forecast
    forecast_demand = fetch_forecast_demand_elexon(plot_start, plot_end)
    
    # Fetch historical data for seasonal baseline - 1 YEAR
    historical_start = (today - timedelta(days=365)).replace(day=1)
    historical_end = gas_day_yesterday - timedelta(days=1)
    historical_demand = fetch_historical_demand_elexon(historical_start, historical_end)
    
    # Calculate seasonal baseline
    current_month = today.month
    baseline = calculate_seasonal_baseline_electricity(historical_demand, current_month)
    baseline_expanded = expand_baseline_to_timeline_electricity(baseline, plot_start, plot_end)
    
    return {
        'actual_demand': actual_demand,
        'forecast_demand': forecast_demand,
        'baseline_expanded': baseline_expanded,
        'plot_start': plot_start,
        'plot_end': plot_end,
        'today_gas_day_start': today_gas_day_start,
        'gas_day_yesterday': gas_day_yesterday,
        'gas_day_today': gas_day_today
    }


def preload_all_data():
    """
    Preload all data sources on app start.
    This function runs once when the app loads and caches results.
    Returns a dictionary with all preloaded data.
    """
    # Use session state to track if we've shown the loading screen
    if 'data_preloaded' not in st.session_state:
        st.session_state.data_preloaded = False
    
    if st.session_state.data_preloaded:
        # Data already preloaded, just return from cache
        return {
            'national_gas': preload_national_gas_data(),
            'gassco': preload_gassco_data(),
            'lng': preload_lng_data(),
            'elexon': preload_elexon_data()
        }
    
    # Show loading UI
    loading_container = st.empty()
    
    with loading_container.container():
        st.markdown('''
        <div class="loading-container">
            <h3>⚡ Loading Market Data...</h3>
            <p>Fetching data from all sources for instant view switching</p>
        </div>
        ''', unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load National Gas data
        status_text.text("📊 Loading National Gas flows...")
        progress_bar.progress(10)
        national_gas_data = preload_national_gas_data()
        progress_bar.progress(25)
        
        # Load GASSCO data
        status_text.text("🔧 Loading GASSCO outages...")
        gassco_data = preload_gassco_data()
        progress_bar.progress(50)
        
        # Load LNG vessel data
        status_text.text("🚢 Loading Milford Haven vessels...")
        lng_data = preload_lng_data()
        progress_bar.progress(65)
        
        # Load Elexon data (this is the slowest due to historical fetch)
        status_text.text("⚡ Loading Elexon electricity data (historical baseline)...")
        elexon_data = preload_elexon_data()
        progress_bar.progress(100)
        
        status_text.text("✅ All data loaded!")
        time.sleep(0.5)
    
    # Clear the loading UI
    loading_container.empty()
    
    # Mark as preloaded
    st.session_state.data_preloaded = True
    
    return {
        'national_gas': national_gas_data,
        'gassco': gassco_data,
        'lng': lng_data,
        'elexon': elexon_data
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # EAGER LOADING: Preload all data on app start
    preloaded_data = preload_all_data()
    
    with st.sidebar:
        st.markdown('<div style="text-align:center;padding:1rem 0;"><h1 style="font-size:1.6rem;margin:0;color:#0097a9 !important;">UK Energy Market</h1><p style="font-size:0.85rem;opacity:0.8;margin-top:0.5rem;color:#0097a9 !important;">Real-time Dashboard</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Data Source")
        data_source = st.radio(
            "Source", 
            ["National Gas", "GASSCO", "Milford Haven LNG", "Elexon"], 
            label_visibility="collapsed", 
            key="ds"
        )
        
        st.markdown("---")
        
        if data_source == "National Gas":
            st.markdown("### Views")
            ng_view = st.radio("View", ["Table", "Supply", "Demand"], label_visibility="collapsed", key="ngv")
            
            if ng_view == "Supply":
                st.markdown("---")
                st.markdown("##### <span style='color:#FFFF00;'>Supply Categories</span>", unsafe_allow_html=True)
                supply_cat = st.radio("Cat", ["LNG", "Storage Withdrawal", "Beach Terminal", "IC Import"], label_visibility="collapsed", key="sc")
            elif ng_view == "Demand":
                st.markdown("---")
                st.markdown("##### <span style='color:#FFFF00;'>Demand Categories</span>", unsafe_allow_html=True)
                demand_cat = st.radio("Cat", ["CCGT", "Storage Injection", "LDZ", "Industrial", "IC Export"], label_visibility="collapsed", key="dc")
        
        elif data_source == "GASSCO":
            st.markdown("### Views")
            gassco_view = st.radio("View", ["Field Outages", "Terminal Outages"], label_visibility="collapsed", key="gv")
        
        elif data_source == "Elexon":
            st.markdown("### Views")
            elexon_view = st.radio("View", ["Electricity Demand"], label_visibility="collapsed", key="ev")
        
        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_preloaded = False
            st.rerun()
        
        st.markdown(f'<div style="text-align:center;padding:1rem 0;font-size:0.8rem;color:#718096;">Last updated:<br>{datetime.now().strftime("%H:%M:%S %d/%m/%Y")}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-header"><h1>UK Energy Market Dashboard</h1><p>Real-time monitoring of UK gas and electricity markets</p></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ELEXON / ELECTRICITY DEMAND VIEW
    # ========================================================================
    if data_source == "Elexon":
        st.markdown('<div class="section-header">UK Electricity Demand: 48-Hour Outlook</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="info-box">
            <strong>Electricity Demand Forecast</strong> — Shows actual demand from yesterday and today, 
            48-hour day-ahead forecast, and seasonal baseline ranges based on historical patterns for the current month.
        </div>
        ''', unsafe_allow_html=True)
        
        # Use preloaded data
        elexon_data = preloaded_data['elexon']
        
        actual_demand = elexon_data['actual_demand']
        forecast_demand = elexon_data['forecast_demand']
        baseline_expanded = elexon_data['baseline_expanded']
        plot_start = elexon_data['plot_start']
        plot_end = elexon_data['plot_end']
        today_gas_day_start = elexon_data['today_gas_day_start']
        
        # Split actual into yesterday and today
        if len(actual_demand) > 0:
            actual_demand_copy = actual_demand.copy()
            actual_demand_copy['timestamp'] = pd.to_datetime(actual_demand_copy['timestamp'], utc=True).dt.tz_localize(None)
            
            # Filter to only show data from plot_start (5am yesterday) onwards
            actual_demand_copy = actual_demand_copy[actual_demand_copy['timestamp'] >= plot_start].copy()
            
            yesterday_actual = actual_demand_copy[
                actual_demand_copy['timestamp'] < today_gas_day_start
            ].copy()
            
            today_actual = actual_demand_copy[
                actual_demand_copy['timestamp'] >= today_gas_day_start
            ].copy()
            
            # Get latest actual time to filter forecast
            if len(today_actual) > 0:
                latest_actual_time = today_actual['timestamp'].max()
            else:
                latest_actual_time = today_gas_day_start
        else:
            yesterday_actual = pd.DataFrame()
            today_actual = pd.DataFrame()
            latest_actual_time = today_gas_day_start
        
        # Filter forecast to only show after latest actual
        if len(forecast_demand) > 0:
            forecast_copy = forecast_demand.copy()
            forecast_copy['timestamp'] = pd.to_datetime(forecast_copy['timestamp'], utc=True).dt.tz_localize(None)
            forecast_plot = forecast_copy[
                (forecast_copy['timestamp'] > latest_actual_time) & 
                (forecast_copy['timestamp'] <= plot_end)
            ].copy()
        else:
            forecast_plot = pd.DataFrame()
        
        # Create the plot
        fig = create_electricity_demand_plot(yesterday_actual, today_actual, forecast_plot, baseline_expanded)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if len(today_actual) > 0:
                current_demand = today_actual['demand_mw'].iloc[-1]
                st.metric("Current Demand", f"{current_demand:,.0f} MW")
            else:
                st.metric("Current Demand", "N/A")
        
        with col2:
            if len(today_actual) > 0:
                avg_today = today_actual['demand_mw'].mean()
                st.metric("Average Today", f"{avg_today:,.0f} MW")
            else:
                st.metric("Average Today", "N/A")
        
        with col3:
            if len(forecast_plot) > 0:
                peak_forecast = forecast_plot['demand_mw'].max()
                st.metric("Peak Forecast", f"{peak_forecast:,.0f} MW")
            else:
                st.metric("Peak Forecast", "N/A")
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    # ========================================================================
    # NATIONAL GAS VIEW
    # ========================================================================
    elif data_source == "National Gas":
        # Use preloaded data
        demand_df, supply_df = preloaded_data['national_gas']
        
        if demand_df is not None and supply_df is not None:
            if 'Storage' in demand_df.columns:
                demand_df = demand_df.copy()
                demand_df.rename(columns={'Storage': 'Storage Injection'}, inplace=True)
            
            n = len(demand_df)
            today = datetime.now().date()
            start = datetime.combine(today, datetime.min.time().replace(hour=5))
            ts = [start + timedelta(minutes=2*i) for i in range(n)]
            
            demand_df = demand_df.copy()
            demand_df['Timestamp'] = ts
            demand_df = demand_df.sort_values('Timestamp').reset_index(drop=True)
            demand_df['next_time'] = demand_df['Timestamp'].shift(-1).fillna(demand_df['Timestamp'].iloc[-1] + timedelta(minutes=2))
            demand_df['interval_seconds'] = (demand_df['next_time'] - demand_df['Timestamp']).dt.total_seconds()
            
            n_s = len(supply_df)
            ts_s = [start + timedelta(minutes=2*i) for i in range(n_s)]
            supply_df = supply_df.copy()
            supply_df['Timestamp'] = ts_s
            supply_df = supply_df.sort_values('Timestamp').reset_index(drop=True)
            supply_df['next_time'] = supply_df['Timestamp'].shift(-1).fillna(supply_df['Timestamp'].iloc[-1] + timedelta(minutes=2))
            supply_df['interval_seconds'] = (supply_df['next_time'] - supply_df['Timestamp']).dt.total_seconds()
            
            if ng_view == "Table":
                st.markdown('<div class="section-header">UK Gas Flows - Supply, Demand & Balance</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box"><strong>Flow Table</strong> shows the current gas day flows. All values in mcm.</div>', unsafe_allow_html=True)
                bal = render_nomination_table(demand_df, supply_df)
            
            elif ng_view == "Supply":
                st.markdown(f'<div class="section-header">Supply - {supply_cat}</div>', unsafe_allow_html=True)
                col_map = {"LNG": "LNG", "Storage Withdrawal": "Storage Withdrawal", "Beach Terminal": "Beach (UKCS/Norway)", "IC Import": None}
                
                if supply_cat == "IC Import":
                    supply_df['Total IC Import'] = supply_df['Bacton BBL Import'] + supply_df['Bacton INT Import']
                    col_name = 'Total IC Import'
                else:
                    col_name = col_map[supply_cat]
                
                if col_name and col_name in supply_df.columns:
                    fig, avg, total, current = create_flow_chart(supply_df, col_name, f'{supply_cat} Flow', '#0097a9')
                    if fig:
                        render_metric_cards([("Average Flow", avg, "mcm"), ("Total So Far", total, "mcm"), ("Current Flow", current, "mcm")])
                        st.plotly_chart(fig, use_container_width=True, theme=None)
            
            elif ng_view == "Demand":
                st.markdown(f'<div class="section-header">Demand - {demand_cat}</div>', unsafe_allow_html=True)
                col_map = {"CCGT": "Power Station", "Storage Injection": "Storage Injection", "LDZ": "LDZ Offtake", "Industrial": "Industrial", "IC Export": None}
                
                if demand_cat == "IC Export":
                    demand_df['Total IC Export'] = demand_df['Bacton BBL Export'] + demand_df['Bacton INT Export'] + demand_df['Moffat Export']
                    col_name = 'Total IC Export'
                else:
                    col_name = col_map[demand_cat]
                
                if col_name and col_name in demand_df.columns:
                    fig, avg, total, current = create_flow_chart(demand_df, col_name, f'{demand_cat} Flow', '#f59e0b')
                    if fig:
                        render_metric_cards([("Average Flow", avg, "mcm"), ("Total So Far", total, "mcm"), ("Current Flow", current, "mcm")])
                        st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            st.error("⚠️ Unable to fetch National Gas data.")
    
    # ========================================================================
    # GASSCO VIEW
    # ========================================================================
    elif data_source == "GASSCO":
        st.markdown(f'<div class="section-header">GASSCO - {gassco_view}</div>', unsafe_allow_html=True)
        
        # Use preloaded data
        fields_proc, terminal_proc = preloaded_data['gassco']
        
        if gassco_view == "Field Outages":
            if fields_proc is not None and len(fields_proc) > 0:
                st.markdown(f'<div class="info-box"><strong>{len(fields_proc)} active field outage(s)</strong> within 14 days.</div>', unsafe_allow_html=True)
                st.plotly_chart(create_gassco_timeline_plot(fields_proc, "Field"), use_container_width=True, theme=None)
                st.plotly_chart(create_gassco_cumulative_plot(fields_proc, "Field"), use_container_width=True, theme=None)
                st.markdown("#### Outage Details")
                render_gassco_table(fields_proc)
            else:
                st.markdown('<div class="no-data"><h3>✅ No Field Outages</h3><p>No active field outages within 14 days.</p></div>', unsafe_allow_html=True)
        else:
            if terminal_proc is not None and len(terminal_proc) > 0:
                st.markdown(f'<div class="info-box"><strong>{len(terminal_proc)} active terminal outage(s)</strong> within 14 days.</div>', unsafe_allow_html=True)
                st.plotly_chart(create_gassco_timeline_plot(terminal_proc, "Terminal"), use_container_width=True, theme=None)
                st.plotly_chart(create_gassco_cumulative_plot(terminal_proc, "Terminal"), use_container_width=True, theme=None)
                st.markdown("#### Outage Details")
                render_gassco_table(terminal_proc)
            else:
                st.markdown('<div class="no-data"><h3>✅ No Terminal Outages</h3><p>No active terminal outages within 14 days.</p></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # LNG VESSELS VIEW
    # ========================================================================
    elif data_source == "Milford Haven LNG":
        st.markdown('<div class="section-header">LNG Vessels - Milford Haven Port</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="info-box">
            <strong>LNG Vessel Tracking</strong> — Shows confirmed LNG tankers arriving at Milford Haven 
            (South Hook & Dragon terminals).
        </div>
        ''', unsafe_allow_html=True)
        
        # Use preloaded basic vessel data, then enrich with details if needed
        vessels_df = preloaded_data['lng']
        
        if vessels_df is not None and len(vessels_df) > 0:
            # Filter for LNG vessels and get details (this part still needs to fetch VesselFinder details)
            ship_type_col = None
            for col in vessels_df.columns:
                if 'type' in col.lower() or 'ship type' in col.lower():
                    ship_type_col = col
                    break
            
            if ship_type_col:
                lng_df = vessels_df[
                    vessels_df[ship_type_col].str.lower().str.contains('lng', na=False)
                ].copy()
                
                if len(lng_df) > 0:
                    # Fetch vessel details (cached per vessel)
                    ship_col = None
                    for col in vessels_df.columns:
                        if col.lower() in ['ship', 'vessel', 'name', 'vessel name', 'ship name']:
                            ship_col = col
                            break
                    if ship_col is None:
                        ship_col = vessels_df.columns[0]
                    
                    with st.spinner("Fetching vessel details from VesselFinder..."):
                        unique_ships = lng_df[ship_col].unique()
                        vessel_details = []
                        for ship in unique_ships:
                            details = get_vessel_info(ship)
                            vessel_details.append(details)
                        
                        details_df = pd.DataFrame(vessel_details)
                        lng_df = lng_df.merge(details_df, left_on=ship_col, right_on='Ship', how='left')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("LNG Vessels Expected", len(lng_df))
                    with col2:
                        found_count = len(lng_df[lng_df.get('Status', '') == 'Found']) if 'Status' in lng_df.columns else 0
                        st.metric("Vessel Details Found", found_count)
                    with col3:
                        unique_flags = lng_df['Flag'].dropna().nunique() if 'Flag' in lng_df.columns else 0
                        st.metric("Unique Flags", unique_flags)
                    
                    st.markdown("---")
                    st.markdown("#### LNG Vessel Arrivals")
                    render_lng_vessel_table(lng_df)
                else:
                    st.markdown('''
                    <div class="no-data">
                        <h3>No LNG Vessels Found</h3>
                        <p>No LNG tankers are currently scheduled.</p>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="no-data">
                    <h3>No LNG Vessels Found</h3>
                    <p>Could not identify ship type column.</p>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="no-data">
                <h3>No LNG Vessels Found</h3>
                <p>No LNG tankers are currently scheduled.</p>
            </div>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

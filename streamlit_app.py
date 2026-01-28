# update: 28/01/2026

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from urllib.parse import quote
import time

# Page configuration
st.set_page_config(
    page_title="UK Gas Market Dashboard",
    page_icon="🛠️",
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
        color: #00d4ff !important;
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
    
    .vessel-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #0097a9;
    }
    
    .vessel-card h4 {
        color: #1e293b;
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
    }
    
    .vessel-detail {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .vessel-detail:last-child {
        border-bottom: none;
    }
    
    .vessel-detail .label {
        color: #64748b;
        font-weight: 500;
    }
    
    .vessel-detail .value {
        color: #1e293b;
        font-weight: 600;
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
    
    .status-confirmed {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-expected {
        background-color: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .data-source-badge {
        display: inline-block;
        background: #e2e8f0;
        color: #475569;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LNG VESSEL TRACKING FUNCTIONS
# ============================================================================

def get_vessel_info(ship_name: str) -> dict:
    """
    Fetch vessel details from VesselFinder website.
    
    Args:
        ship_name: Name of the ship to look up
        
    Returns:
        Dictionary containing vessel information
    """
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
            """Extract value from page using various structures."""
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
        
    except requests.exceptions.Timeout:
        default_result['Status'] = 'Timeout'
        return default_result
    except requests.exceptions.RequestException as e:
        default_result['Status'] = f'Request Error: {str(e)}'
        return default_result
    except Exception as e:
        default_result['Status'] = f'Error: {str(e)}'
        return default_result


@st.cache_data(ttl=300)
def get_milford_haven_vessels() -> pd.DataFrame:
    """
    Scrape the Milford Haven Port Authority website for arriving vessels.
    
    Returns:
        DataFrame containing vessel arrival information
    """
    url = "https://www.mhpa.co.uk/live-information/vessels-arriving/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            st.error(f"Failed to fetch MHPA data: HTTP {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', class_='timetable-table')
        if not table:
            table = soup.find('table')
        
        if not table:
            st.error("Could not find vessel table on MHPA website")
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
        st.error(f"Error fetching MHPA data: {str(e)}")
        return None


@st.cache_data(ttl=300)
def get_lng_vessels_with_details() -> pd.DataFrame:
    """
    Get LNG vessels arriving at Milford Haven with additional details from VesselFinder.
    
    Returns:
        DataFrame containing LNG vessel information with additional details
    """
    vessels_df = get_milford_haven_vessels()
    
    if vessels_df is None or len(vessels_df) == 0:
        return None
    
    # Identify the ship type column
    ship_type_col = None
    for col in vessels_df.columns:
        if 'type' in col.lower() or 'ship type' in col.lower():
            ship_type_col = col
            break
    
    if ship_type_col is None:
        st.warning("Could not identify ship type column. Showing all vessels.")
        return vessels_df
    
    # Filter for LNG tankers
    lng_df = vessels_df[
        vessels_df[ship_type_col].str.lower().str.contains('lng', na=False)
    ].copy()
    
    if len(lng_df) == 0:
        return None
    
    # Identify the ship name column
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
    """
    Render the LNG vessel table with styling.
    
    Args:
        df: DataFrame containing vessel information
    """
    if df is None or len(df) == 0:
        st.markdown('''
        <div class="no-data">
            <h3>No LNG Vessels Found</h3>
            <p>No LNG tankers are currently scheduled to arrive at Milford Haven.</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # Identify ship name column
    ship_col = None
    for col in df.columns:
        if col.lower() in ['ship', 'vessel', 'name', 'vessel name', 'ship name']:
            ship_col = col
            break
    if not ship_col:
        ship_col = df.columns[0]
    
    # Identify "To" / destination column explicitly
    to_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'to':
            to_col = col
            break
    
    # If no exact "To" column, check for destination/berth/terminal
    if to_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['destination', 'berth', 'terminal']):
                to_col = col
                break
    
    # Build display columns - ship name first
    display_cols = [ship_col]
    
    # Add "To" column early if found
    if to_col and to_col not in display_cols:
        display_cols.append(to_col)
    
    # Add schedule/origin columns
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['date', 'time', 'eta', 'arrival', 'from', 'agent']):
            if col not in display_cols:
                display_cols.append(col)
    
    # Add vessel details columns
    detail_cols = ['IMO', 'Flag', 'Deadweight', 'GrossTonnage']
    for col in detail_cols:
        if col in df.columns and col not in display_cols:
            display_cols.append(col)
    
    # Filter to existing columns only
    display_cols = [col for col in display_cols if col in df.columns]
    
    display_df = df[display_cols].copy()
    
    # Rename columns for better display
    rename_map = {
        'GrossTonnage': 'Gross Tonnage',
    }
    # Rename "To" column to "Destination" for clarity if it exists
    if to_col and to_col.lower() == 'to':
        rename_map[to_col] = 'Destination'
    
    display_df = display_df.rename(columns=rename_map)
    
    # Build column config dynamically
    column_config = {
        ship_col: st.column_config.TextColumn("Vessel Name", width="medium"),
        "IMO": st.column_config.TextColumn("IMO Number", width="small"),
        "Flag": st.column_config.TextColumn("Flag", width="small"),
        "Deadweight": st.column_config.TextColumn("DWT", width="small"),
        "Gross Tonnage": st.column_config.TextColumn("GT", width="small"),
        "Destination": st.column_config.TextColumn("Destination", width="medium"),
    }
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )


def render_lng_vessel_cards(df: pd.DataFrame):
    """
    Render LNG vessel information as cards using Streamlit native components.
    
    Args:
        df: DataFrame containing vessel information
    """
    if df is None or len(df) == 0:
        return
    
    # Identify the ship name column
    ship_col = None
    for col in df.columns:
        if col.lower() in ['ship', 'vessel', 'name', 'vessel name', 'ship name']:
            ship_col = col
            break
    if not ship_col:
        ship_col = df.columns[0]
    
    # Identify "To" / destination column
    to_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'to':
            to_col = col
            break
    if to_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['destination', 'berth', 'terminal']):
                to_col = col
                break
    
    for idx, row in df.iterrows():
        ship_name = row[ship_col]
        
        with st.container(border=True):
            st.subheader(ship_name)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pd.notna(row.get('IMO')) and row.get('IMO'):
                    st.markdown(f"**IMO:** {row.get('IMO')}")
                if pd.notna(row.get('MMSI')) and row.get('MMSI'):
                    st.markdown(f"**MMSI:** {row.get('MMSI')}")
                if pd.notna(row.get('Flag')) and row.get('Flag'):
                    st.markdown(f"**Flag:** {row.get('Flag')}")
                if pd.notna(row.get('Deadweight')) and row.get('Deadweight'):
                    st.markdown(f"**Deadweight:** {row.get('Deadweight')}")
                if pd.notna(row.get('GrossTonnage')) and row.get('GrossTonnage'):
                    st.markdown(f"**Gross Tonnage:** {row.get('GrossTonnage')}")
            
            with col2:
                # Show destination (To column)
                if to_col and pd.notna(row.get(to_col)) and row.get(to_col):
                    st.markdown(f"**Destination:** {row.get(to_col)}")
                
                # Add arrival/schedule info from other columns
                for col in df.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['date', 'time', 'eta', 'arrival']):
                        if col != ship_col and pd.notna(row.get(col)) and row.get(col):
                            st.markdown(f"**{col}:** {row.get(col)}")
                    elif 'from' in col_lower:
                        if pd.notna(row.get(col)) and row.get(col):
                            st.markdown(f"**From:** {row.get(col)}")
            
            vf_url = row.get('VesselFinderURL')
            if pd.notna(vf_url) and vf_url:
                st.markdown(f"[View on VesselFinder]({vf_url})")


# ============================================================================
# EXISTING FUNCTIONS (GASSCO AND NATIONAL GAS)
# ============================================================================

@st.cache_data(ttl=120)
def scrape_gassco_data():
    """Scrape GASSCO REMIT data for field and terminal outages."""
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
    """Parse GASSCO table data into DataFrame."""
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
    """Process REMIT data - filter active events within 14 days."""
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
    """
    Fetch gas data from National Gas API with retry logic.
    
    Args:
        request_type: The type of data to request (e.g., 'demandCategoryGraph', 'supplyCategoryGraph')
        max_retries: Number of retry attempts if request fails
        
    Returns:
        DataFrame containing gas data, or None if all attempts fail
    """
    url = "https://data.nationalgas.com/api/gas-system-status-graph"
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, 
                json={"request": request_type}, 
                headers=headers, 
                timeout=30  # Increased timeout for reliability
            )
            response.raise_for_status()  # Raise exception for bad status codes
            return pd.DataFrame(response.json()["data"])
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            st.error(f"⚠️ National Gas API timeout after {max_retries} attempts. The API may be slow or unavailable.")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            st.error(f"⚠️ Error fetching gas data: {str(e)}")
            return None
        except (KeyError, ValueError) as e:
            st.error(f"⚠️ Error parsing gas data response: {str(e)}")
            return None
    
    return None


def get_chart_layout(title="", height=500):
    """Get standardised Plotly chart layout."""
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
    """Create timeline plot for GASSCO outages."""
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
    """Create cumulative unavailability plot for GASSCO outages."""
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
    """Create flow rate chart with average and current markers."""
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
    """Render metric cards in columns."""
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
    """Render the nomination summary table."""
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
    """Render GASSCO outages table."""
    display_df = df.copy()
    for col in ['Publication date/time', 'Event Start', 'Event Stop']:
        if col in display_df.columns:
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M')
    
    cols = ['Affected Asset or Unit', 'Type of Unavailability', 'Event Start', 'Event Stop', 'Unavailable Capacity', 'Duration', 'Reason for the unavailability']
    cols = [c for c in cols if c in display_df.columns]
    
    st.dataframe(display_df[cols], use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    with st.sidebar:
        st.markdown('<div style="text-align:center;padding:1rem 0;"><h1 style="font-size:1.6rem;margin:0;color:#0097a9 !important;"> UK Gas Market</h1><p style="font-size:0.85rem;opacity:0.8;margin-top:0.5rem;color:#0097a9 !important;">Real-time Dashboard</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Data Source")
        data_source = st.radio(
            "Source", 
            ["National Gas", "GASSCO", "Milford Haven LNG"], 
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
        
        elif data_source == "Milford Haven LNG":
            st.markdown("### View Options")
            lng_view = st.radio(
                "Display", 
                ["Table View", "Card View"], 
                label_visibility="collapsed", 
                key="lngv"
            )
            st.markdown("---")
            st.markdown('<div class="info-box" style="background:#1a1a2e;border-left-color:#0097a9;"><small style="color:#0097a9;">Data sourced from Milford Haven Port Authority with vessel details from VesselFinder.</small></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown(f'<div style="text-align:center;padding:1rem 0;font-size:0.8rem;color:#718096;">Last updated:<br>{datetime.now().strftime("%H:%M:%S %d/%m/%Y")}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-header"><h1>UK Gas Market Dashboard</h1><p>Real-time monitoring of UK gas supply, demand, and infrastructure status</p></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # NATIONAL GAS VIEW
    # ========================================================================
    if data_source == "National Gas":
        demand_df = get_gas_data("demandCategoryGraph")
        supply_df = get_gas_data("supplyCategoryGraph")
        
        if demand_df is not None and supply_df is not None:
            if 'Storage' in demand_df.columns:
                demand_df.rename(columns={'Storage': 'Storage Injection'}, inplace=True)
            
            n = len(demand_df)
            today = datetime.now().date()
            start = datetime.combine(today, datetime.min.time().replace(hour=5))
            ts = [start + timedelta(minutes=2*i) for i in range(n)]
            
            demand_df['Timestamp'] = ts
            demand_df = demand_df.sort_values('Timestamp').reset_index(drop=True)
            demand_df['next_time'] = demand_df['Timestamp'].shift(-1).fillna(demand_df['Timestamp'].iloc[-1] + timedelta(minutes=2))
            demand_df['interval_seconds'] = (demand_df['next_time'] - demand_df['Timestamp']).dt.total_seconds()
            
            n_s = len(supply_df)
            ts_s = [start + timedelta(minutes=2*i) for i in range(n_s)]
            supply_df['Timestamp'] = ts_s
            supply_df = supply_df.sort_values('Timestamp').reset_index(drop=True)
            supply_df['next_time'] = supply_df['Timestamp'].shift(-1).fillna(supply_df['Timestamp'].iloc[-1] + timedelta(minutes=2))
            supply_df['interval_seconds'] = (supply_df['next_time'] - supply_df['Timestamp']).dt.total_seconds()
            
            if ng_view == "Table":
                st.markdown('<div class="section-header"> UK Gas Flows - Supply, Demand & Balance</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box"><strong>Flow Table</strong> shows the current gas day flows. All values in mcm.</div>', unsafe_allow_html=True)
                bal = render_nomination_table(demand_df, supply_df)
            
            elif ng_view == "Supply":
                st.markdown(f'<div class="section-header"> Supply - {supply_cat}</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="section-header"> Demand - {demand_cat}</div>', unsafe_allow_html=True)
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
            st.error("⚠️ Unable to fetch National Gas data. Please try refreshing.")
    
    # ========================================================================
    # GASSCO VIEW
    # ========================================================================
    elif data_source == "GASSCO":
        st.markdown(f'<div class="section-header"> GASSCO - {gassco_view}</div>', unsafe_allow_html=True)
        
        with st.spinner("Fetching GASSCO data..."):
            fields_df, terminal_df = scrape_gassco_data()
        
        fields_proc = process_remit_data(fields_df)
        terminal_proc = process_remit_data(terminal_df)
        
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
        st.markdown('<div class="section-header"> LNG Vessels - Milford Haven Port</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="info-box">
            <strong>LNG Vessel Tracking</strong> — Shows confirmed LNG tankers arriving at Milford Haven 
            (South Hook & Dragon terminals) with vessel details from VesselFinder.
        </div>
        ''', unsafe_allow_html=True)
        
        with st.spinner("Fetching LNG vessel data..."):
            lng_df = get_lng_vessels_with_details()
        
        if lng_df is not None and len(lng_df) > 0:
            # Summary metrics
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
            
            if lng_view == "Table View":
                st.markdown("#### LNG Vessel Arrivals")
                render_lng_vessel_table(lng_df)
                
                csv = lng_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"lng_vessels_milford_haven_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            elif lng_view == "Card View":
                st.markdown("#### LNG Vessel Details")
                render_lng_vessel_cards(lng_df)
        
        else:
            st.markdown('''
            <div class="no-data">
                <h3>No LNG Vessels Found</h3>
                <p>No LNG tankers are currently scheduled to arrive at Milford Haven.</p>
                <p style="font-size: 0.85rem; margin-top: 1rem;">This may be due to no scheduled arrivals or a temporary issue fetching data from MHPA.</p>
            </div>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

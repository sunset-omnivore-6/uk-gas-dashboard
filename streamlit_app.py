# update: 27/01/2026

"""
UK Gas Market Dashboard

Requirements:
    pip install streamlit plotly pandas numpy requests beautifulsoup4 lxml
"""
#import the relevant files 
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="UK Gas Market Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# cascading style sheet 
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


#functions to get the data
@st.cache_data(ttl=120)
def scrape_gassco_data():
    try:
        session = requests.Session()
        session.get("https://umm.gassco.no/", timeout=10)
        session.get("https://umm.gassco.no/disclaimer/acceptDisclaimer", timeout=10) # uses this to get around the disclaimer on gassco page 
        response = session.get("https://umm.gassco.no/", timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        msg_tables = soup.find_all('table', class_='msgTable')
        
        fields_df = parse_gassco_table(msg_tables[0]) if len(msg_tables) > 0 else None # checks here if tables exist, this makes sure its pulled in the data correctly 
        terminal_df = parse_gassco_table(msg_tables[1]) if len(msg_tables) > 1 else None
        
        return fields_df, terminal_df
    except:
        return None, None

#create function pull out the desired information from the gassco tables
def parse_gassco_table(table):
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

#process the tables by changing the time formats (chr -> time) and setting a 2 week cutoff for REMIT changes 
def process_remit_data(df):
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
    
    cutoff = datetime.now(df['Event Start'].dt.tz) + timedelta(days=14) # Change here for a longer/shorter view window 
    df = df[(df['Event Start'] <= cutoff) | (df['Event Stop'] <= cutoff)]
    
    if len(df) == 0:
        return None
    
    df = df.drop_duplicates()
    df['Duration'] = (df['Event Stop'] - df['Event Start']).dt.total_seconds() / (24 * 3600)
    df['midpoint'] = df['Event Start'] + (df['Event Stop'] - df['Event Start']) / 2 #define midpoint for the plot annotation
    
    return df.sort_values('Unavailable Capacity')

#set up scrapper to get national gas data, runs every 2 mins 
@st.cache_data(ttl=120) #change time here for faster/slower intervals, limited by the national gas publishing
def get_gas_data(request_type):
    try:
        url = "https://data.nationalgas.com/api/gas-system-status-graph"
        response = requests.post(url, json={"request": request_type}, 
                                headers={"Content-Type": "application/json"}, timeout=10)
        return pd.DataFrame(response.json()["data"])
    except Exception as e:
        st.error(f"Error fetching gas data: {str(e)}")
        return None

# same as above but for nominations - FIXED VERSION
@st.cache_data(ttl=120)
def get_nominations(date_str, category_ids):
    base_url = "https://data.nationalgas.com/api/find-gas-data-download?applicableFor=Y&dateFrom="
    conversion = 11111111.11
    nominations = []
    
    for ids in category_ids:
        try:
            url = f"{base_url}{date_str}&dateTo={date_str}&dateType=GASDAY&latestFlag=Y&ids={ids}&type=CSV"
            df = pd.read_csv(url)
            if len(df) > 0 and 'Value' in df.columns:
                nominations.append(round(df['Value'].sum() / conversion, 2))
            else:
                nominations.append(0)
        except Exception as e:
            print(f"Error fetching nomination for {ids}: {str(e)}")
            nominations.append(0)
    
    return nominations



#functions for making the chart - FIXED VERSION WITH BETTER COLORS
def get_chart_layout(title="", height=500):
    return dict(
        title=dict(text=title, font=dict(size=18, color='#1e293b')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1e293b', size=12),
        hovermode='x unified',
        height=height,
        margin=dict(l=60, r=60, t=100, b=60),
        xaxis=dict(
            gridcolor='#e2e8f0', 
            linecolor='#1e293b',  # Changed to dark color for visibility
            tickfont=dict(color='#1e293b'),  # Changed to dark color
            title_font=dict(color='#1e293b')
        ),
        yaxis=dict(
            gridcolor='#e2e8f0', 
            linecolor='#1e293b',  # Changed to dark color for visibility
            tickfont=dict(color='#1e293b'),  # Changed to dark color
            title_font=dict(color='#1e293b')
        )
    )


def create_gassco_timeline_plot(df, title_prefix):
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
    cols = st.columns(len(metrics))
    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value:.2f} {unit}</div>
            </div>
            """, unsafe_allow_html=True)


# FIXED NOMINATION TABLE FUNCTION
def render_nomination_table(demand_df, supply_df):
    today = datetime.now().strftime("%Y-%m-%d")
    
    demand_ids = ["PUBOBJ1156,PUBOBJ1160,PUBOBJ1157", "PUBOBJ1153", "PUBOBJ1154", "PUBOBJ1155", "PUBOBJ1597", "PUBOBJ1094", "PUBOBJ1093"]
    supply_ids = ["PUBOBJ1149", "PUBOBJ1158,PUBOBJ1150", "PUBOBJ1106", "PUBOBJ1126", "PUBOBJ1147"]
    
    # Fetch nominations with error handling
    try:
        demand_noms = get_nominations(today, demand_ids)
        supply_noms = get_nominations(today, supply_ids)
    except Exception as e:
        st.warning(f"Could not fetch nominations: {str(e)}")
        demand_noms = [0] * len(demand_ids)
        supply_noms = [0] * len(supply_ids)
    
    demand_cols = ["LDZ Offtake", "Power Station", "Industrial", "Storage Injection", "Bacton BBL Export", "Bacton INT Export", "Moffat Export"]
    supply_cols = ["Storage Withdrawal", "LNG", "Bacton BBL Import", "Bacton INT Import", "Beach (UKCS/Norway)"]
    
    def summarise(df, cols, noms):
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
            results.append({"Category": col, "Avg": round(avg, 2), "Comp": round(comp, 2), "Inst": round(inst, 2), "Nom": noms[i] if i < len(noms) else 0})
        return pd.DataFrame(results)
    
    demand_sum = summarise(demand_df, demand_cols, demand_noms)
    supply_sum = summarise(supply_df, supply_cols, supply_noms)
    
    d_tot = demand_sum[["Avg", "Comp", "Inst", "Nom"]].sum()
    s_tot = supply_sum[["Avg", "Comp", "Inst", "Nom"]].sum()
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
        rows.append(f'<tr class="demand"><td>{r["Category"]}</td><td style="text-align:right;">{r["Avg"]:.1f}</td><td style="text-align:right;">{r["Comp"]:.1f}</td><td style="text-align:right;">{r["Inst"]:.1f}</td><td style="text-align:right;">{r["Nom"]:.1f}</td></tr>')
    rows.append(f'<tr class="demand-total"><td><strong>DEMAND TOTAL</strong></td><td style="text-align:right;"><strong>{d_tot["Avg"]:.1f}</strong></td><td style="text-align:right;"><strong>{d_tot["Comp"]:.1f}</strong></td><td style="text-align:right;"><strong>{d_tot["Inst"]:.1f}</strong></td><td style="text-align:right;"><strong>{d_tot["Nom"]:.1f}</strong></td></tr>')
    
    for _, r in supply_sum.iterrows():
        rows.append(f'<tr class="supply"><td>{r["Category"]}</td><td style="text-align:right;">{r["Avg"]:.1f}</td><td style="text-align:right;">{r["Comp"]:.1f}</td><td style="text-align:right;">{r["Inst"]:.1f}</td><td style="text-align:right;">{r["Nom"]:.1f}</td></tr>')
    rows.append(f'<tr class="supply-total"><td><strong>SUPPLY TOTAL</strong></td><td style="text-align:right;"><strong>{s_tot["Avg"]:.1f}</strong></td><td style="text-align:right;"><strong>{s_tot["Comp"]:.1f}</strong></td><td style="text-align:right;"><strong>{s_tot["Inst"]:.1f}</strong></td><td style="text-align:right;"><strong>{s_tot["Nom"]:.1f}</strong></td></tr>')
    rows.append(f'<tr class="balance"><td><strong>BALANCE</strong></td><td style="text-align:right;"><strong>{bal["Avg"]:.1f}</strong></td><td style="text-align:right;"><strong>{bal["Comp"]:.1f}</strong></td><td style="text-align:right;"><strong>{bal["Inst"]:.1f}</strong></td><td style="text-align:right;"><strong>{bal["Nom"]:.1f}</strong></td></tr>')
    
    st.markdown(f"""
    <table class="nomination-table">
        <thead><tr><th>Category</th><th style="text-align:right;">Avg Rate (mcm)</th><th style="text-align:right;">Completed (mcm)</th><th style="text-align:right;">Instant (mcm)</th><th style="text-align:right;">Nominated (mcm)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """, unsafe_allow_html=True)
    
    return bal


def render_gassco_table(df):
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
        st.markdown('<div style="text-align:center;padding:1rem 0;"><h1 style="font-size:1.6rem;margin:0;color:#00d4ff !important;">UK Gas Market</h1><p style="font-size:0.85rem;opacity:0.8;margin-top:0.5rem;color:#a0aec0 !important;">Real-time Dashboard</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Data Source")
        data_source = st.radio("Source", ["National Gas", "GASSCO"], label_visibility="collapsed", key="ds")
        
        st.markdown("---")
        
        if data_source == "National Gas":
            st.markdown("### Views")
            ng_view = st.radio("View", ["Nomination", "Supply", "Demand"], label_visibility="collapsed", key="ngv")
            
            if ng_view == "Supply":
                st.markdown("---")
                st.markdown("##### Supply Categories")
                supply_cat = st.radio("Cat", ["LNG", "Storage Withdrawal", "Beach Terminal", "IC Import"], label_visibility="collapsed", key="sc")
            elif ng_view == "Demand":
                st.markdown("---")
                st.markdown("##### Demand Categories")
                demand_cat = st.radio("Cat", ["CCGT", "Storage Injection", "LDZ", "Industrial", "IC Export"], label_visibility="collapsed", key="dc")
        else:
            st.markdown("### Views")
            gassco_view = st.radio("View", ["Field Outages", "Terminal Outages"], label_visibility="collapsed", key="gv")
        
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown(f'<div style="text-align:center;padding:1rem 0;font-size:0.8rem;color:#718096;">Last updated:<br>{datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-header"><h1>UK Gas Market Dashboard</h1><p>Real-time monitoring of UK gas supply, demand, and infrastructure status</p></div>', unsafe_allow_html=True)
    
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
            
            if ng_view == "Nomination":
                st.markdown('<div class="section-header">📋 UK Gas Flows - Supply, Demand & Balance</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box"><strong>Nomination Table</strong> shows the current gas day flows. All values in mcm.</div>', unsafe_allow_html=True)
                bal = render_nomination_table(demand_df, supply_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Balance", f"{bal['Inst']:.1f} mcm", delta="Surplus" if bal['Inst'] >= 0 else "Deficit")
                with col2:
                    st.metric("Gas Day Progress", f"{((datetime.now() - start).total_seconds() / 86400 * 100):.0f}%")
                with col3:
                    st.metric("Data Points", str(n))
            
            elif ng_view == "Supply":
                st.markdown(f'<div class="section-header">📈 Supply - {supply_cat}</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="section-header">📉 Demand - {demand_cat}</div>', unsafe_allow_html=True)
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
    
    else:
        st.markdown(f'<div class="section-header"> GASSCO - {gassco_view}</div>', unsafe_allow_html=True)
        
        with st.spinner("Fetching GASSCO data..."):
            fields_df, terminal_df = scrape_gassco_data()
        
        fields_proc = process_remit_data(fields_df)
        terminal_proc = process_remit_data(terminal_df)
        
        if gassco_view == "Field Outages":
            if fields_proc is not None and len(fields_proc) > 0:
                st.markdown(f'<div class="info-box"><strong>{len(fields_proc)} active field outage(s)</strong> within 14 days.</div>', unsafe_allow_html=True)
                st.plotly_chart(create_gassco_timeline_plot(fields_proc, "Field"), use_container_width=True)
                st.plotly_chart(create_gassco_cumulative_plot(fields_proc, "Field"), use_container_width=True)
                st.markdown("#### Outages Details")
                render_gassco_table(fields_proc)
            else:
                st.markdown('<div class="no-data"><h3>✅ No Field Outages</h3><p>No active field outages within 14 days.</p></div>', unsafe_allow_html=True)
        else:
            if terminal_proc is not None and len(terminal_proc) > 0:
                st.markdown(f'<div class="info-box"><strong>{len(terminal_proc)} active terminal outage(s)</strong> within 14 days.</div>', unsafe_allow_html=True)
                st.plotly_chart(create_gassco_timeline_plot(terminal_proc, "Terminal"), use_container_width=True)
                st.plotly_chart(create_gassco_cumulative_plot(terminal_proc, "Terminal"), use_container_width=True)
                st.markdown("#### 📋 Outages Details")
                render_gassco_table(terminal_proc)
            else:
                st.markdown('<div class="no-data"><h3>✅ No Terminal Outages</h3><p>No active terminal outages within 14 days.</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# ============================================================================
# UK GAS MARKET DASHBOARD - STREAMLIT VERSION
# ============================================================================
#
# DEPLOYMENT INSTRUCTIONS:
# ------------------------
# 1. Create a GitHub repository and upload these files:
#    - streamlit_app.py (this file)
#    - requirements.txt
#
# 2. Go to https://share.streamlit.io
#    - Sign in with GitHub
#    - Click "New app"
#    - Select your repository
#    - Click "Deploy"
#
# 3. Share the URL with your team (e.g., https://uk-gas-dashboard.streamlit.app)
#
# That's it! Users just click the link - no downloads, no setup required.
# When you update the code on GitHub, the app updates automatically.
#
# ============================================================================

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="UK Gas Market Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHING - Data is cached for 5 minutes to avoid hitting APIs too frequently
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def scrape_gassco_data():
    """Scrape REMIT data from GASSCO website"""
    try:
        session = requests.Session()
        session.get("https://umm.gassco.no/", timeout=10)
        session.get("https://umm.gassco.no/disclaimer/acceptDisclaimer", timeout=10)
        response = session.get("https://umm.gassco.no/", timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        msg_tables = soup.find_all('table', class_='msgTable')
        
        fields_df = None
        terminal_df = None
        
        if len(msg_tables) > 0:
            fields_df = parse_gassco_table(msg_tables[0])
        if len(msg_tables) > 1:
            terminal_df = parse_gassco_table(msg_tables[1])
        
        return fields_df, terminal_df
    except Exception as e:
        st.error(f"Error fetching GASSCO data: {e}")
        return None, None

def parse_gassco_table(table):
    """Parse a GASSCO table into a DataFrame"""
    rows = table.find_all('tr', id=True)
    data = []
    
    for row in rows:
        row_id = row.get('id')
        cells = row.find_all('td')
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        if len(cell_texts) >= 19:
            data.append({
                'Message ID': cell_texts[0],
                'Affected Asset or Unit': cell_texts[1],
                'Event Status': cell_texts[2],
                'Type of Unavailability': cell_texts[3],
                'Type of Event': cell_texts[4],
                'Publication date/time': cell_texts[5],
                'Event Start': cell_texts[6],
                'Event Stop': cell_texts[7],
                'Unit of Measurement': cell_texts[8],
                'Technical Capacity': cell_texts[9],
                'Available Capacity': cell_texts[10],
                'Unavailable Capacity': cell_texts[11],
                'Reason for the unavailability': cell_texts[12],
                'Remarks': cell_texts[13],
                'Balancing Zone': cell_texts[14],
                'Market Participant': cell_texts[15],
                'Market Participant Code': cell_texts[16],
                'Affected Asset or Unit EIC Code': cell_texts[17],
                'Direction': cell_texts[18],
                'url ID': row_id
            })
    
    return pd.DataFrame(data) if data else None

def process_remit_data(df):
    """Process REMIT data - filter, convert dates, calculate metrics"""
    if df is None or len(df) == 0:
        return None
    
    df = df[df['Event Status'] == 'Active'].copy()
    if len(df) == 0:
        return None
    
    df['Publication date/time'] = pd.to_datetime(df['Publication date/time'], format='ISO8601', utc=True)
    df['Event Start'] = pd.to_datetime(df['Event Start'], format='ISO8601', utc=True)
    df['Event Stop'] = pd.to_datetime(df['Event Stop'], format='ISO8601', utc=True)
    
    df['Technical Capacity'] = pd.to_numeric(df['Technical Capacity'], errors='coerce')
    df['Available Capacity'] = pd.to_numeric(df['Available Capacity'], errors='coerce')
    df['Unavailable Capacity'] = pd.to_numeric(df['Unavailable Capacity'], errors='coerce')
    
    cutoff = datetime.now(df['Event Start'].dt.tz) + timedelta(days=14)
    df = df[(df['Event Start'] <= cutoff) | (df['Event Stop'] <= cutoff)]
    
    if len(df) == 0:
        return None
    
    df = df.drop_duplicates()
    df = df.sort_values(['Affected Asset or Unit', 'Publication date/time'], ascending=[True, False])
    
    df['% Unavailable'] = (df['Unavailable Capacity'] / df['Technical Capacity']) * 100
    df['Duration'] = (df['Event Stop'] - df['Event Start']).dt.total_seconds() / (24 * 3600)
    df['midpoint'] = df['Event Start'] + (df['Event Stop'] - df['Event Start']) / 2
    
    columns_to_keep = [
        'Affected Asset or Unit', 'Type of Unavailability', 'Type of Event',
        'Publication date/time', 'Event Start', 'Event Stop',
        'Technical Capacity', 'Available Capacity', 'Unavailable Capacity',
        'Reason for the unavailability', 'Remarks',
        '% Unavailable', 'Duration', 'midpoint'
    ]
    df = df[columns_to_keep]
    df = df.sort_values('Unavailable Capacity')
    
    return df

@st.cache_data(ttl=300)
def get_gas_data(request_type):
    """Fetch gas data from National Gas API"""
    try:
        url = "https://data.nationalgas.com/api/gas-system-status-graph"
        body = {"request": request_type}
        response = requests.post(url, json=body, headers={
            "Content-Type": "application/json",
            "Accept": "application/json"
        }, timeout=10)
        data_json = response.json()
        df = pd.DataFrame(data_json["data"])
        return df
    except Exception as e:
        st.error(f"Error fetching National Gas data: {e}")
        return None

@st.cache_data(ttl=300)
def get_nominations(date_str, category_ids):
    """Fetch nomination data from National Gas API"""
    base_url = "https://data.nationalgas.com/api/find-gas-data-download?applicableFor=Y&dateFrom="
    conversion = 11111111.11
    nominations = []
    for ids in category_ids:
        url = base_url + date_str + "&dateTo=" + date_str + "&dateType=GASDAY&latestFlag=Y&ids=" + ids + "&type=CSV"
        try:
            df = pd.read_csv(url)
            if len(df) > 0:
                value = df['Value'].sum() / conversion
            else:
                value = 0
        except:
            value = 0
        nominations.append(round(value, 2))
    return nominations

# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================

def create_gassco_timeline_plot(df, title_prefix):
    """Create interactive timeline plot for GASSCO data"""
    outage_colors = {'Planned': '#f8b4b4', 'Unplanned': '#7fcdcd'}
    fig = go.Figure()
    shown_legends = set()
    
    for idx, row in df.iterrows():
        color = outage_colors.get(row['Type of Unavailability'], '#cccccc')
        show_legend = row['Type of Unavailability'] not in shown_legends
        
        if show_legend:
            shown_legends.add(row['Type of Unavailability'])
        
        fig.add_trace(go.Scatter(
            x=[row['Event Start'], row['Event Stop']],
            y=[row['Affected Asset or Unit'], row['Affected Asset or Unit']],
            mode='lines',
            line=dict(color=color, width=20),
            name=row['Type of Unavailability'],
            legendgroup=row['Type of Unavailability'],
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{row['Affected Asset or Unit']}</b><br>"
                f"Type: {row['Type of Unavailability']}<br>"
                f"Start: {row['Event Start'].strftime('%d %b %Y %H:%M')}<br>"
                f"Stop: {row['Event Stop'].strftime('%d %b %Y %H:%M')}<br>"
                f"Unavailable: {row['Unavailable Capacity']:.1f} MSm³/d<br>"
                f"Duration: {row['Duration']:.1f} days<extra></extra>"
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=[row['Event Start']], y=[row['Affected Asset or Unit']],
            mode='markers', marker=dict(color=color, size=8, symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[row['Event Stop']], y=[row['Affected Asset or Unit']],
            mode='markers', marker=dict(color=color, size=8, symbol='triangle-right'),
            showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_annotation(
            x=row['midpoint'], y=row['Affected Asset or Unit'],
            text=f"{row['Unavailable Capacity']:.1f} MSm³/d",
            showarrow=False, font=dict(size=12, color='black'),
            yshift=24, xanchor='center'
        )
    
    today = datetime.now(df['Event Start'].dt.tz)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix} Outages Timeline</b><br><sub>Active outages within 14 days | Today: {datetime.now().strftime('%d %b %Y')}</sub>",
            font=dict(size=16)
        ),
        xaxis=dict(title="", type='date', tickformat='%d %b', dtick=3*24*60*60*1000),
        yaxis=dict(title=title_prefix, categoryorder='array', 
                   categoryarray=df['Affected Asset or Unit'].tolist()),
        hovermode='closest',
        shapes=[dict(type='line', x0=today, x1=today, y0=0, y1=1, yref='paper',
                    line=dict(color='gray', width=2, dash='dash'))],
        legend=dict(title=dict(text='Outage Type'), orientation='v', x=1.02, y=1),
        height=max(400, len(df) * 50), plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=150, r=100, t=100, b=50)
    )
    
    fig.add_annotation(
        x=today, y=1.02, yref='paper', text='Today', showarrow=False,
        font=dict(size=14, color='#2c3e50'), bgcolor='white',
        bordercolor='#cccccc', borderwidth=1, borderpad=4
    )
    
    return fig

def create_gassco_cumulative_plot(df, title_prefix):
    """Create cumulative unavailable capacity plot"""
    events = []
    for _, row in df.iterrows():
        events.append({'time': row['Event Start'], 'delta_capacity': -row['Unavailable Capacity']})
        events.append({'time': row['Event Stop'], 'delta_capacity': row['Unavailable Capacity']})
    
    events_df = pd.DataFrame(events)
    events_df = events_df.groupby('time')['delta_capacity'].sum().reset_index()
    events_df = events_df.sort_values('time')
    events_df['cumulative_unavailable'] = events_df['delta_capacity'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=events_df['time'], y=events_df['cumulative_unavailable'],
        mode='lines+markers',
        line=dict(shape='hv', color='#c0392b', width=2),
        marker=dict(size=6, color='#c0392b'),
        hovertemplate="<b>Time:</b> %{x|%d %b %Y %H:%M}<br><b>Cumulative Unavailable:</b> %{y:.1f} MSm³/d<br><extra></extra>"
    ))
    
    today = datetime.now(df['Event Start'].dt.tz)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix} Cumulative Outages</b>",
            font=dict(size=16)
        ),
        xaxis=dict(title="", type='date', tickformat='%d %b', dtick=3*24*60*60*1000),
        yaxis=dict(title='Unavailable Capacity (MSm³/d)', zeroline=True,
                  zerolinecolor='gray', zerolinewidth=1),
        hovermode='closest',
        shapes=[dict(type='line', x0=today, x1=today, y0=0, y1=1, yref='paper',
                    line=dict(color='#2c3e50', width=2, dash='dash'))],
        height=400, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=80, r=50, t=80, b=50)
    )
    
    return fig

def create_flow_chart(df, column_name, chart_title, color='#1f77b4'):
    """Create a flow chart figure"""
    flow_averages = np.average(df[column_name], weights=df['interval_seconds'])
    
    today = datetime.now().date()
    gas_day_start = datetime.combine(today, datetime.min.time().replace(hour=5, minute=0, second=0))
    gas_day_end = gas_day_start + timedelta(days=1)
    now = datetime.now()
    
    elapsed_seconds = (now - gas_day_start).total_seconds()
    total_gas_day_seconds = (gas_day_end - gas_day_start).total_seconds()
    elapsed_pct = max(0, min(1, elapsed_seconds / total_gas_day_seconds))
    
    total_flow_so_far = flow_averages * elapsed_pct
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df[column_name],
        mode='lines', 
        name=chart_title,
        line=dict(color=color, width=2),
        hovertemplate='<b>Time:</b> %{x|%H:%M}<br><b>Flow:</b> %{y:.2f} mcm<extra></extra>'
    ))
    
    fig.add_hline(
        y=flow_averages, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text=f"Avg: {flow_averages:.2f} mcm", 
        annotation_position="right",
        annotation=dict(font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.8)")
    )
    
    now_ms = int(now.timestamp() * 1000)
    
    fig.add_vline(
        x=now_ms, 
        line_color="black", 
        line_width=3,
        annotation_text=f"Now: {total_flow_so_far:.2f} mcm", 
        annotation_position="top",
        annotation=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.8)")
    )
    
    y_max = df[column_name].max()
    if y_max == 0:
        y_max = 1
    
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=16)),
        xaxis=dict(title="", range=[gas_day_start, gas_day_end], tickformat='%H:%M', dtick=7200000),
        yaxis=dict(title="Flow Rate (mcm)", range=[0, y_max * 1.2]),
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified', showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60), height=400
    )
    
    current_flow = df[column_name].iloc[-1]
    
    return fig, flow_averages, total_flow_so_far, current_flow

def create_nomination_summary(demand_df, supply_df):
    """Create nomination summary dataframe"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    demand_ids = [
        "PUBOBJ1156,PUBOBJ1160,PUBOBJ1157",
        "PUBOBJ1153",
        "PUBOBJ1154",
        "PUBOBJ1155",
        "PUBOBJ1597",
        "PUBOBJ1094",
        "PUBOBJ1093"
    ]
    
    supply_ids = [
        "PUBOBJ1149",
        "PUBOBJ1158,PUBOBJ1150",
        "PUBOBJ1106",
        "PUBOBJ1126",
        "PUBOBJ1147"
    ]
    
    demand_nominations = get_nominations(today, demand_ids)
    supply_nominations = get_nominations(today, supply_ids)
    
    def summarise_category(df, category_columns, nominations):
        results = []
        n_obs = len(df)
        pct_day = (n_obs * 2) / 1440
        
        for i, col in enumerate(category_columns):
            if col in df.columns:
                avg_5am_to_now = df[col].mean()
                current_eod = avg_5am_to_now * pct_day
                instant = df[col].iloc[-1]
                
                results.append({
                    "Category": col,
                    "Avg Rate (mcm)": round(avg_5am_to_now, 1),
                    "Completed (mcm)": round(current_eod, 1),
                    "Current (mcm)": round(instant, 1),
                    "Nominated (mcm)": nominations[i] if i < len(nominations) else 0
                })
        
        return pd.DataFrame(results)
    
    demand_columns = ["LDZ Offtake", "Power Station", "Industrial", "Storage Injection",
                      "Bacton BBL Export", "Bacton INT Export", "Moffat Export"]
    supply_columns = ["Storage Withdrawal", "LNG", "Bacton BBL Import", 
                      "Bacton INT Import", "Beach (UKCS/Norway)"]
    
    demand_summary = summarise_category(demand_df, demand_columns, demand_nominations)
    supply_summary = summarise_category(supply_df, supply_columns, supply_nominations)
    
    return demand_summary, supply_summary

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("⛽ UK Gas Market Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data refreshes every 5 minutes")
    
    # Refresh button in sidebar
    with st.sidebar:
        st.header("🔄 Controls")
        if st.button("🔄 Refresh Data Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.header("📊 Navigation")
        page = st.radio(
            "Select View:",
            ["📈 National Gas - Nominations", 
             "📊 National Gas - Supply",
             "📉 National Gas - Demand",
             "🔧 GASSCO - Field Outages",
             "🏭 GASSCO - Terminal Outages"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.caption("Data sources: National Gas, GASSCO")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
        if auto_refresh:
            st.caption("Page will refresh automatically")
            import time
            time.sleep(300)  # 5 minutes
            st.rerun()
    
    # Load data with progress indicator
    with st.spinner("Loading data..."):
        # National Gas data
        demand_df = get_gas_data("demandCategoryGraph")
        supply_df = get_gas_data("supplyCategoryGraph")
        
        if demand_df is not None:
            if 'Storage' in demand_df.columns:
                demand_df = demand_df.rename(columns={'Storage': 'Storage Injection'})
            
            n = len(demand_df)
            today = datetime.now().date()
            start_time = datetime.combine(today, datetime.min.time().replace(hour=5, minute=0, second=0))
            timestamps = [start_time + timedelta(minutes=2*i) for i in range(n)]
            demand_df['Timestamp'] = timestamps
            demand_df = demand_df.sort_values('Timestamp').reset_index(drop=True)
            demand_df['next_time'] = demand_df['Timestamp'].shift(-1)
            demand_df.loc[demand_df.index[-1], 'next_time'] = demand_df['Timestamp'].iloc[-1] + timedelta(minutes=2)
            demand_df['interval_seconds'] = (demand_df['next_time'] - demand_df['Timestamp']).dt.total_seconds()
        
        if supply_df is not None:
            n = len(supply_df)
            timestamps = [start_time + timedelta(minutes=2*i) for i in range(n)]
            supply_df['Timestamp'] = timestamps
            supply_df = supply_df.sort_values('Timestamp').reset_index(drop=True)
            supply_df['next_time'] = supply_df['Timestamp'].shift(-1)
            supply_df.loc[supply_df.index[-1], 'next_time'] = supply_df['Timestamp'].iloc[-1] + timedelta(minutes=2)
            supply_df['interval_seconds'] = (supply_df['next_time'] - supply_df['Timestamp']).dt.total_seconds()
        
        # GASSCO data
        fields_df, terminal_df = scrape_gassco_data()
        fields_processed = process_remit_data(fields_df)
        terminal_processed = process_remit_data(terminal_df)
    
    # Display content based on selection
    if page == "📈 National Gas - Nominations":
        st.header("UK Gas Flows - Supply, Demand & Balance")
        
        if demand_df is not None and supply_df is not None:
            demand_summary, supply_summary = create_nomination_summary(demand_df, supply_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Demand")
                st.dataframe(
                    demand_summary.style.format({
                        'Avg Rate (mcm)': '{:.1f}',
                        'Completed (mcm)': '{:.1f}',
                        'Current (mcm)': '{:.1f}',
                        'Nominated (mcm)': '{:.1f}'
                    }).background_gradient(cmap='Oranges', subset=['Avg Rate (mcm)']),
                    use_container_width=True,
                    hide_index=True
                )
                demand_total = demand_summary['Avg Rate (mcm)'].sum()
                st.metric("Total Demand Rate", f"{demand_total:.1f} mcm")
            
            with col2:
                st.subheader("📥 Supply")
                st.dataframe(
                    supply_summary.style.format({
                        'Avg Rate (mcm)': '{:.1f}',
                        'Completed (mcm)': '{:.1f}',
                        'Current (mcm)': '{:.1f}',
                        'Nominated (mcm)': '{:.1f}'
                    }).background_gradient(cmap='Blues', subset=['Avg Rate (mcm)']),
                    use_container_width=True,
                    hide_index=True
                )
                supply_total = supply_summary['Avg Rate (mcm)'].sum()
                st.metric("Total Supply Rate", f"{supply_total:.1f} mcm")
            
            # Balance
            st.divider()
            balance = supply_total - demand_total
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Supply", f"{supply_total:.1f} mcm")
            col2.metric("Total Demand", f"{demand_total:.1f} mcm")
            col3.metric("Balance", f"{balance:.1f} mcm", delta=f"{balance:.1f} mcm")
        else:
            st.error("Unable to load National Gas data")
    
    elif page == "📊 National Gas - Supply":
        st.header("Supply Categories")
        
        if supply_df is not None:
            tab1, tab2, tab3, tab4 = st.tabs(["LNG", "Storage Withdrawal", "Beach Terminal", "Interconnector Import"])
            
            with tab1:
                fig, avg, total, current = create_flow_chart(supply_df, 'LNG', 'LNG Import Flow', '#1f77b4')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig, avg, total, current = create_flow_chart(supply_df, 'Storage Withdrawal', 'Storage Withdrawal Flow', '#2ca02c')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig, avg, total, current = create_flow_chart(supply_df, 'Beach (UKCS/Norway)', 'Beach Terminal Supply', '#9467bd')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                supply_df['Total IC Import'] = supply_df['Bacton BBL Import'] + supply_df['Bacton INT Import']
                fig, avg, total, current = create_flow_chart(supply_df, 'Total IC Import', 'Interconnector Import', '#d62728')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to load supply data")
    
    elif page == "📉 National Gas - Demand":
        st.header("Demand Categories")
        
        if demand_df is not None:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["CCGT", "Storage Injection", "LDZ", "Industrial", "IC Export"])
            
            with tab1:
                fig, avg, total, current = create_flow_chart(demand_df, 'Power Station', 'CCGT Power Station', '#ff7f0e')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig, avg, total, current = create_flow_chart(demand_df, 'Storage Injection', 'Storage Injection', '#2ca02c')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig, avg, total, current = create_flow_chart(demand_df, 'LDZ Offtake', 'LDZ Offtake', '#1f77b4')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                fig, avg, total, current = create_flow_chart(demand_df, 'Industrial', 'Industrial Demand', '#9467bd')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                demand_df['Total IC Export'] = demand_df['Bacton BBL Export'] + demand_df['Bacton INT Export'] + demand_df['Moffat Export']
                fig, avg, total, current = create_flow_chart(demand_df, 'Total IC Export', 'Interconnector Export', '#d62728')
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Rate", f"{avg:.2f} mcm")
                col2.metric("Total So Far", f"{total:.2f} mcm")
                col3.metric("Current Flow", f"{current:.2f} mcm")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to load demand data")
    
    elif page == "🔧 GASSCO - Field Outages":
        st.header("GASSCO Field Outages")
        st.caption("Active outages within the next 14 days")
        
        if fields_processed is not None and len(fields_processed) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Active Outages", len(fields_processed))
            col2.metric("Total Unavailable", f"{fields_processed['Unavailable Capacity'].sum():.1f} MSm³/d")
            planned = len(fields_processed[fields_processed['Type of Unavailability'] == 'Planned'])
            col3.metric("Planned / Unplanned", f"{planned} / {len(fields_processed) - planned}")
            
            # Timeline chart
            st.plotly_chart(create_gassco_timeline_plot(fields_processed, "Field"), use_container_width=True)
            
            # Cumulative chart
            st.plotly_chart(create_gassco_cumulative_plot(fields_processed, "Field"), use_container_width=True)
            
            # Data table
            st.subheader("Outage Details")
            display_df = fields_processed.copy()
            display_df['Event Start'] = display_df['Event Start'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Event Stop'] = display_df['Event Stop'].dt.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.drop(columns=['midpoint', 'Publication date/time'], errors='ignore')
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("✅ No active field outages in the next 14 days")
    
    elif page == "🏭 GASSCO - Terminal Outages":
        st.header("GASSCO Terminal Outages")
        st.caption("Active outages within the next 14 days")
        
        if terminal_processed is not None and len(terminal_processed) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Active Outages", len(terminal_processed))
            col2.metric("Total Unavailable", f"{terminal_processed['Unavailable Capacity'].sum():.1f} MSm³/d")
            planned = len(terminal_processed[terminal_processed['Type of Unavailability'] == 'Planned'])
            col3.metric("Planned / Unplanned", f"{planned} / {len(terminal_processed) - planned}")
            
            # Timeline chart
            st.plotly_chart(create_gassco_timeline_plot(terminal_processed, "Terminal"), use_container_width=True)
            
            # Cumulative chart
            st.plotly_chart(create_gassco_cumulative_plot(terminal_processed, "Terminal"), use_container_width=True)
            
            # Data table
            st.subheader("Outage Details")
            display_df = terminal_processed.copy()
            display_df['Event Start'] = display_df['Event Start'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Event Stop'] = display_df['Event Stop'].dt.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.drop(columns=['midpoint', 'Publication date/time'], errors='ignore')
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("✅ No active terminal outages in the next 14 days")

if __name__ == "__main__":
    main()

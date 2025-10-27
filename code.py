import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Hardcoded universes (expandable)
SMALL_CAP_TICKERS = ['AVAH', 'STRA', 'ARLO', 'LFST', 'ASGN', 'ROCK', 'VOYG', 'DQ', 'SYBT', 'PRGS',
                     'DX', 'EVTC', 'DV', 'WLY', 'RCUS', 'SDRL', 'PACS', 'BELFB', 'SLDE', 'CLOV',
                     'NTCT', 'LQDA', 'DXPE', 'PLUS', 'NWN', 'VERA', 'ANIP', 'FIHL', 'ADNT', 'TRIP',
                     'SCS', 'CXM', 'ADEA']
MID_CAP_TICKERS = ['CMA', 'IPG', 'PEN', 'ORI', 'AIT', 'KNSL', 'OTEX', 'ALGN', 'RRX', 'IDCC',
                   'CRL', 'OVV', 'SARO', 'AOS', 'DCI', 'CUBE', 'FRHC', 'PSO', 'MKSI', 'SPXC',
                   'EDU', 'BWA', 'MOS', 'AUR', 'LSCC', 'DDS', 'FIGR', 'EGP', 'FYBR', 'MDGL',
                   'ESTC', 'UWMC', 'RGEN']
LARGE_CAP_TICKERS = ['UBER', 'ANET', 'NOW', 'LRCX', 'PDD', 'ISRG', 'INTU', 'BX', 'ARM', 'INTC',
                     'AMAT', 'T', 'C', 'BLK', 'HDB', 'NEE', 'SONY', 'SCHW', 'BKNG', 'MUFG',
                     'BA', 'APH', 'VZ', 'KLAC', 'TJX', 'GEV', 'AMGN', 'ACN', 'DHR', 'UL',
                     'TXN', 'SPGI', 'BSX']

@st.cache_data(ttl=300)  # Cache for 5 min to avoid API spam
def fetch_stock_data(ticker, period='6mo'):
    try:
        data = yf.download(ticker, period=period, progress=False)  # Suppress progress bar
        info = yf.Ticker(ticker).info
        return data, info
    except Exception as e:
        st.warning(f"Fetch failed for {ticker}: {str(e)[:50]}...")  # Log lightly
        return pd.DataFrame(), {}

def calculate_rsi(data, window=14):
    if len(data) < window:
        return 50  # Neutral fallback
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(data, fast=12, slow=26, signal=9):
    if len(data) < slow:
        return 0
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd.iloc[-1] - signal_line.iloc[-1]

def get_profit_potential(data, days=5, target_type='short'):
    if len(data) < days:
        return 0
    current = data['Close'].iloc[-1]
    if target_type == 'short':
        high_recent = data['High'].tail(days).max()
        return ((high_recent - current) / current * 100)
    elif target_type == 'mid':
        macd_score = calculate_macd(data)
        vol = data['Close'].pct_change().std() * np.sqrt(14)  # 2-week annualized vol
        return (macd_score / vol * 10) if vol > 0 else 0  # Scaled to %
    else:  # long
        if len(data) < 50:
            return 0
        roc_50 = ((data['Close'].iloc[-1] / data['Close'].iloc[-50]) - 1) * 100
        return roc_50 * 3  # Simple 3-month projection

def get_additional_metric(ticker, metric='volume_surge'):
    data, info = fetch_stock_data(ticker, '1mo')
    if data.empty:
        return 0
    if metric == 'volume_surge':
        avg_vol = data['Volume'].mean()
        current_vol = data['Volume'].iloc[-1]
        return (current_vol / avg_vol) if avg_vol > 0 else 0
    elif metric == 'rsi':
        return calculate_rsi(data)
    elif metric == 'beta':
        return info.get('beta', 1.0)

# Streamlit App
st.set_page_config(page_title="99 Stocks Dashboard", layout="wide")
st.title("99 Stocks Dashboard: Segmented Opportunities by Market Cap & Horizon")

# Sidebars for sliders
with st.sidebar:
    st.header("Profit Target Adjustments")
    short_target = st.slider("Short-Term (Table 1: 1-3 days)", 0.0, 100.0, 5.0)
    mid_target = st.slider("Mid-Term (Table 2: 1-2 weeks)", 0.0, 100.0, 10.0)
    long_target = st.slider("Long-Term (Table 3: 1-3 months)", 0.0, 100.0, 30.0)
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Selected stock for chart
all_tickers = SMALL_CAP_TICKERS + MID_CAP_TICKERS + LARGE_CAP_TICKERS
selected_ticker = st.sidebar.selectbox("View Candlestick Chart", [""] + all_tickers)

col1, col2, col3 = st.columns(3)

# Function to build and display table
@st.cache_data(ttl=300)
def build_table_df(tickers, cap_type, target):
    df_list = []
    for ticker in tickers:
        data, info = fetch_stock_data(ticker)
        if data.empty:
            continue
        potential = get_profit_potential(data, target_type=cap_type.lower())
        if potential < target:
            continue
        score = potential
        add_metric = get_additional_metric(ticker, 'volume_surge' if cap_type == 'Small' else 'rsi' if cap_type == 'Mid' else 'beta')
        df_list.append({
            'Ticker': ticker,
            'Name': info.get('longName', ticker),
            'Price': round(data['Close'].iloc[-1], 2),
            'Potential %': round(potential, 2),
            'Score': round(score, 2),
            'Add. Metric': round(add_metric, 2)
        })
    if df_list:
        return pd.DataFrame(df_list).sort_values('Score', ascending=False).head(33)
    return pd.DataFrame()

def display_table(tickers, cap_type, target, col):
    with col:
        st.subheader(f"{cap_type} Cap Stocks (Target: {target}%)")
        df = build_table_df(tickers, cap_type, target)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No stocks meet the target—lower the slider or check data fetch.")

# Build tables
display_table(SMALL_CAP_TICKERS, 'Small', short_target, col1)
display_table(MID_CAP_TICKERS, 'Mid', mid_target, col2)
display_table(LARGE_CAP_TICKERS, 'Large', long_target, col3)

# Candlestick Chart
if selected_ticker:
    st.header(f"Candlestick Chart: {selected_ticker}")
    data, _ = fetch_stock_data(selected_ticker, '6mo')
    if not data.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=('Price', 'Volume'), row_width=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'], name='Candlestick'),
                      row=1, col=1)
        colors = ['green' if data['Close'][i] >= data['Open'][i] else 'red' for i in range(len(data))]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                             marker_color='rgba(0,0,0,0.3)'), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Heatmap of All 99
st.header("Profit Potential Heatmap (All 99 Stocks)")
heatmap_data = []
for ticker in all_tickers:
    data, _ = fetch_stock_data(ticker)
    if not data.empty:
        potential = get_profit_potential(data, target_type='short')  # Uniform short-term for viz
        heatmap_data.append({'Ticker': ticker, 'Potential %': potential})

if heatmap_data:
    df_heat = pd.DataFrame(heatmap_data).head(99)
    if not df_heat.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        pivot = df_heat.set_index('Ticker')['Potential %'].to_frame().T
        sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=0, fmt='.1f', ax=ax, cbar_kws={'label': 'Potential %'})
        plt.title('Profit Potential Across All Stocks')
        st.pyplot(fig)
    else:
        st.warning("Insufficient data for heatmap.")
else:
    st.warning("Data fetch issue—check API or increase cache TTL.")

st.caption("Data via yfinance | As of " + datetime.now().strftime("%Y-%m-%d %H:%M") + " | Prototype—enhance with caching/alerts.")

"""Enhanced Streamlit Dashboard for Crypto Market Monitoring."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import io
import base64

# Configuration
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Crypto Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main header gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(102,126,234,0.1);
        border-radius: 10px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Download buttons */
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255,255,255,0.02);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============== Helper Functions ==============

def get_csv_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate CSV download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">📥 Download CSV</a>'


def get_chart_download_button(fig: go.Figure, filename: str, key: str):
    """Add download button for chart as PNG."""
    # Convert to PNG bytes
    img_bytes = fig.to_image(format="png", width=1920, height=1080, scale=2)
    
    st.download_button(
        label="📸 Download Chart (PNG)",
        data=img_bytes,
        file_name=filename,
        mime="image/png",
        key=key
    )


# ============== API Functions ==============

@st.cache_data(ttl=60)
def fetch_assets() -> List[Dict]:
    """Fetch available assets from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/assets", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch assets: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_prices(asset: str, from_date: str = None, to_date: str = None) -> Optional[Dict]:
    """Fetch price history for an asset."""
    try:
        params = {"asset": asset}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        response = requests.get(f"{API_BASE_URL}/prices", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch prices: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_comparison(assets: str, normalize: bool = False, from_date: str = None) -> Optional[Dict]:
    """Fetch comparison data for multiple assets."""
    try:
        params = {"assets": assets, "normalize": str(normalize).lower()}
        if from_date:
            params["from"] = from_date
        
        response = requests.get(f"{API_BASE_URL}/compare", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch comparison: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_anomalies(asset: str, threshold: float = 2.5) -> List[Dict]:
    """Fetch anomalies for an asset."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/anomalies",
            params={"asset": asset, "threshold": threshold},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch anomalies: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_volatility(asset: str, window: int = 20) -> List[Dict]:
    """Fetch volatility data for an asset."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/volatility",
            params={"asset": asset, "window": window},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch volatility: {e}")
        return []


# ============== Enhanced Chart Functions ==============

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI and MACD indicators."""
    # RSI
    delta = df['price_usd'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['price_usd'].ewm(span=12, adjust=False).mean()
    exp2 = df['price_usd'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    return df


def create_comprehensive_price_chart(prices: List[Dict], asset_name: str, anomalies: List[Dict] = None) -> go.Figure:
    """Create comprehensive price chart with multiple indicators."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate indicators
    df['MA7'] = df['price_usd'].rolling(window=7, min_periods=1).mean()
    df['MA20'] = df['price_usd'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['price_usd'].rolling(window=50, min_periods=1).mean()
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['price_usd'].rolling(window=20, min_periods=1).mean()
    df['BB_std'] = df['price_usd'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    fig = go.Figure()
    
    # Bollinger Bands fill
    fig.add_trace(go.Scatter(
        x=pd.concat([df['timestamp'], df['timestamp'][::-1]]),
        y=pd.concat([df['BB_upper'], df['BB_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(102,126,234,0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Bollinger Bands',
        showlegend=True,
    ))
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price_usd'],
        mode='lines',
        name='Price (USD)',
        line=dict(color='#667eea', width=3),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>'
    ))
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['MA7'],
        mode='lines', name='MA7',
        line=dict(color='#f59e0b', width=1.5, dash='dot'),
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['MA20'],
        mode='lines', name='MA20',
        line=dict(color='#22c55e', width=1.5, dash='dot'),
    ))
    
    if len(df) > 50:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['MA50'],
            mode='lines', name='MA50',
            line=dict(color='#ef4444', width=1.5, dash='dot'),
        ))
    
    # Add anomaly markers
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        if not anomaly_df.empty:
            anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'])
            
            spikes = anomaly_df[anomaly_df['anomaly_type'] == 'spike']
            drops = anomaly_df[anomaly_df['anomaly_type'] == 'drop']
            
            if not spikes.empty:
                fig.add_trace(go.Scatter(
                    x=spikes['timestamp'], y=spikes['price'],
                    mode='markers', name='🔺 Spike',
                    marker=dict(color='#22c55e', size=15, symbol='triangle-up',
                               line=dict(color='white', width=2)),
                ))
            
            if not drops.empty:
                fig.add_trace(go.Scatter(
                    x=drops['timestamp'], y=drops['price'],
                    mode='markers', name='🔻 Drop',
                    marker=dict(color='#ef4444', size=15, symbol='triangle-down',
                               line=dict(color='white', width=2)),
                ))
    
    fig.update_layout(
        title=dict(
            text=f"📈 {asset_name} Price Analysis",
            font=dict(size=24, color='white'),
            x=0.5
        ),
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        yaxis=dict(tickformat="$,.0f"),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


def create_animated_bubble_chart(prices: List[Dict]) -> go.Figure:
    """Create animated bubble chart showing Price vs Volume over time."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Calculate daily percent change for color
    df['pct_change'] = df['price_usd'].pct_change() * 100
    df['pct_change'] = df['pct_change'].fillna(0)
    
    fig = px.scatter(
        df,
        x="price_usd",
        y="volume_24h",
        animation_frame="date_str",
        animation_group="symbol" if 'symbol' in df.columns else None,
        size="market_cap" if 'market_cap' in df.columns and not df['market_cap'].isna().all() else None,
        color="pct_change",
        range_x=[df['price_usd'].min()*0.9, df['price_usd'].max()*1.1],
        range_y=[df['volume_24h'].min()*0.9 if 'volume_24h' in df else 0, df['volume_24h'].max()*1.1 if 'volume_24h' in df else 100],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title="✨ Dynamic Market Motion (Price vs Volume)",
        template="plotly_dark",
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Price (USD)",
        yaxis_title="24h Volume",
        coloraxis_colorbar_title="Change %"
    )
    
    return fig


def create_gauge_chart(value: float, title: str, min_val: float, max_val: float, thresholds: List[float] = None) -> go.Figure:
    """Create a gauge chart (speedometer style)."""
    
    steps = []
    if thresholds:
        steps = [
            {'range': [min_val, thresholds[0]], 'color': "green"},
            {'range': [thresholds[0], thresholds[1]], 'color': "gray"},
            {'range': [thresholds[1], max_val], 'color': "red"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "white"},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    
    return fig


def create_correlation_heatmap(comparison_data: Dict) -> go.Figure:
    """Create correlation matrix heatmap from comparison data."""
    if not comparison_data or not comparison_data.get('data'):
        return go.Figure()
        
    # Restructure data into DataFrame
    data_points = []
    for point in comparison_data['data']:
        row = {'timestamp': point['timestamp']}
        row.update(point['prices'])
        data_points.append(row)
    
    df = pd.DataFrame(data_points)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Calculate correlation
    corr = df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="🔗 Asset Correlation Matrix",
        template="plotly_dark",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_forecast_chart(prices: List[Dict], asset_name: str, days_forward: int = 7) -> go.Figure:
    """Create a simple linear forecast chart."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_numeric'] = df['timestamp'].astype(int) / 10**9
    
    # Simple Linear Regression
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Use last 100 points for trend to make it relevant
    lookback = min(len(df), 100) 
    df_subset = df.iloc[-lookback:]
    
    X = df_subset['timestamp_numeric'].values.reshape(-1, 1)
    y = df_subset['price_usd'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates
    last_date = df['timestamp'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_forward + 1)]
    future_X = np.array([d.timestamp() for d in future_dates]).reshape(-1, 1)
    
    future_pred = model.predict(future_X)
    
    # Create figure with historical data
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price_usd'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#667eea', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred,
        mode='lines+markers',
        name='Linear Forecast',
        line=dict(color='#22c55e', width=2, dash='dash'),
        marker=dict(symbol='star')
    ))
    
    # Confidence Interval (Simplified cone)
    std_err = np.std(y - model.predict(X))
    upper_bound = future_pred + (std_err * 2)
    lower_bound = future_pred - (std_err * 2)
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(34, 197, 94, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence'
    ))
    
    fig.update_layout(
        title=f"🔮 {asset_name} Price Forecast ({days_forward} days)",
        template="plotly_dark",
        height=500,
        yaxis_title="Price (USD)"
    )
    
    return fig


def create_technical_indicators_chart(prices: List[Dict]) -> go.Figure:
    """Create chart for RSI and MACD."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5],
        subplot_titles=("RSI (14)", "MACD (12, 26, 9)")
    )
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['RSI'],
        mode='lines', name='RSI',
        line=dict(color='#a855f7', width=2),
    ), row=1, col=1)
    
    # RSI Bounds
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['MACD'],
        mode='lines', name='MACD',
        line=dict(color='#00e5ff', width=1.5),
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['Signal_Line'],
        mode='lines', name='Signal',
        line=dict(color='#ffea00', width=1.5),
    ), row=2, col=1)
    
    # MACD Histogram
    colors = ['#22c55e' if val >= 0 else '#ef4444' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['MACD_Hist'],
        name='Histogram',
        marker_color=colors,
    ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_candlestick_style_chart(prices: List[Dict], asset_name: str) -> go.Figure:
    """Create OHLC-style visualization from price data."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by minute intervals to create OHLC-like data if granular enough
    # If not granular, use hourly
    if len(df) > 1000:
        freq = '15min'
    elif len(df) > 200:
        freq = '5min'
    else:
        freq = '1min' # Won't aggregate much if ingestion is 5min
        
    df = df.set_index('timestamp')
    ohlc = df['price_usd'].resample(freq).agg(['first', 'max', 'min', 'last']).dropna()
    ohlc.columns = ['open', 'high', 'low', 'close']
    ohlc = ohlc.reset_index()
    
    if ohlc.empty:
         # Fallback for sparse data
        ohlc = df.reset_index()[['timestamp', 'price_usd']].copy()
        ohlc['open'] = ohlc['price_usd']
        ohlc['high'] = ohlc['price_usd']
        ohlc['low'] = ohlc['price_usd']
        ohlc['close'] = ohlc['price_usd']
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=ohlc['timestamp'],
        open=ohlc['open'],
        high=ohlc['high'],
        low=ohlc['low'],
        close=ohlc['close'],
        name='Price',
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444',
    ))
    
    fig.update_layout(
        title=dict(
            text=f"🕯️ {asset_name} Candle View",
            font=dict(size=24, color='white'),
            x=0.5
        ),
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=True,
        yaxis=dict(tickformat="$,.0f"),
    )
    
    return fig


def create_volume_market_cap_chart(prices: List[Dict], asset_name: str) -> go.Figure:
    """Create dual-axis volume and market cap chart."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna(subset=['volume_24h'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"📊 24h Trading Volume", f"💰 Market Cap"),
        row_heights=[0.5, 0.5]
    )
    
    # Volume bars with gradient colors
    colors = ['rgba(102,126,234,0.8)' if i % 2 == 0 else 'rgba(118,75,162,0.8)' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume_24h'],
        name='Volume',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Volume: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Market cap area chart
    if 'market_cap' in df.columns and not df['market_cap'].isna().all():
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['market_cap'],
            mode='lines',
            name='Market Cap',
            line=dict(color='#f093fb', width=2),
            fill='tozeroy',
            fillcolor='rgba(240,147,251,0.2)',
            hovertemplate='<b>%{x}</b><br>Market Cap: $%{y:,.0f}<extra></extra>'
        ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )
    
    fig.update_yaxes(tickformat="$,.0f")
    
    return fig


def create_volatility_heatmap(prices: List[Dict], asset_name: str) -> go.Figure:
    """Create volatility visualization with multiple metrics."""
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate various volatility metrics
    df['returns'] = df['price_usd'].pct_change() * 100
    df['volatility_5'] = df['returns'].rolling(window=5, min_periods=1).std()
    df['volatility_10'] = df['returns'].rolling(window=10, min_periods=1).std()
    df['volatility_20'] = df['returns'].rolling(window=20, min_periods=1).std()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("📉 Price Returns (%)", "🌡️ Rolling Volatility"),
        row_heights=[0.4, 0.6]
    )
    
    # Returns bar chart
    colors = ['#22c55e' if r >= 0 else '#ef4444' for r in df['returns'].fillna(0)]
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['returns'],
        name='Returns',
        marker_color=colors,
    ), row=1, col=1)
    
    # Volatility lines
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['volatility_5'],
        mode='lines', name='Vol (5)',
        line=dict(color='#f59e0b', width=2),
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['volatility_10'],
        mode='lines', name='Vol (10)',
        line=dict(color='#22c55e', width=2),
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['volatility_20'],
        mode='lines', name='Vol (20)',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102,126,234,0.2)',
    ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )
    
    return fig


def create_comparison_radar(assets_data: Dict) -> go.Figure:
    """Create radar chart comparing asset metrics."""
    if not assets_data or not assets_data.get('data'):
        return go.Figure()
    
    # Get latest data for each asset
    latest_prices = {}
    for point in assets_data['data'][-1:]:
        latest_prices = point.get('prices', {})
    
    if not latest_prices:
        return go.Figure()
    
    assets = list(latest_prices.keys())
    values = list(latest_prices.values())
    
    # Normalize values for radar chart
    max_val = max(values) if values else 1
    normalized = [v / max_val * 100 for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]],  # Close the polygon
        theta=assets + [assets[0]],
        fill='toself',
        fillcolor='rgba(102,126,234,0.3)',
        line=dict(color='#667eea', width=2),
        name='Relative Price'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)'
        ),
        template="plotly_dark",
        height=500,
        title=dict(
            text="🎯 Asset Comparison Radar",
            font=dict(size=20, color='white'),
            x=0.5
        ),
    )
    
    return fig


def create_multi_asset_area(comparison_data: Dict, normalized: bool) -> go.Figure:
    """Create stacked area chart for multiple assets."""
    if not comparison_data or not comparison_data.get('data'):
        return go.Figure()
    
    data = comparison_data['data']
    assets = comparison_data.get('assets', [])
    
    fig = go.Figure()
    
    colors = ['#667eea', '#22c55e', '#f59e0b', '#ef4444', '#a855f7', '#ec4899']
    
    for i, asset in enumerate(assets):
        timestamps = []
        prices = []
        
        for point in data:
            if asset in point['prices']:
                timestamps.append(point['timestamp'])
                prices.append(point['prices'][asset])
        
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(timestamps),
            y=prices,
            mode='lines',
            name=asset,
            line=dict(color=colors[i % len(colors)], width=2),
            stackgroup='one' if not normalized else None,
            fill='tonexty' if normalized else None,
        ))
    
    ylabel = "% Change" if normalized else "Price (USD)"
    title = "📊 Normalized Performance Comparison" if normalized else "📊 Price Comparison"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color='white'), x=0.5),
        xaxis_title="Time",
        yaxis_title=ylabel,
        template="plotly_dark",
        height=600,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    
    return fig


def create_price_distribution(prices: List[Dict], asset_name: str) -> go.Figure:
    """Create price distribution histogram."""
    df = pd.DataFrame(prices)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📊 Price Distribution", "📈 Returns Distribution"),
        horizontal_spacing=0.1
    )
    
    # Price histogram
    fig.add_trace(go.Histogram(
        x=df['price_usd'],
        nbinsx=30,
        name='Price',
        marker_color='rgba(102,126,234,0.7)',
        hovertemplate='Price: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)
    
    # Returns histogram
    returns = df['price_usd'].pct_change() * 100
    returns = returns.dropna()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        name='Returns',
        marker_color='rgba(118,75,162,0.7)',
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ), row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
    )
    
    return fig


# ============== Main Dashboard ==============

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Live Crypto Market Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time cryptocurrency analytics with advanced visualizations</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Dashboard Controls")
    
    # Fetch available assets
    assets = fetch_assets()
    
    if not assets:
        st.warning("⚠️ No assets available. Make sure the API server is running.")
        st.code("python -m app.main", language="bash")
        return
    
    asset_options = {f"{a['symbol']} - {a['name']}": a['symbol'] for a in assets}
    
    # Asset selector
    st.sidebar.markdown("### 🪙 Select Asset")
    selected_display = st.sidebar.selectbox(
        "Primary Asset",
        options=list(asset_options.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_asset = asset_options[selected_display]
    
    # Date range
    st.sidebar.markdown("### 📅 Time Range")
    date_range = st.sidebar.selectbox(
        "Quick Select",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
        index=0,
        label_visibility="collapsed"
    )
    
    from_date = None
    if date_range == "Last 24 Hours":
        from_date = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    elif date_range == "Last 7 Days":
        from_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
    elif date_range == "Last 30 Days":
        from_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
    
    # Anomaly settings
    st.sidebar.markdown("### 🎯 Anomaly Detection")
    anomaly_threshold = st.sidebar.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)
    
    # Comparison settings
    st.sidebar.markdown("### 📊 Comparison")
    compare_assets = st.sidebar.multiselect(
        "Compare With",
        options=[a['symbol'] for a in assets if a['symbol'] != selected_asset],
        default=[]
    )
    normalize_comparison = st.sidebar.checkbox("Normalize Prices", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Data refreshes automatically. Use the download buttons to export charts/data.")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Forecast", 
        "📉 Technical Level", 
        "✨ Market Dynamics", 
        "🔍 Advanced Comparison",
        "⚠️ Anomalies & Data"
    ])
    
    # Fetch data
    price_data = fetch_prices(selected_asset, from_date=from_date)
    anomalies = fetch_anomalies(selected_asset, anomaly_threshold) if show_anomalies else []
    
    # ============== TAB 1: Price & Forecast ==============
    with tab1:
        if price_data and price_data.get('prices'):
            prices = price_data['prices']
            asset_info = price_data['asset']
            
            # Top metrics row
            st.markdown("### 📊 Key Metrics")
            if prices:
                latest = prices[-1]
                high = max(p['price_usd'] for p in prices)
                low = min(p['price_usd'] for p in prices)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "💵 Current Price",
                        f"${latest['price_usd']:,.2f}",
                        f"{latest.get('price_change_24h', 0):.2f}%" if latest.get('price_change_24h') else None
                    )
                
                with col2:
                    st.metric("💰 Market Cap", f"${latest.get('market_cap', 0)/1e9:.2f}B" if latest.get('market_cap') else "N/A")
                
                with col3:
                    st.metric("📈 24h Volume", f"${latest.get('volume_24h', 0)/1e6:.2f}M" if latest.get('volume_24h') else "N/A")
                
                with col4:
                    st.metric("🔼 High", f"${high:,.2f}")
                
                with col5:
                    st.metric("🔽 Low", f"${low:,.2f}")
            
            st.markdown("---")
            
            # Main price chart
            st.markdown("### 📈 Comprehensive Price Analysis")
            fig_price = create_comprehensive_price_chart(prices, asset_info['name'], anomalies)
            st.plotly_chart(fig_price, use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                get_chart_download_button(fig_price, f"{selected_asset}_price_chart.png", "price_png")
            
            st.markdown("---")
            
            # Forecast & Candlestick Split
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔮 Price Forecast (7 Days)")
                try:
                    fig_forecast = create_forecast_chart(prices, asset_info['name'], days_forward=7)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    get_chart_download_button(fig_forecast, f"{selected_asset}_forecast.png", "forecast_png")
                except Exception as e:
                    st.warning(f"Forecast unavailable: Install scikit-learn ({e})")
            
            with col2:
                st.markdown("### 🕯️ Candlestick View")
                fig_candle = create_candlestick_style_chart(prices, asset_info['name'])
                st.plotly_chart(fig_candle, use_container_width=True)
                get_chart_download_button(fig_candle, f"{selected_asset}_candle.png", "candle_png")
            
        else:
            st.info(f"📊 No price data available for {selected_asset}. Data collection in progress...")
    
    # ============== TAB 2: Technical Level ==============
    with tab2:
        if price_data and price_data.get('prices'):
            prices = price_data['prices']
            asset_info = price_data['asset']
            
            st.markdown("### 📉 Technical Indicators (RSI & MACD)")
            fig_tech = create_technical_indicators_chart(prices)
            st.plotly_chart(fig_tech, use_container_width=True)
            get_chart_download_button(fig_tech, f"{selected_asset}_tech_indicators.png", "tech_png")
            
            st.markdown("---")
            
            # Gauge Charts for Current Status
            df_tech = calculate_technical_indicators(pd.DataFrame(prices))
            current_rsi = df_tech['RSI'].iloc[-1] if not df_tech['RSI'].isna().all() else 50
            
            st.markdown("### 🧭 Live Technical Gauges")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(create_gauge_chart(current_rsi, "RSI (14)", 0, 100, [30, 70]), use_container_width=True)
            
            with col2:
                # Volatility Gauge
                returns = df_tech['price_usd'].pct_change() * 100
                curr_vol = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0
                st.plotly_chart(create_gauge_chart(curr_vol, "Volatility (20d)", 0, 5, [1, 3]), use_container_width=True)
            
            with col3:
                # Price relative to MA50
                curr_price = df_tech['price_usd'].iloc[-1]
                ma50 = df_tech['price_usd'].rolling(50).mean().iloc[-1] if len(df_tech) > 50 else curr_price
                pct_diff = ((curr_price - ma50) / ma50) * 100 if ma50 else 0
                st.metric("Price vs MA50", f"{pct_diff:.2f}%", delta_color="normal")
            
        else:
            st.info("No data available")
    
    # ============== TAB 3: Market Dynamics ==============
    with tab3:
        if price_data and price_data.get('prices'):
            prices = price_data['prices']
            asset_info = price_data['asset']
            
            st.markdown("### ✨ Animated Market Motion")
            st.info("Press Play to watch the price vs volume evolution over time")
            fig_bubble = create_animated_bubble_chart(prices)
            st.plotly_chart(fig_bubble, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Volume & Market Cap")
                fig_vol = create_volume_market_cap_chart(prices, asset_info['name'])
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col2:
                st.markdown("### 🌡️ Volatility Heatmap")
                fig_volatility = create_volatility_heatmap(prices, asset_info['name'])
                st.plotly_chart(fig_volatility, use_container_width=True)
        else:
            st.info("No data available")
    
    # ============== TAB 4: Advanced Comparison ==============
    with tab4:
        st.markdown("### 🔍 Multi-Asset Analysis")
        
        if compare_assets:
            all_assets = [selected_asset] + compare_assets
            comparison_data = fetch_comparison(",".join(all_assets), normalize_comparison, from_date)
            
            if comparison_data and comparison_data.get('data'):
                # Area chart
                st.markdown("#### Performance Comparison")
                fig_compare = create_multi_asset_area(comparison_data, normalize_comparison)
                st.plotly_chart(fig_compare, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Asset Radar")
                    fig_radar = create_comparison_radar(comparison_data)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🔗 Correlation Matrix")
                    fig_corr = create_correlation_heatmap(comparison_data)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Download
                all_data = []
                for point in comparison_data['data']:
                    row = {'timestamp': point['timestamp']}
                    row.update(point['prices'])
                    all_data.append(row)
                df_all = pd.DataFrame(all_data)
                csv = df_all.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data",
                    data=csv,
                    file_name="comparison_data.csv",
                    mime="text/csv",
                    key="compare_csv"
                )
        else:
            st.info("👈 Select assets to compare from the sidebar to enable advanced correlation and radar analysis.")
    
    # ============== TAB 5: Anomalies & Data ==============
    with tab5:
        st.markdown("### ⚠️ Detected Anomalies")
        
        if anomalies:
            df_anomalies = pd.DataFrame(anomalies)
            df_anomalies['timestamp'] = pd.to_datetime(df_anomalies['timestamp'])
            
            st.dataframe(
                df_anomalies[['timestamp', 'anomaly_type', 'price', 'severity', 'price_change']],
                use_container_width=True,
                column_config={
                    "timestamp": "Time",
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "severity": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                    "price_change": st.column_config.NumberColumn("Change %", format="%.2f%%")
                },
                hide_index=True
            )
            
            csv = df_anomalies.to_csv(index=False)
            st.download_button("📥 Download Anomalies", csv, f"{selected_asset}_anomalies.csv", "text/csv")
        else:
            st.success(f"✅ No anomalies detected for {selected_asset}")
        
        st.markdown("---")
        
        # Raw data viewer
        st.markdown("### 📋 Raw Data Explorer")
        if price_data and price_data.get('prices'):
            df_raw = pd.DataFrame(price_data['prices'])
            st.dataframe(df_raw, use_container_width=True, height=400)
            
            csv = df_raw.to_csv(index=False)
            st.download_button("📥 Download Full Dataset", csv, f"{selected_asset}_full_data.csv", "text/csv")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        f"Powered by CoinGecko API • Last updated: {datetime.now().strftime('%H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

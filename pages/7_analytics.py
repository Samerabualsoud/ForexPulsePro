"""
Advanced Analytics & Performance Dashboard - Professional Trading Analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

# Add imports
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.auth import require_authentication, render_user_info
    from utils.cache import get_cached_performance_stats, get_cached_market_data
    from components.advanced_charts import chart_builder
    
    # Require authentication 
    user_info = require_authentication()
    render_user_info()
    imports_available = True
except ImportError as e:
    st.warning(f"⚠️ Import error: {e} - running in demo mode")
    user_info = {"username": "demo", "role": "admin"}
    chart_builder = None
    imports_available = False

# Professional Analytics Page Styling
st.markdown("""
<style>
    .analytics-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e1e8ed;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .analytics-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<h1 class="analytics-title">📊 Advanced Trading Analytics</h1>', unsafe_allow_html=True)
st.markdown("### *Professional Performance Analysis & Strategy Insights*")
st.markdown("---")

# Generate demo analytics data
@st.cache_data(ttl=300)
def generate_analytics_data():
    """Generate comprehensive analytics data"""
    # Strategy performance metrics
    strategies = ['ema_rsi', 'donchian_atr', 'meanrev_bb', 'momentum', 'breakout', 'scalping', 'swing']
    
    strategy_metrics = {}
    for strategy in strategies:
        np.random.seed(hash(strategy) % 1000)
        strategy_metrics[strategy] = {
            'total_trades': np.random.randint(50, 200),
            'win_rate': np.random.uniform(0.45, 0.75),
            'avg_pnl': np.random.uniform(-50, 150),
            'profit_factor': np.random.uniform(0.8, 2.5),
            'sharpe_ratio': np.random.uniform(-0.5, 2.0),
            'max_drawdown': np.random.uniform(0.05, 0.25),
            'avg_trade_duration': np.random.uniform(30, 240)  # minutes
        }
    
    # Generate equity curve
    equity_curve = [0]
    for i in range(100):
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        equity_curve.append(equity_curve[-1] + daily_return * 10000)  # $10k account
    
    # Risk metrics
    risk_data = {
        'current_volatility': np.random.uniform(1.0, 3.5),
        'portfolio_var': np.random.uniform(500, 1500),
        'correlation_risk': np.random.uniform(0.3, 0.8),
        'concentration_risk': np.random.uniform(0.2, 0.6)
    }
    
    return {
        'strategy_metrics': strategy_metrics,
        'equity_curve': equity_curve,
        'risk_data': risk_data
    }

# Load analytics data
analytics_data = generate_analytics_data()

# Key Performance Indicators
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.markdown("### 🎯 **Key Performance Indicators**")

col1, col2, col3, col4 = st.columns(4)

# Calculate overall metrics
all_strategies = analytics_data['strategy_metrics']
total_trades = sum(s['total_trades'] for s in all_strategies.values())
weighted_win_rate = sum(s['win_rate'] * s['total_trades'] for s in all_strategies.values()) / total_trades
total_pnl = sum(s['avg_pnl'] * s['total_trades'] for s in all_strategies.values())
avg_sharpe = np.mean([s['sharpe_ratio'] for s in all_strategies.values()])

with col1:
    st.metric(
        "Total Trades",
        f"{total_trades:,}",
        delta=f"+{np.random.randint(5, 25)} today"
    )

with col2:
    st.metric(
        "Overall Win Rate",
        f"{weighted_win_rate:.1%}",
        delta=f"{np.random.uniform(-2, 5):.1f}%"
    )

with col3:
    st.metric(
        "Total P&L",
        f"${total_pnl:,.0f}",
        delta=f"${np.random.uniform(-200, 800):.0f}"
    )

with col4:
    st.metric(
        "Avg Sharpe Ratio",
        f"{avg_sharpe:.2f}",
        delta=f"{np.random.uniform(-0.1, 0.3):.2f}"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Charts Section
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    st.markdown("### 📈 **Portfolio Equity Curve**")
    
    # Create equity curve chart
    if chart_builder and imports_available:
        equity_fig = chart_builder.create_equity_curve_chart({
            'equity_curve': analytics_data['equity_curve']
        })
        st.plotly_chart(equity_fig, use_container_width=True)
    else:
        # Fallback simple chart
        import plotly.graph_objects as go
        dates = pd.date_range(start=datetime.now() - timedelta(days=len(analytics_data['equity_curve'])), 
                            end=datetime.now(), periods=len(analytics_data['equity_curve']))
        fig = go.Figure(data=go.Scatter(x=dates, y=analytics_data['equity_curve'], mode='lines', 
                                       name='Equity Curve', line=dict(color='#667eea', width=3)))
        fig.update_layout(title='Portfolio Equity Curve', xaxis_title='Date', yaxis_title='P&L', 
                         template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    st.markdown("### 🔥 **Strategy Performance Heatmap**")
    
    # Create strategy heatmap
    if chart_builder and imports_available:
        heatmap_fig = chart_builder.create_strategy_heatmap(all_strategies)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        # Fallback simple heatmap
        import plotly.graph_objects as go
        strategies = list(all_strategies.keys())
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        matrix = [[s['win_rate']*100 for s in all_strategies.values()] for _ in symbols]
        fig = go.Figure(data=go.Heatmap(z=matrix, x=strategies, y=symbols, colorscale='RdYlGn'))
        fig.update_layout(title='Strategy Performance Heatmap', height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Performance Analysis
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.markdown("### ⚡ **Strategy Performance Radar**")

# Create performance radar chart
radar_metrics = {
    'win_rate': weighted_win_rate,
    'profit_factor': np.mean([s['profit_factor'] for s in all_strategies.values()]),
    'sharpe_ratio': avg_sharpe,
    'max_drawdown': np.mean([s['max_drawdown'] for s in all_strategies.values()]),
    'avg_trade_duration': np.mean([s['avg_trade_duration'] for s in all_strategies.values()]),
    'total_pnl': total_pnl
}

if chart_builder and imports_available:
    radar_fig = chart_builder.create_performance_metrics_chart(radar_metrics)
    st.plotly_chart(radar_fig, use_container_width=True)
else:
    # Fallback metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Win Rate", f"{radar_metrics['win_rate']:.1%}")
        st.metric("Profit Factor", f"{radar_metrics['profit_factor']:.2f}")
    with col2:
        st.metric("Sharpe Ratio", f"{radar_metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{radar_metrics['max_drawdown']:.1%}")
    with col3:
        st.metric("Avg Duration", f"{radar_metrics['avg_trade_duration']:.0f}m")
        st.metric("Total P&L", f"${radar_metrics['total_pnl']:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# Risk Management Dashboard
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.markdown("### 🛡️ **Comprehensive Risk Dashboard**")

if chart_builder and imports_available:
    risk_fig = chart_builder.create_risk_metrics_dashboard(analytics_data['risk_data'])
    st.plotly_chart(risk_fig, use_container_width=True)
else:
    # Fallback risk metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Volatility", f"{analytics_data['risk_data']['current_volatility']:.1f}%")
    with col2:
        st.metric("Portfolio VaR", f"${analytics_data['risk_data']['portfolio_var']:,.0f}")
    with col3:
        st.metric("Correlation Risk", f"{analytics_data['risk_data']['correlation_risk']:.1%}")
    with col4:
        st.metric("Concentration Risk", f"{analytics_data['risk_data']['concentration_risk']:.1%}")
st.markdown('</div>', unsafe_allow_html=True)

# Strategy Breakdown Table
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.markdown("### 📋 **Detailed Strategy Breakdown**")

# Create detailed strategy table
strategy_df = pd.DataFrame(all_strategies).T
strategy_df.columns = [
    'Total Trades', 'Win Rate', 'Avg P&L', 'Profit Factor', 
    'Sharpe Ratio', 'Max DD', 'Avg Duration'
]

# Format the dataframe for display
strategy_df['Win Rate'] = strategy_df['Win Rate'].apply(lambda x: f"{x:.1%}")
strategy_df['Avg P&L'] = strategy_df['Avg P&L'].apply(lambda x: f"${x:.0f}")
strategy_df['Profit Factor'] = strategy_df['Profit Factor'].apply(lambda x: f"{x:.2f}")
strategy_df['Sharpe Ratio'] = strategy_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
strategy_df['Max DD'] = strategy_df['Max DD'].apply(lambda x: f"{x:.1%}")
strategy_df['Avg Duration'] = strategy_df['Avg Duration'].apply(lambda x: f"{x:.0f}m")

# Color-code the table based on performance
def color_performance(val):
    """Color code performance metrics"""
    if 'Win Rate' in str(val) or 'Profit Factor' in str(val):
        if val == strategy_df['Win Rate'].max() or val == strategy_df['Profit Factor'].max():
            return 'background-color: #d4edda; color: #155724;'
        elif val == strategy_df['Win Rate'].min() or val == strategy_df['Profit Factor'].min():
            return 'background-color: #f8d7da; color: #721c24;'
    return ''

st.dataframe(
    strategy_df,
    use_container_width=True,
    height=300
)
st.markdown('</div>', unsafe_allow_html=True)

# Advanced Insights
st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
st.markdown("### 🧠 **AI-Powered Insights**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎯 **Performance Insights**")
    
    # Generate insights based on data
    best_strategy = max(all_strategies.items(), key=lambda x: x[1]['profit_factor'])
    worst_strategy = min(all_strategies.items(), key=lambda x: x[1]['win_rate'])
    
    insights = [
        f"🥇 **Best Performer**: {best_strategy[0].title()} strategy with {best_strategy[1]['profit_factor']:.2f} profit factor",
        f"⚠️ **Needs Attention**: {worst_strategy[0].title()} strategy has {worst_strategy[1]['win_rate']:.1%} win rate",
        f"📊 **Portfolio Health**: Current Sharpe ratio of {avg_sharpe:.2f} indicates {'strong' if avg_sharpe > 1 else 'moderate' if avg_sharpe > 0.5 else 'weak'} risk-adjusted returns",
        f"💰 **Capital Efficiency**: {total_trades:,} trades generated ${total_pnl:,.0f} in returns"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")

with col2:
    st.markdown("#### 🔮 **Optimization Recommendations**")
    
    recommendations = [
        "🎛️ **Increase allocation** to high-performing strategies with Sharpe > 1.5",
        "⏰ **Optimize timing** - Consider reducing trade frequency during high volatility periods",
        "🎯 **Risk Management** - Current max drawdown acceptable, maintain 2-3% risk per trade",
        "📈 **Diversification** - Consider adding mean reversion strategies for market balance"
    ]
    
    for rec in recommendations:
        st.markdown(f"• {rec}")

st.markdown('</div>', unsafe_allow_html=True)

# Export Options
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Performance Report", use_container_width=True):
        st.success("📄 Performance report generated! (Demo)")

with col2:
    if st.button("📧 Email Analytics Summary", use_container_width=True):
        st.success("📧 Analytics summary sent! (Demo)")

with col3:
    if st.button("🔄 Refresh Analytics", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">Advanced Analytics Dashboard - Real-time Strategy Performance Analysis ⚡</p>',
    unsafe_allow_html=True
)
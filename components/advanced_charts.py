"""
Professional Financial Visualizations - Bloomberg-Quality Charts
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class ProfessionalChartBuilder:
    """Build professional-grade financial charts"""
    
    def __init__(self):
        self.theme_colors = {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'dark': '#343a40',
            'light': '#f8f9fa'
        }
        
        self.chart_config = {
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'displaylogo': False
        }
    
    def create_candlestick_chart(
        self,
        ohlc_data: List[Dict],
        symbol: str,
        signals: Optional[List[Dict]] = None,
        indicators: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create professional candlestick chart with indicators and signals
        
        Args:
            ohlc_data: OHLC price data
            symbol: Currency pair symbol
            signals: Trading signals to overlay
            indicators: Technical indicators to plot
            
        Returns:
            Plotly candlestick chart
        """
        try:
            if not ohlc_data:
                return self._create_empty_chart("No market data available")
            
            df = pd.DataFrame(ohlc_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create subplots for price + volume
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.8, 0.2],
                subplot_titles=[f'{symbol} Price Chart', 'Volume'],
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Main candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol,
                    increasing_line_color=self.theme_colors['success'],
                    decreasing_line_color=self.theme_colors['danger']
                ),
                row=1, col=1
            )
            
            # Add technical indicators if provided
            if indicators:
                self._add_indicators_to_chart(fig, df, indicators)
            
            # Add trading signals if provided
            if signals:
                self._add_signals_to_chart(fig, signals)
            
            # Volume bars
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df.get('volume', [1000] * len(df)),
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Professional Trading Chart',
                xaxis_title='Time',
                yaxis_title='Price',
                template='plotly_white',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Remove range slider for cleaner look
            fig.update_layout(xaxis_rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            return self._create_empty_chart(f"Chart error: {str(e)}")
    
    def _add_indicators_to_chart(self, fig: go.Figure, df: pd.DataFrame, indicators: Dict):
        """Add technical indicators to the chart"""
        try:
            # Moving averages
            if 'ema_short' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=indicators['ema_short'],
                        mode='lines',
                        name='EMA 12',
                        line=dict(color=self.theme_colors['primary'], width=2)
                    ),
                    row=1, col=1
                )
            
            if 'ema_long' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=indicators['ema_long'],
                        mode='lines',
                        name='EMA 26',
                        line=dict(color=self.theme_colors['secondary'], width=2)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=indicators['bb_upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color=self.theme_colors['info'], width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=indicators['bb_lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color=self.theme_colors['info'], width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(23, 162, 184, 0.1)',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
    
    def _add_signals_to_chart(self, fig: go.Figure, signals: List[Dict]):
        """Add trading signals as markers on the chart"""
        try:
            buy_signals = [s for s in signals if s.get('action') == 'BUY']
            sell_signals = [s for s in signals if s.get('action') == 'SELL']
            
            # Buy signals
            if buy_signals:
                buy_times = [pd.to_datetime(s['issued_at']) for s in buy_signals]
                buy_prices = [s['price'] for s in buy_signals]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        mode='markers',
                        name='BUY Signals',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.theme_colors['success'],
                            line=dict(width=2, color='white')
                        )
                    ),
                    row=1, col=1
                )
            
            # Sell signals
            if sell_signals:
                sell_times = [pd.to_datetime(s['issued_at']) for s in sell_signals]
                sell_prices = [s['price'] for s in sell_signals]
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        mode='markers',
                        name='SELL Signals',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.theme_colors['danger'],
                            line=dict(width=2, color='white')
                        )
                    ),
                    row=1, col=1
                )
                
        except Exception as e:
            logger.error(f"Error adding signals to chart: {e}")
    
    def create_equity_curve_chart(self, backtest_results: Dict) -> go.Figure:
        """Create professional equity curve visualization"""
        try:
            equity_curve = backtest_results.get('equity_curve', [0])
            if len(equity_curve) < 2:
                return self._create_empty_chart("No equity data available")
            
            # Create time series for equity curve
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=len(equity_curve)),
                end=datetime.now(),
                periods=len(equity_curve)
            )
            
            fig = go.Figure()
            
            # Main equity curve
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity_curve,
                    mode='lines',
                    name='Equity Curve',
                    line=dict(color=self.theme_colors['primary'], width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                )
            )
            
            # Add running maximum (for drawdown visualization)
            running_max = np.maximum.accumulate(equity_curve)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=running_max,
                    mode='lines',
                    name='Running Max',
                    line=dict(color=self.theme_colors['success'], width=1, dash='dash'),
                    opacity=0.7
                )
            )
            
            # Highlight drawdown periods
            drawdowns = [(running_max[i] - equity_curve[i]) / running_max[i] if running_max[i] > 0 else 0 
                        for i in range(len(equity_curve))]
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=[running_max[i] - (drawdowns[i] * running_max[i]) for i in range(len(drawdowns))],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=self.theme_colors['danger'], width=1),
                    fill='tonexty',
                    fillcolor='rgba(220, 53, 69, 0.2)'
                )
            )
            
            # Layout
            fig.update_layout(
                title='Portfolio Equity Curve & Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Cumulative P&L',
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating equity curve: {e}")
            return self._create_empty_chart(f"Equity curve error: {str(e)}")
    
    def create_strategy_heatmap(self, strategy_performance: Dict) -> go.Figure:
        """Create strategy performance heatmap"""
        try:
            if not strategy_performance:
                return self._create_empty_chart("No strategy performance data")
            
            # Prepare data for heatmap
            strategies = list(strategy_performance.keys())
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']  # Common pairs
            
            # Create synthetic performance matrix
            np.random.seed(42)  # For consistent demo data
            performance_matrix = []
            
            for strategy in strategies:
                strategy_row = []
                base_performance = strategy_performance[strategy].get('win_rate', 50) / 100
                
                for symbol in symbols:
                    # Add some variation per symbol
                    symbol_perf = base_performance + np.random.normal(0, 0.1)
                    symbol_perf = max(0, min(1, symbol_perf))  # Clamp to 0-1
                    strategy_row.append(symbol_perf * 100)  # Convert to percentage
                
                performance_matrix.append(strategy_row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=performance_matrix,
                x=symbols,
                y=strategies,
                colorscale=[
                    [0, '#dc3545'],     # Red for poor performance
                    [0.5, '#ffc107'],   # Yellow for average
                    [1, '#28a745']      # Green for good performance
                ],
                text=[[f"{val:.1f}%" for val in row] for row in performance_matrix],
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False,
                colorbar=dict(
                    title="Win Rate %",
                    titleside="right"
                )
            ))
            
            fig.update_layout(
                title='Strategy Performance by Currency Pair',
                xaxis_title='Currency Pairs',
                yaxis_title='Trading Strategies',
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating strategy heatmap: {e}")
            return self._create_empty_chart(f"Heatmap error: {str(e)}")
    
    def create_risk_metrics_dashboard(self, risk_data: Dict) -> go.Figure:
        """Create comprehensive risk metrics visualization"""
        try:
            # Create subplot grid for multiple risk metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Volatility Regime Analysis',
                    'Position Size Distribution', 
                    'Risk-Adjusted Returns',
                    'Correlation Matrix'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "heatmap"}]
                ]
            )
            
            # Volatility gauge
            current_vol = risk_data.get('current_volatility', 1.5)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_vol,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Current Volatility %"},
                    delta={'reference': 2.0},
                    gauge={
                        'axis': {'range': [None, 5]},
                        'bar': {'color': self.theme_colors['primary']},
                        'steps': [
                            {'range': [0, 1.5], 'color': self.theme_colors['success']},
                            {'range': [1.5, 3], 'color': self.theme_colors['warning']},
                            {'range': [3, 5], 'color': self.theme_colors['danger']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 3
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Position size distribution
            position_sizes = [0.1, 0.2, 0.15, 0.25, 0.3, 0.1, 0.05]  # Demo data
            size_labels = ['0.01-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '0.3+']
            
            fig.add_trace(
                go.Bar(
                    x=size_labels,
                    y=position_sizes,
                    name='Position Size Distribution',
                    marker_color=self.theme_colors['info']
                ),
                row=1, col=2
            )
            
            # Risk-adjusted returns scatter
            returns = np.random.normal(0.02, 0.15, 50)
            risks = np.random.uniform(0.01, 0.05, 50)
            
            fig.add_trace(
                go.Scatter(
                    x=risks,
                    y=returns,
                    mode='markers',
                    name='Risk vs Return',
                    marker=dict(
                        size=8,
                        color=self.theme_colors['primary'],
                        opacity=0.7
                    )
                ),
                row=2, col=1
            )
            
            # Correlation matrix
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
            correlation_matrix = [
                [1.0, 0.7, -0.3],
                [0.7, 1.0, -0.4],
                [-0.3, -0.4, 1.0]
            ]
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=pairs,
                    y=pairs,
                    colorscale='RdBu',
                    zmid=0,
                    text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Comprehensive Risk Management Dashboard',
                height=600,
                showlegend=False,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk metrics dashboard: {e}")
            return self._create_empty_chart(f"Risk dashboard error: {str(e)}")
    
    def create_performance_metrics_chart(self, metrics: Dict) -> go.Figure:
        """Create performance metrics visualization"""
        try:
            # Create radar chart for key metrics
            categories = [
                'Win Rate', 'Profit Factor', 'Sharpe Ratio',
                'Max Drawdown', 'Avg Trade Duration', 'Risk-Adj Return'
            ]
            
            # Normalize metrics to 0-100 scale for radar chart
            values = [
                metrics.get('win_rate', 0.5) * 100,
                min(100, metrics.get('profit_factor', 1.0) * 50),
                min(100, max(0, (metrics.get('sharpe_ratio', 0) + 1) * 50)),
                max(0, 100 - (metrics.get('max_drawdown', 0.2) * 500)),
                min(100, 100 - (metrics.get('avg_trade_duration', 60) / 10)),
                min(100, max(0, (metrics.get('total_pnl', 0) / 1000 + 1) * 50))
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Performance Profile',
                line_color=self.theme_colors['primary']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Strategy Performance Profile",
                height=500,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance metrics chart: {e}")
            return self._create_empty_chart(f"Performance chart error: {str(e)}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color=self.theme_colors['dark'])
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

# Global chart builder instance
chart_builder = ProfessionalChartBuilder()
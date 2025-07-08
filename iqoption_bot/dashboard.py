import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import time
import json
import queue
import threading
from api import IQOptionAPI
from mindset import TradingPsychology

class ProgressReporter:
    def __init__(self):
        self.queue = queue.Queue()
        self.current_step = 0
        self.steps = [
            {"name": "Waiting for new candle data", "percent": 20},
            {"name": "Preparing candle DataFrame", "percent": 10},
            {"name": "Calculating indicators", "percent": 20},
            {"name": "Generating signal", "percent": 20},
            {"name": "Placing trade", "percent": 20},
            {"name": "Checking trade result", "percent": 10}
        ]
    
    def reset(self):
        """Reset progress for a new cycle"""
        self.current_step = 0
        self.queue.queue.clear()
    
    def next_step(self):
        """Move to the next step in the process"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
        self.report_progress()
    
    def set_step(self, step_index):
        """Set specific step"""
        if 0 <= step_index < len(self.steps):
            self.current_step = step_index
        self.report_progress()
    
    def report_progress(self):
        """Report current progress to the queue"""
        step = self.steps[self.current_step]
        total_percent = sum(s["percent"] for s in self.steps[:self.current_step])
        self.queue.put({
            "step": self.current_step + 1,
            "name": step["name"],
            "current_percent": total_percent + step["percent"],
            "description": f"Step {self.current_step + 1}: {step['name']} ({total_percent + step['percent']}%)"
        })
    
    def get_progress(self):
        """Get current progress from the queue"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

class TradingDashboard:
    def __init__(self, api, psychology):
        """
        Initialize dashboard with API and psychology components
        
        Args:
            api: IQOptionAPI instance
            psychology: TradingPsychology instance
        """
        self.api = api
        self.psychology = psychology
        self.progress_reporter = ProgressReporter()
        self.setup_page()
        
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="IQ Option Trading Bot Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .metric-container {
                border: 1px solid #e1e4e8;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #f8f9fa;
            }
            .metric-value {
                font-size: 1.5rem;
                font-weight: bold;
            }
            .metric-label {
                font-size: 0.9rem;
                color: #6c757d;
            }
            .positive {
                color: #28a745;
                font-weight: bold;
            }
            .negative {
                color: #dc3545;
                font-weight: bold;
            }
            .status-active {
                color: #28a745;
                font-weight: bold;
            }
            .status-paused {
                color: #dc3545;
                font-weight: bold;
            }
            .progress-container {
                margin: 20px 0;
                padding: 15px;
                border-radius: 10px;
                background-color: #f0f2f6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .progress-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
            }
            .progress-description {
                font-size: 1.1rem;
                font-weight: bold;
                color: #2c3e50;
            }
            .progress-percent {
                font-weight: bold;
                color: #3498db;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def display_progress(self):
        """Display the progress bar with current status"""
        progress_data = self.progress_reporter.get_progress()
        
        if progress_data:
            st.session_state.progress = progress_data
        
        if "progress" in st.session_state:
            progress = st.session_state["progress"]
            
            st.markdown(
                f"""
                <div class="progress-container">
                    <div class="progress-header">
                        <div class="progress-description">Trade Cycle Progress</div>
                        <div class="progress-percent">{progress['current_percent']}%</div>
                    </div>
                    <div class="progress-description">{progress['description']}</div>
                    <div style="margin-top: 10px;">
                        {self.create_progress_bar(progress['current_percent'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def create_progress_bar(self, percent):
        """Create HTML-based progress bar"""
        return f"""
        <div style="
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        ">
            <div style="
                height: 100%;
                width: {percent}%;
                background-color: #3498db;
                transition: width 0.5s ease-in-out;
            "></div>
        </div>
        """
    
    def display_metrics(self):
        """Display key trading metrics with custom styling"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Account Balance
        with col1:
            balance = self.api.account_balance or 0
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Account Balance</div>
                    <div class="metric-value">${balance:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Today's P/L
        with col2:
            profit = self.psychology.profit_today
            profit_class = "positive" if profit >= 0 else "negative"
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Today's P/L</div>
                    <div class="metric-value {profit_class}">${profit:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Trades Today
        with col3:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Trades Today</div>
                    <div class="metric-value">
                        {self.psychology.trade_count}/{self.psychology.daily_trade_limit}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Trading Status
        with col4:
            status = "Active" if self.psychology.trading_enabled else "Paused"
            status_class = "status-active" if self.psychology.trading_enabled else "status-paused"
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Trading Status</div>
                    <div class="metric-value {status_class}">{status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def display_controls(self):
        """Display trading controls"""
        st.sidebar.header("🚀 Trading Controls")
        
        # Trading toggle button
        toggle_text = "🟢 Enable Trading" if not self.psychology.trading_enabled else "🔴 Disable Trading"
        if st.sidebar.button(toggle_text, use_container_width=True):
            if self.psychology.trading_enabled:
                self.psychology.disable_trading()
            else:
                self.psychology.enable_trading()
            st.experimental_rerun()
        
        # Reset daily stats button
        if st.sidebar.button("🔄 Reset Daily Stats", use_container_width=True):
            self.psychology.reset_daily_stats()
            st.experimental_rerun()
        
        # Trade amount slider
        st.sidebar.slider("💰 Trade Amount ($)", 1, 20, 5, key="trade_amount")
        
        # Asset selection
        self.selected_asset = st.sidebar.selectbox("📈 Select Asset", self.api.active_assets)
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
            st.experimental_rerun()
        
        # Simulate trade cycle button
        if st.sidebar.button("▶️ Simulate Trade Cycle", use_container_width=True):
            threading.Thread(target=self.simulate_trade_cycle, daemon=True).start()
    
    def simulate_trade_cycle(self):
        """Simulate a full trade cycle to demonstrate progress"""
        try:
            self.progress_reporter.reset()
            
            # Step 1: Waiting for candle data
            self.progress_reporter.set_step(0)
            time.sleep(2)  # Simulate waiting
            
            # Step 2: Prepare DataFrame
            self.progress_reporter.next_step()
            time.sleep(1)
            
            # Step 3: Calculate indicators
            self.progress_reporter.next_step()
            time.sleep(3)
            
            # Step 4: Generate signal
            self.progress_reporter.next_step()
            time.sleep(2)
            
            # Step 5: Place trade
            self.progress_reporter.next_step()
            time.sleep(2)
            
            # Step 6: Check result
            self.progress_reporter.next_step()
            time.sleep(1)
            
            # Reset for next cycle
            time.sleep(1)
            self.progress_reporter.reset()
            self.progress_reporter.report_progress()  # Send reset notification
            
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
    
    def display_trade_history(self):
        """Display recent trade history"""
        st.header("📊 Trade History")
        
        try:
            # Load trade history
            trades = pd.read_csv("data/trades_log.csv")
            
            if not trades.empty:
                # Convert timestamp to datetime
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
                trades = trades.sort_values('timestamp', ascending=False)
                
                # Display recent trades
                st.dataframe(
                    trades.head(20).style.apply(
                        lambda row: ['background-color: #d4edda' if row.result == 'WIN' 
                                    else 'background-color: #f8d7da' for _ in row],
                        axis=1
                    ),
                    height=600,
                    use_container_width=True
                )
                
                # Display performance chart
                if len(trades) > 1:
                    fig = px.line(
                        trades, 
                        x='timestamp', 
                        y='balance', 
                        title="Account Balance Over Time",
                        markers=True
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Balance ($)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to display performance chart")
            else:
                st.info("No trades recorded yet")
        except Exception as e:
            st.error(f"Error loading trade history: {str(e)}")
    
    def display_market_data(self):
        """Display current market data"""
        st.header("📈 Market Data")
        
        asset = self.selected_asset
        candle = self.api.get_candle_data(asset)
        
        if candle:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price display
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-label">{asset} Price</div>
                        <div class="metric-value">{candle['close']:.5f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Candle color indicator
                color_class = "positive" if candle['color'] == 'green' else "negative"
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-label">Candle Color</div>
                        <div class="metric-value {color_class}">
                            {candle['color'].upper()} ({'Bullish' if candle['color'] == 'green' else 'Bearish'})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                # Candle details in JSON
                st.subheader("Candle Details")
                candle_details = {
                    "Open": candle['open'],
                    "High": candle['high'],
                    "Low": candle['low'],
                    "Close": candle['close'],
                    "Volume": candle['volume'],
                    "Timestamp": datetime.datetime.fromtimestamp(candle['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                }
                st.json(candle_details)
            
            # Display historical candles
            history = self.api.get_candle_history(asset, 10)
            if history:
                st.subheader("Recent Candles")
                history_df = pd.DataFrame(history)
                
                # Format timestamp
                history_df['time'] = history_df['timestamp'].apply(
                    lambda ts: datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                )
                
                # Display with color coding
                styled_df = history_df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
                styled_df['color'] = styled_df.apply(
                    lambda row: 'background-color: #d4edda' if row['open'] < row['close'] 
                    else 'background-color: #f8d7da', axis=1
                )
                
                st.dataframe(
                    styled_df.style.apply(lambda x: x['color'], axis=1),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning(f"No data available for {asset}")
    
    def display_psychology_info(self):
        """Display psychology and risk management information"""
        st.sidebar.header("🧠 Psychology & Risk")
        
        # Consecutive losses
        st.sidebar.metric(
            "Consecutive Losses", 
            self.psychology.consecutive_losses,
            f"Max: {self.psychology.loss_streak_threshold}"
        )
        
        # Loss streak status
        if self.psychology.consecutive_losses >= self.psychology.loss_streak_threshold:
            if self.psychology.last_trade_time:
                time_since_loss = time.time() - self.psychology.last_trade_time
                remaining = max(0, self.psychology.wait_after_loss - time_since_loss)
                
                # Create progress bar
                progress = max(0, min(1.0, time_since_loss / self.psychology.wait_after_loss))
                st.sidebar.progress(progress, text=f"Cooling down: {int(remaining)}s remaining")
        
        # Daily trade limit progress
        trade_progress = min(1.0, self.psychology.trade_count / self.psychology.daily_trade_limit)
        st.sidebar.progress(
            trade_progress, 
            text=f"Trades: {self.psychology.trade_count}/{self.psychology.daily_trade_limit}"
        )
        
        # Profit/Loss progress
        profit_pct = min(1.0, self.psychology.profit_today / self.psychology.daily_profit_target)
        loss_pct = min(1.0, abs(self.psychology.profit_today) / abs(self.psychology.daily_loss_limit))
        
        st.sidebar.progress(
            profit_pct, 
            text=f"Profit: ${self.psychology.profit_today:.2f}/${self.psychology.daily_profit_target:.2f}"
        )
        st.sidebar.progress(
            loss_pct, 
            text=f"Loss: ${abs(self.psychology.profit_today):.2f}/${abs(self.psychology.daily_loss_limit):.2f}"
        )
        
        # Motivational messages
        if self.psychology.consecutive_losses > 0:
            st.sidebar.subheader("💪 Motivation")
            st.sidebar.info(self.psychology.motivational_messages[
                self.psychology.consecutive_losses % len(self.psychology.motivational_messages)
            ])
    
    def auto_refresh(self):
        """Create auto-refresh mechanism"""
        if "refresh_counter" not in st.session_state:
            st.session_state.refresh_counter = 0
        
        # Placeholder to trigger refresh
        refresh_placeholder = st.empty()
        refresh_placeholder.write(f"Last update: {datetime.datetime.now().strftime('%H:%M:%S')}")
        st.session_state.refresh_counter += 1
    
    def run(self):
        """Run the dashboard"""
        st.title("📈 IQ Option Trading Bot Dashboard")
        
        # Display progress bar at the top
        self.display_progress()
        
        # Create main layout columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Sidebar sections
            self.display_controls()
            self.display_psychology_info()
        
        with col2:
            # Main content tabs
            tab1, tab2 = st.tabs(["📊 Market Data", "📋 Trade History"])
            
            with tab1:
                self.display_market_data()
            
            with tab2:
                self.display_trade_history()
        
        # Display metrics at the bottom
        self.display_metrics()
        
        # Auto-refresh every 2 seconds
        time.sleep(2)
        st.experimental_rerun()

        # In your dashboard:
def display_strategy_metrics(self):
    st.subheader("Strategy Metrics")
    
    # Confidence score
    confidence = self.bot.strategy.last_confidence
    st.metric("Signal Confidence", f"{confidence}%", 
              delta="Strong" if confidence > 75 else "Weak" if confidence < 50 else "Moderate")
    
    # Progress bar
    progress = self.bot.strategy.execution_progress
    st.progress(progress)
    st.caption(f"Strategy Execution: {progress}% complete")
    
    # Indicators visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['MACD Hist', 'BB Distance', 'Stoch RSI', 'ADX', 'ATR'],
        y=[
            metadata['indicators']['macd_hist'],
            metadata['indicators']['bollinger_band'],
            metadata['indicators']['stoch_rsi'],
            metadata['indicators']['adx'],
            metadata['indicators']['atr']
        ],
        name='Indicator Values'
    ))
    st.plotly_chart(fig)

if __name__ == "__main__":
    # Initialize API and psychology
    # Note: We're not connecting to the API for dashboard-only use
    api = IQOptionAPI("email", "password")
    psychology = TradingPsychology()
    
    # Create dashboard instance
    dashboard = TradingDashboard(api, psychology)
    dashboard.run()
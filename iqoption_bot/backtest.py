import pandas as pd
import numpy as np
from logic import TradingStrategy
from mindset import TradingPsychology
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class Backtester:
    def __init__(self):
        self.strategy = TradingStrategy()
        self.psychology = TradingPsychology()
        self.results = []
        self.trade_history = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load historical data from CSV"""
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV missing required columns")
            
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def run_backtest(self, data: pd.DataFrame, trade_amount: float = 5.0) -> Dict:
        """Run backtest on historical data"""
        balance = 1000  # Starting balance
        self.results = []
        self.trade_history = []
        
        for i in range(len(data)-1):
            current_candle = data.iloc[i]
            next_candle = data.iloc[i+1]
            
            # Prepare candle data for indicators
            candle_data = data.iloc[:i+1].copy()
            
            # Calculate indicators
            indicators = self.strategy.calculate_indicators(candle_data)
            
            # Generate signal
            signal = self.strategy.generate_signal(indicators)
            
            if signal and self.psychology.can_trade():
                # Determine if trade would be successful
                if signal == "CALL":
                    win = next_candle['close'] > current_candle['close']
                else:  # PUT
                    win = next_candle['close'] < current_candle['close']
                
                # Calculate P&L
                payout = trade_amount * (0.8 if win else -1)  # Assuming 80% payout
                balance += payout
                
                # Record trade
                trade_record = {
                    'timestamp': current_candle.name,
                    'signal': signal,
                    'amount': trade_amount,
                    'result': 'WIN' if win else 'LOSS',
                    'payout': payout,
                    'balance': balance,
                    'rsi': indicators['rsi'],
                    'ema': indicators['ema']
                }
                
                self.trade_history.append(trade_record)
                self.psychology.update_trade_stats({
                    'win': win,
                    'amount': abs(payout)
                })
        
        # Generate performance metrics
        if self.trade_history:
            wins = len([t for t in self.trade_history if t['result'] == 'WIN'])
            losses = len(self.trade_history) - wins
            win_rate = (wins / len(self.trade_history)) * 100
            profit = balance - 1000
            max_drawdown = self.calculate_max_drawdown()
            
            metrics = {
                'initial_balance': 1000,
                'final_balance': balance,
                'total_profit': profit,
                'total_trades': len(self.trade_history),
                'win_rate': win_rate,
                'profit_factor': self.calculate_profit_factor(),
                'max_drawdown': max_drawdown,
                'average_win': np.mean([t['payout'] for t in self.trade_history if t['result'] == 'WIN']),
                'average_loss': np.mean([t['payout'] for t in self.trade_history if t['result'] == 'LOSS']),
                'largest_win': max([t['payout'] for t in self.trade_history if t['result'] == 'WIN'], default=0),
                'largest_loss': min([t['payout'] for t in self.trade_history if t['result'] == 'LOSS'], default=0),
                'longest_win_streak': self.calculate_streak('WIN'),
                'longest_loss_streak': self.calculate_streak('LOSS')
            }
            
            return metrics
        return {}
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown during backtest"""
        if not self.trade_history:
            return 0.0
            
        balances = [t['balance'] for t in self.trade_history]
        peak = balances[0]
        max_drawdown = 0.0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)"""
        if not self.trade_history:
            return 0.0
            
        gross_wins = sum(t['payout'] for t in self.trade_history if t['result'] == 'WIN')
        gross_losses = abs(sum(t['payout'] for t in self.trade_history if t['result'] == 'LOSS'))
        
        if gross_losses == 0:
            return float('inf')
            
        return gross_wins / gross_losses
    
    def calculate_streak(self, result_type: str) -> int:
        """Calculate longest streak of WIN or LOSS trades"""
        if not self.trade_history:
            return 0
            
        current_streak = 0
        max_streak = 0
        
        for trade in self.trade_history:
            if trade['result'] == result_type:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0
                
        return max_streak
    
    def save_results(self, filename: str):
        """Save backtest results to file"""
        if not self.trade_history:
            return
            
        # Ensure directory exists
        os.makedirs('data/backtest_results', exist_ok=True)
        
        # Save trade history to CSV
        pd.DataFrame(self.trade_history).to_csv(
            f'data/backtest_results/{filename}_trades.csv', index=False
        )
        
        # Save metrics to JSON
        metrics = self.run_backtest(pd.DataFrame(self.trade_history))
        with open(f'data/backtest_results/{filename}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate and save plots
        self.generate_plots(filename)
    
    def generate_plots(self, filename: str):
        """Generate and save performance plots"""
        if not self.trade_history:
            return
            
        df = pd.DataFrame(self.trade_history)
        
        # Balance curve
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='timestamp', y='balance')
        plt.title('Account Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance ($)')
        plt.tight_layout()
        plt.savefig(f'data/backtest_results/{filename}_balance.png')
        plt.close()
        
        # RSI distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df['rsi'], bins=30, kde=True)
        plt.title('RSI Distribution at Trade Entry')
        plt.xlabel('RSI Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'data/backtest_results/{filename}_rsi_dist.png')
        plt.close()
        
        # Win/Loss distribution
        plt.figure(figsize=(8, 6))
        df['result'].value_counts().plot(kind='bar')
        plt.title('Win/Loss Distribution')
        plt.xlabel('Result')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'data/backtest_results/{filename}_winloss.png')
        plt.close()

if __name__ == "__main__":
    backtester = Backtester()
    
    # Example usage:
    # data = backtester.load_data('path/to/historical_data.csv')
    # results = backtester.run_backtest(data)
    # backtester.save_results('my_backtest')
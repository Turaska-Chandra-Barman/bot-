import numpy as np
import pandas as pd
import talib
import logging
from datetime import datetime, time, timedelta
import requests
import json
from scipy.stats import linregress
from typing import Dict, Optional, Tuple, List
import random

class AdvancedTradingStrategy:
    def __init__(self, api=None, pairs=None, timeframe=None, initial_balance=None):
        # Preserve existing parameters
        self.ema_period = 9
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Advanced strategy parameters
        self.logger = logging.getLogger("STRATEGY")
        self.api_keys = {
            'news': 'your_news_api_key',
            'sentiment': 'your_sentiment_api_key',
            'social': 'your_social_api_key'
        }
        
        # Strategy weights (adjust based on backtesting)
        self.strategy_weights = {
            'smart_money': 0.9,
            'liquidity_sniping': 0.85,
            'order_block': 0.8,
            'market_structure': 0.95,
            'fair_value_gap': 0.75,
            'vwap_reversion': 0.7,
            'momentum_divergence': 0.8,
            'hidden_divergence': 0.75,
            'stop_hunt': 0.85,
            'sentiment': 0.65
        }
        
        # Market session hours (UTC)
        self.session_hours = {
            'asian': (time(22, 0), time(7, 0)),
            'london': (time(7, 0), time(16, 0)),
            'new_york': (time(12, 0), time(20, 0))
        }
        
        # Initialize state variables
        self.consecutive_losses = 0
        self.daily_profit = 0
        self.last_trade_result = None
        self.last_trade_time = None
        self.signal_history = []
        self.initial_balance = initial_balance or 10000
        self.current_balance = self.initial_balance

    """Core Strategy Methods"""
    
    def calculate_indicators(self, candle_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from candle data"""
        if candle_data.empty:
            self.logger.warning("No candle data available")
            return {}
            
        closes = candle_data['close'].values
        opens = candle_data['open'].values
        
        # Calculate indicators
        rsi = talib.RSI(closes, timeperiod=self.rsi_period)
        ema = talib.EMA(closes, timeperiod=self.ema_period)
        
        # Get latest values
        latest_rsi = rsi[-1] if len(rsi) > 0 else 50
        latest_ema = ema[-1] if len(ema) > 0 else closes[-1]
        latest_close = closes[-1] if len(closes) > 0 else 0
        latest_open = opens[-1] if len(opens) > 0 else 0
        candle_color = 'green' if latest_close > latest_open else 'red'
        
        return {
            'rsi': latest_rsi,
            'ema': latest_ema,
            'price': latest_close,
            'candle_color': candle_color
        }
    
    def generate_signal(self, indicators: Dict) -> Tuple[Optional[str], Dict]:
        """
        Generate trading signal with advanced features
        Returns tuple: (signal, metadata)
        """
        if not indicators:
            return None, {'confidence': 0, 'reason': 'no_indicators'}
        
        # 1. Check trading session
        current_time = datetime.utcnow()
        if not self.in_trading_session(current_time):
            self.logger.debug("Outside active trading session")
            return None, {'confidence': 0, 'reason': 'session'}
        
        # 2. Apply risk management
        if not self.risk_management_check():
            self.logger.debug("Risk management block")
            return None, {'confidence': 0, 'reason': 'risk'}
        
        # 3. Calculate base signal
        base_signal = self.generate_base_signal(indicators)
        if not base_signal:
            return None, {'confidence': 0, 'reason': 'no_base_signal'}
        
        # 4. Calculate advanced confidence
        try:
            strategy_confidence = self.calculate_strategy_confidence(candle_data=indicators)
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {str(e)}")
            strategy_confidence = 0.7  # Default confidence
        
        confidence_percent = int(strategy_confidence * 100)
        
        # 5. Generate final signal
        if confidence_percent >= 70:
            return base_signal, {
                'confidence': confidence_percent,
                'base_signal': base_signal,
                'strategies_used': len(self.strategy_weights),
                'timestamp': current_time
            }
        
        return None, {
            'confidence': confidence_percent,
            'reason': 'confidence',
            'timestamp': current_time
        }
    
    def generate_base_signal(self, indicators: Dict) -> Optional[str]:
        """Generate trading signal based on indicators"""
        try:
            rsi = indicators['rsi']
            ema = indicators['ema']
            price = indicators['price']
            candle_color = indicators['candle_color']
            
            # CALL signal conditions
            if (rsi < self.rsi_oversold and 
                candle_color == 'green' and 
                price > ema):
                return "CALL"
            
            # PUT signal conditions
            elif (rsi > self.rsi_overbought and 
                  candle_color == 'red' and 
                  price < ema):
                return "PUT"
            
            return None
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return None

    """Advanced Strategy Implementations"""
    
    # 1. Smart Money Order Flow Tracking
    def detect_smart_money(self, df: pd.DataFrame) -> float:
        """Detect institutional entries using price/volume spikes"""
        if df.empty:
            return 0.0
            
        # Calculate volume spike (3x average)
        avg_volume = df['volume'].rolling(20).mean()
        volume_spike = (df['volume'] > 3 * avg_volume).iloc[-1] if not avg_volume.empty else False
        
        # Calculate price spike (2x ATR)
        atr = talib.ATR(df['high'], df['low'], df['close'], 14)
        price_spike = (df['close'].pct_change().abs() > 2 * atr).iloc[-1] if not atr.empty else False
        
        return 1.0 if volume_spike and price_spike else 0.0
    
    # 2. Liquidity Sniping
    def liquidity_sniping(self, df: pd.DataFrame) -> float:
        """Detect liquidity pools near support/resistance"""
        if df.empty or len(df) < 5:
            return 0.0
            
        # Identify recent swing highs/lows
        swing_high = df['high'].rolling(5).max().iloc[-1]
        swing_low = df['low'].rolling(5).min().iloc[-1]
        
        # Check if price is approaching these levels
        current_price = df['close'].iloc[-1]
        threshold = 0.001  # 0.1%
        
        near_swing_high = abs(current_price - swing_high) / swing_high < threshold
        near_swing_low = abs(current_price - swing_low) / swing_low < threshold
        
        return 1.0 if near_swing_high or near_swing_low else 0.0
    
    # 3. Order Block Reversal
    def order_block_reversal(self, df: pd.DataFrame) -> float:
        """Identify key order blocks with reversal confirmation"""
        if df.empty or len(df) < 3:
            return 0.0
            
        # Detect large candle with high volume
        body_size = abs(df['close'] - df['open'])
        avg_body = body_size.rolling(10).mean()
        large_candle = (body_size > 2 * avg_body).iloc[-1] if not avg_body.empty else False
        high_volume = (df['volume'] > 1.5 * df['volume'].rolling(20).mean()).iloc[-1] if len(df) > 20 else False
        
        # Check if next candle reverses
        if large_candle and high_volume:
            prev_close = df['close'].iloc[-2]
            current_open = df['open'].iloc[-1]
            reversal = (current_open < prev_close) if df['close'].iloc[-2] > df['open'].iloc[-2] else (current_open > prev_close)
            return 1.0 if reversal else 0.0
        return 0.0
    
    # 4. Market Structure Break
    def market_structure_break(self, df: pd.DataFrame) -> float:
        """Detect breaks of higher time frame market structure"""
        if df.empty or len(df) < 20:
            return 0.0
            
        # Check for break of 20-period high/low
        high_20 = df['high'].rolling(20).max().iloc[-2]  # Previous period high
        low_20 = df['low'].rolling(20).min().iloc[-2]    # Previous period low
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        break_high = current_high > high_20
        break_low = current_low < low_20
        
        return 1.0 if break_high or break_low else 0.0
    
    # 5. Fair Value Gap Filling
    def fair_value_gap(self, df: pd.DataFrame) -> float:
        """Detect and trade fair value gaps"""
        if df.empty:
            return 0.0
            
        # Calculate gap between current close and 20-period EMA
        ema_20 = talib.EMA(df['close'], 20).iloc[-1]
        current_price = df['close'].iloc[-1]
        gap = abs(current_price - ema_20) / ema_20
        
        return min(1.0, gap * 10)  # Normalize to 0-1 range
    
    # 6. VWAP Reversion
    def vwap_reversion(self, df: pd.DataFrame) -> float:
        """Trade reversion to VWAP"""
        if df.empty:
            return 0.0
            
        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Distance from VWAP
        current_price = df['close'].iloc[-1]
        distance = abs(current_price - vwap.iloc[-1]) / vwap.iloc[-1]
        
        return min(1.0, distance * 10)  # Normalize to 0-1 range
    
    # 7. Momentum Divergence
    def momentum_divergence(self, df: pd.DataFrame) -> float:
        """Detect RSI/MACD divergence with volume confirmation"""
        if df.empty or len(df) < 20:
            return 0.0
            
        # Calculate RSI and MACD
        rsi = talib.RSI(df['close'], 14)
        macd, signal, _ = talib.MACD(df['close'])
        
        # Look for divergence
        price_high = df['close'].rolling(5).max()
        rsi_high = rsi.rolling(5).max()
        bearish_div = (price_high.iloc[-1] > price_high.iloc[-6]) and (rsi_high.iloc[-1] < rsi_high.iloc[-6])
        
        price_low = df['close'].rolling(5).min()
        rsi_low = rsi.rolling(5).min()
        bullish_div = (price_low.iloc[-1] < price_low.iloc[-6]) and (rsi_low.iloc[-1] > rsi_low.iloc[-6])
        
        # Volume confirmation
        volume_spike = df['volume'].iloc[-1] > 1.5 * df['volume'].rolling(20).mean().iloc[-1]
        
        return 1.0 if (bearish_div or bullish_div) and volume_spike else 0.0
    
    # 8. Hidden Divergence
    def hidden_divergence(self, df: pd.DataFrame) -> float:
        """Detect hidden divergences on multiple timeframes"""
        if df.empty or len(df) < 10:
            return 0.0
            
        rsi = talib.RSI(df['close'], 14)
        
        # Bullish hidden divergence
        price_higher_low = df['close'].iloc[-1] > df['close'].iloc[-3]
        rsi_lower_low = rsi.iloc[-1] < rsi.iloc[-3]
        bullish_hidden = price_higher_low and rsi_lower_low
        
        # Bearish hidden divergence
        price_lower_high = df['close'].iloc[-1] < df['close'].iloc[-3]
        rsi_higher_high = rsi.iloc[-1] > rsi.iloc[-3]
        bearish_hidden = price_lower_high and rsi_higher_high
        
        return 1.0 if bullish_hidden or bearish_hidden else 0.0
    
    # 9. Stop Hunt Detection
    def detect_stop_hunt(self, df: pd.DataFrame) -> float:
        """Identify potential stop loss runs"""
        if df.empty or len(df) < 20:
            return 0.0
            
        # Look for long wicks beyond support/resistance
        recent_high = df['high'].rolling(20).max().iloc[-2]
        recent_low = df['low'].rolling(20).min().iloc[-2]
        
        wick_high = (df['high'].iloc[-1] > recent_high) and (df['close'].iloc[-1] < recent_high)
        wick_low = (df['low'].iloc[-1] < recent_low) and (df['close'].iloc[-1] > recent_low)
        
        return 1.0 if wick_high or wick_low else 0.0
    
    # 10. Sentiment-based Confirmation
    def sentiment_confirmation(self) -> float:
        """Use market sentiment for confirmation"""
        try:
            # In a real implementation, this would call sentiment APIs
            # Simplified implementation for demo
            return random.uniform(0.7, 1.0)
        except:
            self.logger.warning("Sentiment API error")
            return 0.5
    
    """Signal Integration and Risk Management"""
    
    def calculate_strategy_confidence(self, candle_data: Dict) -> float:
        """Calculate weighted confidence score from all strategies"""
        strategies = {
            'smart_money': self.detect_smart_money,
            'liquidity_sniping': self.liquidity_sniping,
            'order_block': self.order_block_reversal,
            'market_structure': self.market_structure_break,
            'fair_value_gap': self.fair_value_gap,
            'vwap_reversion': self.vwap_reversion,
            'momentum_divergence': self.momentum_divergence,
            'hidden_divergence': self.hidden_divergence,
            'stop_hunt': self.detect_stop_hunt,
            'sentiment': self.sentiment_confirmation,
        }
        
        total_weight = 0
        weighted_score = 0
        
        for name, method in strategies.items():
            try:
                weight = self.strategy_weights.get(name, 0.7)
                
                # Handle candle_data format
                if isinstance(candle_data, dict):
                    # If we have multi-timeframe data, use the primary (5m) for strategies
                    primary_df = candle_data.get('5m', pd.DataFrame())
                    score = method(primary_df) if not primary_df.empty else 0.0
                else:
                    score = method(candle_data)
                
                total_weight += weight
                weighted_score += score * weight
                
                self.logger.debug(f"Strategy {name}: score={score:.2f}, weight={weight:.2f}")
            except Exception as e:
                self.logger.error(f"Error in {name} strategy: {str(e)}")
        
        if total_weight == 0:
            return 0.0
        
        confidence = weighted_score / total_weight
        return max(0.0, min(1.0, confidence))  # Ensure 0-1 range
    
    def in_trading_session(self, dt: datetime) -> bool:
        """Check if current time is in active trading session"""
        current_time = dt.time()
        for session, (start, end) in self.session_hours.items():
            if start < end:
                if start <= current_time <= end:
                    return True
            else:  # Overnight session
                if current_time >= start or current_time <= end:
                    return True
        return False
    
    def risk_management_check(self) -> bool:
        """Apply risk management rules before trading"""
        # Daily loss limit
        if self.daily_profit < -1000:  # $1000 daily loss limit
            self.logger.warning("Daily loss limit reached")
            return False
            
        # Cooldown after loss
        if self.last_trade_result == 'loss':
            if self.last_trade_time and (datetime.utcnow() - self.last_trade_time).seconds < 300:  # 5 min cooldown
                return False
                
        return True
    
    def _calculate_trade_amount(self, base_amount: float) -> float:
        """Calculate trade amount based on risk management"""
        # Simple risk management: 1% of current balance
        risk_amount = self.current_balance * 0.01
        return min(base_amount, risk_amount)
    
    def update_trade_result(self, result: str, profit: float):
        """Update strategy with trade result"""
        self.last_trade_result = result
        self.last_trade_time = datetime.utcnow()
        self.daily_profit += profit
        self.current_balance += profit
        
        if result == 'loss':
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_profit = 0
        self.consecutive_losses = 0
        self.last_trade_result = None
        self.current_balance = self.initial_balance
import time
from datetime import datetime, timedelta
import random
import logging
from typing import Dict

class TradingPsychology:
    def __init__(self):
        self.daily_trade_limit = 10
        self.daily_profit_target = 50
        self.daily_loss_limit = -30
        self.cooldown_period = 60  # seconds
        self.loss_streak_threshold = 2
        self.wait_after_loss = 3600  # 1 hour in seconds
        
        self.trade_count = 0
        self.profit_today = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.trading_enabled = True
        self.logger = logging.getLogger("PSYCHOLOGY")
        
        # Motivational messages
        self.motivational_messages = [
            "Losses are part of the game. Stay disciplined!",
            "Stick to your strategy. The edge will play out.",
            "One trade doesn't define you. Keep going!",
            "The market gives and takes. Stay balanced.",
            "Emotions are the enemy. Follow your rules."
        ]
    
    def can_trade(self) -> bool:
        """Check if trading is allowed based on all rules"""
        if not self.trading_enabled:
            self.logger.warning("Trading manually disabled")
            return False
            
        # Check daily trade limit
        if self.trade_count >= self.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached ({self.daily_trade_limit})")
            self.trading_enabled = False
            return False
            
        # Check profit target
        if self.profit_today >= self.daily_profit_target:
            self.logger.warning(f"Daily profit target reached (${self.daily_profit_target})")
            self.trading_enabled = False
            return False
            
        # Check loss limit
        if self.profit_today <= self.daily_loss_limit:
            self.logger.warning(f"Daily loss limit hit (${self.daily_loss_limit})")
            self.trading_enabled = False
            return False
            
        # Check loss streak
        if self.consecutive_losses >= self.loss_streak_threshold:
            if self.last_trade_time and (time.time() - self.last_trade_time) < self.wait_after_loss:
                remaining = int(self.wait_after_loss - (time.time() - self.last_trade_time))
                self.logger.warning(f"Waiting after loss streak ({remaining}s remaining)")
                return False
                
        # Check cooldown
        if self.last_trade_time and (time.time() - self.last_trade_time) < self.cooldown_period:
            remaining = int(self.cooldown_period - (time.time() - self.last_trade_time))
            self.logger.info(f"Cooldown period ({remaining}s remaining)")
            return False
            
        return True
    
    def update_trade_stats(self, result: Dict):
        """Update statistics after a trade"""
        self.trade_count += 1
        self.last_trade_time = time.time()
        
        if result['win']:
            self.consecutive_losses = 0
            self.profit_today += result['amount']
        else:
            self.consecutive_losses += 1
            self.profit_today -= result['amount']
            
            # Show motivational message on loss
            msg = random.choice(self.motivational_messages)
            self.logger.info(f"Motivation: {msg}")
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)"""
        now = datetime.now()
        self.trade_count = 0
        self.profit_today = 0
        self.consecutive_losses = 0
        self.logger.info(f"Daily stats reset at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def enable_trading(self):
        """Manually enable trading"""
        self.trading_enabled = True
        self.logger.info("Trading manually enabled")
    
    def disable_trading(self):
        """Manually disable trading"""
        self.trading_enabled = False
        self.logger.info("Trading manually disabled")
import logging
import time
import csv
import os
import pandas as pd
from datetime import datetime
from api import IQOptionAPI
from logic import AdvancedTradingStrategy
from mindset import TradingPsychology

class BinaryTradingBot:
    def __init__(self, email, password):
        self.api = IQOptionAPI(email, password)
        self.strategy = AdvancedTradingStrategy(
            api=self.api,
            pairs=["EURUSD", "GBPUSD"],
            timeframe=60,
            initial_balance=10000
        )
        self.psychology = TradingPsychology()
        self.trade_amount = 5
        self.trade_duration = 1
        self.assets = ["EURUSD", "GBPUSD"]
        self.setup_logging()
        self.setup_data_directory()
        self.logger = logging.getLogger("BOT")
        self.min_history_candles = 20
        self.otc_assets = ["OIL", "GOLD"]  # Add your actual OTC assets here
        self.base_amount = 5  # Base trade amount for trade cycle calculations
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/bot_log.txt'),
                logging.StreamHandler()
            ]
        )
    
    def setup_data_directory(self):
        if not os.path.exists('data'):
            os.makedirs('data')
            
        if not os.path.exists('data/trades_log.csv'):
            with open('data/trades_log.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'action', 'amount', 
                    'result', 'payout', 'balance', 'rsi', 'ema',
                    'confidence', 'progress'
                ])
    
    def log_trade(self, trade_data):
        with open('data/trades_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data['timestamp'],
                trade_data['asset'],
                trade_data['action'],
                trade_data['amount'],
                trade_data['result'],
                trade_data['payout'],
                trade_data['balance'],
                trade_data['rsi'],
                trade_data['ema'],
                trade_data.get('confidence', 0),
                trade_data.get('progress', 0)
            ])
    
    def wait_for_candle_data(self):
        self.logger.info("Waiting for candle data...")
        for asset in self.assets:
            start_time = time.time()
            max_wait = 30
            
            while time.time() - start_time < max_wait:
                candle = self.api.get_candle_data(asset)
                if candle:
                    self.logger.info(f"Received initial candle for {asset}")
                    break
                else:
                    self.logger.debug(f"Still waiting for {asset} candle data...")
                    time.sleep(1)
            else:
                self.logger.error(f"Failed to get initial candle for {asset} after {max_wait} seconds")
    
    def is_weekend_asset(self, asset):
        """Check if an asset trades on weekends"""
        return asset in self.otc_assets
    
    def get_multi_timeframe_data(self, asset):
        """Get data for multiple timeframes (1m, 5m, 15m)"""
        data = {}
        try:
            # Get 1-minute data
            data['1m'] = pd.DataFrame(self.api.get_candle_history(asset, self.min_history_candles))
            
            # Get 5-minute data (approximately 5x fewer candles)
            data['5m'] = pd.DataFrame(self.api.get_candle_history(
                asset, 
                count=int(self.min_history_candles/5),
                timeframe=5  # Assuming your API supports timeframe parameter
            ))
            
            # Get 15-minute data
            data['15m'] = pd.DataFrame(self.api.get_candle_history(
                asset, 
                count=int(self.min_history_candles/15),
                timeframe=15
            ))
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data: {str(e)}")
            data = {'5m': pd.DataFrame()}  # Fallback to empty DataFrame
        
        return data
    
    def run(self):
        self.logger.info("Starting trading bot")
        
        if not self.api.connect():
            self.logger.error("API connection failed")
            return
            
        self.wait_for_candle_data()
        
        try:
            while True:
                # Get current time
                now = datetime.utcnow()
                is_weekend = now.weekday() >= 5  # Saturday(5) or Sunday(6)
                
                # Check if we can trade based on psychology rules
                if not self.psychology.can_trade():
                    time.sleep(1)
                    continue
                
                # Process each asset
                for asset in self.assets:
                    try:
                        # Skip non-OTC assets on weekends
                        if is_weekend and not self.is_weekend_asset(asset):
                            self.logger.debug(f"Skipping {asset} on weekend")
                            continue
                            
                        # Get multi-timeframe data
                        candle_data = self.get_multi_timeframe_data(asset)
                        
                        # Get latest candle
                        candle = self.api.get_candle_data(asset)
                        if not candle:
                            self.logger.warning(f"No candle for {asset}")
                            continue
                        
                        # Calculate indicators
                        indicators = self.strategy.calculate_indicators(candle_data['5m'])
                        if not indicators:
                            self.logger.warning(f"No indicators for {asset}")
                            continue
                        
                        # Generate signal with metadata
                        signal, metadata = self.strategy.generate_signal(indicators)
                        if not signal:
                            continue
                            
                        self.logger.info(f"Signal: {asset} {signal} (Confidence: {metadata['confidence']}%)")
                        
                        # Calculate trade amount based on strategy
                        trade_amount = self.strategy._calculate_trade_amount(self.base_amount)
                        
                        # Place trade
                        trade_id = self.api.place_trade(
                            asset, signal, trade_amount, self.trade_duration
                        )
                        
                        if not trade_id:
                            self.logger.error("Trade placement failed")
                            continue
                            
                        # Wait for trade to complete
                        time.sleep(self.trade_duration * 60)
                        
                        # Check result
                        result = self.api.check_trade_result(trade_id)
                        if not result:
                            self.logger.error("Trade result check failed")
                            continue
                            
                        # Update psychology
                        self.psychology.update_trade_stats({
                            'win': result['win'],
                            'amount': result['amount']
                        })
                        
                        # Update strategy
                        self.strategy.update_trade_result(
                            'win' if result['win'] else 'loss', 
                            result['amount'] if result['win'] else -trade_amount
                        )
                        
                        # Log trade
                        trade_log = {
                            'timestamp': datetime.now().isoformat(),
                            'asset': asset,
                            'action': signal,
                            'amount': trade_amount,
                            'result': 'WIN' if result['win'] else 'LOSS',
                            'payout': result['amount'] if result['win'] else 0,
                            'balance': self.api.get_balance(),
                            'rsi': indicators['rsi'],
                            'ema': indicators['ema'],
                            'confidence': metadata['confidence'],
                        }
                        self.log_trade(trade_log)
                        
                        self.logger.info(f"Trade {'WIN' if result['win'] else 'LOSS'} ${result['amount']}")
                    
                    except Exception as e:
                        self.logger.error(f"Asset {asset} error: {str(e)}")
                        time.sleep(5)
                
                # Pause between cycles
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        finally:
            self.api.disconnect()
            self.logger.info("Bot shutdown")

if __name__ == "__main__":
    email = "barmanjatin372@gmail.com"
    password = "Uiop9900&&"
    bot = BinaryTradingBot(email, password)
    bot.run()
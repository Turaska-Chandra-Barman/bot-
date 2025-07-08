import time
import logging
from iqoptionapi.stable_api import IQ_Option
from threading import Thread, Event
from queue import Queue
from datetime import datetime

class IQOptionAPI:
    def __init__(self, email, password, balance_type="PRACTICE"):
        self.email = email
        self.password = password
        self.balance_type = balance_type
        self.api = None
        self.connected = False
        self.account_balance = 0
        self.active_assets = ["EURUSD", "GBPUSD"]
        self.active_timeframe = 1  # 1 minute
        self.logger = logging.getLogger("API")
        self.candle_queues = {asset: Queue() for asset in self.active_assets}
        self.stop_event = Event()
        self.candle_thread = None
        self.max_candles = 100
        self.candle_history = {asset: [] for asset in self.active_assets}
        
    def connect(self):
        """Connect to IQ Option API"""
        try:
            self.logger.info("Connecting to IQ Option API...")
            self.api = IQ_Option(self.email, self.password)
            self.connected = self.api.connect()
            
            if self.connected:
                if not self.api.check_connect():
                    self.logger.error("Connection verification failed")
                    self.connected = False
                    return False
                
                self.api.change_balance(self.balance_type)
                self.account_balance = self.api.get_balance()
                self.logger.info(f"Connected successfully. Balance: ${self.account_balance}")
                
                self.start_candle_streaming()
                return True
            else:
                self.logger.error("Connection failed")
                return False
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False
    
    def start_candle_streaming(self):
        """Start candle streaming in background thread"""
        if self.candle_thread and self.candle_thread.is_alive():
            self.stop_candle_streaming()
            
        self.stop_event.clear()
        self.candle_thread = Thread(target=self._candle_stream_worker, daemon=True)
        self.candle_thread.start()
        self.logger.info("Candle streaming started")
    
    def stop_candle_streaming(self):
        """Stop candle streaming"""
        self.stop_event.set()
        if self.candle_thread and self.candle_thread.is_alive():
            self.candle_thread.join(timeout=5)
        self.logger.info("Candle streaming stopped")
    
    def _candle_stream_worker(self):
        """Background worker for candle data"""
        try:
            # Subscribe to candles
            for asset in self.active_assets:
                self.api.start_candles_stream(asset, self.active_timeframe, self.max_candles)
                self.logger.info(f"Subscribed to {asset} candles")
                time.sleep(0.5)
            
            # Main processing loop
            while not self.stop_event.is_set():
                for asset in self.active_assets:
                    try:
                        candles = self.api.get_realtime_candles(asset, self.active_timeframe)
                        if candles:
                            # Process all new candles
                            for candle_id, candle_data in candles.items():
                                # Skip already processed candles
                                if candle_id in [c['id'] for c in self.candle_history[asset]]:
                                    continue
                                
                                # Create candle object
                                candle = {
                                    'id': candle_id,
                                    'open': float(candle_data['open']),
                                    'close': float(candle_data['close']),
                                    'high': float(candle_data['max']),
                                    'low': float(candle_data['min']),
                                    'volume': float(candle_data['volume']),
                                    'timestamp': candle_data['from'],
                                    'color': 'green' if float(candle_data['open']) < float(candle_data['close']) else 'red',
                                    'asset': asset
                                }
                                
                                # Add to history
                                self.candle_history[asset].append(candle)
                                
                                # Put in queue (only latest)
                                if self.candle_queues[asset].empty():
                                    self.candle_queues[asset].put(candle)
                    except Exception as e:
                        self.logger.error(f"Error processing {asset}: {str(e)}")
                
                time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Candle worker error: {str(e)}")
    
    def get_candle_data(self, asset, timeout=3):
        """Get latest candle with timeout handling"""
        if asset not in self.active_assets:
            self.logger.error(f"Invalid asset: {asset}")
            return None
            
        try:
            return self.candle_queues[asset].get(timeout=timeout)
        except:
            if self.candle_history[asset]:
                return self.candle_history[asset][-1]
            self.logger.warning(f"Timeout for {asset} candle")
            return None
    
    def get_candle_history(self, asset, count=20):
        """Get historical candles"""
        if asset in self.candle_history and self.candle_history[asset]:
            return self.candle_history[asset][-count:]
        return []
    
    def is_asset_open(self, asset):
        """Check if an asset is open for trading"""
        try:
            # Get all open times
            open_times = self.api.get_all_open_time()
            
            # Check binary options
            if 'binary' in open_times and asset in open_times['binary']:
                return open_times['binary'][asset]['open']
            
            # Check digital options
            if 'digital' in open_times and asset in open_times['digital']:
                return open_times['digital'][asset]['open']
            
            return True  # Default to open if no info available
        except Exception as e:
            self.logger.error(f"Asset open check error: {str(e)}")
            return True  # Default to open on error
    
    def place_trade(self, asset, action, amount, duration=1):
        """Place trade with enhanced error handling"""
        if not self.connected:
            self.logger.error("Not connected to API")
            return False
            
        action = action.lower()
        if action not in ["put", "call"]:
            self.logger.error(f"Invalid action: {action}")
            return False
            
        if asset not in self.active_assets:
            self.logger.error(f"Asset not available: {asset}")
            return False
            
        try:
            # Place trade
            status, trade_id = self.api.buy(float(amount), asset, action, duration)
            
            if trade_id:
                self.logger.info(f"Trade placed: {asset} {action} ${amount}")
                return trade_id
            else:
                error_reason = "Unknown error"
                if status == "error":
                    error_reason = "Invalid parameters"
                elif status == "failed":
                    error_reason = "Trade execution failed"
                    
                self.logger.error(f"Trade placement failed: {error_reason}")
                return False
        except Exception as e:
            self.logger.error(f"Trade error: {str(e)}")
            return False
    
    def check_trade_result(self, trade_id, max_attempts=5):
        """Check trade result with retries"""
        if not trade_id:
            return None
            
        for attempt in range(max_attempts):
            try:
                result = self.api.check_win_v3(trade_id)
                if result is not None:
                    return {
                        'win': result > 0,
                        'amount': abs(result),
                        'trade_id': trade_id,
                        'timestamp': datetime.now().timestamp()
                    }
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Result check error: {str(e)}")
                time.sleep(1)
                
        self.logger.error(f"Failed to get trade result after {max_attempts} attempts")
        return None
    
    def get_balance(self):
        """Get current balance"""
        try:
            self.account_balance = self.api.get_balance()
            return self.account_balance
        except:
            return None
    
    def disconnect(self):
        """Clean disconnection"""
        try:
            self.stop_candle_streaming()
            if self.connected:
                self.api.disconnect()
                self.connected = False
                self.logger.info("Disconnected")
        except Exception as e:
            self.logger.error(f"Disconnect error: {str(e)}")
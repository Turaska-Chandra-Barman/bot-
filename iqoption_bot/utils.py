import logging
import time
from datetime import datetime
import sounddevice as sd
import numpy as np
import requests

class BotUtils:
    @staticmethod
    def play_sound(frequency: float = 440.0, duration: float = 0.5):
        """Play a simple alert sound"""
        try:
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t)
            sd.play(wave, sample_rate)
            time.sleep(duration)
            sd.stop()
        except Exception as e:
            logging.getLogger("UTILS").warning(f"Couldn't play sound: {str(e)}")
    
    @staticmethod
    def send_telegram_alert(message: str, bot_token: str, chat_id: str):
        """Send notification via Telegram bot"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            params = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, params=params)
            if response.status_code != 200:
                logging.getLogger("UTILS").warning(f"Telegram alert failed: {response.text}")
        except Exception as e:
            logging.getLogger("UTILS").warning(f"Telegram alert error: {str(e)}")
    
    @staticmethod
    def time_until_market_open():
        """Calculate time until next market open (Monday 00:00 UTC)"""
        now = datetime.utcnow()
        
        # If it's Friday after close or weekend
        if now.weekday() >= 4:  # Friday(4), Saturday(5), Sunday(6)
            days_until_monday = (7 - now.weekday()) % 7
            next_open = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_until_monday)
        else:
            # For trading days, just wait until next day
            next_open = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        
        return (next_open - now).total_seconds()
    
    @staticmethod
    def validate_credentials(email: str, password: str) -> bool:
        """Validate IQ Option credentials format"""
        if not email or not password:
            return False
        if "@" not in email or "." not in email:
            return False
        if len(password) < 6:
            return False
        return True
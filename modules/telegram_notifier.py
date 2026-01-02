#!/usr/bin/env python3
"""
Telegram Notifier Module for CivicCam

Sends real-time Telegram alerts when littering incidents are detected.
Includes evidence photos, license plate info, timestamp, and location.

Usage:
    notifier = TelegramNotifier(bot_token, chat_id, location="Camera 1")
    await notifier.send_incident_alert(event_data, image_path)
"""

import os
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    TelegramError = Exception

from modules.utils import setup_logger

logger = setup_logger(name="civiccam.telegram", level="INFO", console_output=True)


class TelegramNotifier:
    """
    Handles Telegram notifications for CivicCam incidents.
    
    Features:
    - Sends photo with caption containing incident details
    - Rate limiting to prevent spam
    - Graceful error handling
    - Async operation
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        location: str = "CivicCam Station",
        rate_limit_seconds: float = 10.0,
        include_evidence_photo: bool = True
    ):
        """
        Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram bot API token (or set TELEGRAM_BOT_TOKEN env var)
            chat_id: Target chat/group ID (or set TELEGRAM_CHAT_ID env var)
            location: Descriptive location name for messages
            rate_limit_seconds: Minimum seconds between notifications
            include_evidence_photo: Whether to attach evidence images
        """
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.location = location
        self.rate_limit_seconds = rate_limit_seconds
        self.include_evidence_photo = include_evidence_photo
        
        self.last_notification_time = 0.0
        self.bot: Optional[Bot] = None
        self.enabled = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the Telegram bot."""
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not installed. Telegram notifications disabled.")
            return
        
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set. Telegram notifications disabled.")
            return
        
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set. Telegram notifications disabled.")
            return
        
        try:
            self.bot = Bot(token=self.bot_token)
            self.enabled = True
            logger.info(f"✅ Telegram notifier initialized for chat: {self.chat_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enabled = False
    
    def _format_message(self, event_data: Dict[str, Any]) -> str:
        """
        Format the incident notification message.
        
        Args:
            event_data: Dictionary containing event information
            
        Returns:
            Formatted message string
        """
        timestamp = event_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        object_type = event_data.get("objectType", "unknown object")
        confidence = event_data.get("confidence", 0)
        confidence_pct = int(confidence * 100) if confidence <= 1 else int(confidence)
        
        vehicle_detected = event_data.get("vehicleDetected", False)
        plate_number = event_data.get("plateNumber", None)
        
        # Build message
        lines = [
            "🚨 *LITTERING INCIDENT DETECTED*",
            "",
            f"📅 *Time:* {timestamp}",
            f"📍 *Location:* {self.location}",
            f"🎯 *Object:* {object_type}",
            f"📊 *Confidence:* {confidence_pct}%",
        ]
        
        # Add vehicle/plate info if available
        if vehicle_detected:
            if plate_number:
                lines.append(f"🚗 *License Plate:* `{plate_number}`")
            else:
                lines.append("🚗 *Vehicle:* Detected (plate not readable)")
        
        # Add hashtags for searchability
        lines.extend([
            "",
            "#civiccam #littering #alert"
        ])
        
        return "\n".join(lines)
    
    def _can_send(self) -> bool:
        """Check if we can send a notification (rate limiting)."""
        current_time = time.time()
        if current_time - self.last_notification_time < self.rate_limit_seconds:
            return False
        return True
    
    async def send_incident_alert(
        self,
        event_data: Dict[str, Any],
        image_path: Optional[Path] = None
    ) -> bool:
        """
        Send an incident alert to Telegram.
        
        Args:
            event_data: Dictionary containing event information
            image_path: Path to the evidence image (optional)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Telegram notifications disabled, skipping alert")
            return False
        
        if not self._can_send():
            logger.debug("Rate limited, skipping Telegram notification")
            return False
        
        try:
            message = self._format_message(event_data)
            
            # Send with photo if available and enabled
            if self.include_evidence_photo and image_path and image_path.exists():
                with open(image_path, "rb") as photo:
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=message,
                        parse_mode="Markdown"
                    )
                logger.info(f"📱 Telegram alert sent with photo: {image_path.name}")
            else:
                # Send text-only message
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
                logger.info("📱 Telegram alert sent (text only)")
            
            self.last_notification_time = time.time()
            return True
            
        except TelegramError as e:
            logger.error(f"Telegram API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    def send_incident_alert_sync(
        self,
        event_data: Dict[str, Any],
        image_path: Optional[Path] = None
    ) -> bool:
        """
        Synchronous wrapper for send_incident_alert.
        Use this when calling from non-async code.
        
        Args:
            event_data: Dictionary containing event information
            image_path: Path to the evidence image (optional)
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Try to get the running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, schedule the coroutine
                future = asyncio.run_coroutine_threadsafe(
                    self.send_incident_alert(event_data, image_path),
                    loop
                )
                return future.result(timeout=10)
            except RuntimeError:
                # No running loop, create a new one
                return asyncio.run(self.send_incident_alert(event_data, image_path))
        except Exception as e:
            logger.error(f"Error in sync wrapper: {e}")
            return False
    
    async def send_test_message(self) -> bool:
        """Send a test message to verify the bot is working."""
        if not self.enabled:
            logger.error("Cannot send test message: Telegram not enabled")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text="🔔 *CivicCam Test Message*\n\nTelegram notifications are working correctly!",
                parse_mode="Markdown"
            )
            logger.info("✅ Test message sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send test message: {e}")
            return False


# Factory function for easy creation from config
def create_notifier_from_config(config: Dict[str, Any]) -> Optional[TelegramNotifier]:
    """
    Create a TelegramNotifier from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'telegram' section
        
    Returns:
        TelegramNotifier instance or None if disabled
    """
    telegram_config = config.get("telegram", {})
    
    if not telegram_config.get("enable", False):
        logger.info("Telegram notifications disabled in config")
        return None
    
    return TelegramNotifier(
        bot_token=telegram_config.get("bot_token"),
        chat_id=telegram_config.get("chat_id"),
        location=telegram_config.get("location", "CivicCam Station"),
        rate_limit_seconds=telegram_config.get("rate_limit_seconds", 10.0),
        include_evidence_photo=telegram_config.get("include_evidence_photo", True)
    )

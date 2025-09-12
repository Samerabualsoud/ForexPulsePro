"""
WhatsApp Cloud API Service
"""
import os
import requests
import asyncio
import hmac
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..models import Signal
from ..signals.utils import format_signal_message
from ..logs.logger import get_logger

logger = get_logger(__name__)

class WhatsAppService:
    """WhatsApp Cloud API integration service"""
    
    def __init__(self):
        self.token = os.getenv("WHATSAPP_TOKEN")
        self.phone_id = os.getenv("WHATSAPP_PHONE_ID")
        self.recipients = self._parse_recipients(os.getenv("WHATSAPP_TO", ""))
        self.base_url = "https://graph.facebook.com/v19.0"
        self.enabled = bool(self.token and self.phone_id and self.recipients)
        
        if not self.enabled:
            logger.warning("WhatsApp service disabled - missing configuration")
        else:
            logger.info(f"WhatsApp service enabled for {len(self.recipients)} recipients")
    
    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse comma-separated recipient numbers"""
        if not recipients_str:
            return []
        
        recipients = []
        for number in recipients_str.split(','):
            number = number.strip()
            if number:
                # Ensure E.164 format
                if not number.startswith('+'):
                    number = '+' + number
                recipients.append(number)
        
        return recipients
    
    async def send_signal(self, signal: Signal) -> Dict[str, Any]:
        """Send trading signal to WhatsApp"""
        if not self.enabled:
            raise Exception("WhatsApp service not configured")
        
        try:
            # Format signal message
            signal_data = {
                'symbol': signal.symbol,
                'action': signal.action,
                'price': signal.price,
                'sl': signal.sl,
                'tp': signal.tp,
                'confidence': signal.confidence,
                'strategy': signal.strategy
            }
            
            message = format_signal_message(signal_data)
            
            # Add timestamp and additional info
            timestamp = datetime.utcnow().strftime("%H:%M UTC")
            formatted_message = f"🚨 FOREX SIGNAL\n\n{message}\n\nTime: {timestamp}\nExpires: {signal.expires_at.strftime('%H:%M UTC') if signal.expires_at else 'N/A'}"
            
            # Send to all recipients
            results = []
            for recipient in self.recipients:
                result = await self._send_message(recipient, formatted_message)
                results.append(result)
            
            logger.info(f"Signal sent to {len(self.recipients)} WhatsApp recipients")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp signal: {e}")
            raise
    
    async def send_test_message(self) -> Dict[str, Any]:
        """Send test message to verify connectivity"""
        if not self.enabled:
            raise Exception("WhatsApp service not configured")
        
        test_message = f"🧪 WhatsApp Test Message\n\nFX Signal Dashboard is connected and working!\n\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        results = []
        for recipient in self.recipients:
            result = await self._send_message(recipient, test_message)
            results.append(result)
        
        return {"status": "success", "results": results}
    
    async def _send_message(self, recipient: str, message: str) -> Dict[str, Any]:
        """Send individual message to WhatsApp API"""
        try:
            url = f"{self.base_url}/{self.phone_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": recipient,
                "type": "text",
                "text": {
                    "body": message
                }
            }
            
            # Add retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    if "messages" in result and len(result["messages"]) > 0:
                        message_id = result["messages"][0].get("id")
                        logger.debug(f"Message sent to {recipient}: {message_id}")
                        return {
                            "recipient": recipient,
                            "message_id": message_id,
                            "status": "sent"
                        }
                    else:
                        raise Exception(f"Unexpected response format: {result}")
                
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
        except Exception as e:
            logger.error(f"Failed to send message to {recipient}: {e}")
            return {
                "recipient": recipient,
                "error": str(e),
                "status": "failed"
            }
    
    def generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook verification"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def verify_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self.generate_webhook_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)
    
    async def send_bulk_message(self, message: str, recipients: Optional[List[str]] = None) -> Dict[str, Any]:
        """Send message to multiple recipients"""
        if not self.enabled:
            raise Exception("WhatsApp service not configured")
        
        target_recipients = recipients or self.recipients
        
        results = []
        for recipient in target_recipients:
            result = await self._send_message(recipient, message)
            results.append(result)
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        success_count = len([r for r in results if r.get("status") == "sent"])
        
        return {
            "status": "completed",
            "total_recipients": len(target_recipients),
            "successful_sends": success_count,
            "failed_sends": len(target_recipients) - success_count,
            "results": results
        }
    
    def is_configured(self) -> bool:
        """Check if WhatsApp service is properly configured"""
        return self.enabled
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get detailed configuration status"""
        return {
            "enabled": self.enabled,
            "has_token": bool(self.token),
            "has_phone_id": bool(self.phone_id),
            "recipient_count": len(self.recipients),
            "recipients_masked": [
                f"{r[:3]}***{r[-3:]}" if len(r) > 6 else "***"
                for r in self.recipients
            ]
        }

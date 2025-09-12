"""
WhatsApp Service Tests
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
import os
from datetime import datetime, timedelta

# Import modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.whatsapp import WhatsAppService
from backend.models import Signal

class TestWhatsAppService:
    """Test WhatsApp Cloud API service"""
    
    @pytest.fixture
    def whatsapp_service(self):
        """Create WhatsApp service instance with test config"""
        with patch.dict(os.environ, {
            'WHATSAPP_TOKEN': 'test_token_123',
            'WHATSAPP_PHONE_ID': 'test_phone_id_456', 
            'WHATSAPP_TO': '+1234567890,+0987654321'
        }):
            return WhatsAppService()
    
    @pytest.fixture
    def whatsapp_service_disabled(self):
        """Create disabled WhatsApp service (no config)"""
        with patch.dict(os.environ, {}, clear=True):
            return WhatsAppService()
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample signal for testing"""
        signal = Signal(
            id=123,
            symbol='EURUSD',
            action='BUY',
            price=1.08523,
            sl=1.08323,
            tp=1.08723,
            confidence=0.72,
            strategy='ema_rsi',
            expires_at=datetime.utcnow() + timedelta(hours=1),
            issued_at=datetime.utcnow()
        )
        return signal
    
    def test_whatsapp_service_initialization_enabled(self, whatsapp_service):
        """Test WhatsApp service initializes when properly configured"""
        assert whatsapp_service.token == 'test_token_123'
        assert whatsapp_service.phone_id == 'test_phone_id_456'
        assert whatsapp_service.recipients == ['+1234567890', '+0987654321']
        assert whatsapp_service.enabled == True
        assert whatsapp_service.is_configured() == True
    
    def test_whatsapp_service_initialization_disabled(self, whatsapp_service_disabled):
        """Test WhatsApp service initializes as disabled when not configured"""
        assert whatsapp_service_disabled.enabled == False
        assert whatsapp_service_disabled.is_configured() == False
        assert whatsapp_service_disabled.recipients == []
    
    def test_parse_recipients_valid(self, whatsapp_service):
        """Test parsing valid recipient numbers"""
        recipients_str = "+1234567890, +0987654321, 1122334455"
        recipients = whatsapp_service._parse_recipients(recipients_str)
        
        assert len(recipients) == 3
        assert recipients[0] == '+1234567890'
        assert recipients[1] == '+0987654321'  
        assert recipients[2] == '+1122334455'  # Should add + prefix
    
    def test_parse_recipients_empty(self, whatsapp_service):
        """Test parsing empty recipients string"""
        recipients = whatsapp_service._parse_recipients("")
        assert recipients == []
        
        recipients = whatsapp_service._parse_recipients("   ")
        assert recipients == []
    
    def test_parse_recipients_malformed(self, whatsapp_service):
        """Test parsing malformed recipients"""
        recipients_str = "+1234567890,, ,+0987654321,   "
        recipients = whatsapp_service._parse_recipients(recipients_str)
        
        assert len(recipients) == 2
        assert recipients[0] == '+1234567890'
        assert recipients[1] == '+0987654321'
    
    @pytest.mark.asyncio
    async def test_send_signal_disabled_service(self, whatsapp_service_disabled, sample_signal):
        """Test sending signal when service is disabled"""
        with pytest.raises(Exception) as exc_info:
            await whatsapp_service_disabled.send_signal(sample_signal)
        
        assert "WhatsApp service not configured" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_send_signal_success(self, whatsapp_service, sample_signal):
        """Test successful signal sending"""
        mock_response = {
            'messages': [{'id': 'msg_id_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await whatsapp_service.send_signal(sample_signal)
            
            assert result['status'] == 'success'
            assert len(result['results']) == 2  # Two recipients
            
            # Check API was called correctly
            assert mock_post.call_count == 2  # One call per recipient
            
            # Check first call
            first_call = mock_post.call_args_list[0]
            assert first_call[1]['json']['messaging_product'] == 'whatsapp'
            assert first_call[1]['json']['to'] == '+1234567890'
            assert first_call[1]['json']['type'] == 'text'
            assert 'EURUSD BUY @ 1.08523' in first_call[1]['json']['text']['body']
    
    @pytest.mark.asyncio
    async def test_send_signal_api_error(self, whatsapp_service, sample_signal):
        """Test signal sending with API error"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 400
            mock_post.return_value.raise_for_status.side_effect = Exception("API Error")
            
            result = await whatsapp_service.send_signal(sample_signal)
            
            assert result['status'] == 'success'  # Overall status
            # But individual results should show failures
            for res in result['results']:
                assert res['status'] == 'failed'
                assert 'error' in res
    
    @pytest.mark.asyncio
    async def test_send_signal_network_error(self, whatsapp_service, sample_signal):
        """Test signal sending with network error"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            result = await whatsapp_service.send_signal(sample_signal)
            
            for res in result['results']:
                assert res['status'] == 'failed'
                assert 'Network error' in res['error']
    
    @pytest.mark.asyncio
    async def test_send_signal_retry_logic(self, whatsapp_service, sample_signal):
        """Test retry logic on failed requests"""
        call_count = 0
        
        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise Exception("Temporary error")
            else:  # Succeed on 3rd attempt
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'messages': [{'id': 'msg_123'}]}
                return mock_response
        
        with patch('requests.post', side_effect=mock_post_side_effect):
            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                result = await whatsapp_service.send_signal(sample_signal)
                
                # Should eventually succeed after retries
                success_results = [r for r in result['results'] if r['status'] == 'sent']
                assert len(success_results) == 2  # Both recipients should succeed
    
    @pytest.mark.asyncio
    async def test_send_test_message_success(self, whatsapp_service):
        """Test sending test message successfully"""
        mock_response = {
            'messages': [{'id': 'test_msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            result = await whatsapp_service.send_test_message()
            
            assert result['status'] == 'success'
            assert len(result['results']) == 2
            
            # Check message content
            first_call = mock_post.call_args_list[0]
            message_body = first_call[1]['json']['text']['body']
            assert 'WhatsApp Test Message' in message_body
            assert 'FX Signal Dashboard' in message_body
    
    @pytest.mark.asyncio
    async def test_send_test_message_disabled(self, whatsapp_service_disabled):
        """Test sending test message when service is disabled"""
        with pytest.raises(Exception) as exc_info:
            await whatsapp_service_disabled.send_test_message()
        
        assert "WhatsApp service not configured" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_send_bulk_message_success(self, whatsapp_service):
        """Test sending bulk message to multiple recipients"""
        mock_response = {
            'messages': [{'id': 'bulk_msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            with patch('asyncio.sleep'):  # Mock sleep for rate limiting
                
                result = await whatsapp_service.send_bulk_message("Test bulk message")
                
                assert result['status'] == 'completed'
                assert result['total_recipients'] == 2
                assert result['successful_sends'] == 2
                assert result['failed_sends'] == 0
    
    @pytest.mark.asyncio
    async def test_send_bulk_message_custom_recipients(self, whatsapp_service):
        """Test sending bulk message to custom recipient list"""
        custom_recipients = ['+1111111111', '+2222222222']
        mock_response = {
            'messages': [{'id': 'bulk_msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            with patch('asyncio.sleep'):
                
                result = await whatsapp_service.send_bulk_message(
                    "Test message", 
                    recipients=custom_recipients
                )
                
                assert result['total_recipients'] == 2
                assert result['successful_sends'] == 2
                
                # Check correct recipients were used
                calls = mock_post.call_args_list
                assert calls[0][1]['json']['to'] == '+1111111111'
                assert calls[1][1]['json']['to'] == '+2222222222'
    
    def test_generate_webhook_signature(self, whatsapp_service):
        """Test webhook signature generation"""
        payload = '{"test": "data"}'
        secret = 'webhook_secret'
        
        signature = whatsapp_service.generate_webhook_signature(payload, secret)
        
        assert signature is not None
        assert len(signature) == 64  # SHA256 hex length
        assert signature.isalnum()  # Should be alphanumeric hex
    
    def test_verify_webhook_signature_valid(self, whatsapp_service):
        """Test webhook signature verification with valid signature"""
        payload = '{"test": "data"}'
        secret = 'webhook_secret'
        
        signature = whatsapp_service.generate_webhook_signature(payload, secret)
        is_valid = whatsapp_service.verify_webhook_signature(payload, signature, secret)
        
        assert is_valid == True
    
    def test_verify_webhook_signature_invalid(self, whatsapp_service):
        """Test webhook signature verification with invalid signature"""
        payload = '{"test": "data"}'
        secret = 'webhook_secret'
        invalid_signature = 'invalid_signature_123'
        
        is_valid = whatsapp_service.verify_webhook_signature(payload, invalid_signature, secret)
        
        assert is_valid == False
    
    def test_verify_webhook_signature_wrong_secret(self, whatsapp_service):
        """Test webhook signature verification with wrong secret"""
        payload = '{"test": "data"}'
        secret = 'webhook_secret'
        wrong_secret = 'wrong_secret'
        
        signature = whatsapp_service.generate_webhook_signature(payload, secret)
        is_valid = whatsapp_service.verify_webhook_signature(payload, signature, wrong_secret)
        
        assert is_valid == False
    
    def test_get_configuration_status_enabled(self, whatsapp_service):
        """Test getting configuration status when enabled"""
        status = whatsapp_service.get_configuration_status()
        
        assert status['enabled'] == True
        assert status['has_token'] == True
        assert status['has_phone_id'] == True
        assert status['recipient_count'] == 2
        assert len(status['recipients_masked']) == 2
        
        # Check recipients are properly masked
        assert status['recipients_masked'][0] == '+12***890'
        assert status['recipients_masked'][1] == '+09***321'
    
    def test_get_configuration_status_disabled(self, whatsapp_service_disabled):
        """Test getting configuration status when disabled"""
        status = whatsapp_service_disabled.get_configuration_status()
        
        assert status['enabled'] == False
        assert status['has_token'] == False
        assert status['has_phone_id'] == False
        assert status['recipient_count'] == 0
        assert status['recipients_masked'] == []
    
    def test_format_signal_message_content(self, whatsapp_service, sample_signal):
        """Test signal message formatting contains correct information"""
        # Mock the format_signal_message function
        from backend.signals.utils import format_signal_message
        
        signal_data = {
            'symbol': sample_signal.symbol,
            'action': sample_signal.action,
            'price': sample_signal.price,
            'sl': sample_signal.sl,
            'tp': sample_signal.tp,
            'confidence': sample_signal.confidence,
            'strategy': sample_signal.strategy
        }
        
        message = format_signal_message(signal_data)
        
        assert 'EURUSD BUY @ 1.08523' in message
        assert 'SL 1.08323' in message
        assert 'TP 1.08723' in message
        assert 'conf 0.72' in message
        assert 'ema_rsi' in message
    
    @pytest.mark.asyncio
    async def test_send_message_rate_limiting(self, whatsapp_service):
        """Test rate limiting delays between messages"""
        mock_response = {
            'messages': [{'id': 'msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            with patch('asyncio.sleep') as mock_sleep:
                
                await whatsapp_service.send_bulk_message("Test message")
                
                # Should have called sleep between messages for rate limiting
                assert mock_sleep.call_count >= 1
                sleep_call = mock_sleep.call_args_list[0]
                assert sleep_call[0][0] == 0.1  # 0.1 second delay
    
    def test_whatsapp_service_environment_variables(self):
        """Test service reads environment variables correctly"""
        test_env = {
            'WHATSAPP_TOKEN': 'env_token_789',
            'WHATSAPP_PHONE_ID': 'env_phone_123',
            'WHATSAPP_TO': '+1111111111'
        }
        
        with patch.dict(os.environ, test_env):
            service = WhatsAppService()
            
            assert service.token == 'env_token_789'
            assert service.phone_id == 'env_phone_123'
            assert service.recipients == ['+1111111111']
            assert service.enabled == True
    
    def test_whatsapp_service_partial_configuration(self):
        """Test service handles partial configuration correctly"""
        # Missing phone ID
        partial_env = {
            'WHATSAPP_TOKEN': 'token_123',
            'WHATSAPP_TO': '+1234567890'
        }
        
        with patch.dict(os.environ, partial_env, clear=True):
            service = WhatsAppService()
            
            assert service.enabled == False  # Should be disabled
            assert service.token == 'token_123'
            assert service.phone_id is None
    
    @pytest.mark.asyncio
    async def test_send_signal_message_formatting(self, whatsapp_service, sample_signal):
        """Test that signal messages are formatted with proper structure"""
        mock_response = {
            'messages': [{'id': 'msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            await whatsapp_service.send_signal(sample_signal)
            
            # Get the message that was sent
            call_args = mock_post.call_args_list[0]
            message_body = call_args[1]['json']['text']['body']
            
            # Check message structure
            assert '🚨 FOREX SIGNAL' in message_body
            assert 'EURUSD BUY @ 1.08523' in message_body
            assert 'Time:' in message_body
            assert 'UTC' in message_body
            assert 'Expires:' in message_body
    
    @pytest.mark.asyncio
    async def test_api_request_headers(self, whatsapp_service):
        """Test that API requests include correct headers"""
        mock_response = {
            'messages': [{'id': 'msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            await whatsapp_service.send_test_message()
            
            # Check headers
            call_args = mock_post.call_args_list[0]
            headers = call_args[1]['headers']
            
            assert headers['Authorization'] == 'Bearer test_token_123'
            assert headers['Content-Type'] == 'application/json'
    
    @pytest.mark.asyncio 
    async def test_api_endpoint_url(self, whatsapp_service):
        """Test that correct API endpoint URL is used"""
        mock_response = {
            'messages': [{'id': 'msg_123'}]
        }
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            
            await whatsapp_service.send_test_message()
            
            # Check URL
            call_args = mock_post.call_args_list[0]
            url = call_args[0][0]  # First positional argument
            
            expected_url = 'https://graph.facebook.com/v19.0/test_phone_id_456/messages'
            assert url == expected_url

class TestWhatsAppErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.fixture
    def whatsapp_service(self):
        with patch.dict(os.environ, {
            'WHATSAPP_TOKEN': 'test_token',
            'WHATSAPP_PHONE_ID': 'test_phone', 
            'WHATSAPP_TO': '+1234567890'
        }):
            return WhatsAppService()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, whatsapp_service):
        """Test handling of request timeouts"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Timeout")
            
            result = await whatsapp_service.send_test_message()
            
            assert len(result['results']) == 1
            assert result['results'][0]['status'] == 'failed'
            assert 'Timeout' in result['results'][0]['error']
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, whatsapp_service):
        """Test handling of malformed API responses"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'unexpected': 'format'}
            mock_post.return_value = mock_response
            
            result = await whatsapp_service.send_test_message()
            
            # Should handle gracefully
            assert result['results'][0]['status'] == 'failed'
    
    @pytest.mark.asyncio
    async def test_json_decode_error(self, whatsapp_service):
        """Test handling of JSON decode errors"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response
            
            result = await whatsapp_service.send_test_message()
            
            assert result['results'][0]['status'] == 'failed'

if __name__ == "__main__":
    pytest.main([__file__])

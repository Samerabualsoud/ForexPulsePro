"""
Kill Switch Component
"""
import streamlit as st
import requests
from typing import Optional

def render_kill_switch(
    current_status: bool,
    auth_token: Optional[str] = None,
    show_details: bool = True
) -> None:
    """
    Render kill switch control component
    
    Args:
        current_status: Current kill switch status (True = enabled)
        auth_token: JWT token for authentication
        show_details: Whether to show detailed information
    """
    
    st.subheader("🚨 Emergency Kill Switch")
    
    # Status display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if current_status:
            st.error("🔴 **KILL SWITCH ACTIVE** - All signals blocked")
            st.caption("Signal generation and WhatsApp delivery are completely disabled")
        else:
            st.success("🟢 **SYSTEM ACTIVE** - Normal operation")
            st.caption("Signal generation and delivery are running normally")
    
    with col2:
        # Status indicator
        status_text = "BLOCKING" if current_status else "ACTIVE"
        st.metric("System Status", status_text)
    
    # Control buttons (admin only)
    if auth_token and st.session_state.get('user_role') == 'admin':
        st.markdown("---")
        st.subheader("🎛️ Control Panel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if current_status:
                if st.button(
                    "🟢 DISABLE Kill Switch",
                    type="primary",
                    use_container_width=True,
                    help="Resume normal signal generation and delivery"
                ):
                    if toggle_kill_switch(False, auth_token):
                        st.success("✅ Kill switch disabled - signals will resume")
                        st.rerun()
                    else:
                        st.error("❌ Failed to disable kill switch")
            else:
                if st.button(
                    "🔴 ENABLE Kill Switch",
                    type="secondary",
                    use_container_width=True,
                    help="Immediately stop all signal generation and delivery"
                ):
                    # Confirmation dialog
                    if st.session_state.get('confirm_kill_switch'):
                        if toggle_kill_switch(True, auth_token):
                            st.success("✅ Kill switch enabled - all signals blocked")
                            st.session_state.confirm_kill_switch = False
                            st.rerun()
                        else:
                            st.error("❌ Failed to enable kill switch")
                    else:
                        st.session_state.confirm_kill_switch = True
                        st.warning("⚠️ This will immediately stop all signal generation. Click again to confirm.")
        
        with col2:
            if st.button(
                "🧪 Test WhatsApp",
                use_container_width=True,
                help="Send test message to verify WhatsApp connectivity"
            ):
                if test_whatsapp_connection(auth_token):
                    st.success("✅ WhatsApp test message sent successfully")
                else:
                    st.error("❌ WhatsApp test failed - check configuration")
    
    else:
        if not auth_token:
            st.info("🔒 Admin authentication required to control kill switch")
        else:
            st.info("🔒 Admin privileges required to control kill switch")
    
    # Show impact details if requested
    if show_details:
        render_kill_switch_details(current_status)

def render_kill_switch_details(enabled: bool) -> None:
    """Render detailed information about kill switch impact"""
    
    st.markdown("---")
    
    with st.expander("ℹ️ Kill Switch Information"):
        if enabled:
            st.markdown("""
            **Current Impact:**
            - 🚫 Signal generation is completely stopped
            - 🚫 WhatsApp messages are blocked
            - 🚫 No new signals will be created
            - ✅ Existing signals remain in database
            - ✅ API endpoints remain accessible
            - ✅ Dashboard functionality is normal
            
            **When to Disable:**
            - Market conditions return to normal
            - System maintenance is complete
            - Emergency situation is resolved
            """)
        else:
            st.markdown("""
            **Normal Operation:**
            - ✅ Signal generation runs every minute
            - ✅ WhatsApp delivery is active
            - ✅ Risk management filters are active
            - ✅ All strategies are processing
            
            **When to Enable Kill Switch:**
            - High-impact news events (NFP, FOMC, etc.)
            - Unusual market volatility
            - System maintenance required
            - Emergency situations
            - API connectivity issues
            """)
        
        st.markdown("""
        **Technical Details:**
        - Kill switch takes effect immediately
        - No restart required
        - Can be toggled remotely via API
        - All admin users can control the switch
        - Status is logged for audit purposes
        """)

def toggle_kill_switch(enabled: bool, auth_token: str) -> bool:
    """
    Toggle kill switch via API
    
    Args:
        enabled: Whether to enable (True) or disable (False) kill switch
        auth_token: JWT authentication token
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        response = requests.post(
            "http://localhost:8000/api/risk/killswitch",
            json={"enabled": enabled},
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return False
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return False

def test_whatsapp_connection(auth_token: str) -> bool:
    """
    Test WhatsApp connection via API
    
    Args:
        auth_token: JWT authentication token
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        response = requests.post(
            "http://localhost:8000/api/whatsapp/test",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Show detailed results
            if result.get('status') == 'success':
                results = result.get('results', [])
                success_count = len([r for r in results if r.get('status') == 'sent'])
                total_count = len(results)
                
                if success_count == total_count:
                    return True
                else:
                    st.warning(f"Partial success: {success_count}/{total_count} messages sent")
                    return False
            else:
                return False
        else:
            st.error(f"WhatsApp test failed: {response.status_code}")
            return False
    
    except requests.exceptions.RequestException as e:
        st.error(f"WhatsApp test connection error: {e}")
        return False

def render_kill_switch_status_indicator(enabled: bool) -> None:
    """
    Render a compact kill switch status indicator for headers/sidebars
    
    Args:
        enabled: Current kill switch status
    """
    
    if enabled:
        st.error("🔴 KILL SWITCH ACTIVE")
    else:
        st.success("🟢 System Active")

def render_kill_switch_quick_toggle(
    current_status: bool,
    auth_token: Optional[str] = None
) -> None:
    """
    Render a quick toggle button for the kill switch
    
    Args:
        current_status: Current kill switch status
        auth_token: JWT authentication token
    """
    
    if not auth_token or st.session_state.get('user_role') != 'admin':
        return
    
    if current_status:
        if st.button("🟢 Resume Signals", use_container_width=True):
            if toggle_kill_switch(False, auth_token):
                st.success("Signals resumed")
                st.rerun()
    else:
        if st.button("🔴 Emergency Stop", use_container_width=True):
            if toggle_kill_switch(True, auth_token):
                st.success("Emergency stop activated")
                st.rerun()

def get_kill_switch_status() -> Optional[bool]:
    """
    Get current kill switch status from API
    
    Returns:
        True if enabled, False if disabled, None if error
    """
    
    try:
        response = requests.get("http://localhost:8000/api/risk/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('kill_switch_enabled', False)
        else:
            return None
    
    except requests.exceptions.RequestException:
        return None

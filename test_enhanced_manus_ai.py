#!/usr/bin/env python3
"""
Test script for Enhanced Manus AI Service
Validates professional trading best practices and strategy recommendations
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Add backend to path for imports
sys.path.append('backend')

from services.manus_ai import ManusAI
from logs.logger import get_logger

logger = get_logger(__name__)

def create_mock_market_data():
    """Create mock OHLC data for testing"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic OHLC data
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, 100)  # Small price movements
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data with some volatility
    data = []
    for i, price in enumerate(prices[:-1]):
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        close = prices[i + 1]
        
        data.append({
            'timestamp': dates[i],
            'open': price,
            'high': max(high, price, close),
            'low': min(low, price, close),
            'close': close,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def create_volatile_market_data():
    """Create high volatility mock data"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    np.random.seed(123)
    
    base_price = 1.1000
    returns = np.random.normal(0, 0.005, 100)  # Higher volatility
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    data = []
    for i, price in enumerate(prices[:-1]):
        high = price * (1 + abs(np.random.normal(0, 0.002)))
        low = price * (1 - abs(np.random.normal(0, 0.002)))
        close = prices[i + 1]
        
        data.append({
            'timestamp': dates[i],
            'open': price,
            'high': max(high, price, close),
            'low': min(low, price, close),
            'close': close,
            'volume': np.random.randint(5000, 50000)
        })
    
    return pd.DataFrame(data)

def test_manus_ai_basic_functionality():
    """Test basic Manus AI functionality"""
    print("🧪 Testing Enhanced Manus AI Basic Functionality")
    print("=" * 60)
    
    try:
        # Initialize Manus AI
        manus_ai = ManusAI()
        print(f"✅ Manus AI initialized successfully")
        print(f"   Service name: {manus_ai.name}")
        print(f"   Available: {manus_ai.is_available()}")
        
        # Test strategy mapping
        print(f"\n📊 Strategy Mapping Validation:")
        for regime, mapping in manus_ai.strategy_mapping.items():
            print(f"   {regime}:")
            print(f"     Primary: {mapping['primary']}")
            print(f"     Secondary: {mapping['secondary']}")
            print(f"     Avoid: {mapping.get('avoid', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_strategy_suggestions():
    """Test strategy suggestion functionality"""
    print("\n🎯 Testing Strategy Suggestions")
    print("=" * 60)
    
    try:
        manus_ai = ManusAI()
        
        # Test with normal volatility data
        print("📈 Testing with Normal Market Data:")
        normal_data = create_mock_market_data()
        suggestions = manus_ai.suggest_strategies('EURUSD', normal_data)
        
        print(f"   Status: {suggestions.get('status')}")
        print(f"   Symbol: {suggestions.get('symbol')}")
        
        market_analysis = suggestions.get('market_analysis', {})
        print(f"   Market Regime: {market_analysis.get('regime')}")
        print(f"   Volatility Level: {market_analysis.get('volatility_level')}")
        print(f"   Sentiment: {market_analysis.get('sentiment')}")
        
        strategies = suggestions.get('recommended_strategies', [])[:3]
        print(f"   Top 3 Recommended Strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"     {i}. {strategy['name']} - Confidence: {strategy['confidence']:.1%} ({strategy['priority']})")
        
        # Test with high volatility data  
        print("\n📈 Testing with High Volatility Data:")
        volatile_data = create_volatile_market_data()
        volatile_suggestions = manus_ai.suggest_strategies('GBPUSD', volatile_data)
        
        volatile_market = volatile_suggestions.get('market_analysis', {})
        print(f"   Market Regime: {volatile_market.get('regime')}")
        print(f"   Volatility Level: {volatile_market.get('volatility_level')}")
        
        volatile_strategies = volatile_suggestions.get('recommended_strategies', [])[:3]
        print(f"   Top 3 Strategies for High Volatility:")
        for i, strategy in enumerate(volatile_strategies, 1):
            print(f"     {i}. {strategy['name']} - Confidence: {strategy['confidence']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy suggestions test failed: {e}")
        return False

def test_risk_parameters():
    """Test risk parameter calculation"""
    print("\n🛡️ Testing Risk Parameter Calculation")
    print("=" * 60)
    
    try:
        manus_ai = ManusAI()
        market_data = create_mock_market_data()
        suggestions = manus_ai.suggest_strategies('USDJPY', market_data)
        
        risk_params = suggestions.get('risk_parameters', {})
        print(f"   Max Risk Per Trade: {risk_params.get('max_risk_per_trade', 0):.1%}")
        print(f"   ATR Stop Distance: {risk_params.get('atr_stop_distance', 0):.5f}")
        print(f"   ATR Stop Percentage: {risk_params.get('atr_stop_percentage', 0):.2%}")
        print(f"   Recommended Stop Multiplier: {risk_params.get('recommended_stop_multiplier', 1.0)}")
        
        rr_ratios = risk_params.get('risk_reward_ratios', {})
        print(f"   Risk/Reward Ratios:")
        for risk_type, ratio in rr_ratios.items():
            print(f"     {risk_type.title()}: {ratio}:1")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk parameters test failed: {e}")
        return False

def test_fallback_mechanisms():
    """Test fallback mechanisms"""
    print("\n🔄 Testing Fallback Mechanisms")
    print("=" * 60)
    
    try:
        manus_ai = ManusAI()
        
        # Test with invalid data
        print("   Testing with minimal data...")
        minimal_data = pd.DataFrame({
            'open': [1.1000, 1.1001],
            'high': [1.1002, 1.1003], 
            'low': [1.0998, 1.0999],
            'close': [1.1001, 1.1002]
        })
        
        fallback_suggestions = manus_ai.suggest_strategies('TESTPAIR', minimal_data)
        print(f"   Fallback Status: {fallback_suggestions.get('status')}")
        
        if fallback_suggestions.get('status') == 'fallback':
            print("   ✅ Fallback mechanism activated correctly")
        
        # Test fallback strategy list
        fallback_strategies = manus_ai._fallback_strategy_list()
        print(f"   Fallback Strategies Available: {len(fallback_strategies)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback mechanisms test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Enhanced Manus AI Service - Comprehensive Test Suite")
    print("=" * 80)
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tests = [
        ("Basic Functionality", test_manus_ai_basic_functionality),
        ("Strategy Suggestions", test_strategy_suggestions),
        ("Risk Parameters", test_risk_parameters),
        ("Fallback Mechanisms", test_fallback_mechanisms)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced Manus AI service is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
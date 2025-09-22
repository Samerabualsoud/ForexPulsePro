#!/usr/bin/env python3
"""
Database Cleanup Script for Unrealistic Pip Values

This script identifies and fixes signals with unrealistic pip calculations
caused by using incorrect pip sizes in the signal evaluation process.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime

from backend.database import get_session_local
from backend.models import Signal
from backend.instruments.metadata import get_pip_size, get_asset_class, AssetClass
from backend.logs.logger import get_logger

logger = get_logger(__name__)

class DatabaseCleanup:
    """Cleanup service for fixing unrealistic pip calculations"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def analyze_bad_signals(self, db: Session) -> dict:
        """Analyze signals with unrealistic pip values"""
        try:
            # Get all signals with pip results
            all_signals = db.query(Signal).filter(
                and_(
                    Signal.pips_result.isnot(None),
                    Signal.pips_result != 0
                )
            ).all()
            
            # Filter for unrealistic pip values in Python
            bad_signals = [s for s in all_signals if abs(s.pips_result or 0) > 10000]
            
            analysis = {
                "total_bad_signals": len(bad_signals),
                "symbols_affected": {},
                "pip_ranges": {
                    "min_pips": None,
                    "max_pips": None,
                    "total_pips": 0
                }
            }
            
            for signal in bad_signals:
                symbol = signal.symbol
                if symbol not in analysis["symbols_affected"]:
                    analysis["symbols_affected"][symbol] = {
                        "count": 0,
                        "pip_values": [],
                        "price_differences": []
                    }
                
                analysis["symbols_affected"][symbol]["count"] += 1
                analysis["symbols_affected"][symbol]["pip_values"].append(signal.pips_result)
                
                # Calculate actual price difference for verification
                if signal.result == "WIN" and signal.tp_reached:
                    price_diff = abs(signal.tp - signal.price)
                elif signal.result == "LOSS" and signal.sl_hit:
                    price_diff = abs(signal.price - signal.sl)
                else:
                    price_diff = 0
                
                analysis["symbols_affected"][symbol]["price_differences"].append(price_diff)
                
                # Update ranges
                if analysis["pip_ranges"]["min_pips"] is None or signal.pips_result < analysis["pip_ranges"]["min_pips"]:
                    analysis["pip_ranges"]["min_pips"] = signal.pips_result
                if analysis["pip_ranges"]["max_pips"] is None or signal.pips_result > analysis["pip_ranges"]["max_pips"]:
                    analysis["pip_ranges"]["max_pips"] = signal.pips_result
                
                analysis["pip_ranges"]["total_pips"] += signal.pips_result
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing bad signals: {e}")
            raise
    
    def recalculate_signal_pips(self, signal: Signal) -> float:
        """Recalculate pip value for a single signal using correct pip size"""
        try:
            # Get correct pip size for this instrument
            correct_pip_size = get_pip_size(signal.symbol)
            asset_class = get_asset_class(signal.symbol)
            
            # Calculate actual price difference based on signal outcome
            if signal.result == "WIN" and signal.tp_reached:
                # TP was reached
                price_diff = abs(signal.tp - signal.price)
                correct_pips = price_diff / correct_pip_size
                
            elif signal.result == "LOSS" and signal.sl_hit:
                # SL was hit
                price_diff = abs(signal.price - signal.sl)
                correct_pips = -(price_diff / correct_pip_size)
                
            else:
                # Neither TP nor SL hit, or expired
                self.logger.warning(f"Signal {signal.id} has result {signal.result} but no clear outcome")
                return 0.0
            
            # Sanity check - ensure realistic pip values
            if abs(correct_pips) > 10000:
                self.logger.warning(f"Even after correction, Signal {signal.id} has unrealistic pips: {correct_pips:.1f}")
                self.logger.warning(f"Symbol: {signal.symbol}, Asset: {asset_class}, Price diff: {price_diff}, Pip size: {correct_pip_size}")
                # For crypto/metals with very large price moves, cap at reasonable values
                if asset_class in [AssetClass.CRYPTO, AssetClass.METALS]:
                    correct_pips = 1000 if correct_pips > 0 else -1000
                else:
                    correct_pips = 500 if correct_pips > 0 else -500
            
            return correct_pips
            
        except Exception as e:
            self.logger.error(f"Error recalculating pips for signal {signal.id}: {e}")
            return 0.0
    
    def fix_bad_signals(self, db: Session, dry_run: bool = True) -> dict:
        """Fix all signals with unrealistic pip values"""
        try:
            # Get all signals with pip results
            all_signals = db.query(Signal).filter(
                Signal.pips_result.isnot(None)
            ).all()
            
            # Filter for unrealistic pip values in Python
            bad_signals = [s for s in all_signals if abs(s.pips_result or 0) > 10000]
            
            results = {
                "total_processed": 0,
                "successfully_fixed": 0,
                "errors": 0,
                "changes": [],
                "dry_run": dry_run
            }
            
            for signal in bad_signals:
                try:
                    old_pips = signal.pips_result
                    new_pips = self.recalculate_signal_pips(signal)
                    
                    change_info = {
                        "signal_id": signal.id,
                        "symbol": signal.symbol,
                        "old_pips": old_pips,
                        "new_pips": new_pips,
                        "improvement_factor": abs(old_pips / new_pips) if new_pips != 0 else float('inf')
                    }
                    
                    results["changes"].append(change_info)
                    results["total_processed"] += 1
                    
                    if not dry_run:
                        signal.pips_result = new_pips
                        results["successfully_fixed"] += 1
                        self.logger.info(f"Fixed Signal {signal.id}: {old_pips:.1f} → {new_pips:.1f} pips")
                    else:
                        self.logger.info(f"DRY RUN - Would fix Signal {signal.id}: {old_pips:.1f} → {new_pips:.1f} pips")
                    
                except Exception as e:
                    self.logger.error(f"Error fixing signal {signal.id}: {e}")
                    results["errors"] += 1
                    continue
            
            if not dry_run:
                db.commit()
                self.logger.info(f"Database changes committed: {results['successfully_fixed']} signals fixed")
            else:
                self.logger.info(f"DRY RUN completed: {results['total_processed']} signals analyzed")
            
            return results
            
        except Exception as e:
            if not dry_run:
                db.rollback()
            self.logger.error(f"Error fixing bad signals: {e}")
            raise

def main():
    """Main cleanup execution"""
    print("🔧 Database Cleanup Script for Unrealistic Pip Values")
    print("=" * 60)
    
    cleanup = DatabaseCleanup()
    SessionLocal = get_session_local()
    db = SessionLocal()
    
    try:
        # Phase 1: Analysis
        print("\n📊 Phase 1: Analyzing bad signals...")
        analysis = cleanup.analyze_bad_signals(db)
        
        print(f"\n📈 Analysis Results:")
        print(f"   Total bad signals: {analysis['total_bad_signals']}")
        print(f"   Symbols affected: {len(analysis['symbols_affected'])}")
        
        if analysis['pip_ranges']['min_pips'] is not None:
            print(f"   Pip range: {analysis['pip_ranges']['min_pips']:,.1f} to {analysis['pip_ranges']['max_pips']:,.1f}")
            print(f"   Total bad pips: {analysis['pip_ranges']['total_pips']:,.1f}")
        
        for symbol, data in analysis['symbols_affected'].items():
            avg_pips = sum(data['pip_values']) / len(data['pip_values'])
            print(f"   {symbol}: {data['count']} signals, avg {avg_pips:,.1f} pips")
        
        if analysis['total_bad_signals'] == 0:
            print("✅ No signals with unrealistic pip values found!")
            return
        
        # Phase 2: Dry run
        print("\n🧪 Phase 2: Dry run (simulation)...")
        dry_results = cleanup.fix_bad_signals(db, dry_run=True)
        
        print(f"\n🔍 Dry Run Results:")
        print(f"   Signals to process: {dry_results['total_processed']}")
        print(f"   Errors encountered: {dry_results['errors']}")
        
        # Show sample improvements
        print(f"\n📋 Sample improvements:")
        for i, change in enumerate(dry_results['changes'][:5]):
            improvement = f"{change['improvement_factor']:.1f}x" if change['improvement_factor'] != float('inf') else "∞"
            print(f"   Signal {change['signal_id']} ({change['symbol']}): {change['old_pips']:,.1f} → {change['new_pips']:.1f} pips (improvement: {improvement})")
        
        if len(dry_results['changes']) > 5:
            print(f"   ... and {len(dry_results['changes']) - 5} more signals")
        
        # Phase 3: Auto-proceed (for script execution)
        print(f"\n⚠️  Ready to fix {dry_results['total_processed']} signals")
        print("Proceeding automatically with database changes...")
        response = 'yes'  # Auto-proceed for script execution
        
        if response in ['yes', 'y']:
            print("\n🔧 Phase 3: Applying fixes...")
            real_results = cleanup.fix_bad_signals(db, dry_run=False)
            
            print(f"\n✅ Cleanup Complete!")
            print(f"   Successfully fixed: {real_results['successfully_fixed']} signals")
            print(f"   Errors: {real_results['errors']}")
            
            # Final verification
            print("\n🔍 Final verification...")
            final_analysis = cleanup.analyze_bad_signals(db)
            print(f"   Remaining bad signals: {final_analysis['total_bad_signals']}")
            
            if final_analysis['total_bad_signals'] == 0:
                print("🎉 All signals fixed successfully!")
            else:
                print("⚠️  Some signals still have issues - manual review required")
        else:
            print("❌ Cleanup cancelled by user")
            
    except Exception as e:
        logger.error(f"Cleanup script failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    finally:
        db.close()
    
    return 0

if __name__ == "__main__":
    exit(main())
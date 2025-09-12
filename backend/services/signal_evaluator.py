"""
Signal Outcome Evaluation Service

Tracks signal performance and determines success rates
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models import Signal
from ..database import SessionLocal
from ..logs.logger import get_logger

logger = get_logger(__name__)

class SignalEvaluator:
    """Service to evaluate signal outcomes and track success rates"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def evaluate_expired_signals(self, db: Session = None) -> dict:
        """
        Evaluate signals that have expired but not been marked as evaluated
        """
        if not db:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False
        
        try:
            now = datetime.utcnow()
            
            # Find expired signals that haven't been evaluated
            expired_signals = db.query(Signal).filter(
                and_(
                    Signal.expires_at < now,
                    Signal.result == "PENDING",
                    Signal.blocked_by_risk == False
                )
            ).all()
            
            results = {
                "evaluated_count": 0,
                "expired_count": 0,
                "success_count": 0,
                "loss_count": 0
            }
            
            for signal in expired_signals:
                # For now, mark expired signals as EXPIRED
                # In a real system, you'd check against actual market data
                signal.result = "EXPIRED"
                signal.evaluated_at = now
                signal.tp_reached = False
                signal.sl_hit = False
                results["expired_count"] += 1
                
                self.logger.info(f"Signal {signal.id} marked as EXPIRED after {signal.expires_at}")
            
            results["evaluated_count"] = len(expired_signals)
            db.commit()
            
            return results
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error evaluating expired signals: {e}")
            raise
        finally:
            if should_close:
                db.close()
    
    def simulate_signal_outcome(self, signal: Signal, db: Session) -> None:
        """
        Simulate signal outcome based on strategy confidence
        This is a simulation - in production, you'd use real market data
        """
        try:
            # Simple simulation based on confidence
            # Higher confidence = higher chance of success
            import random
            
            # Base success rate on confidence (60-90% range)
            base_success_rate = 0.6 + (signal.confidence * 0.3)
            
            # Add some randomness
            outcome_random = random.random()
            
            if outcome_random < base_success_rate:
                # SUCCESS - TP reached
                signal.result = "WIN"
                signal.tp_reached = True
                signal.sl_hit = False
                
                # Calculate simulated pips result
                # For forex, 1 pip = 0.0001 for most pairs (except JPY pairs where 1 pip = 0.01)
                # JPY pairs: 1 pip = 0.01, others: 1 pip = 0.0001
                if 'JPY' in signal.symbol:
                    pip_value = 0.01
                else:
                    pip_value = 0.0001
                
                price_diff = abs(signal.tp - signal.price)
                signal.pips_result = price_diff / pip_value
                    
            else:
                # LOSS - SL hit
                signal.result = "LOSS"
                signal.tp_reached = False
                signal.sl_hit = True
                
                # Calculate simulated pips loss
                # For forex, 1 pip = 0.0001 for most pairs (except JPY pairs where 1 pip = 0.01)
                # JPY pairs: 1 pip = 0.01, others: 1 pip = 0.0001
                if 'JPY' in signal.symbol:
                    pip_value = 0.01
                else:
                    pip_value = 0.0001
                
                price_diff = abs(signal.price - signal.sl)
                signal.pips_result = -(price_diff / pip_value)
            
            signal.evaluated_at = datetime.utcnow()
            db.commit()
            
            self.logger.info(f"Signal {signal.id} evaluated: {signal.result} ({signal.pips_result:.1f} pips)")
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error simulating signal outcome: {e}")
            raise
    
    def get_success_rate_stats(self, db: Session = None, days: int = 30) -> dict:
        """
        Get success rate statistics for the specified period
        """
        if not db:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False
        
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Get all evaluated signals
            total_signals = db.query(Signal).filter(
                and_(
                    Signal.issued_at >= since_date,
                    Signal.result != "PENDING",
                    Signal.blocked_by_risk == False
                )
            ).count()
            
            # Get successful signals (TP reached)
            successful_signals = db.query(Signal).filter(
                and_(
                    Signal.issued_at >= since_date,
                    Signal.result == "WIN",
                    Signal.blocked_by_risk == False
                )
            ).count()
            
            # Get losing signals
            losing_signals = db.query(Signal).filter(
                and_(
                    Signal.issued_at >= since_date,
                    Signal.result == "LOSS",
                    Signal.blocked_by_risk == False
                )
            ).count()
            
            # Get expired signals
            expired_signals = db.query(Signal).filter(
                and_(
                    Signal.issued_at >= since_date,
                    Signal.result == "EXPIRED",
                    Signal.blocked_by_risk == False
                )
            ).count()
            
            # Calculate success rate
            success_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0
            
            # Get total pips
            pips_query = db.query(Signal).filter(
                and_(
                    Signal.issued_at >= since_date,
                    Signal.result != "PENDING",
                    Signal.pips_result.isnot(None),
                    Signal.blocked_by_risk == False
                )
            ).all()
            
            total_pips = sum(signal.pips_result for signal in pips_query if signal.pips_result)
            
            return {
                "total_signals": total_signals,
                "successful_signals": successful_signals,
                "losing_signals": losing_signals,
                "expired_signals": expired_signals,
                "success_rate": round(success_rate, 2),
                "total_pips": round(total_pips, 1),
                "avg_pips_per_trade": round(total_pips / total_signals, 1) if total_signals > 0 else 0,
                "days": days
            }
            
        except Exception as e:
            self.logger.error(f"Error getting success rate stats: {e}")
            return {
                "total_signals": 0,
                "successful_signals": 0,
                "losing_signals": 0,
                "expired_signals": 0,
                "success_rate": 0,
                "total_pips": 0,
                "avg_pips_per_trade": 0,
                "days": days
            }
        finally:
            if should_close:
                db.close()

# Global evaluator instance
evaluator = SignalEvaluator()
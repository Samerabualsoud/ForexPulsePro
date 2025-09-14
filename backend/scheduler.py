"""
Signal Generation Scheduler
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import asyncio
import threading
from sqlalchemy.orm import sessionmaker

from .database import engine
from .signals.engine import SignalEngine
from .services.signal_evaluator import evaluator
from .logs.logger import get_logger

logger = get_logger(__name__)

class SignalScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone='UTC')
        self.signal_engine = SignalEngine()
        self.SessionLocal = sessionmaker(bind=engine)
        
    def start(self):
        """Start the scheduler"""
        try:
            # Crypto-only configuration with AI-optimized 5-minute frequency
            # High-frequency analysis for crypto volatility capture
            self.scheduler.add_job(
                func=self._run_crypto_signal_generation,
                trigger=IntervalTrigger(minutes=5),  # AI-recommended for crypto
                id='crypto_signal_generation', 
                name='Generate Crypto Signals',
                replace_existing=True
            )
            
            # Add job to evaluate signals every minute (offset by 30 seconds)
            self.scheduler.add_job(
                func=self._run_signal_evaluation,
                trigger=IntervalTrigger(minutes=1),
                id='signal_evaluation',
                name='Evaluate Crypto Signals',
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info("Signal scheduler started successfully")
            
            # Run crypto signal generation immediately on start
            self._run_crypto_signal_generation()
            # Run evaluation 10 seconds after to allow signals to be created first
            self.scheduler.add_job(
                func=self._run_signal_evaluation,
                trigger='date',
                run_date=datetime.utcnow().replace(microsecond=0) + timedelta(seconds=10),
                id='initial_evaluation',
                name='Initial Signal Evaluation'
            )
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("Signal scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
    
    # Removed _run_signal_generation - crypto-only configuration
    
    def _run_crypto_signal_generation(self):
        """Run crypto signal generation with AI-optimized 5-minute frequency"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._generate_crypto_signals())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async)
        thread.start()
    
    # Removed _run_metals_signal_generation - crypto-only configuration
    
    # Removed _generate_signals - using crypto-specific generation only
    
    async def _generate_crypto_signals(self):
        """Generate signals specifically for crypto pairs with high-frequency analysis"""
        db = self.SessionLocal()
        try:
            # Required crypto vs USD pairs only
            crypto_symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'ADAUSD', 'SOLUSD']
            
            for symbol in crypto_symbols:
                try:
                    await self.signal_engine.process_symbol(symbol, db)
                    logger.debug(f"Processed crypto signals for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing crypto {symbol}: {e}")
            
            logger.info(f"Crypto signal generation completed at {datetime.utcnow()}")
            
        except Exception as e:
            logger.error(f"Crypto signal generation failed: {e}")
        finally:
            db.close()
    
    # Removed _generate_metals_signals - crypto-only configuration
    
    def _run_signal_evaluation(self):
        """Run signal evaluation in a separate thread"""
        def run_evaluation():
            db = self.SessionLocal()
            try:
                # First, evaluate expired signals
                results = evaluator.evaluate_expired_signals(db)
                
                # Then simulate outcomes for newly expired signals
                if results['expired_count'] > 0:
                    # Get the signals that were just marked as EXPIRED and simulate their outcomes
                    from .models import Signal
                    expired_signals = db.query(Signal).filter(
                        Signal.result == "EXPIRED",
                        Signal.evaluated_at.isnot(None)
                    ).order_by(Signal.evaluated_at.desc()).limit(results['expired_count']).all()
                    
                    for signal in expired_signals:
                        try:
                            evaluator.simulate_signal_outcome(signal, db)
                        except Exception as e:
                            logger.error(f"Error simulating outcome for signal {signal.id}: {e}")
                
                logger.info(f"Signal evaluation completed: {results}")
                
            except Exception as e:
                logger.error(f"Signal evaluation failed: {e}")
            finally:
                db.close()
        
        thread = threading.Thread(target=run_evaluation)
        thread.start()
    
    def get_status(self):
        """Get scheduler status"""
        return {
            "running": self.scheduler.running,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ]
        }

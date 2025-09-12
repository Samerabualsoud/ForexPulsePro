"""
Signal Generation Scheduler
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import asyncio
import threading
from sqlalchemy.orm import sessionmaker

from .database import engine
from .signals.engine import SignalEngine
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
            # Add job to run every minute
            self.scheduler.add_job(
                func=self._run_signal_generation,
                trigger=IntervalTrigger(minutes=1),
                id='signal_generation',
                name='Generate Forex Signals',
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info("Signal scheduler started successfully")
            
            # Run immediately on start
            self._run_signal_generation()
            
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
    
    def _run_signal_generation(self):
        """Run signal generation in a separate thread"""
        def run_async():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._generate_signals())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async)
        thread.start()
    
    async def _generate_signals(self):
        """Generate signals for all symbols"""
        db = self.SessionLocal()
        try:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            
            for symbol in symbols:
                try:
                    await self.signal_engine.process_symbol(symbol, db)
                    logger.debug(f"Processed signals for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            logger.info(f"Signal generation completed at {datetime.utcnow()}")
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
        finally:
            db.close()
    
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

"""
Signal Generation Scheduler
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import asyncio
import threading
from sqlalchemy.orm import sessionmaker

from .database import get_engine
from .signals.engine import SignalEngine, is_forex_market_open
from .services.signal_evaluator import evaluator
from .logs.logger import get_logger

logger = get_logger(__name__)

class SignalScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone='UTC')
        self.signal_engine = SignalEngine()
        self.SessionLocal = sessionmaker(bind=get_engine())
        
    def start(self):
        """Start the scheduler"""
        try:
            # Major forex & crypto pairs configuration with 1-minute frequency
            # Professional forex and crypto trading analysis for comprehensive market coverage
            self.scheduler.add_job(
                func=self._run_forex_signal_generation,
                trigger=IntervalTrigger(minutes=1),  # User requested 1-minute frequency
                id='forex_signal_generation', 
                name='Generate Forex & Crypto Signals',
                replace_existing=True
            )
            
            # Add job to evaluate signals every minute (offset by 30 seconds)
            self.scheduler.add_job(
                func=self._run_signal_evaluation,
                trigger=IntervalTrigger(minutes=1),
                id='signal_evaluation',
                name='Evaluate Signals',
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info("Signal scheduler started successfully")
            
            # Run forex signal generation immediately on start
            self._run_forex_signal_generation()
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
    
    # Removed _run_signal_generation - forex-only configuration
    
    def _run_forex_signal_generation(self):
        """Run forex signal generation with institutional 15-minute frequency"""
        def run_async():
            try:
                # Use asyncio.run() which properly manages event loop lifecycle
                asyncio.run(self._generate_signals())
            except Exception as e:
                logger.error(f"Error in signal generation thread: {e}")
        
        thread = threading.Thread(target=run_async)
        thread.daemon = True  # Make thread daemon so it doesn't block shutdown
        thread.start()
    
    # Removed _run_metals_signal_generation - forex-only configuration
    
    # Removed _generate_signals - using forex-specific generation only
    
    async def _generate_signals(self):
        """Generate signals for major forex pairs and cryptocurrency pairs with comprehensive analysis"""
        db = self.SessionLocal()
        try:
            # Check forex market hours before processing
            forex_market_open = is_forex_market_open()
            logger.info(f"🕐 Forex market status: {'OPEN' if forex_market_open else 'CLOSED'}")
            # All major forex pairs for comprehensive trading coverage
            forex_symbols = [
                # USD Major Pairs
                'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
                # EUR Cross Pairs
                'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
                # GBP Cross Pairs
                'GBPJPY', 'GBPAUD', 'GBPCHF', 'GBPCAD',
                # JPY Cross Pairs
                'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY',
                # Other Major Cross Pairs
                'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'NZDCAD', 'NZDCHF'
            ]
            
            # All major cryptocurrency pairs for 24/7 crypto trading coverage
            crypto_symbols = [
                'BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD',
                'SOLUSD', 'BNBUSD', 'XRPUSD', 'MATICUSD'
            ]
            
            # Metals & Oil symbols for commodities trading
            metals_oil_symbols = [
                'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD',  # Gold, Silver, Platinum, Palladium
                'USOIL', 'UKOUSD', 'WTIUSD', 'XBRUSD'    # Oil futures
            ]
            
            # Process forex pairs (only if market is open)
            forex_processed = 0
            if forex_market_open:
                logger.info(f"🔄 Starting forex processing: {len(forex_symbols)} symbols")
                for i, symbol in enumerate(forex_symbols):
                    try:
                        logger.debug(f"Processing forex {i+1}/{len(forex_symbols)}: {symbol}")
                        await asyncio.wait_for(
                            self.signal_engine.process_symbol(symbol, db),
                            timeout=30  # 30 second timeout per symbol
                        )
                        forex_processed += 1
                        logger.debug(f"✅ Processed forex signals for {symbol} ({forex_processed}/{len(forex_symbols)})")
                    except asyncio.TimeoutError:
                        logger.warning(f"⏰ Timeout processing forex {symbol} - continuing to next symbol")
                    except Exception as e:
                        logger.error(f"❌ Error processing forex {symbol}: {e}")
                logger.info(f"📊 Forex processing completed: {forex_processed}/{len(forex_symbols)} symbols processed")
            else:
                logger.info(f"⏸️ Skipping forex processing - market is closed")
            
            # Process crypto pairs (24/7 availability)
            logger.info(f"🪙 Starting crypto processing: {len(crypto_symbols)} symbols")
            crypto_processed = 0
            for i, symbol in enumerate(crypto_symbols):
                try:
                    logger.debug(f"Processing crypto {i+1}/{len(crypto_symbols)}: {symbol}")
                    await asyncio.wait_for(
                        self.signal_engine.process_symbol(symbol, db),
                        timeout=30  # 30 second timeout per symbol
                    )
                    crypto_processed += 1
                    logger.debug(f"✅ Processed crypto signals for {symbol} ({crypto_processed}/{len(crypto_symbols)})")
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout processing crypto {symbol} - continuing to next symbol")
                except Exception as e:
                    logger.error(f"❌ Error processing crypto {symbol}: {e}")
            logger.info(f"📊 Crypto processing completed: {crypto_processed}/{len(crypto_symbols)} symbols processed")
            
            # Process metals & oil pairs (extended hours availability)
            logger.info(f"🥇 Starting metals/oil processing: {len(metals_oil_symbols)} symbols")
            metals_processed = 0
            for i, symbol in enumerate(metals_oil_symbols):
                try:
                    logger.debug(f"Processing metals/oil {i+1}/{len(metals_oil_symbols)}: {symbol}")
                    await asyncio.wait_for(
                        self.signal_engine.process_symbol(symbol, db),
                        timeout=30  # 30 second timeout per symbol
                    )
                    metals_processed += 1
                    logger.debug(f"✅ Processed metals/oil signals for {symbol} ({metals_processed}/{len(metals_oil_symbols)})")
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout processing metals/oil {symbol} - continuing to next symbol")
                except Exception as e:
                    logger.error(f"❌ Error processing metals/oil {symbol}: {e}")
            logger.info(f"📊 Metals/oil processing completed: {metals_processed}/{len(metals_oil_symbols)} symbols processed")
            
            total_processed = forex_processed + crypto_processed + metals_processed
            total_symbols = len(forex_symbols) + len(crypto_symbols) + len(metals_oil_symbols)
            logger.info(f"🏁 SIGNAL GENERATION COMPLETED: {total_processed}/{total_symbols} symbols processed")
            logger.info(f"📈 Breakdown: {forex_processed}/{len(forex_symbols)} forex + {crypto_processed}/{len(crypto_symbols)} crypto + {metals_processed}/{len(metals_oil_symbols)} metals/oil")
            logger.info(f"⏰ Completion time: {datetime.utcnow()}")
            
        except Exception as e:
            logger.error(f"Forex & Crypto signal generation failed: {e}")
        finally:
            db.close()
    
    # Removed _generate_metals_signals - forex-only configuration
    
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

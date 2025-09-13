"""
Comprehensive Sentiment Analysis Service for Financial News

This service provides multiple sentiment analysis methods optimized for financial text:
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- TextBlob (Rule-based sentiment analysis)
- Custom Financial Keywords Analysis
"""

from typing import Dict, List, Optional, Tuple, Union
import re
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from ..logs.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Comprehensive sentiment analysis service for financial news"""
    
    def __init__(self):
        """Initialize the sentiment analyzer with all required models"""
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize VADER analyzer
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize VADER analyzer: {e}")
            self.vader_analyzer = None
        
        # Financial sentiment keywords
        self.bullish_keywords = {
            # Strong positive
            'surge', 'soar', 'rally', 'boom', 'bullish', 'breakout', 'strength', 'momentum',
            'uptrend', 'buy', 'accumulate', 'outperform', 'record high', 'all-time high',
            'gains', 'profit', 'earnings beat', 'revenue growth', 'strong', 'robust',
            'expansion', 'upgrade', 'bull market', 'upturn', 'recovery', 'rebound',
            'optimistic', 'positive', 'confidence', 'boost', 'rise', 'increase',
            'growth', 'improve', 'exceed', 'beat expectations', 'overweight', 'recommend',
            
            # Moderate positive
            'steady', 'stable', 'hold', 'maintain', 'supported', 'resilient',
            'opportunity', 'potential', 'favorable', 'benefit', 'advantage'
        }
        
        self.bearish_keywords = {
            # Strong negative
            'crash', 'plunge', 'dive', 'collapse', 'bearish', 'breakdown', 'weakness',
            'downtrend', 'sell', 'dump', 'underperform', 'record low', 'losses', 'loss',
            'earnings miss', 'revenue decline', 'weak', 'fragile', 'contraction',
            'downgrade', 'bear market', 'downturn', 'recession', 'decline', 'fall',
            'pessimistic', 'negative', 'concern', 'worry', 'drop', 'decrease',
            'shrink', 'miss', 'below expectations', 'underweight', 'avoid',
            
            # Moderate negative
            'caution', 'risk', 'uncertainty', 'volatile', 'challenge', 'pressure',
            'headwind', 'obstacle', 'difficulty', 'struggle', 'concern'
        }
        
        # Currency and forex specific terms
        self.forex_bullish = {
            'strengthen', 'appreciate', 'gain ground', 'upward pressure', 'buying interest',
            'safe haven', 'flight to quality', 'central bank support', 'rate hike',
            'hawkish', 'tighten policy', 'intervention buying'
        }
        
        self.forex_bearish = {
            'weaken', 'depreciate', 'lose ground', 'downward pressure', 'selling pressure',
            'risk off', 'dovish', 'rate cut', 'loose policy', 'intervention selling',
            'capital outflow', 'devaluation'
        }
        
        # Weight multipliers for different types of keywords
        self.keyword_weights = {
            'strong_bullish': 0.8,
            'moderate_bullish': 0.4,
            'strong_bearish': -0.8,
            'moderate_bearish': -0.4,
            'forex_bullish': 0.6,
            'forex_bearish': -0.6
        }
        
        self.logger.info("Sentiment analyzer initialized with financial keyword dictionaries")
    
    def analyze_sentiment(self, text: str, method: str = "combined") -> Dict:
        """
        Analyze sentiment using specified method or combined approach
        
        Args:
            text: Text to analyze
            method: Analysis method ('vader', 'textblob', 'keywords', 'combined')
            
        Returns:
            Dict with score (-1 to 1), confidence (0 to 1), label, and method details
        """
        if not text or not isinstance(text, str):
            return self._empty_result("Invalid input text")
        
        text_clean = self._preprocess_text(text)
        
        try:
            if method == "vader":
                return self._analyze_vader(text_clean)
            elif method == "textblob":
                return self._analyze_textblob(text_clean)
            elif method == "keywords":
                return self._analyze_keywords(text_clean)
            elif method == "combined":
                return self._analyze_combined(text_clean)
            else:
                self.logger.warning(f"Unknown method '{method}', using combined approach")
                return self._analyze_combined(text_clean)
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return self._empty_result(f"Analysis error: {str(e)}")
    
    def _analyze_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        if not self.vader_analyzer:
            return self._empty_result("VADER analyzer not available")
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # VADER returns compound score between -1 and 1
            compound_score = scores['compound']
            
            # Calculate confidence based on the strength of positive/negative scores
            confidence = max(scores['pos'], scores['neg'])
            
            # Determine label
            if compound_score >= 0.05:
                label = "POSITIVE"
            elif compound_score <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            return {
                "score": round(compound_score, 3),
                "confidence": round(confidence, 3),
                "label": label,
                "method": "vader",
                "details": {
                    "positive": round(scores['pos'], 3),
                    "neutral": round(scores['neu'], 3),
                    "negative": round(scores['neg'], 3),
                    "compound": round(compound_score, 3)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"VADER analysis failed: {e}")
            return self._empty_result(f"VADER error: {str(e)}")
    
    def _analyze_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            
            # TextBlob returns polarity between -1 and 1
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Use subjectivity as confidence measure (more subjective = more confident)
            confidence = subjectivity
            
            # Determine label
            if polarity > 0.1:
                label = "POSITIVE"
            elif polarity < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            return {
                "score": round(polarity, 3),
                "confidence": round(confidence, 3),
                "label": label,
                "method": "textblob",
                "details": {
                    "polarity": round(polarity, 3),
                    "subjectivity": round(subjectivity, 3)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"TextBlob analysis failed: {e}")
            return self._empty_result(f"TextBlob error: {str(e)}")
    
    def _analyze_keywords(self, text: str) -> Dict:
        """Analyze sentiment using custom financial keywords"""
        try:
            text_lower = text.lower()
            
            # Count keyword matches
            bullish_matches = self._count_keyword_matches(text_lower, self.bullish_keywords)
            bearish_matches = self._count_keyword_matches(text_lower, self.bearish_keywords)
            forex_bull_matches = self._count_keyword_matches(text_lower, self.forex_bullish)
            forex_bear_matches = self._count_keyword_matches(text_lower, self.forex_bearish)
            
            # Calculate weighted sentiment score
            total_score = 0
            total_matches = 0
            
            if bullish_matches > 0:
                total_score += bullish_matches * self.keyword_weights['strong_bullish']
                total_matches += bullish_matches
            
            if bearish_matches > 0:
                total_score += bearish_matches * self.keyword_weights['strong_bearish']
                total_matches += bearish_matches
            
            if forex_bull_matches > 0:
                total_score += forex_bull_matches * self.keyword_weights['forex_bullish']
                total_matches += forex_bull_matches
                
            if forex_bear_matches > 0:
                total_score += forex_bear_matches * self.keyword_weights['forex_bearish']
                total_matches += forex_bear_matches
            
            # Normalize score to -1 to 1 range
            if total_matches > 0:
                # Use tanh to ensure score stays within bounds
                import math
                score = math.tanh(total_score / total_matches)
                confidence = min(total_matches / 10.0, 1.0)  # More matches = higher confidence
            else:
                score = 0.0
                confidence = 0.0
            
            # Determine label
            if score > 0.1:
                label = "POSITIVE"
            elif score < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            return {
                "score": round(score, 3),
                "confidence": round(confidence, 3),
                "label": label,
                "method": "keywords",
                "details": {
                    "bullish_matches": bullish_matches,
                    "bearish_matches": bearish_matches,
                    "forex_bullish_matches": forex_bull_matches,
                    "forex_bearish_matches": forex_bear_matches,
                    "total_matches": total_matches,
                    "raw_score": round(total_score, 3)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Keyword analysis failed: {e}")
            return self._empty_result(f"Keywords error: {str(e)}")
    
    def _analyze_combined(self, text: str) -> Dict:
        """Analyze sentiment using combined approach with weighted average"""
        try:
            # Get results from all methods
            vader_result = self._analyze_vader(text)
            textblob_result = self._analyze_textblob(text)
            keywords_result = self._analyze_keywords(text)
            
            # Define weights for each method
            weights = {
                'vader': 0.4,      # Good for general sentiment
                'textblob': 0.25,  # Additional general sentiment validation
                'keywords': 0.35   # Financial-specific sentiment
            }
            
            # Calculate weighted average score
            total_weight = 0
            weighted_score = 0
            
            results = [
                (vader_result, weights['vader']),
                (textblob_result, weights['textblob']),
                (keywords_result, weights['keywords'])
            ]
            
            valid_results = []
            for result, weight in results:
                if result['confidence'] > 0:  # Only include results with confidence
                    weighted_score += result['score'] * weight * result['confidence']
                    total_weight += weight * result['confidence']
                    valid_results.append(result)
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                
                # Calculate confidence as average of individual confidences
                avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
                
                # Boost confidence if multiple methods agree
                if len(valid_results) > 1:
                    agreement_bonus = 0.1 * (len(valid_results) - 1)
                    avg_confidence = min(avg_confidence + agreement_bonus, 1.0)
            else:
                final_score = 0.0
                avg_confidence = 0.0
            
            # Determine label
            if final_score > 0.05:
                label = "POSITIVE"
            elif final_score < -0.05:
                label = "NEGATIVE" 
            else:
                label = "NEUTRAL"
            
            return {
                "score": round(final_score, 3),
                "confidence": round(avg_confidence, 3),
                "label": label,
                "method": "combined",
                "details": {
                    "vader": vader_result,
                    "textblob": textblob_result,
                    "keywords": keywords_result,
                    "weights_used": weights,
                    "valid_methods": len(valid_results)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Combined analysis failed: {e}")
            return self._empty_result(f"Combined error: {str(e)}")
    
    def analyze_batch(self, texts: List[str], method: str = "combined") -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        if not texts:
            return []
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text, method)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch analysis failed for text {i}: {e}")
                error_result = self._empty_result(f"Batch error: {str(e)}")
                error_result['batch_index'] = i
                results.append(error_result)
        
        return results
    
    def get_sentiment_summary(self, texts: List[str], method: str = "combined") -> Dict:
        """Get aggregated sentiment summary for multiple texts"""
        if not texts:
            return {"error": "No texts provided"}
        
        results = self.analyze_batch(texts, method)
        
        # Calculate aggregated metrics
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')
        neutral_count = sum(1 for r in results if r['label'] == 'NEUTRAL')
        
        scores = [r['score'] for r in results if 'score' in r]
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Overall sentiment
        if avg_score > 0.05:
            overall_sentiment = "POSITIVE"
        elif avg_score < -0.05:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": round(avg_score, 3),
            "average_confidence": round(avg_confidence, 3),
            "total_texts": len(texts),
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "method": method,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common financial abbreviations
        abbreviations = {
            'IPO': 'initial public offering',
            'M&A': 'merger and acquisition',
            'CEO': 'chief executive officer',
            'CFO': 'chief financial officer',
            'YoY': 'year over year',
            'QoQ': 'quarter over quarter'
        }
        
        for abbr, expansion in abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _count_keyword_matches(self, text: str, keywords: set) -> int:
        """Count matches for a set of keywords in text"""
        count = 0
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(keyword)}\b'
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            count += matches
        return count
    
    def _empty_result(self, error_msg: str = "No analysis performed") -> Dict:
        """Return empty result with error message"""
        return {
            "score": 0.0,
            "confidence": 0.0,
            "label": "NEUTRAL",
            "method": "error",
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def health_check(self) -> Dict:
        """Check if all sentiment analysis components are working"""
        health = {
            "vader_available": self.vader_analyzer is not None,
            "textblob_available": True,  # Always available if imported
            "keywords_loaded": len(self.bullish_keywords) + len(self.bearish_keywords) > 0,
            "service_ready": True
        }
        
        # Test with simple text
        try:
            test_result = self.analyze_sentiment("The market is performing well with strong gains.")
            health["test_analysis_working"] = test_result['confidence'] > 0
        except:
            health["test_analysis_working"] = False
            health["service_ready"] = False
        
        return health


# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()
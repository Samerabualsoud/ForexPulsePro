# Forex Signal Dashboard

## Overview

A production-ready Forex trading signal dashboard that automatically generates trading signals using technical analysis, applies comprehensive risk management controls, and delivers signals via WhatsApp Cloud API. The system features a Streamlit frontend for dashboard interaction, FastAPI backend for REST API functionality, and APScheduler for real-time signal processing every minute. Built with three integrated trading strategies (EMA+RSI, Donchian+ATR, Mean Reversion+Bollinger Bands) and includes JWT authentication, role-based access control, and comprehensive logging.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application with component-based architecture
- **Pages**: Overview dashboard, strategy configuration, risk management, API keys, logs viewer, and documentation
- **Components**: Reusable components for signal tables, kill switches, log viewers, and strategy forms
- **State Management**: Streamlit session state for authentication and caching
- **API Integration**: Direct HTTP requests to FastAPI backend with smart fallback to demo data for published apps

### Backend Architecture
- **Framework**: FastAPI with asynchronous request handling
- **Authentication**: JWT-based authentication with role-based access (admin/viewer)
- **API Design**: RESTful endpoints following standard HTTP conventions
- **Middleware**: CORS middleware for cross-origin requests
- **Error Handling**: Structured error responses with appropriate HTTP status codes

### Data Processing
- **Signal Engine**: APScheduler-based system running signal analysis every 5 minutes (reduced from 1 minute to prevent noise)
- **Strategy Pattern**: Pluggable strategy architecture with seven advanced trading strategies
- **Cross-Strategy Consensus**: Intelligent conflict resolution requiring 80%+ confidence for opposing signals
- **Signal Cooldown**: 15-minute cooldown per symbol to prevent whipsaw trading
- **Technical Indicators**: TA-Lib integration for professional-grade technical analysis
- **Data Providers**: Pluggable provider system with MockDataProvider (CSV/synthetic data) and AlphaVantageProvider stub

### Database Architecture
- **Primary**: PostgreSQL with SQLAlchemy ORM
- **Fallback**: SQLite for development/testing environments
- **Models**: User, Signal, Strategy, RiskConfig entities with proper relationships
- **Session Management**: Connection pooling and session lifecycle management

### Risk Management System
- **Kill Switch**: Global emergency stop for all signal generation
- **Daily Loss Limits**: Configurable maximum daily loss thresholds
- **Volatility Guard**: ATR-based signal filtering during high volatility periods
- **Signal Quality Control**: Confidence-based filtering and validation

### Monitoring and Observability
- **Logging**: Structured JSON logging with structlog and rotating file handlers
- **Metrics**: Prometheus metrics for signal generation, WhatsApp delivery, and system performance
- **Health Checks**: System status monitoring with detailed health endpoints

### Deployment Architecture
- **Containerization**: Docker support with multi-service docker-compose setup
- **Process Management**: Background thread management for scheduler and FastAPI server
- **Environment Configuration**: Environment variable-based configuration for different deployment environments

## External Dependencies

### Third-Party Services
- **WhatsApp Cloud API**: Signal delivery via Facebook Graph API v19.0 with message templates and delivery confirmation
- **Alpha Vantage API**: Market data provider (stubbed implementation, configurable via ALPHAVANTAGE_KEY)

### Python Libraries
- **FastAPI**: Modern web framework for REST API development
- **Streamlit**: Interactive web dashboard framework
- **SQLAlchemy**: Database ORM with PostgreSQL and SQLite support
- **APScheduler**: Background job scheduling for signal generation
- **TA-Lib**: Technical analysis library for trading indicators
- **pandas/numpy**: Data manipulation and numerical computing
- **PyJWT**: JSON Web Token implementation for authentication
- **structlog**: Structured logging with JSON output
- **requests**: HTTP client for external API integration
- **prometheus_client**: Metrics collection and monitoring

### Database Systems
- **PostgreSQL**: Primary production database (configurable via DATABASE_URL)
- **SQLite**: Development and fallback database option

### Development and Testing
- **pytest**: Test framework with API and engine testing
- **Docker**: Containerization for consistent deployment
- **GitHub Actions**: CI/CD pipeline for linting, testing, and building (referenced in specs)

### Data Sources
- **Mock Data Provider**: CSV-based historical data or synthetic data generation for EURUSD, GBPUSD, USDJPY
- **Alpha Vantage**: External market data API (optional, disabled by default)
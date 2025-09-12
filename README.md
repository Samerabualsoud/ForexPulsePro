# Forex Signal Dashboard

A production-ready Forex Signal Dashboard that automatically generates trading signals, applies risk management, and delivers signals via WhatsApp Cloud API. Built with Streamlit frontend and FastAPI backend.

![Dashboard Overview](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### 📊 Signal Generation
- **Three Built-in Strategies**: EMA+RSI, Donchian+ATR, Mean Reversion+Bollinger Bands
- **1-minute Signal Processing**: APScheduler runs signal analysis every minute
- **Technical Indicators**: Powered by pandas, numpy, and TA-Lib
- **Configurable Parameters**: Per-symbol strategy configuration with real-time updates

### 📱 WhatsApp Integration
- **WhatsApp Cloud API**: Automatic signal delivery to configured recipients
- **Message Templates**: Professional signal formatting with SL/TP levels
- **Delivery Confirmation**: Track message delivery status and errors
- **Bulk Messaging**: Send signals to multiple recipients simultaneously

### 🛡️ Risk Management
- **Kill Switch**: Emergency stop for all signal generation
- **Daily Loss Limits**: Configurable maximum daily loss thresholds
- **Volatility Guard**: ATR-based signal filtering during high volatility
- **Signal Quality Control**: Confidence-based filtering and validation

### 🎛️ Multi-Page Dashboard
- **Overview**: Real-time signal monitoring and system status
- **Strategies**: Configure trading strategies per symbol
- **Risk Management**: Control risk parameters and emergency stops
- **API Keys**: Manage WhatsApp and database configurations
- **Logs**: View system events and signal history
- **Documentation**: Complete API and setup documentation

### 🔧 Technical Features
- **JWT Authentication**: Role-based access (admin/viewer)
- **Database Support**: PostgreSQL with SQLite fallback
- **REST API**: Complete API for external integration
- **Prometheus Metrics**: System monitoring and performance tracking
- **Structured Logging**: JSON logging with rotation
- **Docker Support**: Ready for containerized deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [WhatsApp Setup](#whatsapp-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Trading Strategies](#trading-strategies)
- [Risk Management](#risk-management)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) PostgreSQL database
- (Optional) Docker for containerized deployment

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd forex-signal-dashboard

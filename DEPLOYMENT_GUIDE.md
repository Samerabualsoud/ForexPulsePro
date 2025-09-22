# ForexPulsePro Deployment Guide

## Digital Ocean App Platform Deployment

### Prerequisites
- Digital Ocean account
- GitHub repository: https://github.com/Samerabualsoud/ForexPulsePro

### Deployment Steps

#### Option 1: Using Digital Ocean App Platform (Recommended)

1. **Login to Digital Ocean**
   - Go to https://cloud.digitalocean.com/
   - Navigate to "Apps" section

2. **Create New App**
   - Click "Create App"
   - Choose "GitHub" as source
   - Select repository: `Samerabualsoud/ForexPulsePro`
   - Branch: `main`

3. **Configure App Settings**
   - **Name**: `forexpulsepro`
   - **Region**: Choose closest to your users
   - **Plan**: Basic ($5/month recommended for testing)

4. **Environment Variables**
   Add these environment variables in the App Platform:
   ```
   BACKEND_URL=http://localhost:8080
   BACKEND_HOST=localhost
   BACKEND_PORT=8080
   FRONTEND_PORT=5000
   JWT_SECRET=your_secure_jwt_secret_change_in_production
   LOG_LEVEL=INFO
   CORS_ORIGINS=*
   ```

5. **Build Configuration**
   - Build Command: `pip install -r requirements.txt`
   - Run Command: See the startup script in `.do/app.yaml`

6. **Deploy**
   - Click "Create Resources"
   - Wait for deployment to complete (5-10 minutes)

#### Option 2: Using Docker (Alternative)

1. **Build Docker Image**
   ```bash
   docker build -t forexpulsepro .
   ```

2. **Run Container**
   ```bash
   docker run -p 5000:5000 -p 8080:8080 forexpulsepro
   ```

3. **Deploy to Digital Ocean Container Registry**
   ```bash
   # Tag and push to DO registry
   docker tag forexpulsepro registry.digitalocean.com/your-registry/forexpulsepro
   docker push registry.digitalocean.com/your-registry/forexpulsepro
   ```

### Post-Deployment Configuration

#### 1. Environment Variables Setup
After deployment, configure these additional variables for full functionality:

**WhatsApp Integration** (Optional):
```
WHATSAPP_TOKEN=your_whatsapp_cloud_api_token
WHATSAPP_PHONE_ID=your_phone_number_id
WHATSAPP_TO=+1234567890,+0987654321
```

**Database** (Optional - defaults to SQLite):
```
DATABASE_URL=postgresql://username:password@hostname:port/database
```

**Market Data Providers** (Optional):
```
POLYGON_API_KEY=your_polygon_api_key
ALPHAVANTAGE_KEY=your_alphavantage_key
FINNHUB_API_KEY=your_finnhub_key
```

#### 2. Access Your Application
- Frontend (Streamlit): `https://your-app-name.ondigitalocean.app`
- Backend API: `https://your-app-name.ondigitalocean.app:8080/docs`

#### 3. Default Login Credentials
- **Admin**: username: `admin`, password: `admin123`
- **Viewer**: username: `viewer`, password: `viewer123`

**⚠️ IMPORTANT**: Change these default credentials immediately after first login!

### Troubleshooting

#### Common Issues

1. **TA-Lib Installation Errors**
   - The Dockerfile handles TA-Lib installation automatically
   - If issues persist, check the build logs for compilation errors

2. **Port Binding Issues**
   - Ensure both ports 5000 and 8080 are properly configured
   - Check that the startup script runs both services

3. **Environment Variables**
   - Verify all required environment variables are set
   - Check Digital Ocean App Platform environment section

4. **Database Connection**
   - SQLite is used by default (no configuration needed)
   - For PostgreSQL, ensure DATABASE_URL is properly formatted

#### Health Checks
- Backend health: `GET /api/health`
- Frontend: Access the main dashboard
- Metrics: `GET /metrics`

### Scaling and Production

#### Performance Optimization
- Upgrade to Professional plan for better performance
- Enable auto-scaling based on CPU/memory usage
- Consider using managed PostgreSQL for database

#### Security Recommendations
1. Change default JWT secret
2. Use strong passwords for admin accounts
3. Enable HTTPS (automatic with Digital Ocean App Platform)
4. Regularly update dependencies
5. Monitor application logs

#### Monitoring
- Use Digital Ocean's built-in monitoring
- Access Prometheus metrics at `/metrics`
- Check application logs in Digital Ocean dashboard

### Support
- GitHub Repository: https://github.com/Samerabualsoud/ForexPulsePro
- Digital Ocean Documentation: https://docs.digitalocean.com/products/app-platform/

### Cost Estimation
- **Basic Plan**: $5/month (512MB RAM, 1 vCPU)
- **Professional Plan**: $12/month (1GB RAM, 1 vCPU)
- **Additional costs**: Database, storage, bandwidth (minimal for typical usage)

The application is optimized for the Basic plan and should handle moderate traffic efficiently.

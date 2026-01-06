# Monitoring Setup - Prometheus + Grafana

## Quick Start

### 1. Start All Services
```bash
docker-compose up -d
```

This starts:
- **Heart Disease App** on http://localhost:8080
- **Prometheus** on http://localhost:9090
- **Grafana** on http://localhost:3000

### 2. Verify Services

Check all containers are running:
```bash
docker-compose ps
```

Check app metrics endpoint:
```bash
curl http://localhost:8080/metrics
```

### 3. Access Grafana

1. Open browser: http://localhost:3000
2. Login with:
   - **Username:** admin
   - **Password:** admin
3. Skip password change (or set new password)

### 4. Create Dashboard

#### Import Pre-built Dashboard:
1. Click **"+"** → **Import**
2. Enter dashboard ID: **10842** (Flask Exporter Dashboard)
3. Click **Load**
4. Select **Prometheus** as data source
5. Click **Import**

#### Or Create Custom Dashboard:
1. Click **"+"** → **Dashboard** → **Add visualization**
2. Select **Prometheus** data source
3. Add these queries:

**Request Rate:**
```promql
rate(flask_http_request_total[5m])
```

**Request Duration:**
```promql
flask_http_request_duration_seconds_sum / flask_http_request_duration_seconds_count
```

**Error Rate:**
```promql
rate(flask_http_request_total{status=~"4..|5.."}[5m])
```

**Active Requests:**
```promql
flask_http_request_duration_seconds_count
```

**Success Rate:**
```promql
sum(rate(flask_http_request_total{status="200"}[5m])) / sum(rate(flask_http_request_total[5m])) * 100
```

### 5. Generate Test Traffic

Run predictions to generate metrics:
```bash
# Health check
curl http://localhost:8080/health

# Successful prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'

# Generate multiple requests
for i in {1..50}; do
  curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}' &
done
```

### 6. View Metrics

#### Prometheus:
1. Open http://localhost:9090
2. Go to **Status** → **Targets** (verify app is UP)
3. Go to **Graph** tab
4. Try queries from step 4 above

#### Grafana:
1. Open http://localhost:3000
2. Go to your dashboard
3. See real-time metrics and graphs

## Available Metrics

The app exposes these metrics at `/metrics`:

- `flask_http_request_total` - Total HTTP requests
- `flask_http_request_duration_seconds` - Request duration
- `flask_http_request_created` - Request creation timestamp
- `flask_exporter_info` - Exporter version info
- `app_info` - Application and model info

## Monitoring

### View Logs
```bash
# App logs
docker-compose logs -f app

# Prometheus logs
docker-compose logs -f prometheus

# Grafana logs
docker-compose logs -f grafana

# All logs
docker-compose logs -f
```

### Restart Services
```bash
docker-compose restart app
```

### Stop All Services
```bash
docker-compose down
```

### Stop and Remove All Data
```bash
docker-compose down -v
```

## Troubleshooting

### App not showing in Prometheus targets:
```bash
# Check network connectivity
docker-compose exec prometheus wget -O- http://app:8080/metrics

# Restart prometheus
docker-compose restart prometheus
```

### Cannot access Grafana:
```bash
# Check if running
docker-compose ps grafana

# View logs
docker-compose logs grafana

# Restart
docker-compose restart grafana
```

### Metrics not updating:
```bash
# Generate test traffic
curl http://localhost:8080/health

# Check metrics endpoint
curl http://localhost:8080/metrics | grep flask_http_request_total
```

## Dashboard Screenshots Commands

Run these for documentation:
```bash
# 1. Check all services
docker-compose ps

# 2. View metrics endpoint
curl http://localhost:8080/metrics

# 3. Generate load
for i in {1..100}; do
  curl -s http://localhost:8080/health > /dev/null
done

# 4. Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

Then take screenshots of:
1. Grafana dashboard showing metrics
2. Prometheus targets page
3. App logs showing requests
4. Metrics endpoint output

# CivicCam Deployment Guide

## Quick Deploy to VPS

### 1. Get a VPS
- DigitalOcean Droplet (4GB RAM, $24/month)
- Or AWS EC2, Linode, Vultr

### 2. Clone & Deploy Backend

SSH into your VPS and run:
```bash
git clone <your-repo-url> civiccam
cd civiccam/civiccam
chmod +x deploy.sh
./deploy.sh
```

### 3. Configure IP Camera

Edit `config.yaml`:
```yaml
camera:
  source: "rtsp://user:pass@camera-ip:554/stream"
```

Restart:
```bash
docker compose restart
```

### 4. Deploy Frontend to Vercel

On your local machine:
```bash
cd /Users/ar/Documents/Civic-Cam/code

# Create .env.local with your VPS IP
echo "REACT_APP_API_URL=http://YOUR_VPS_IP:8000" > .env.local

# Build and deploy
npm run build
npx vercel --prod
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /api/video` | MJPEG live stream |
| `GET /api/events` | List events |
| `GET /api/stats` | Statistics |
| `WS /ws` | Real-time events |

---

## Troubleshooting

```bash
# View logs
docker compose logs -f

# Restart
docker compose restart

# Rebuild
docker compose build --no-cache
docker compose up -d
```

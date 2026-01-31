# 🌤️ Clima AI

AI-powered weather analysis and energy prediction API for sustainable building management.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Features

- **Real-time Weather Data** - Fetches current weather from Open-Meteo API
- **7-Day Forecasts** - Hourly and daily weather predictions
- **AI Weather Analysis** - LLM-powered impact analysis for building operations
- **Energy Optimization** - Smart recommendations for HVAC and energy savings
- **Anomaly Detection** - AI-powered sensor anomaly detection
- **ML Predictions** - Linear regression energy predictions with feature importance

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/weather` | GET | Current weather for location |
| `/weather/forecast` | GET | 7-day weather forecast |
| `/analyze` | POST | AI weather impact analysis |
| `/optimize` | POST | AI energy optimization |
| `/anomaly` | POST | AI anomaly detection |
| `/predict` | POST | ML predictions + AI insights |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/mayankbaluni/clima-ai.git
cd clima-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## ⚙️ Configuration

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from [Groq Console](https://console.groq.com).

## 🏃 Running

```bash
python main.py
```

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📖 Usage Examples

### Get Current Weather
```bash
curl "http://localhost:8000/weather?lat=28.6139&lon=77.2090"
```

### Get Energy Predictions with AI Insights
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"lat": 28.6139, "lon": 77.2090}'
```

### Get AI Optimization Recommendations
```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 28.6139,
    "lon": 77.2090,
    "current_energy_kwh": 120,
    "target_temperature": 24,
    "budget_priority": "balanced"
  }'
```

## 🌐 Deployment

### Vercel

This project is configured for Vercel deployment:

```bash
vercel deploy
```

## 🤖 AI Models

- **LLM**: Groq's `llama-3.3-70b-versatile` for intelligent insights
- **ML**: Scikit-learn Linear Regression for energy predictions

## 📦 Tech Stack

- **FastAPI** - Modern Python web framework
- **Groq** - Fast LLM inference
- **Open-Meteo** - Free weather API
- **Scikit-learn** - ML predictions
- **Pandas** - Data processing

## 📄 License

MIT License - feel free to use this project for your own purposes.

---

Built with ❤️ for sustainable building management

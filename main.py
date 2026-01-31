import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

# =============================================================================
# FastAPI App Configuration
# =============================================================================
app = FastAPI(
    title="Weather AI API",
    description="AI-powered weather analysis and energy prediction API for sustainable building management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pydantic Models
# =============================================================================
class Location(BaseModel):
    lat: float = Field(default=28.6139, description="Latitude (default: Delhi)")
    lon: float = Field(default=77.2090, description="Longitude (default: Delhi)")

class SensorData(BaseModel):
    indoor_temperature: float = Field(..., description="Indoor temperature in °C")
    energy_usage_kwh: float = Field(..., description="Current energy usage in kWh")
    occupancy: Optional[int] = Field(default=None, description="Number of occupants")
    hvac_mode: Optional[str] = Field(default="auto", description="HVAC mode: auto/cooling/heating/off")

class AnalysisRequest(BaseModel):
    lat: float = Field(default=28.6139, description="Latitude")
    lon: float = Field(default=77.2090, description="Longitude")
    building_type: Optional[str] = Field(default="commercial", description="Building type: commercial/residential/industrial")
    
class OptimizationRequest(BaseModel):
    lat: float = Field(default=28.6139, description="Latitude")
    lon: float = Field(default=77.2090, description="Longitude")
    current_energy_kwh: float = Field(default=100.0, description="Current hourly energy consumption")
    target_temperature: float = Field(default=24.0, description="Target indoor temperature in °C")
    budget_priority: Optional[str] = Field(default="balanced", description="Priority: cost/comfort/balanced")

class AnomalyRequest(BaseModel):
    lat: float = Field(default=28.6139, description="Latitude")
    lon: float = Field(default=77.2090, description="Longitude")
    sensor_readings: List[SensorData] = Field(default=None, description="List of sensor readings")

# =============================================================================
# Helper Functions
# =============================================================================
def get_groq_client():
    """Get Groq client with API key validation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY not configured. Please set a valid API key in .env file."
        )
    return Groq(api_key=api_key)

def get_current_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch current weather from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m,pressure_msl",
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        return {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "pressure": current.get("pressure_msl"),
            "timezone": data.get("timezone"),
            "location": {"lat": lat, "lon": lon}
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Weather API timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

def get_weather_forecast(lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """Fetch weather forecast from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "forecast_days": days,
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "hourly": data.get("hourly", {}),
            "daily": data.get("daily", {}),
            "timezone": data.get("timezone"),
            "location": {"lat": lat, "lon": lon}
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Weather API timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

def get_mock_sensor_data() -> pd.DataFrame:
    """Generate mock sensor data for demonstration."""
    np.random.seed(42)
    hours = 24
    base_temp = 22
    base_energy = 100
    
    return pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), 
                                   periods=hours, freq='h'),
        'indoor_temperature': base_temp + np.random.randn(hours) * 2,
        'energy_usage_kwh': base_energy + np.random.randn(hours) * 15 + np.sin(np.linspace(0, 2*np.pi, hours)) * 20,
        'occupancy': np.random.randint(0, 50, hours)
    })

# =============================================================================
# AI Generation Functions
# =============================================================================
def generate_ai_insight(predictions: list, weather: dict) -> str:
    """Generate AI insight for energy predictions."""
    client = get_groq_client()
    prompt = f"""You are an energy efficiency expert. Analyze this data and provide actionable advice.

Weather Conditions:
- Temperature: {weather.get('temperature', 'N/A')}°C (feels like {weather.get('apparent_temperature', 'N/A')}°C)
- Humidity: {weather.get('humidity', 'N/A')}%
- Wind Speed: {weather.get('wind_speed', 'N/A')} km/h

Predicted Energy Usage (kWh): {[round(p, 2) for p in predictions[:5]]}

Provide 2-3 specific, actionable recommendations for sustainable building management. Be concise."""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.7
    )
    return chat_completion.choices[0].message.content

def generate_weather_analysis(weather: dict, forecast: dict, building_type: str) -> str:
    """Generate comprehensive AI weather analysis for building operations."""
    client = get_groq_client()
    
    daily = forecast.get("daily", {})
    upcoming_temps = daily.get("temperature_2m_max", [])[:3]
    
    prompt = f"""You are a building operations consultant specializing in weather impact analysis.

Current Weather:
- Temperature: {weather.get('temperature')}°C
- Humidity: {weather.get('humidity')}%
- Wind: {weather.get('wind_speed')} km/h

3-Day Forecast (Max Temps): {upcoming_temps}°C

Building Type: {building_type}

Analyze the weather impact on this {building_type} building and provide:
1. Immediate operational considerations
2. Energy demand forecast
3. Recommended HVAC settings

Be specific and concise (max 150 words)."""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=250,
        temperature=0.7
    )
    return chat_completion.choices[0].message.content

def generate_optimization_recommendations(weather: dict, forecast: dict, request: OptimizationRequest) -> Dict[str, Any]:
    """Generate AI-powered energy optimization recommendations."""
    client = get_groq_client()
    
    prompt = f"""You are an energy optimization AI for smart buildings.

Current Conditions:
- Outdoor: {weather.get('temperature')}°C, {weather.get('humidity')}% humidity
- Target Indoor: {request.target_temperature}°C
- Current Usage: {request.current_energy_kwh} kWh/hour
- Priority: {request.budget_priority}

Provide a JSON response with optimization recommendations:
{{
    "recommended_setpoint": <optimal temperature>,
    "estimated_savings_percent": <0-30>,
    "hvac_mode": "cooling|heating|auto|eco",
    "scheduling": "<brief scheduling advice>",
    "immediate_actions": ["action1", "action2", "action3"]
}}

Return ONLY valid JSON, no markdown."""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.5
    )
    
    response_text = chat_completion.choices[0].message.content
    try:
        # Try to parse JSON from response
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Return as text if not valid JSON
        return {"recommendations": response_text}

def generate_anomaly_analysis(weather: dict, sensor_stats: dict) -> Dict[str, Any]:
    """Generate AI-powered anomaly detection analysis."""
    client = get_groq_client()
    
    prompt = f"""You are an AI anomaly detection system for building sensors.

Weather: {weather.get('temperature')}°C, {weather.get('humidity')}% humidity

Sensor Statistics (24h):
- Avg Indoor Temp: {sensor_stats.get('avg_temp', 'N/A'):.1f}°C
- Temp Std Dev: {sensor_stats.get('std_temp', 'N/A'):.2f}°C
- Avg Energy: {sensor_stats.get('avg_energy', 'N/A'):.1f} kWh
- Energy Std Dev: {sensor_stats.get('std_energy', 'N/A'):.2f} kWh
- Max Energy Spike: {sensor_stats.get('max_energy', 'N/A'):.1f} kWh

Analyze for anomalies and respond with JSON:
{{
    "anomalies_detected": true|false,
    "severity": "low|medium|high|none",
    "findings": ["finding1", "finding2"],
    "root_cause": "<likely cause if anomaly detected>",
    "recommended_actions": ["action1", "action2"]
}}

Return ONLY valid JSON."""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.5
    )
    
    response_text = chat_completion.choices[0].message.content
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"analysis": response_text}

# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    """Serve the beautiful landing page."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()
    return """
    <html>
        <head><title>Clima AI</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>🌤️ Clima AI</h1>
            <p>AI-powered weather analysis API</p>
            <a href="/docs">View API Documentation</a>
        </body>
    </html>
    """

@app.get("/api", tags=["General"])
async def api_info():
    """API information endpoint."""
    return {
        "name": "Weather AI API",
        "version": "1.0.0",
        "description": "AI-powered weather analysis and energy prediction for sustainable buildings",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "weather": "/weather",
            "forecast": "/weather/forecast",
            "analyze": "/analyze",
            "optimize": "/optimize",
            "anomaly": "/anomaly",
            "predict": "/predict"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    groq_configured = os.getenv("GROQ_API_KEY") not in [None, "", "your_groq_key_here"]
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "weather_api": "operational",
            "ai_service": "configured" if groq_configured else "not_configured"
        }
    }

@app.get("/weather", tags=["Weather"])
async def get_weather(
    lat: float = Query(default=28.6139, description="Latitude"),
    lon: float = Query(default=77.2090, description="Longitude")
):
    """Get current weather for a location."""
    weather = get_current_weather(lat, lon)
    return {"status": "success", "data": weather}

@app.get("/weather/forecast", tags=["Weather"])
async def get_forecast(
    lat: float = Query(default=28.6139, description="Latitude"),
    lon: float = Query(default=77.2090, description="Longitude"),
    days: int = Query(default=7, ge=1, le=14, description="Forecast days (1-14)")
):
    """Get weather forecast for a location."""
    forecast = get_weather_forecast(lat, lon, days)
    return {"status": "success", "data": forecast}

@app.post("/analyze", tags=["AI Analysis"])
async def analyze_weather_impact(request: AnalysisRequest):
    """AI-powered weather impact analysis for building operations."""
    weather = get_current_weather(request.lat, request.lon)
    forecast = get_weather_forecast(request.lat, request.lon, days=3)
    
    analysis = generate_weather_analysis(weather, forecast, request.building_type)
    
    return {
        "status": "success",
        "data": {
            "current_weather": weather,
            "building_type": request.building_type,
            "ai_analysis": analysis
        }
    }

@app.post("/optimize", tags=["AI Analysis"])
async def optimize_energy(request: OptimizationRequest):
    """AI-powered energy optimization recommendations."""
    weather = get_current_weather(request.lat, request.lon)
    forecast = get_weather_forecast(request.lat, request.lon, days=1)
    
    recommendations = generate_optimization_recommendations(weather, forecast, request)
    
    return {
        "status": "success",
        "data": {
            "current_conditions": {
                "outdoor_temp": weather.get("temperature"),
                "humidity": weather.get("humidity"),
                "current_usage": request.current_energy_kwh
            },
            "target_temperature": request.target_temperature,
            "priority": request.budget_priority,
            "ai_recommendations": recommendations
        }
    }

@app.post("/anomaly", tags=["AI Analysis"])
async def detect_anomalies(request: AnomalyRequest):
    """AI-powered anomaly detection in sensor data."""
    weather = get_current_weather(request.lat, request.lon)
    
    # Use provided sensor data or generate mock data
    if request.sensor_readings:
        df = pd.DataFrame([s.model_dump() for s in request.sensor_readings])
    else:
        df = get_mock_sensor_data()
    
    # Calculate statistics
    sensor_stats = {
        "avg_temp": df['indoor_temperature'].mean(),
        "std_temp": df['indoor_temperature'].std(),
        "avg_energy": df['energy_usage_kwh'].mean(),
        "std_energy": df['energy_usage_kwh'].std(),
        "max_energy": df['energy_usage_kwh'].max(),
        "min_energy": df['energy_usage_kwh'].min()
    }
    
    analysis = generate_anomaly_analysis(weather, sensor_stats)
    
    return {
        "status": "success",
        "data": {
            "weather": weather,
            "sensor_statistics": sensor_stats,
            "ai_analysis": analysis
        }
    }

@app.post("/predict", tags=["AI Analysis"])
async def predict_energy(location: Location):
    """Enhanced energy prediction with ML model and AI insights."""
    weather = get_current_weather(location.lat, location.lon)
    
    # Get sensor data
    sensor_data = get_mock_sensor_data()
    
    # Add weather features
    sensor_data['outdoor_temperature'] = weather['temperature']
    sensor_data['humidity'] = weather['humidity']
    
    # Calculate adjusted energy
    sensor_data['energy_adjusted'] = sensor_data['energy_usage_kwh'] * (
        1 + (sensor_data['humidity'] / 100) * 0.1 + 
        (sensor_data['outdoor_temperature'] - sensor_data['indoor_temperature']) * 0.02
    )
    
    # Handle missing values
    sensor_data.fillna(sensor_data.mean(numeric_only=True), inplace=True)
    
    # Train model
    X = sensor_data[['indoor_temperature', 'outdoor_temperature', 'humidity']]
    y = sensor_data['energy_adjusted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test).tolist()
    mse = mean_squared_error(y_test, predictions)
    
    # Generate AI insight
    ai_insight = generate_ai_insight(predictions, weather)
    
    return {
        "status": "success",
        "data": {
            "weather": weather,
            "model_performance": {
                "mse": round(mse, 4),
                "r2_score": round(model.score(X_test, y_test), 4)
            },
            "predictions": [round(p, 2) for p in predictions],
            "feature_importance": {
                "indoor_temperature": round(model.coef_[0], 4),
                "outdoor_temperature": round(model.coef_[1], 4),
                "humidity": round(model.coef_[2], 4)
            },
            "ai_insight": ai_insight
        }
    }

# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Weather AI API...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("💚 Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
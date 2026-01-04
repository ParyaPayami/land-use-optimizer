# API Server Reference

::: pimaluos.api.server

---

## REST Endpoints

### Cities

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/cities` | List available cities |
| `GET` | `/cities/{city}/config` | Get city configuration |
| `GET` | `/cities/{city}/parcels` | Get parcel GeoJSON |

### Scenarios

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scenarios/simulate` | Run physics simulation |
| `GET` | `/scenarios/{id}` | Get scenario details |
| `POST` | `/scenarios/{id}/optimize` | Run multi-agent optimization |

### Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/agents/profiles` | Get stakeholder profiles |

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/export/geojson/{id}` | Export GeoJSON |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |

---

## WebSocket

### Real-time Simulation Updates

Connect to `/ws/simulation` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Simulation update:', data);
};

// Start simulation
ws.send(JSON.stringify({
  type: 'start_simulation',
  city: 'manhattan',
  scenario_id: 'scenario-123'
}));
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `start_simulation` | Client → Server | Start simulation |
| `stop_simulation` | Client → Server | Stop simulation |
| `progress` | Server → Client | Simulation progress |
| `result` | Server → Client | Final results |
| `error` | Server → Client | Error message |

---

## Running the Server

```bash
# Development
python -m pimaluos.api.server

# With uvicorn
uvicorn pimaluos.api.server:app --reload --port 8000

# Production
gunicorn pimaluos.api.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## Request/Response Models

### SimulationRequest

```python
{
    "city": "manhattan",
    "parcel_updates": [
        {
            "parcel_id": 123,
            "use": "residential",
            "far": 2.0,
            "height_ft": 65
        }
    ]
}
```

### SimulationResponse

```python
{
    "scenario_id": "sim-abc123",
    "traffic": {
        "avg_congestion_ratio": 1.23,
        "oversaturated_links": 5
    },
    "hydrology": {
        "peak_runoff_cfs": 45.6,
        "capacity_utilization": 0.78
    },
    "solar": {
        "avg_shadow_pct": 12.3,
        "violations": 2
    },
    "violations": {
        "total_violations": 3
    }
}
```

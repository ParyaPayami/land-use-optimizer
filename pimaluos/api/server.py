"""
PIMALUOS FastAPI Server

REST API and WebSocket server for the PIMALUOS dashboard.

Endpoints:
- GET /cities - List available cities
- GET /cities/{city}/parcels - Get parcel data
- POST /scenarios/simulate - Run simulation
- WS /ws/simulation - Real-time simulation updates
- GET /export/geojson/{scenario_id} - Export as GeoJSON
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn


# ===== Pydantic Models =====

class ParcelUpdate(BaseModel):
    """Single parcel land use update."""
    parcel_id: int
    use: str = "residential"
    far: float = 1.0
    height_ft: float = 35.0


class ScenarioInput(BaseModel):
    """Scenario simulation request."""
    city: str = "manhattan"
    parcel_updates: List[ParcelUpdate] = Field(default_factory=list)
    run_physics: bool = True
    stakeholder_weights: Optional[Dict[str, float]] = None


class SimulationResult(BaseModel):
    """Simulation result response."""
    scenario_id: str
    city: str
    num_parcels: int
    traffic: Dict[str, float]
    hydrology: Dict[str, float]
    solar: Dict[str, float]
    violations: int
    is_valid: bool


class AgentConfig(BaseModel):
    """Stakeholder agent configuration."""
    agent_type: str
    weight: float = 0.2
    awareness: Optional[Dict[str, float]] = None


# ===== Application Factory =====

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="PIMALUOS API",
        description="Physics Informed Multi-Agent Land Use Optimization Software",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Storage for scenarios
    scenarios: Dict[str, Dict] = {}
    
    # WebSocket connections
    ws_connections: List[WebSocket] = []
    
    
    # ===== City Endpoints =====
    
    @app.get("/cities", tags=["Cities"])
    async def list_cities() -> List[Dict[str, str]]:
        """List available cities."""
        return [
            {"id": "manhattan", "name": "Manhattan, NYC", "status": "available"},
            {"id": "chicago", "name": "Chicago, IL", "status": "coming_soon"},
            {"id": "la", "name": "Los Angeles, CA", "status": "coming_soon"},
            {"id": "boston", "name": "Boston, MA", "status": "coming_soon"},
        ]
    
    
    @app.get("/cities/{city}/config", tags=["Cities"])
    async def get_city_config(city: str) -> Dict[str, Any]:
        """Get city configuration."""
        try:
            from pimaluos.config.settings import get_city_config as load_config
            config = load_config(city)
            return {
                "name": config.name,
                "display_name": config.display_name,
                "latitude": config.latitude,
                "longitude": config.longitude,
                "crs": config.crs,
                "default_zoom": config.default_zoom,
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"City not found: {city}")
    
    
    @app.get("/cities/{city}/parcels", tags=["Cities"])
    async def get_parcels(
        city: str,
        limit: int = Query(100, le=10000),
        offset: int = Query(0, ge=0)
    ) -> Dict[str, Any]:
        """
        Get parcel data for a city.
        
        Returns GeoJSON FeatureCollection.
        """
        # Placeholder - would load from actual data
        features = [
            {
                "type": "Feature",
                "id": i,
                "properties": {
                    "parcel_id": i,
                    "address": f"{100 + i} Example St",
                    "land_use": "residential",
                    "zone": "R6",
                    "far": 1.5 + (i % 5) * 0.5,
                    "height_ft": 35 + (i % 10) * 10,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-73.99 + i * 0.001, 40.75],
                        [-73.99 + i * 0.001 + 0.0005, 40.75],
                        [-73.99 + i * 0.001 + 0.0005, 40.7505],
                        [-73.99 + i * 0.001, 40.7505],
                        [-73.99 + i * 0.001, 40.75],
                    ]]
                }
            }
            for i in range(offset, min(offset + limit, 1000))
        ]
        
        return {
            "type": "FeatureCollection",
            "city": city,
            "total": 1000,
            "offset": offset,
            "limit": limit,
            "features": features,
        }
    
    
    # ===== Scenario Endpoints =====
    
    @app.post("/scenarios/simulate", tags=["Scenarios"])
    async def run_simulation(scenario: ScenarioInput) -> SimulationResult:
        """
        Run physics simulation on proposed land use scenario.
        """
        scenario_id = str(uuid.uuid4())[:8]
        
        # Placeholder simulation results
        result = SimulationResult(
            scenario_id=scenario_id,
            city=scenario.city,
            num_parcels=len(scenario.parcel_updates) or 100,
            traffic={
                "avg_congestion_ratio": 1.2,
                "max_congestion_ratio": 1.8,
                "oversaturated_links": 5,
            },
            hydrology={
                "peak_runoff_cfs": 85.5,
                "capacity_utilization": 0.75,
            },
            solar={
                "avg_shadow_pct": 35.0,
                "num_violations": 2,
            },
            violations=0,
            is_valid=True,
        )
        
        # Store scenario
        scenarios[scenario_id] = {
            "input": scenario.dict(),
            "result": result.dict(),
        }
        
        return result
    
    
    @app.get("/scenarios/{scenario_id}", tags=["Scenarios"])
    async def get_scenario(scenario_id: str) -> Dict[str, Any]:
        """Get scenario details."""
        if scenario_id not in scenarios:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return scenarios[scenario_id]
    
    
    @app.post("/scenarios/{scenario_id}/optimize", tags=["Scenarios"])
    async def optimize_scenario(
        scenario_id: str,
        agent_configs: List[AgentConfig]
    ) -> Dict[str, Any]:
        """
        Run multi-agent optimization on scenario.
        """
        if scenario_id not in scenarios:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        # Placeholder optimization results
        return {
            "scenario_id": scenario_id,
            "status": "completed",
            "iterations": 50,
            "final_utility": {
                "resident": 0.72,
                "developer": 0.68,
                "planner": 0.75,
                "environmentalist": 0.65,
                "equity_advocate": 0.70,
            },
            "pareto_points": [
                {"x": 0.7, "y": 0.8, "z": 0.65},
                {"x": 0.75, "y": 0.7, "z": 0.72},
                {"x": 0.68, "y": 0.75, "z": 0.78},
            ],
        }
    
    
    # ===== Export Endpoints =====
    
    @app.get("/export/geojson/{scenario_id}", tags=["Export"])
    async def export_geojson(scenario_id: str) -> Dict[str, Any]:
        """Export scenario results as GeoJSON."""
        if scenario_id not in scenarios:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        return {
            "type": "FeatureCollection",
            "scenario_id": scenario_id,
            "features": [],  # Would include actual parcel features
        }
    
    
    # ===== WebSocket Endpoints =====
    
    @app.websocket("/ws/simulation")
    async def simulation_websocket(websocket: WebSocket):
        """
        WebSocket for real-time simulation updates.
        
        Messages:
        - {"type": "start", "scenario_id": "..."}
        - {"type": "progress", "iteration": N, "metrics": {...}}
        - {"type": "complete", "result": {...}}
        """
        await websocket.accept()
        ws_connections.append(websocket)
        
        try:
            while True:
                data = await websocket.receive_json()
                
                if data.get("type") == "start":
                    # Send progress updates
                    for i in range(10):
                        await websocket.send_json({
                            "type": "progress",
                            "iteration": i + 1,
                            "total": 10,
                            "metrics": {
                                "loss": 0.5 - i * 0.04,
                                "violations": max(0, 5 - i),
                            }
                        })
                    
                    await websocket.send_json({
                        "type": "complete",
                        "result": {"status": "success"}
                    })
                    
        except WebSocketDisconnect:
            ws_connections.remove(websocket)
    
    
    # ===== Agent Endpoints =====
    
    @app.get("/agents/profiles", tags=["Agents"])
    async def get_agent_profiles() -> Dict[str, Dict]:
        """Get default stakeholder agent profiles."""
        return {
            "resident": {
                "name": "Resident",
                "description": "Housing affordability + amenity access",
                "awareness": {"self": 0.5, "local": 0.3, "global": 0.1, "equity": 0.1},
                "color": "#4CAF50",
            },
            "developer": {
                "name": "Developer", 
                "description": "Development ROI + market conditions",
                "awareness": {"self": 0.7, "local": 0.2, "global": 0.05, "equity": 0.05},
                "color": "#2196F3",
            },
            "planner": {
                "name": "City Planner",
                "description": "Tax revenue + sustainability",
                "awareness": {"self": 0.1, "local": 0.2, "global": 0.5, "equity": 0.2},
                "color": "#9C27B0",
            },
            "environmentalist": {
                "name": "Environmentalist",
                "description": "Carbon reduction + green space",
                "awareness": {"self": 0.1, "local": 0.2, "global": 0.4, "equity": 0.3},
                "color": "#8BC34A",
            },
            "equity_advocate": {
                "name": "Equity Advocate",
                "description": "Fairness + inclusion",
                "awareness": {"self": 0.1, "local": 0.2, "global": 0.2, "equity": 0.5},
                "color": "#FF9800",
            },
        }
    
    
    # ===== Health Check =====
    
    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}
    
    
    return app


# Create app instance
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

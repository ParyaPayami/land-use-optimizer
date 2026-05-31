# PIMALUOS Dashboard

Interactive Next.js 14 dashboard for urban land-use optimization.

## Features

- 🗺️ **2D Map View** - deck.gl visualization with parcel selection
- 🌐 **3D Globe View** - CesiumJS digital twin (optional)
- 🤖 **Agent Panel** - Configure stakeholder weights
- 🎛️ **Scenario Builder** - Modify land use and run simulations
- 📊 **Metrics Bar** - Real-time traffic, drainage, solar metrics

## Setup

```bash
# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your API tokens:
# - NEXT_PUBLIC_MAPBOX_TOKEN (for 2D maps)
# - NEXT_PUBLIC_CESIUM_TOKEN (for 3D globe)

# Start development server
npm run dev
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_MAPBOX_TOKEN` | Mapbox GL access token | Recommended |
| `NEXT_PUBLIC_CESIUM_TOKEN` | Cesium Ion access token | Optional |

## Architecture

```
src/
├── app/
│   ├── layout.tsx      # Root layout
│   ├── page.tsx        # Main dashboard
│   └── globals.css     # Global styles
├── components/
│   ├── MapView.tsx     # deck.gl 2D map
│   ├── GlobeView.tsx   # CesiumJS 3D globe
│   ├── AgentPanel.tsx  # Stakeholder controls
│   ├── ControlPanel.tsx # Scenario builder
│   ├── MetricsBar.tsx  # Bottom metrics
│   └── Sidebar.tsx     # Side panel wrapper
└── lib/
    └── store.ts        # Zustand state management
```

## Backend Integration

The dashboard proxies API requests to FastAPI backend (port 8000):

- `GET /api/cities` - List available cities
- `GET /api/cities/{city}/parcels` - Get parcel GeoJSON
- `POST /api/scenarios/simulate` - Run physics simulation
- `WS /ws/simulation` - Real-time updates

## Development

```bash
# Start frontend
npm run dev        # http://localhost:3000

# Start backend (from pimaluos root)
python -m pimaluos.api.server  # http://localhost:8000
```

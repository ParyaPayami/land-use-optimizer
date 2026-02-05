import { create } from 'zustand';

interface Agent {
    type: string;
    name: string;
    weight: number;
    awareness: {
        self: number;
        local: number;
        global: number;
        equity: number;
    };
    color: string;
}

interface Parcel {
    id: number;
    geometry: GeoJSON.Polygon;
    properties: {
        address: string;
        zone: string;
        currentFar: number;
        proposedFar: number;
        landUse: string;
        height: number;
    };
}

interface SimulationResult {
    traffic: { avgCongestion: number; oversaturated: number };
    hydrology: { runoff: number; utilization: number };
    solar: { shadowPct: number; violations: number };
    violations: number;
}

interface Store {
    // City selection
    selectedCity: string;
    setSelectedCity: (city: string) => void;

    // Agent configuration
    agents: Agent[];
    updateAgentWeight: (type: string, weight: number) => void;

    // Parcel selection
    selectedParcels: number[];
    selectParcel: (id: number) => void;
    clearSelection: () => void;

    // Simulation state
    simulationRunning: boolean;
    setSimulationRunning: (running: boolean) => void;
    simulationResult: SimulationResult | null;
    setSimulationResult: (result: SimulationResult) => void;

    // View state
    viewState: {
        longitude: number;
        latitude: number;
        zoom: number;
        pitch: number;
        bearing: number;
    };
    setViewState: (state: Partial<Store['viewState']>) => void;
}

export const useStore = create<Store>((set) => ({
    // City
    selectedCity: 'manhattan',
    setSelectedCity: (city) => set({ selectedCity: city }),

    // Agents - default configuration
    agents: [
        {
            type: 'resident',
            name: 'Resident',
            weight: 0.2,
            awareness: { self: 0.5, local: 0.3, global: 0.1, equity: 0.1 },
            color: '#4ade80',
        },
        {
            type: 'developer',
            name: 'Developer',
            weight: 0.2,
            awareness: { self: 0.7, local: 0.2, global: 0.05, equity: 0.05 },
            color: '#60a5fa',
        },
        {
            type: 'planner',
            name: 'City Planner',
            weight: 0.3,
            awareness: { self: 0.1, local: 0.2, global: 0.5, equity: 0.2 },
            color: '#a78bfa',
        },
        {
            type: 'environmentalist',
            name: 'Environmentalist',
            weight: 0.15,
            awareness: { self: 0.1, local: 0.2, global: 0.4, equity: 0.3 },
            color: '#34d399',
        },
        {
            type: 'equity_advocate',
            name: 'Equity Advocate',
            weight: 0.15,
            awareness: { self: 0.1, local: 0.2, global: 0.2, equity: 0.5 },
            color: '#fbbf24',
        },
    ],
    updateAgentWeight: (type, weight) =>
        set((state) => ({
            agents: state.agents.map((a) =>
                a.type === type ? { ...a, weight } : a
            ),
        })),

    // Selection
    selectedParcels: [],
    selectParcel: (id) =>
        set((state) => ({
            selectedParcels: state.selectedParcels.includes(id)
                ? state.selectedParcels.filter((p) => p !== id)
                : [...state.selectedParcels, id],
        })),
    clearSelection: () => set({ selectedParcels: [] }),

    // Simulation
    simulationRunning: false,
    setSimulationRunning: (running) => set({ simulationRunning: running }),
    simulationResult: null,
    setSimulationResult: (result) => set({ simulationResult: result }),

    // View
    viewState: {
        longitude: -73.985,
        latitude: 40.748,
        zoom: 14,
        pitch: 45,
        bearing: 0,
    },
    setViewState: (state) =>
        set((prev) => ({
            viewState: { ...prev.viewState, ...state },
        })),
}));

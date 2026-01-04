'use client';

import { useStore } from '@/lib/store';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Download, Settings } from 'lucide-react';

export function ControlPanel() {
    const {
        selectedParcels,
        clearSelection,
        simulationRunning,
        setSimulationRunning,
        simulationResult,
    } = useStore();

    const handleRunSimulation = async () => {
        setSimulationRunning(true);

        // Call backend API
        try {
            const response = await fetch('/api/scenarios/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    city: 'manhattan',
                    parcel_updates: selectedParcels.map((id) => ({
                        parcel_id: id,
                        use: 'residential',
                        far: 2.0,
                    })),
                }),
            });

            const result = await response.json();
            console.log('Simulation result:', result);
        } catch (error) {
            console.error('Simulation failed:', error);
        } finally {
            setSimulationRunning(false);
        }
    };

    return (
        <div className="space-y-4">
            <h2 className="text-lg font-semibold gradient-text">Scenario Builder</h2>

            {/* Selected Parcels */}
            <div className="p-3 glass-subtle">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm">Selected Parcels</span>
                    <span className="text-xs text-gray-400">
                        {selectedParcels.length} selected
                    </span>
                </div>

                {selectedParcels.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                        {selectedParcels.slice(0, 10).map((id) => (
                            <span
                                key={id}
                                className="px-2 py-0.5 bg-primary-500/20 text-primary-400 text-xs rounded"
                            >
                                #{id}
                            </span>
                        ))}
                        {selectedParcels.length > 10 && (
                            <span className="text-xs text-gray-400">
                                +{selectedParcels.length - 10} more
                            </span>
                        )}
                    </div>
                ) : (
                    <p className="text-xs text-gray-500">Click parcels on the map to select</p>
                )}

                {selectedParcels.length > 0 && (
                    <button
                        onClick={clearSelection}
                        className="mt-2 text-xs text-gray-400 hover:text-white transition"
                    >
                        Clear selection
                    </button>
                )}
            </div>

            {/* Land Use Modification */}
            <div className="p-3 glass-subtle">
                <h3 className="text-sm font-medium mb-2">Proposed Change</h3>

                <div className="space-y-2">
                    <div>
                        <label className="text-xs text-gray-400">Land Use</label>
                        <select className="w-full mt-1 bg-white/5 border border-white/10 rounded px-3 py-1.5 text-sm">
                            <option value="residential">Residential</option>
                            <option value="commercial">Commercial</option>
                            <option value="mixed">Mixed Use</option>
                            <option value="industrial">Industrial</option>
                        </select>
                    </div>

                    <div>
                        <label className="text-xs text-gray-400">FAR</label>
                        <input
                            type="range"
                            min="0.5"
                            max="15"
                            step="0.5"
                            defaultValue="2"
                            className="w-full mt-1"
                        />
                        <div className="flex justify-between text-xs text-gray-400">
                            <span>0.5</span>
                            <span>15.0</span>
                        </div>
                    </div>

                    <div>
                        <label className="text-xs text-gray-400">Height (ft)</label>
                        <input
                            type="range"
                            min="35"
                            max="500"
                            step="10"
                            defaultValue="65"
                            className="w-full mt-1"
                        />
                    </div>
                </div>
            </div>

            {/* Simulation Controls */}
            <div className="space-y-2">
                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleRunSimulation}
                    disabled={simulationRunning}
                    className="w-full py-3 bg-gradient-to-r from-primary-500 to-accent-500 rounded-lg font-medium flex items-center justify-center gap-2 disabled:opacity-50"
                >
                    <Play size={16} />
                    {simulationRunning ? 'Simulating...' : 'Run Simulation'}
                </motion.button>

                <div className="grid grid-cols-3 gap-2">
                    <button className="py-2 glass-subtle flex items-center justify-center gap-1 text-xs hover:bg-white/10 transition">
                        <RotateCcw size={14} />
                        Reset
                    </button>
                    <button className="py-2 glass-subtle flex items-center justify-center gap-1 text-xs hover:bg-white/10 transition">
                        <Download size={14} />
                        Export
                    </button>
                    <button className="py-2 glass-subtle flex items-center justify-center gap-1 text-xs hover:bg-white/10 transition">
                        <Settings size={14} />
                        Settings
                    </button>
                </div>
            </div>

            {/* Results Preview */}
            {simulationResult && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 glass-subtle"
                >
                    <h3 className="text-sm font-medium mb-2">Last Result</h3>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                            <span className="text-gray-400">Congestion</span>
                            <div className="font-mono">{simulationResult.traffic.avgCongestion.toFixed(2)}</div>
                        </div>
                        <div>
                            <span className="text-gray-400">Runoff</span>
                            <div className="font-mono">{simulationResult.hydrology.runoff.toFixed(1)} cfs</div>
                        </div>
                        <div>
                            <span className="text-gray-400">Shadow</span>
                            <div className="font-mono">{simulationResult.solar.shadowPct.toFixed(1)}%</div>
                        </div>
                        <div>
                            <span className="text-gray-400">Violations</span>
                            <div className={`font-mono ${simulationResult.violations > 0 ? 'text-red-400' : 'text-green-400'}`}>
                                {simulationResult.violations}
                            </div>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}

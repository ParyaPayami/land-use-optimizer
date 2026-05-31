'use client';

import { useMemo, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    ZAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from 'recharts';
import { useStore } from '@/lib/store';

interface ParetoPoint {
    id: string;
    resident: number;
    developer: number;
    planner: number;
    environmentalist: number;
    equity: number;
    isPareto: boolean;
}

// Generate sample Pareto data
function generateParetoData(n: number = 100): ParetoPoint[] {
    const points: ParetoPoint[] = [];

    for (let i = 0; i < n; i++) {
        const point = {
            id: `point-${i}`,
            resident: Math.random() * 0.4 + 0.5,
            developer: Math.random() * 0.4 + 0.5,
            planner: Math.random() * 0.4 + 0.5,
            environmentalist: Math.random() * 0.4 + 0.5,
            equity: Math.random() * 0.4 + 0.5,
            isPareto: false,
        };
        points.push(point);
    }

    // Find Pareto frontier
    for (const point of points) {
        let dominated = false;
        for (const other of points) {
            if (
                other.resident >= point.resident &&
                other.developer >= point.developer &&
                other.planner >= point.planner &&
                (other.resident > point.resident ||
                    other.developer > point.developer ||
                    other.planner > point.planner)
            ) {
                dominated = true;
                break;
            }
        }
        point.isPareto = !dominated;
    }

    return points;
}

export function ParetoExplorer() {
    const { agents } = useStore();
    const [xAxis, setXAxis] = useState<keyof ParetoPoint>('resident');
    const [yAxis, setYAxis] = useState<keyof ParetoPoint>('developer');
    const [selectedPoint, setSelectedPoint] = useState<ParetoPoint | null>(null);

    const data = useMemo(() => generateParetoData(200), []);

    const paretoPoints = useMemo(() => data.filter((p) => p.isPareto), [data]);
    const dominatedPoints = useMemo(() => data.filter((p) => !p.isPareto), [data]);

    const axisOptions = ['resident', 'developer', 'planner', 'environmentalist', 'equity'] as const;

    const handleClick = useCallback((point: ParetoPoint) => {
        setSelectedPoint(point);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-4 z-50 glass overflow-hidden"
        >
            <div className="h-full flex flex-col p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h2 className="text-xl font-bold gradient-text">Pareto Frontier Explorer</h2>
                        <p className="text-sm text-gray-400">
                            Explore trade-offs between stakeholder objectives
                        </p>
                    </div>
                    <button className="p-2 hover:bg-white/10 rounded-lg transition">
                        ✕
                    </button>
                </div>

                {/* Axis Selection */}
                <div className="flex gap-4 mb-4">
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">X-Axis:</span>
                        <select
                            value={xAxis}
                            onChange={(e) => setXAxis(e.target.value as keyof ParetoPoint)}
                            className="bg-white/5 border border-white/10 rounded px-2 py-1 text-sm"
                        >
                            {axisOptions.map((opt) => (
                                <option key={opt} value={opt}>
                                    {opt.charAt(0).toUpperCase() + opt.slice(1)}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">Y-Axis:</span>
                        <select
                            value={yAxis}
                            onChange={(e) => setYAxis(e.target.value as keyof ParetoPoint)}
                            className="bg-white/5 border border-white/10 rounded px-2 py-1 text-sm"
                        >
                            {axisOptions.map((opt) => (
                                <option key={opt} value={opt}>
                                    {opt.charAt(0).toUpperCase() + opt.slice(1)}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Chart */}
                <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                            <XAxis
                                dataKey={xAxis}
                                type="number"
                                domain={[0.4, 1]}
                                name={xAxis}
                                tick={{ fill: '#9ca3af', fontSize: 12 }}
                                axisLine={{ stroke: '#374151' }}
                                label={{
                                    value: xAxis.charAt(0).toUpperCase() + xAxis.slice(1) + ' Utility',
                                    position: 'bottom',
                                    fill: '#9ca3af',
                                }}
                            />
                            <YAxis
                                dataKey={yAxis}
                                type="number"
                                domain={[0.4, 1]}
                                name={yAxis}
                                tick={{ fill: '#9ca3af', fontSize: 12 }}
                                axisLine={{ stroke: '#374151' }}
                                label={{
                                    value: yAxis.charAt(0).toUpperCase() + yAxis.slice(1) + ' Utility',
                                    angle: -90,
                                    position: 'left',
                                    fill: '#9ca3af',
                                }}
                            />
                            <ZAxis range={[50, 200]} />
                            <Tooltip
                                content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        const point = payload[0].payload as ParetoPoint;
                                        return (
                                            <div className="glass p-3 text-xs">
                                                <div className="font-medium mb-1">
                                                    {point.isPareto ? '⭐ Pareto Optimal' : 'Dominated'}
                                                </div>
                                                <div>Resident: {(point.resident * 100).toFixed(1)}%</div>
                                                <div>Developer: {(point.developer * 100).toFixed(1)}%</div>
                                                <div>Planner: {(point.planner * 100).toFixed(1)}%</div>
                                                <div>Environ.: {(point.environmentalist * 100).toFixed(1)}%</div>
                                                <div>Equity: {(point.equity * 100).toFixed(1)}%</div>
                                            </div>
                                        );
                                    }
                                    return null;
                                }}
                            />

                            {/* Dominated points */}
                            <Scatter name="Dominated" data={dominatedPoints} fill="#4b5563" opacity={0.3} />

                            {/* Pareto frontier */}
                            <Scatter name="Pareto" data={paretoPoints}>
                                {paretoPoints.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill="#0ea5e9"
                                        stroke="#38bdf8"
                                        strokeWidth={2}
                                        onClick={() => handleClick(entry)}
                                        style={{ cursor: 'pointer' }}
                                    />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>

                {/* Legend & Stats */}
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
                    <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-primary-500" />
                            <span className="text-xs text-gray-400">Pareto Optimal ({paretoPoints.length})</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-gray-600" />
                            <span className="text-xs text-gray-400">Dominated ({dominatedPoints.length})</span>
                        </div>
                    </div>

                    {selectedPoint && (
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="flex items-center gap-4"
                        >
                            <span className="text-xs text-gray-400">Selected:</span>
                            <button className="px-3 py-1 bg-primary-500 rounded text-xs">
                                Apply Configuration
                            </button>
                        </motion.div>
                    )}
                </div>
            </div>
        </motion.div>
    );
}

export default ParetoExplorer;

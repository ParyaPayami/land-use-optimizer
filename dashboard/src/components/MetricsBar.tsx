'use client';

import { useStore } from '@/lib/store';
import { motion } from 'framer-motion';
import { Activity, Droplet, Sun, AlertTriangle } from 'lucide-react';

export function MetricsBar() {
    const { simulationResult } = useStore();

    const metrics = [
        {
            label: 'Traffic',
            icon: Activity,
            value: simulationResult?.traffic.avgCongestion.toFixed(2) ?? '--',
            unit: 'congestion',
            color: '#60a5fa',
            threshold: 1.5,
            current: simulationResult?.traffic.avgCongestion ?? 0,
        },
        {
            label: 'Drainage',
            icon: Droplet,
            value: simulationResult?.hydrology.utilization
                ? (simulationResult.hydrology.utilization * 100).toFixed(0)
                : '--',
            unit: '% capacity',
            color: '#4ade80',
            threshold: 100,
            current: (simulationResult?.hydrology.utilization ?? 0) * 100,
        },
        {
            label: 'Solar',
            icon: Sun,
            value: simulationResult?.solar.shadowPct.toFixed(1) ?? '--',
            unit: '% shadow',
            color: '#fbbf24',
            threshold: 50,
            current: simulationResult?.solar.shadowPct ?? 0,
        },
        {
            label: 'Violations',
            icon: AlertTriangle,
            value: simulationResult?.violations ?? '--',
            unit: 'total',
            color: simulationResult?.violations && simulationResult.violations > 0 ? '#ef4444' : '#4ade80',
            threshold: 1,
            current: simulationResult?.violations ?? 0,
        },
    ];

    return (
        <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="fixed bottom-0 left-0 right-0 h-24 glass z-40"
        >
            <div className="h-full max-w-6xl mx-auto px-6 flex items-center justify-between">
                {metrics.map((metric, index) => {
                    const Icon = metric.icon;
                    const isWarning = metric.current > metric.threshold;

                    return (
                        <motion.div
                            key={metric.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="flex items-center gap-4"
                        >
                            <div
                                className="p-3 rounded-xl"
                                style={{ backgroundColor: `${metric.color}20` }}
                            >
                                <Icon size={24} style={{ color: metric.color }} />
                            </div>
                            <div>
                                <div className="text-xs text-gray-400">{metric.label}</div>
                                <div className="flex items-baseline gap-1">
                                    <span
                                        className={`text-2xl font-bold ${isWarning ? 'text-red-400' : 'text-white'}`}
                                    >
                                        {metric.value}
                                    </span>
                                    <span className="text-xs text-gray-500">{metric.unit}</span>
                                </div>
                            </div>
                        </motion.div>
                    );
                })}

                {/* Status Indicator */}
                <div className="flex items-center gap-2 px-4 py-2 glass-subtle">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <span className="text-xs text-gray-400">System Ready</span>
                </div>
            </div>
        </motion.div>
    );
}

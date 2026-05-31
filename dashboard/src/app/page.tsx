'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';
import { Sidebar } from '@/components/Sidebar';
import { ControlPanel } from '@/components/ControlPanel';
import { MetricsBar } from '@/components/MetricsBar';
import { CitySelector } from '@/components/CitySelector';
import { AgentPanel } from '@/components/AgentPanel';
import { useStore } from '@/lib/store';

// Dynamic imports for heavy 3D components
const MapView = dynamic(() => import('@/components/MapView'), { ssr: false });
const GlobeView = dynamic(() => import('@/components/GlobeView'), { ssr: false });

export default function Home() {
    const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');
    const { selectedCity, simulationRunning } = useStore();

    return (
        <main className="h-screen w-screen overflow-hidden bg-gray-950">
            {/* Header */}
            <header className="absolute top-0 left-0 right-0 z-50 h-16 glass flex items-center px-6 justify-between">
                <div className="flex items-center gap-4">
                    <h1 className="text-xl font-bold gradient-text">PIMALUOS</h1>
                    <span className="text-xs text-gray-400 hidden md:block">
                        Physics Informed Multi-Agent Land Use Optimization
                    </span>
                </div>

                <div className="flex items-center gap-4">
                    <CitySelector />

                    {/* View Toggle */}
                    <div className="flex bg-white/10 rounded-lg p-1">
                        <button
                            onClick={() => setViewMode('2d')}
                            className={`px-3 py-1 rounded text-sm transition ${viewMode === '2d' ? 'bg-primary-500 text-white' : 'text-gray-400'
                                }`}
                        >
                            2D Map
                        </button>
                        <button
                            onClick={() => setViewMode('3d')}
                            className={`px-3 py-1 rounded text-sm transition ${viewMode === '3d' ? 'bg-primary-500 text-white' : 'text-gray-400'
                                }`}
                        >
                            3D Globe
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Map/Globe View */}
            <div className="absolute inset-0 pt-16 pb-24">
                {viewMode === '2d' ? <MapView /> : <GlobeView />}
            </div>

            {/* Left Sidebar - Agent Control */}
            <Sidebar position="left">
                <AgentPanel />
            </Sidebar>

            {/* Right Sidebar - Scenario Control */}
            <Sidebar position="right">
                <ControlPanel />
            </Sidebar>

            {/* Bottom Metrics Bar */}
            <MetricsBar />

            {/* Simulation Running Indicator */}
            {simulationRunning && (
                <motion.div
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="fixed bottom-28 left-1/2 -translate-x-1/2 glass px-6 py-3 flex items-center gap-3"
                >
                    <div className="w-3 h-3 bg-primary-500 rounded-full animate-pulse" />
                    <span className="text-sm">Running simulation...</span>
                </motion.div>
            )}
        </main>
    );
}

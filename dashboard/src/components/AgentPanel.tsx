'use client';

import { useStore } from '@/lib/store';
import { motion } from 'framer-motion';

export function AgentPanel() {
    const { agents, updateAgentWeight } = useStore();

    return (
        <div className="space-y-4">
            <h2 className="text-lg font-semibold gradient-text">Stakeholder Agents</h2>
            <p className="text-xs text-gray-400">
                Adjust weights to influence optimization priority
            </p>

            <div className="space-y-3">
                {agents.map((agent, index) => (
                    <motion.div
                        key={agent.type}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-3 glass-subtle"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <div
                                    className="w-3 h-3 rounded-full"
                                    style={{ backgroundColor: agent.color }}
                                />
                                <span className="font-medium text-sm">{agent.name}</span>
                            </div>
                            <span className="text-xs text-gray-400">
                                {(agent.weight * 100).toFixed(0)}%
                            </span>
                        </div>

                        {/* Weight Slider */}
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={agent.weight * 100}
                            onChange={(e) =>
                                updateAgentWeight(agent.type, parseInt(e.target.value) / 100)
                            }
                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                            style={{
                                accentColor: agent.color,
                            }}
                        />

                        {/* Awareness Breakdown */}
                        <div className="mt-2 grid grid-cols-4 gap-1 text-[10px]">
                            {Object.entries(agent.awareness).map(([key, value]) => (
                                <div key={key} className="text-center">
                                    <div className="text-gray-500 capitalize">{key}</div>
                                    <div className="text-gray-300">{(value * 100).toFixed(0)}%</div>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Consensus Mode */}
            <div className="mt-6 p-3 glass-subtle">
                <h3 className="text-sm font-medium mb-2">Consensus Mode</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                    {['Majority', 'Weighted', 'Soft', 'Nash'].map((mode) => (
                        <button
                            key={mode}
                            className="py-1.5 px-3 rounded bg-white/5 hover:bg-white/10 transition"
                        >
                            {mode}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}

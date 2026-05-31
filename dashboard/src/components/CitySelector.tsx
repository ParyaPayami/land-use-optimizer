'use client';

import { useStore } from '@/lib/store';
import { ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const cities = [
    { id: 'manhattan', name: 'Manhattan, NYC', coords: [-73.985, 40.748] },
    { id: 'chicago', name: 'Chicago, IL', coords: [-87.6298, 41.8781], disabled: true },
    { id: 'la', name: 'Los Angeles, CA', coords: [-118.2437, 34.0522], disabled: true },
    { id: 'boston', name: 'Boston, MA', coords: [-71.0589, 42.3601], disabled: true },
];

export function CitySelector() {
    const [open, setOpen] = useState(false);
    const { selectedCity, setSelectedCity, setViewState } = useStore();

    const current = cities.find((c) => c.id === selectedCity);

    const handleSelect = (city: typeof cities[0]) => {
        if (city.disabled) return;
        setSelectedCity(city.id);
        setViewState({
            longitude: city.coords[0],
            latitude: city.coords[1],
            zoom: 14,
        });
        setOpen(false);
    };

    return (
        <div className="relative">
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-2 px-4 py-2 glass-subtle hover:bg-white/10 transition"
            >
                <span className="text-sm font-medium">{current?.name}</span>
                <ChevronDown
                    size={16}
                    className={`transition-transform ${open ? 'rotate-180' : ''}`}
                />
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full mt-2 right-0 w-48 glass overflow-hidden z-50"
                    >
                        {cities.map((city) => (
                            <button
                                key={city.id}
                                onClick={() => handleSelect(city)}
                                disabled={city.disabled}
                                className={`
                  w-full px-4 py-2 text-left text-sm
                  ${city.id === selectedCity ? 'bg-primary-500/20 text-primary-400' : ''}
                  ${city.disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-white/10'}
                  transition
                `}
                            >
                                <span>{city.name}</span>
                                {city.disabled && (
                                    <span className="ml-2 text-xs text-gray-500">Coming soon</span>
                                )}
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

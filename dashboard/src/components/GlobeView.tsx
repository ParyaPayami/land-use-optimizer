'use client';

import { useEffect, useRef } from 'react';
import { useStore } from '@/lib/store';

// Note: Full CesiumJS integration requires CESIUM_ACCESS_TOKEN
// This is a placeholder that shows the structure

export default function GlobeView() {
    const containerRef = useRef<HTMLDivElement>(null);
    const { selectedCity, viewState } = useStore();

    useEffect(() => {
        // Dynamic import Cesium only on client
        const initCesium = async () => {
            if (typeof window === 'undefined') return;

            try {
                // @ts-ignore - Cesium types
                const Cesium = await import('cesium');

                if (!containerRef.current) return;

                // Note: Requires CESIUM_ACCESS_TOKEN environment variable
                const token = process.env.NEXT_PUBLIC_CESIUM_TOKEN;
                if (token) {
                    Cesium.Ion.defaultAccessToken = token;
                }

                const viewer = new Cesium.Viewer(containerRef.current, {
                    terrainProvider: Cesium.createWorldTerrain(),
                    animation: false,
                    timeline: false,
                    baseLayerPicker: false,
                    fullscreenButton: false,
                    vrButton: false,
                    geocoder: false,
                    homeButton: false,
                    infoBox: false,
                    sceneModePicker: false,
                    selectionIndicator: false,
                    navigationHelpButton: false,
                });

                // Fly to city location
                viewer.camera.flyTo({
                    destination: Cesium.Cartesian3.fromDegrees(
                        viewState.longitude,
                        viewState.latitude,
                        5000 // Height in meters
                    ),
                    orientation: {
                        heading: Cesium.Math.toRadians(viewState.bearing),
                        pitch: Cesium.Math.toRadians(-45),
                        roll: 0.0,
                    },
                    duration: 2,
                });

                // Cleanup
                return () => {
                    viewer.destroy();
                };
            } catch (error) {
                console.error('Failed to initialize Cesium:', error);
            }
        };

        initCesium();
    }, [selectedCity, viewState]);

    return (
        <div className="w-full h-full relative">
            <div ref={containerRef} className="w-full h-full" />

            {/* Cesium token warning */}
            {!process.env.NEXT_PUBLIC_CESIUM_TOKEN && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                    <div className="glass p-6 max-w-md text-center">
                        <h3 className="text-lg font-semibold mb-2">3D Globe View</h3>
                        <p className="text-sm text-gray-400 mb-4">
                            Set <code className="text-primary-400">NEXT_PUBLIC_CESIUM_TOKEN</code> to enable the 3D globe view with CesiumJS.
                        </p>
                        <p className="text-xs text-gray-500">
                            Get a free token at <a href="https://cesium.com/ion/" className="text-primary-400 hover:underline" target="_blank">cesium.com/ion</a>
                        </p>
                    </div>
                </div>
            )}

            {/* Globe Controls */}
            <div className="absolute top-4 right-4 glass p-2">
                <div className="flex flex-col gap-1">
                    <button className="p-2 hover:bg-white/10 rounded transition text-xs">
                        🏠 Home
                    </button>
                    <button className="p-2 hover:bg-white/10 rounded transition text-xs">
                        🌍 2D
                    </button>
                    <button className="p-2 hover:bg-white/10 rounded transition text-xs">
                        🌐 3D
                    </button>
                </div>
            </div>
        </div>
    );
}

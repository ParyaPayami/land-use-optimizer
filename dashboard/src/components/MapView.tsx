'use client';

import { useCallback, useMemo } from 'react';
import { Map } from 'react-map-gl';
import { DeckGL } from '@deck.gl/react';
import { GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import { useStore } from '@/lib/store';
import useSWR from 'swr';

const MAPBOX_STYLE = 'mapbox://styles/mapbox/dark-v11';
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

// Fetcher for SWR
const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function MapView() {
    const { viewState, setViewState, selectedCity, selectedParcels, selectParcel } = useStore();

    // Fetch parcel data
    const { data: parcelData } = useSWR(
        `/api/cities/${selectedCity}/parcels?limit=500`,
        fetcher,
        { revalidateOnFocus: false }
    );

    // Color by land use
    const getLandUseColor = useCallback((feature: any) => {
        const use = feature?.properties?.land_use || 'residential';
        const isSelected = selectedParcels.includes(feature?.properties?.parcel_id);

        const colors: Record<string, [number, number, number, number]> = {
            residential: [74, 222, 128, isSelected ? 255 : 180],
            commercial: [96, 165, 250, isSelected ? 255 : 180],
            industrial: [245, 158, 11, isSelected ? 255 : 180],
            mixed: [167, 139, 250, isSelected ? 255 : 180],
        };

        return colors[use] || colors.residential;
    }, [selectedParcels]);

    // Parcel layer
    const layers = useMemo(() => {
        if (!parcelData?.features) return [];

        return [
            new GeoJsonLayer({
                id: 'parcels',
                data: parcelData,
                filled: true,
                stroked: true,
                extruded: true,
                getFillColor: getLandUseColor,
                getLineColor: [255, 255, 255, 100],
                getElevation: (d: any) => (d.properties?.height_ft || 35) * 0.3048, // Convert to meters
                lineWidthMinPixels: 1,
                pickable: true,
                onClick: ({ object }: any) => {
                    if (object?.properties?.parcel_id) {
                        selectParcel(object.properties.parcel_id);
                    }
                },
                updateTriggers: {
                    getFillColor: [selectedParcels],
                },
            }),
        ];
    }, [parcelData, getLandUseColor, selectParcel, selectedParcels]);

    const handleViewStateChange = useCallback(
        ({ viewState: newViewState }: any) => {
            setViewState(newViewState);
        },
        [setViewState]
    );

    return (
        <div className="w-full h-full relative">
            <DeckGL
                viewState={viewState}
                onViewStateChange={handleViewStateChange}
                controller={true}
                layers={layers}
                getCursor={({ isHovering }) => (isHovering ? 'pointer' : 'grab')}
            >
                <Map
                    mapboxAccessToken={MAPBOX_TOKEN}
                    mapStyle={MAPBOX_STYLE}
                    reuseMaps
                />
            </DeckGL>

            {/* Legend */}
            <div className="absolute bottom-4 right-4 glass p-3">
                <h4 className="text-xs font-medium mb-2 text-gray-400">Land Use</h4>
                <div className="space-y-1">
                    {[
                        { name: 'Residential', color: 'bg-green-400' },
                        { name: 'Commercial', color: 'bg-blue-400' },
                        { name: 'Industrial', color: 'bg-yellow-500' },
                        { name: 'Mixed Use', color: 'bg-purple-400' },
                    ].map((item) => (
                        <div key={item.name} className="flex items-center gap-2 text-xs">
                            <div className={`w-3 h-3 rounded ${item.color}`} />
                            <span>{item.name}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* No Mapbox Token Warning */}
            {!MAPBOX_TOKEN && (
                <div className="absolute top-4 left-4 glass p-3 max-w-xs">
                    <p className="text-xs text-yellow-400">
                        ⚠️ Set NEXT_PUBLIC_MAPBOX_TOKEN for map tiles
                    </p>
                </div>
            )}
        </div>
    );
}

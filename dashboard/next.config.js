/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,

    // Enable static export for Cesium assets
    transpilePackages: ['cesium', 'resium'],

    // API routes proxy to FastAPI backend
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: 'http://localhost:8000/:path*',
            },
        ];
    },

    // WebSocket proxy
    async headers() {
        return [
            {
                source: '/:path*',
                headers: [
                    { key: 'Access-Control-Allow-Origin', value: '*' },
                ],
            },
        ];
    },

    // Enable source maps in development
    productionBrowserSourceMaps: true,

    // Cesium configuration
    webpack: (config, { isServer }) => {
        // Cesium needs special handling
        config.resolve.fallback = {
            ...config.resolve.fallback,
            fs: false,
            path: false,
        };

        return config;
    },
};

module.exports = nextConfig;

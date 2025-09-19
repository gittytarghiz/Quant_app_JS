/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // Point rewrites to the API server; default to localhost for dev
    const API = process.env.API_URL || "http://localhost:8000";
    return [
      { source: "/opt/:path*",  destination: `${API}/opt/:path*` },
      { source: "/data/:path*", destination: `${API}/data/:path*` },
    ];
  },
};

module.exports = nextConfig;

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "frame-src https://*.e2b.dev https://*.e2b.app https://*.onkernel.com https://va.vercel-scripts.com 'self'",
              "frame-ancestors 'self' https://*.e2b.dev https://*.e2b.app https://*.onkernel.com",
              "connect-src 'self' http://localhost:8000 ws://localhost:8000 https://*.e2b.dev https://*.e2b.app https://*.onkernel.com wss://*.onkernel.com",
              "img-src 'self' data: https://*.e2b.dev https://*.e2b.app https://*.onkernel.com",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.e2b.dev https://*.e2b.app https://*.onkernel.com https://va.vercel-scripts.com",
              "style-src 'self' 'unsafe-inline'",
            ].join("; "),
          },
          {
            key: "X-Frame-Options",
            value: "SAMEORIGIN",
          },
        ],
      },
    ];
  },
};

export default nextConfig;

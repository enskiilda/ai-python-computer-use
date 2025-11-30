import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { Analytics } from "@vercel/analytics/react"

export const metadata: Metadata = {
  title: "AI Computer Use - Kernel SDK + Llama 4",
  description: "A real-time AI agent that controls a browser using Kernel SDK and NVIDIA Llama 4.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased font-sans">
        {children}
        <Toaster />
        <Analytics />
      </body>
    </html>
  );
}

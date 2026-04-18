import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Flora Asset Pipeline",
  description: "Automated botanical data pipeline for the Flora iOS app",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 min-h-screen">
        <header className="bg-white border-b border-gray-200 px-6 py-4">
          <h1 className="text-lg font-semibold tracking-tight">Flora Asset Pipeline</h1>
        </header>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Link from "next/link";
import { 
  Activity, 
  Upload, 
  Terminal, 
  Microscope, 
  Fingerprint, 
  Cpu, 
  Database 
} from "lucide-react";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MedAlign Research",
  description: "Medical AI Evaluation & Alignment Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} font-sans bg-white text-gray-900 min-h-screen flex flex-col selection:bg-blue-100 selection:text-blue-900 antialiased`}>
        <main className="w-full max-w-7xl mx-auto flex-1">
          {children}
        </main>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";
import "./research-views.css";
import "./workbench.css";
import "./dream-theme.css";
import "./editorial.css";

export const metadata: Metadata = {
  title: "NextTransformer · Training Observatory",
  description: "An open observatory for recurrent, sparse and linear Transformers: learning curves, model cards and architecture visualizations.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  openGraph: {
    title: "NextTransformer · Training Observatory",
    description: "Toward the next generation of Transformers, compared under one fixed training budget.",
    images: ["/brand/observatory-sky.jpg"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}

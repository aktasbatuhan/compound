import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Compound",
  description: "Turn production traces into gated optimization evidence",
};

const NAV = [
  { href: "/review", label: "Review" },
  { href: "/cases", label: "Cases" },
  { href: "/matrix", label: "Model matrix" },
  { href: "/gates", label: "Gates" },
  { href: "/judges", label: "Judges" },
  { href: "/traces", label: "Traces" },
  { href: "/imports", label: "Imports" },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="layout">
          <aside className="sidebar">
            <h1>Compound</h1>
            <nav>
              {NAV.map((item) => (
                <a key={item.href} href={item.href}>
                  {item.label}
                </a>
              ))}
            </nav>
          </aside>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}

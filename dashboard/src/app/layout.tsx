import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
    title: 'PIMALUOS | Urban Land Use Optimization',
    description: 'Physics Informed Multi-Agent Land Use Optimization Software',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className={`${inter.className} bg-gray-950 text-white antialiased`}>
                {children}
            </body>
        </html>
    );
}

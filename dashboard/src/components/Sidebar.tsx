'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface SidebarProps {
    position: 'left' | 'right';
    children: ReactNode;
}

export function Sidebar({ position, children }: SidebarProps) {
    return (
        <motion.aside
            initial={{ x: position === 'left' ? -300 : 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={`
        fixed top-20 bottom-28 w-80 z-40
        ${position === 'left' ? 'left-4' : 'right-4'}
        glass overflow-hidden
      `}
        >
            <div className="h-full overflow-y-auto p-4">
                {children}
            </div>
        </motion.aside>
    );
}

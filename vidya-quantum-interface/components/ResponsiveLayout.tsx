"use client";

import React, { ReactNode } from 'react';
import { useResponsive, createResponsiveStyles } from '@/lib/responsive';
import type { QuantumEffectQuality } from '@/lib/responsive';
import { getBatteryOptimizedSettings } from '@/lib/responsive';
import { useQuantumState } from '@/lib/state';
import DeviceAdaptation from './DeviceAdaptation';

interface ResponsiveLayoutProps {
  children: ReactNode;
}

export default function ResponsiveLayout({ children }: ResponsiveLayoutProps) {
  const { breakpoint, width, height, isMobile, isTablet } = useResponsive();
  const setQuantumQuality = useQuantumState((s) => s.setQuantumQuality);

  // Hydration-safe default: use a stable quality on server, compute real value on client
  const [quality, setQuality] = React.useState<QuantumEffectQuality>('medium');

  React.useEffect(() => {
    // Compute client-only quality to avoid SSR hydration mismatch
    const q = getBatteryOptimizedSettings();
    setQuality(q);
    setQuantumQuality(q);
  }, [setQuantumQuality]);

  const layoutStyles = createResponsiveStyles(
    {
      height: '100vh',
      width: '100vw',
      overflow: 'hidden',
      background: '#07090d',
      position: 'relative',
    },
    {
      mobile: {
        fontSize: '14px',
        padding: '8px',
      },
      tablet: {
        fontSize: '15px',
        padding: '12px',
      },
      desktop: {
        fontSize: '16px',
        padding: '16px',
      },
      wide: {
        fontSize: '18px',
        padding: '20px',
      },
    }
  );

  return (
    <main 
      style={layoutStyles(breakpoint)}
      data-breakpoint={breakpoint}
      data-quantum-quality={quality}
      className={`responsive-layout ${breakpoint} quantum-${quality}`}
      suppressHydrationWarning
    >
      {/* Responsive viewport meta information for debugging */}
      {process.env.NODE_ENV === 'development' && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            zIndex: 9999,
            background: 'rgba(0,0,0,0.8)',
            color: 'white',
            padding: '4px 8px',
            fontSize: '10px',
            fontFamily: 'monospace',
            pointerEvents: 'none',
          }}
          suppressHydrationWarning
        >
          {width}×{height} | {breakpoint} | {quality}
        </div>
      )}
      
      {/* Device adaptation wrapper with touch gestures and accessibility */}
      <DeviceAdaptation>
        {children}
      </DeviceAdaptation>
      
      {/* Responsive CSS custom properties */}
      <style jsx>{`
        :global(:root) {
          --viewport-width: ${width}px;
          --viewport-height: ${height}px;
          --breakpoint: ${breakpoint};
          --is-mobile: ${isMobile ? 1 : 0};
          --is-tablet: ${isTablet ? 1 : 0};
          --quantum-quality: ${quality};
        }

        .responsive-layout {
          /* Responsive quantum effect scaling */
          --quantum-scale: ${isMobile ? 0.6 : isTablet ? 0.8 : 1.0};
          
          /* Responsive interaction zones */
          --touch-target-size: ${isMobile ? '44px' : isTablet ? '40px' : '36px'};
          
          /* Responsive spacing */
          --spacing-xs: ${isMobile ? '4px' : '6px'};
          --spacing-sm: ${isMobile ? '8px' : '12px'};
          --spacing-md: ${isMobile ? '16px' : '20px'};
          --spacing-lg: ${isMobile ? '24px' : '32px'};
          --spacing-xl: ${isMobile ? '32px' : '48px'};
        }

        /* Responsive quantum effects */
        .quantum-minimal {
          --particle-count: 0;
          --shader-complexity: 0;
          --animation-intensity: 0;
        }

        .quantum-low {
          --particle-count: ${isMobile ? 50 : 100};
          --shader-complexity: 1;
          --animation-intensity: 0.3;
        }

        .quantum-medium {
          --particle-count: ${isMobile ? 150 : isTablet ? 250 : 400};
          --shader-complexity: 2;
          --animation-intensity: 0.6;
        }

        .quantum-high {
          --particle-count: ${isMobile ? 200 : isTablet ? 400 : 800};
          --shader-complexity: 3;
          --animation-intensity: 1.0;
        }

        /* Mobile-specific optimizations */
        .mobile {
          --neural-network-lod: 0.3;
          --quantum-field-resolution: 0.5;
          --text-rendering: optimizeSpeed;
        }

        /* Tablet-specific optimizations */
        .tablet {
          --neural-network-lod: 0.6;
          --quantum-field-resolution: 0.75;
          --text-rendering: optimizeLegibility;
        }

        /* Desktop optimizations */
        .desktop,
        .wide {
          --neural-network-lod: 1.0;
          --quantum-field-resolution: 1.0;
          --text-rendering: geometricPrecision;
        }
      `}</style>
    </main>
  );
}
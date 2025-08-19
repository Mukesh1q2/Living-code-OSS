"use client";

import React, { useEffect, useState } from 'react';
import { useResponsive, getDeviceCapabilities, getBatteryOptimizedSettings, getProgressiveEnhancementLevel } from '@/lib/responsive';
import { useTouchGestures } from '@/lib/touch-gestures';
import { useAccessibility } from '@/lib/accessibility';
import { useQuantumState } from '@/lib/state';
import { usePerformanceMonitor } from '@/lib/performance-monitor';

interface DeviceAdaptationProps {
  children: React.ReactNode;
}

export default function DeviceAdaptation({ children }: DeviceAdaptationProps) {
  const { breakpoint, isMobile, isTablet, width, height } = useResponsive();
  const { settings: accessibilitySettings, announce, describeQuantumState, provideFeedback } = useAccessibility();
  const performanceMonitor = usePerformanceMonitor();
  
  const {
    setQuantumQuality,
    setSuperposition,
    createEntanglement,
    initiateTeleportation,
    setShowPlanPanel,
  } = useQuantumState();

  // Hydration-safe defaults; compute real values on client to avoid SSR mismatches
  const [deviceCapabilities, setDeviceCapabilities] = useState(() => ({
    hasWebGL: false,
    hasWebGL2: false,
    hasTouch: false,
    hasPointerEvents: false,
    maxTextureSize: 0,
    supportsComplexShaders: false,
    deviceMemory: 4,
    hardwareConcurrency: 4,
    connectionType: 'unknown' as any,
    batteryLevel: 1,
    isCharging: true,
    devicePixelRatio: 1,
    prefersReducedMotion: false,
    prefersHighContrast: false,
    screenOrientation: 'landscape' as any,
  }));
  const [enhancementLevel, setEnhancementLevel] = useState<'basic' | 'standard' | 'enhanced' | 'full'>(() => 'standard');

  // Touch gesture handlers for quantum interactions
  const gestureElementRef = useTouchGestures({
    onTap: (point) => {
      provideFeedback('light');
      announce('Quantum interaction point selected');
    },
    
    onDoubleTap: (point) => {
      provideFeedback('medium');
      announce('Double tap detected, activating quantum measurement');
    },
    
    onLongPress: (point) => {
      provideFeedback('heavy');
      announce('Long press detected, opening quantum context menu');
      setShowPlanPanel(true);
    },
    
    onQuantumCollapse: (point) => {
      provideFeedback('medium');
      describeQuantumState('collapse');
      // Trigger superposition collapse at touch point
      setSuperposition(false);
    },
    
    onQuantumEntangle: (points) => {
      provideFeedback('heavy');
      describeQuantumState('entanglement');
      announce('Creating quantum entanglement between two points');
      // Create entanglement between the two touch points
      createEntanglement(`node_${points[0].id}`, `node_${points[1].id}`, 0.8);
    },
    
    onQuantumTeleport: (direction, gesture) => {
      provideFeedback('heavy');
      describeQuantumState('teleportation');
      announce(`Initiating quantum teleportation ${direction}`);
      
      // Calculate teleportation target based on direction
      const target: [number, number, number] = [
        direction === 'left' ? -10 : direction === 'right' ? 10 : 0,
        direction === 'up' ? 10 : direction === 'down' ? -10 : 0,
        0
      ];
      
      initiateTeleportation(target, 'touch-gesture');
    },
    
    onDimensionalShift: (scale) => {
      provideFeedback('heavy');
      describeQuantumState('dimensionalShift');
      announce(`Shifting dimensions with scale factor ${scale.toFixed(2)}`);
      
      // Trigger dimensional shift based on pinch scale
      document.dispatchEvent(new CustomEvent('dimensional-shift', { 
        detail: { scale, trigger: 'touch-gesture' } 
      }));
    },
    
    onSwipe: (direction, gesture) => {
      provideFeedback('light');
      announce(`Swiping ${direction} through quantum space`);
      
      // Navigate through neural network based on swipe direction
      document.dispatchEvent(new CustomEvent('neural-network-navigate', {
        detail: { direction, velocity: gesture.velocity }
      }));
    },
  });

  // Monitor device capabilities and battery status
  useEffect(() => {
    const updateDeviceStatus = () => {
      const newCapabilities = getDeviceCapabilities();
      setDeviceCapabilities(newCapabilities);
      
      const newEnhancementLevel = getProgressiveEnhancementLevel(newCapabilities);
      setEnhancementLevel(newEnhancementLevel);
      
      // Update quantum quality based on current device state
      const optimizedQuality = getBatteryOptimizedSettings(newCapabilities);
      setQuantumQuality(optimizedQuality);
    };

    // Initial compute on client after mount to avoid hydration mismatch
    updateDeviceStatus();

    // Update on visibility change (battery optimization)
    document.addEventListener('visibilitychange', updateDeviceStatus);
    
    // Update on orientation change
    window.addEventListener('orientationchange', updateDeviceStatus);
    
    // Update on network change
    if ('connection' in navigator) {
      (navigator as any).connection.addEventListener('change', updateDeviceStatus);
    }

    // Periodic battery check
    const batteryInterval = setInterval(updateDeviceStatus, 30000); // Every 30 seconds

    return () => {
      document.removeEventListener('visibilitychange', updateDeviceStatus);
      window.removeEventListener('orientationchange', updateDeviceStatus);
      if ('connection' in navigator) {
        (navigator as any).connection.removeEventListener('change', updateDeviceStatus);
      }
      clearInterval(batteryInterval);
    };
  }, [setQuantumQuality]);

  // Performance monitoring and adaptation
  useEffect(() => {
    performanceMonitor.start();
    
    const unsubscribeMetrics = performanceMonitor.onMetricsUpdate((metrics) => {
      // Announce performance issues to screen readers
      if (metrics.fps < 20 && accessibilitySettings.voiceAnnouncements) {
        announce('Performance optimization in progress', 'polite');
      }
    });

    const unsubscribeQuality = performanceMonitor.onQualityChange((quality, settings) => {
      announce(`Quantum effect quality adjusted to ${quality} level`);
      setQuantumQuality(quality);
    });

    return () => {
      performanceMonitor.stop();
      unsubscribeMetrics();
      unsubscribeQuality();
    };
  }, [performanceMonitor, accessibilitySettings.voiceAnnouncements, announce, setQuantumQuality]);

  // Responsive quantum effect scaling with enhanced device detection
  const getQuantumEffectScale = () => {
    const baseScale = isMobile ? 0.6 : isTablet ? 0.8 : 1.0;
    
    // Further adjust based on device capabilities
    if (deviceCapabilities.deviceMemory < 2) return baseScale * 0.7;
    if (deviceCapabilities.hardwareConcurrency < 4) return baseScale * 0.8;
    if (!deviceCapabilities.hasWebGL2) return baseScale * 0.9;
    
    return baseScale;
  };

  // Enhanced touch target sizing for accessibility
  const getTouchTargetSize = () => {
    if (isMobile) return Math.max(44, width * 0.08); // Minimum 44px, scale with screen
    if (isTablet) return Math.max(40, width * 0.06);
    return 36;
  };

  // Device-specific CSS custom properties with enhanced responsive values
  const deviceStyles = {
    '--device-scale': getQuantumEffectScale(),
    '--enhancement-level': enhancementLevel,
    '--battery-level': deviceCapabilities.batteryLevel,
    '--connection-quality': deviceCapabilities.connectionType === '4g' ? 1 : 
                           deviceCapabilities.connectionType === '3g' ? 0.7 : 0.4,
    '--device-memory': deviceCapabilities.deviceMemory,
    '--hardware-concurrency': deviceCapabilities.hardwareConcurrency,
    '--max-texture-size': deviceCapabilities.maxTextureSize,
    '--device-pixel-ratio': deviceCapabilities.devicePixelRatio,
    '--touch-target-size': `${getTouchTargetSize()}px`,
    '--viewport-width': `${width}px`,
    '--viewport-height': `${height}px`,
    '--safe-area-inset-top': 'env(safe-area-inset-top, 0px)',
    '--safe-area-inset-bottom': 'env(safe-area-inset-bottom, 0px)',
    '--safe-area-inset-left': 'env(safe-area-inset-left, 0px)',
    '--safe-area-inset-right': 'env(safe-area-inset-right, 0px)',
  } as React.CSSProperties;

  return (
    <div
      ref={gestureElementRef}
      className={`device-adaptation ${breakpoint} enhancement-${enhancementLevel}`}
      style={deviceStyles}
      data-device-capabilities={JSON.stringify({
        webgl: deviceCapabilities.hasWebGL,
        webgl2: deviceCapabilities.hasWebGL2,
        touch: deviceCapabilities.hasTouch,
        memory: deviceCapabilities.deviceMemory,
        battery: Math.round(deviceCapabilities.batteryLevel * 100),
        connection: deviceCapabilities.connectionType,
      })}
      suppressHydrationWarning
      // Accessibility attributes
      role="application"
      aria-label="Vidya Quantum Sanskrit AI Interface"
      aria-describedby="quantum-description"
      tabIndex={0}
    >
      {/* Hidden description for screen readers */}
      <div
        id="quantum-description"
        className="sr-only"
        aria-live="polite"
      >
        Interactive quantum consciousness interface with Sanskrit AI processing. 
        Use keyboard navigation with arrow keys, space to collapse quantum states, 
        and enter to activate elements. Touch gestures include tap for selection, 
        double-tap for measurement, long-press for context menu, two-finger tap for entanglement, 
        three-finger swipe for teleportation, and four-finger pinch for dimensional shifts.
      </div>

      {/* Progressive enhancement indicators */}
      {process.env.NODE_ENV === 'development' && (
        <div className="device-debug-info">
          <div>Enhancement: {enhancementLevel}</div>
          <div>Battery: {Math.round(deviceCapabilities.batteryLevel * 100)}%</div>
          <div>Connection: {deviceCapabilities.connectionType}</div>
          <div>Memory: {deviceCapabilities.deviceMemory}GB</div>
          <div>Cores: {deviceCapabilities.hardwareConcurrency}</div>
          <div>WebGL2: {deviceCapabilities.hasWebGL2 ? 'Yes' : 'No'}</div>
        </div>
      )}

      {children}

      {/* Device-specific styles */}
      <style jsx>{`
        .device-adaptation {
          position: relative;
          width: 100%;
          height: 100%;
          touch-action: none; /* Prevent default touch behaviors */
        }

        .device-adaptation.mobile {
          --quantum-particle-count: 100;
          --neural-network-complexity: 0.3;
          --shader-quality: low;
        }

        .device-adaptation.tablet {
          --quantum-particle-count: 300;
          --neural-network-complexity: 0.6;
          --shader-quality: medium;
        }

        .device-adaptation.desktop,
        .device-adaptation.wide {
          --quantum-particle-count: 500;
          --neural-network-complexity: 1.0;
          --shader-quality: high;
        }

        .enhancement-basic {
          --animation-duration: 0s;
          --particle-effects: none;
          --quantum-effects: minimal;
        }

        .enhancement-standard {
          --animation-duration: 0.3s;
          --particle-effects: basic;
          --quantum-effects: standard;
        }

        .enhancement-enhanced {
          --animation-duration: 0.5s;
          --particle-effects: enhanced;
          --quantum-effects: enhanced;
        }

        .enhancement-full {
          --animation-duration: 0.8s;
          --particle-effects: full;
          --quantum-effects: full;
        }

        .device-debug-info {
          position: fixed;
          top: 40px;
          left: 10px;
          background: rgba(0, 0, 0, 0.8);
          color: white;
          padding: 8px;
          font-size: 10px;
          font-family: monospace;
          border-radius: 4px;
          z-index: 10000;
          pointer-events: none;
        }

        .sr-only {
          position: absolute;
          width: 1px;
          height: 1px;
          padding: 0;
          margin: -1px;
          overflow: hidden;
          clip: rect(0, 0, 0, 0);
          white-space: nowrap;
          border: 0;
        }

        /* High contrast mode styles */
        :global(.high-contrast) .device-adaptation {
          --quantum-color-primary: #ffffff;
          --quantum-color-secondary: #000000;
          --quantum-color-accent: #ffff00;
          filter: contrast(150%);
        }

        /* Reduced motion styles */
        :global(.reduced-motion) .device-adaptation {
          --animation-duration: 0s;
          --particle-effects: none;
          --quantum-transitions: none;
        }

        /* Color blind friendly styles */
        :global(.color-blind-friendly) .device-adaptation {
          --quantum-color-primary: #0066cc;
          --quantum-color-secondary: #ff6600;
          --quantum-color-accent: #009900;
        }

        /* Font size adjustments */
        :global(.font-large) .device-adaptation {
          --base-font-size: 18px;
        }

        :global(.font-extra-large) .device-adaptation {
          --base-font-size: 24px;
        }

        /* Focus indicators */
        :global(.focus-indicators) .device-adaptation *:focus {
          outline: 3px solid var(--quantum-color-accent, #7BE1FF);
          outline-offset: 2px;
        }
      `}</style>
    </div>
  );
}
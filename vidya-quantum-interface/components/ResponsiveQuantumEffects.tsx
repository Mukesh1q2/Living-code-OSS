"use client";

import React, { useEffect, useState, useMemo } from 'react';
import { useResponsive, getDeviceCapabilities, getBatteryOptimizedSettings } from '@/lib/responsive';
import { useAccessibility } from '@/lib/accessibility';
import { useQuantumState } from '@/lib/state';

interface ResponsiveQuantumEffectsProps {
  children: React.ReactNode;
  forceQuality?: 'minimal' | 'low' | 'medium' | 'high';
}

export default function ResponsiveQuantumEffects({ 
  children, 
  forceQuality 
}: ResponsiveQuantumEffectsProps) {
  const { breakpoint, isMobile, isTablet, width, height } = useResponsive();
  const { settings: accessibilitySettings } = useAccessibility();
  const { quantumQuality, setQuantumQuality } = useQuantumState();
  
  const [deviceCapabilities, setDeviceCapabilities] = useState(() => getDeviceCapabilities());
  const [performanceMetrics, setPerformanceMetrics] = useState({
    fps: 60,
    memoryUsage: 0,
    renderTime: 0,
  });

  // Calculate optimal quantum effects configuration
  const quantumConfig = useMemo(() => {
    const quality = forceQuality || getBatteryOptimizedSettings(deviceCapabilities);
    
    // Base configurations for each quality level
    const configs = {
      minimal: {
        particleCount: 0,
        shaderComplexity: 0,
        animationIntensity: 0,
        fieldResolution: 0.25,
        enableAdvancedEffects: false,
        enableParticlePhysics: false,
        enableQuantumField: false,
        maxEntanglements: 0,
      },
      low: {
        particleCount: isMobile ? 50 : 100,
        shaderComplexity: 1,
        animationIntensity: 0.3,
        fieldResolution: 0.5,
        enableAdvancedEffects: false,
        enableParticlePhysics: false,
        enableQuantumField: true,
        maxEntanglements: 2,
      },
      medium: {
        particleCount: isMobile ? 150 : isTablet ? 250 : 400,
        shaderComplexity: 2,
        animationIntensity: 0.6,
        fieldResolution: 0.75,
        enableAdvancedEffects: true,
        enableParticlePhysics: true,
        enableQuantumField: true,
        maxEntanglements: 5,
      },
      high: {
        particleCount: isMobile ? 200 : isTablet ? 400 : 800,
        shaderComplexity: 3,
        animationIntensity: 1.0,
        fieldResolution: 1.0,
        enableAdvancedEffects: true,
        enableParticlePhysics: true,
        enableQuantumField: true,
        maxEntanglements: 10,
      },
    };

    let config = configs[quality];

    // Apply accessibility overrides
    if (accessibilitySettings.reducedMotion) {
      config = {
        ...config,
        animationIntensity: 0,
        particleCount: 0,
        enableParticlePhysics: false,
      };
    }

    if (accessibilitySettings.highContrastMode) {
      config = {
        ...config,
        shaderComplexity: Math.min(config.shaderComplexity, 1),
      };
    }

    // Apply device-specific optimizations
    if (deviceCapabilities.deviceMemory < 2) {
      config.particleCount = Math.floor(config.particleCount * 0.5);
      config.fieldResolution *= 0.7;
    }

    if (deviceCapabilities.hardwareConcurrency < 4) {
      config.enableParticlePhysics = false;
      config.particleCount = Math.floor(config.particleCount * 0.8);
    }

    if (!deviceCapabilities.hasWebGL2) {
      config.shaderComplexity = Math.min(config.shaderComplexity, 1);
      config.enableAdvancedEffects = false;
    }

    // Battery optimization
    if (!deviceCapabilities.isCharging && deviceCapabilities.batteryLevel < 0.3) {
      config.particleCount = Math.floor(config.particleCount * 0.6);
      config.animationIntensity *= 0.7;
      config.fieldResolution *= 0.8;
    }

    return config;
  }, [
    forceQuality, 
    deviceCapabilities, 
    accessibilitySettings, 
    isMobile, 
    isTablet,
    performanceMetrics.fps
  ]);

  // Performance monitoring and adaptive quality
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measurePerformance = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        
        setPerformanceMetrics(prev => ({
          ...prev,
          fps,
          renderTime: currentTime - lastTime,
        }));

        // Adaptive quality adjustment
        if (fps < 30 && quantumQuality !== 'minimal') {
          const qualityLevels = ['high', 'medium', 'low', 'minimal'];
          const currentIndex = qualityLevels.indexOf(quantumQuality);
          if (currentIndex < qualityLevels.length - 1) {
            setQuantumQuality(qualityLevels[currentIndex + 1] as any);
          }
        } else if (fps > 55 && quantumQuality !== 'high') {
          const qualityLevels = ['minimal', 'low', 'medium', 'high'];
          const currentIndex = qualityLevels.indexOf(quantumQuality);
          if (currentIndex < qualityLevels.length - 1) {
            setQuantumQuality(qualityLevels[currentIndex + 1] as any);
          }
        }

        frameCount = 0;
        lastTime = currentTime;
      }

      animationId = requestAnimationFrame(measurePerformance);
    };

    animationId = requestAnimationFrame(measurePerformance);

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [quantumQuality, setQuantumQuality]);

  // Update device capabilities periodically
  useEffect(() => {
    const updateCapabilities = () => {
      setDeviceCapabilities(getDeviceCapabilities());
    };

    const interval = setInterval(updateCapabilities, 30000); // Every 30 seconds
    
    // Update on visibility change
    document.addEventListener('visibilitychange', updateCapabilities);
    
    // Update on orientation change
    window.addEventListener('orientationchange', updateCapabilities);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', updateCapabilities);
      window.removeEventListener('orientationchange', updateCapabilities);
    };
  }, []);

  // Responsive scaling factors
  const scalingFactors = useMemo(() => {
    const baseScale = isMobile ? 0.7 : isTablet ? 0.85 : 1.0;
    const dprScale = Math.min(deviceCapabilities.devicePixelRatio, 2) / 2;
    const memoryScale = Math.min(deviceCapabilities.deviceMemory / 4, 1);
    
    return {
      particle: baseScale * memoryScale,
      field: baseScale * dprScale,
      animation: baseScale,
      interaction: Math.max(baseScale, 0.8), // Ensure interactions remain responsive
    };
  }, [isMobile, isTablet, deviceCapabilities]);

  // CSS custom properties for quantum effects
  const quantumStyles = {
    '--quantum-particle-count': quantumConfig.particleCount,
    '--quantum-shader-complexity': quantumConfig.shaderComplexity,
    '--quantum-animation-intensity': quantumConfig.animationIntensity,
    '--quantum-field-resolution': quantumConfig.fieldResolution,
    '--quantum-max-entanglements': quantumConfig.maxEntanglements,
    '--quantum-scale-particle': scalingFactors.particle,
    '--quantum-scale-field': scalingFactors.field,
    '--quantum-scale-animation': scalingFactors.animation,
    '--quantum-scale-interaction': scalingFactors.interaction,
    '--quantum-enable-advanced': quantumConfig.enableAdvancedEffects ? 1 : 0,
    '--quantum-enable-physics': quantumConfig.enableParticlePhysics ? 1 : 0,
    '--quantum-enable-field': quantumConfig.enableQuantumField ? 1 : 0,
    '--performance-fps': performanceMetrics.fps,
    '--performance-memory': performanceMetrics.memoryUsage,
    '--device-orientation': window.screen?.orientation?.angle || 0,
  } as React.CSSProperties;

  return (
    <div
      className={`responsive-quantum-effects ${breakpoint} quality-${quantumConfig.shaderComplexity}`}
      style={quantumStyles}
      data-quantum-quality={forceQuality || getBatteryOptimizedSettings(deviceCapabilities)}
      data-performance-fps={performanceMetrics.fps}
      data-device-memory={deviceCapabilities.deviceMemory}
      data-battery-level={Math.round(deviceCapabilities.batteryLevel * 100)}
    >
      {children}

      {/* Performance indicator for development */}
      {process.env.NODE_ENV === 'development' && (
        <div className="quantum-performance-indicator">
          <div>FPS: {performanceMetrics.fps}</div>
          <div>Particles: {quantumConfig.particleCount}</div>
          <div>Quality: {forceQuality || getBatteryOptimizedSettings(deviceCapabilities)}</div>
          <div>Battery: {Math.round(deviceCapabilities.batteryLevel * 100)}%</div>
          <div>Memory: {deviceCapabilities.deviceMemory}GB</div>
        </div>
      )}

      <style jsx>{`
        .responsive-quantum-effects {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
        }

        /* Quality-based styles */
        .quality-0 {
          --quantum-glow-intensity: 0;
          --quantum-blur-radius: 0px;
          --quantum-shadow-spread: 0px;
        }

        .quality-1 {
          --quantum-glow-intensity: 0.3;
          --quantum-blur-radius: 2px;
          --quantum-shadow-spread: 4px;
        }

        .quality-2 {
          --quantum-glow-intensity: 0.6;
          --quantum-blur-radius: 4px;
          --quantum-shadow-spread: 8px;
        }

        .quality-3 {
          --quantum-glow-intensity: 1.0;
          --quantum-blur-radius: 6px;
          --quantum-shadow-spread: 12px;
        }

        /* Responsive breakpoint styles */
        .mobile {
          --quantum-ui-scale: 0.9;
          --quantum-text-scale: 0.95;
          --quantum-spacing-scale: 0.8;
        }

        .tablet {
          --quantum-ui-scale: 0.95;
          --quantum-text-scale: 0.98;
          --quantum-spacing-scale: 0.9;
        }

        .desktop,
        .wide {
          --quantum-ui-scale: 1.0;
          --quantum-text-scale: 1.0;
          --quantum-spacing-scale: 1.0;
        }

        /* Performance indicator */
        .quantum-performance-indicator {
          position: fixed;
          top: 80px;
          right: 10px;
          background: rgba(0, 0, 0, 0.8);
          color: #7BE1FF;
          padding: 8px;
          border-radius: 4px;
          font-size: 10px;
          font-family: monospace;
          z-index: 10000;
          pointer-events: none;
          border: 1px solid rgba(123, 225, 255, 0.3);
        }

        .quantum-performance-indicator div {
          margin: 2px 0;
        }

        /* Adaptive quantum field effects */
        :global(.quantum-field) {
          opacity: calc(var(--quantum-enable-field) * var(--quantum-animation-intensity));
          transform: scale(var(--quantum-scale-field));
        }

        :global(.quantum-particle) {
          opacity: calc(var(--quantum-enable-physics) * var(--quantum-animation-intensity));
          transform: scale(var(--quantum-scale-particle));
        }

        :global(.quantum-interaction-zone) {
          transform: scale(var(--quantum-scale-interaction));
          min-width: var(--touch-target-size, 44px);
          min-height: var(--touch-target-size, 44px);
        }

        /* Orientation-specific adjustments */
        @media (orientation: portrait) {
          .responsive-quantum-effects {
            --quantum-layout-direction: column;
            --quantum-aspect-ratio: portrait;
          }
        }

        @media (orientation: landscape) {
          .responsive-quantum-effects {
            --quantum-layout-direction: row;
            --quantum-aspect-ratio: landscape;
          }
        }

        /* Safe area adjustments for mobile devices */
        @supports (padding: env(safe-area-inset-top)) {
          .mobile .responsive-quantum-effects {
            padding-top: var(--safe-area-inset-top);
            padding-bottom: var(--safe-area-inset-bottom);
            padding-left: var(--safe-area-inset-left);
            padding-right: var(--safe-area-inset-right);
          }
        }

        /* High DPI display optimizations */
        @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
          .responsive-quantum-effects {
            --quantum-texture-quality: high;
            --quantum-line-width: 0.5px;
          }
        }

        /* Reduced data mode */
        @media (prefers-reduced-data: reduce) {
          .responsive-quantum-effects {
            --quantum-particle-count: 0;
            --quantum-enable-field: 0;
            --quantum-enable-physics: 0;
          }
        }

        /* Print styles */
        @media print {
          .responsive-quantum-effects {
            --quantum-particle-count: 0;
            --quantum-enable-field: 0;
            --quantum-animation-intensity: 0;
          }
          
          .quantum-performance-indicator {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}
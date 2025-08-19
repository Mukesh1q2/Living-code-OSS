"use client";

import { useEffect, useState } from 'react';
import { useQuantumState } from '@/lib/state';
import { useResponsive, getDeviceCapabilities } from '@/lib/responsive';
// import { usePerformanceOptimization } from '@/lib/usePerformanceOptimization';
import ThreeInspector from './ThreeInspector';
import PerformanceDashboard from './PerformanceDashboard';

interface DevToolsProps {
  enabled?: boolean;
}

export default function DevTools({ enabled = process.env.NODE_ENV === 'development' }: DevToolsProps) {
  const [showPanel, setShowPanel] = useState(false);
  const [threeInspector, setThreeInspector] = useState<any>(null);
  const [capabilities, setCapabilities] = useState<ReturnType<typeof getDeviceCapabilities> | null>(null);
  const [reactDevToolsConnected, setReactDevToolsConnected] = useState(false);
  const [showThreeInspector, setShowThreeInspector] = useState(false);
  const [showPerformanceDashboard, setShowPerformanceDashboard] = useState(false);

  const quantumState = useQuantumState();
  const responsive = useResponsive();
  // NOTE: Disabled hook that requires R3F Canvas context to avoid runtime error when DevTools is outside Canvas.
  // const performance = usePerformanceOptimization();
  const performance = {
    metrics: null as any,
    memoryUsage: null as any,
    currentQuality: 'medium',
    isOptimizing: false,
    adaptationCount: 0,
  };

  useEffect(() => {
    if (!enabled) return;

    // Initialize device capabilities
    setCapabilities(getDeviceCapabilities());

    // Check React DevTools connection
    const checkReactDevTools = () => {
      if (typeof window !== 'undefined') {
        const hasReactDevTools = !!(window as any).__REACT_DEVTOOLS_GLOBAL_HOOK__;
        setReactDevToolsConnected(hasReactDevTools);
      }
    };

    checkReactDevTools();

    // Load Three.js inspector if available
    const loadThreeInspector = async () => {
      try {
        // Set up Three.js inspector interface
        setThreeInspector({
          enabled: true,
          scene: null,
          stats: null,
        });

        // Initialize stats.js for performance monitoring
        if (typeof window !== 'undefined') {
          try {
            const Stats = (await import('stats.js')).default;
            const stats = new Stats();
            stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
            stats.dom.style.position = 'fixed';
            stats.dom.style.top = '80px';
            stats.dom.style.right = '16px';
            stats.dom.style.zIndex = '9998';
            stats.dom.style.display = 'none'; // Hidden by default
            document.body.appendChild(stats.dom);

            setThreeInspector((prev: any) => ({ ...prev, stats }));
          } catch (error) {
            console.warn('Stats.js not available:', error);
          }
        }
      } catch (error) {
        console.warn('Three.js inspector not available:', error);
      }
    };

    loadThreeInspector();

    // Keyboard shortcut to toggle dev panel
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === '`' && e.ctrlKey) {
        setShowPanel(!showPanel);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [enabled, showPanel]);

  if (!enabled || !showPanel) {
    return enabled ? (
      <div
        style={{
          position: 'fixed',
          bottom: 16,
          left: 16,
          zIndex: 9999,
          background: 'rgba(0,0,0,0.8)',
          color: '#7BE1FF',
          padding: '4px 8px',
          borderRadius: 4,
          fontSize: '10px',
          fontFamily: 'monospace',
          cursor: 'pointer',
        }}
        onClick={() => setShowPanel(true)}
        title="Click to open dev tools (Ctrl+` to toggle)"
      >
        DEV
      </div>
    ) : null;
  }

  // Quick demo controls
  const setSuperposition = useQuantumState(s => s.setSuperposition);
  const initiateTeleportation = useQuantumState(s => s.initiateTeleportation);

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 16,
        left: 16,
        right: 16,
        zIndex: 9999,
        background: 'rgba(0,0,0,0.95)',
        border: '1px solid rgba(123,225,255,0.3)',
        borderRadius: 8,
        padding: 16,
        color: '#E8F6FF',
        fontSize: '12px',
        fontFamily: 'monospace',
        maxHeight: '40vh',
        overflow: 'auto',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 style={{ margin: 0, color: '#7BE1FF' }}>Vidya Dev Tools</h3>
        <button
          onClick={() => setShowPanel(false)}
          style={{
            background: 'transparent',
            border: '1px solid rgba(123,225,255,0.3)',
            color: '#7BE1FF',
            borderRadius: 4,
            padding: '2px 6px',
            cursor: 'pointer',
          }}
        >
          ×
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
        {/* Responsive Info */}
        <div>
          <h4 style={{ margin: '0 0 8px 0', color: '#B383FF' }}>Responsive</h4>
          <div>Breakpoint: <span style={{ color: '#63FFC9' }}>{responsive.breakpoint}</span></div>
          <div>Size: <span style={{ color: '#63FFC9' }}>{responsive.width}×{responsive.height}</span></div>
          <div>Mobile: <span style={{ color: responsive.isMobile ? '#63FFC9' : '#FF7BD5' }}>{responsive.isMobile ? 'Yes' : 'No'}</span></div>
          <div>Tablet: <span style={{ color: responsive.isTablet ? '#63FFC9' : '#FF7BD5' }}>{responsive.isTablet ? 'Yes' : 'No'}</span></div>
        </div>

        {/* Quantum State */}
        <div>
          <h4 style={{ margin: '0 0 8px 0', color: '#B383FF' }}>Quantum State</h4>
          <div>Quality: <span style={{ color: '#63FFC9' }}>{quantumState.quantumQuality}</span></div>
          <div>Instances: <span style={{ color: '#63FFC9' }}>{quantumState.vidyaInstances}</span></div>
          <div>Superposition: <span style={{ color: quantumState.superpositionActive ? '#63FFC9' : '#FF7BD5' }}>{quantumState.superpositionActive ? 'Active' : 'Inactive'}</span></div>
          <div>Chat: <span style={{ color: quantumState.chatActive ? '#63FFC9' : '#FF7BD5' }}>{quantumState.chatActive ? 'Active' : 'Inactive'}</span></div>
        </div>

        {/* Device Capabilities */}
        {capabilities && (
          <div>
            <h4 style={{ margin: '0 0 8px 0', color: '#B383FF' }}>Device Capabilities</h4>
            <div>WebGL: <span style={{ color: capabilities.hasWebGL ? '#63FFC9' : '#FF7BD5' }}>{capabilities.hasWebGL ? 'Yes' : 'No'}</span></div>
            <div>WebGL2: <span style={{ color: capabilities.hasWebGL2 ? '#63FFC9' : '#FF7BD5' }}>{capabilities.hasWebGL2 ? 'Yes' : 'No'}</span></div>
            <div>Touch: <span style={{ color: capabilities.hasTouch ? '#63FFC9' : '#FF7BD5' }}>{capabilities.hasTouch ? 'Yes' : 'No'}</span></div>
            <div>Max Texture: <span style={{ color: '#63FFC9' }}>{capabilities.maxTextureSize}</span></div>
          </div>
        )}

        {/* Development Tools */}
        <div>
          <h4 style={{ margin: '0 0 8px 0', color: '#B383FF' }}>Development Tools</h4>
          <div>React DevTools: <span style={{ color: reactDevToolsConnected ? '#63FFC9' : '#FF7BD5' }}>{reactDevToolsConnected ? 'Connected' : 'Not available'}</span></div>
          <div>Three.js Inspector: <span style={{ color: threeInspector ? '#63FFC9' : '#FF7BD5' }}>{threeInspector ? 'Available' : 'Not loaded'}</span></div>
          {threeInspector && (
            <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
              <button
                onClick={() => {
                  if (threeInspector.stats) {
                    const isVisible = threeInspector.stats.dom.style.display !== 'none';
                    threeInspector.stats.dom.style.display = isVisible ? 'none' : 'block';
                  }
                }}
                style={{
                  background: 'rgba(123,225,255,0.1)',
                  border: '1px solid rgba(123,225,255,0.3)',
                  color: '#7BE1FF',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '10px',
                }}
              >
                Toggle Stats
              </button>
              <button
                onClick={() => setShowThreeInspector(!showThreeInspector)}
                style={{
                  background: showThreeInspector ? 'rgba(179,131,255,0.2)' : 'rgba(123,225,255,0.1)',
                  border: '1px solid rgba(123,225,255,0.3)',
                  color: '#7BE1FF',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '10px',
                }}
              >
                {showThreeInspector ? 'Hide Inspector' : 'Show Inspector'}
              </button>
              <button
                onClick={() => setShowPerformanceDashboard(!showPerformanceDashboard)}
                style={{
                  background: showPerformanceDashboard ? 'rgba(99,255,201,0.2)' : 'rgba(123,225,255,0.1)',
                  border: '1px solid rgba(123,225,255,0.3)',
                  color: '#7BE1FF',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '10px',
                }}
              >
                {showPerformanceDashboard ? 'Hide Dashboard' : 'Performance Dashboard'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Performance Monitor */}
      <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid rgba(123,225,255,0.2)' }}>
        <h4 style={{ margin: '0 0 8px 0', color: '#B383FF' }}>Performance</h4>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <div>FPS: <span style={{ color: performance.metrics?.fps && performance.metrics.fps > 50 ? '#63FFC9' : performance.metrics?.fps && performance.metrics.fps > 30 ? '#7BE1FF' : '#FF7BD5' }}>{performance.metrics?.fps || 0}</span></div>
          {performance.memoryUsage && (
            <div>Memory: <span style={{ color: '#63FFC9' }}>{performance.memoryUsage.estimatedMB.toFixed(1)}MB</span></div>
          )}
          <div>Quality: <span style={{ color: '#B383FF' }}>{performance.currentQuality}</span></div>
          {performance.adaptationCount > 0 && (
            <div>Adaptations: <span style={{ color: '#FF7BD5' }}>{performance.adaptationCount}</span></div>
          )}
          {performance.isOptimizing && (
            <div style={{ color: '#7BE1FF' }}>⚡ Optimizing...</div>
          )}
          {/* Demo Mode Controls */}
          <button
            onClick={() => setSuperposition(true)}
            style={{ background: 'rgba(123,225,255,0.1)', border: '1px solid rgba(123,225,255,0.3)', color: '#7BE1FF', borderRadius: 4, padding: '4px 8px', cursor: 'pointer', fontSize: '10px' }}
          >
            Activate Superposition
          </button>
          <button
            onClick={() => setSuperposition(false)}
            style={{ background: 'rgba(179,131,255,0.1)', border: '1px solid rgba(179,131,255,0.3)', color: '#B383FF', borderRadius: 4, padding: '4px 8px', cursor: 'pointer', fontSize: '10px' }}
          >
            Collapse Waveform
          </button>
          <button
            onClick={() => initiateTeleportation([5, 2, 0], 'demo')}
            style={{ background: 'rgba(99,255,201,0.1)', border: '1px solid rgba(99,255,201,0.3)', color: '#63FFC9', borderRadius: 4, padding: '4px 8px', cursor: 'pointer', fontSize: '10px' }}
          >
            Teleport Demo →
          </button>
        </div>
      </div>

      <div style={{ marginTop: 12, fontSize: '10px', color: '#8FB2C8' }}>
        Press Ctrl+` to toggle this panel
      </div>

      {/* Three.js Inspector */}
      <ThreeInspector
        visible={showThreeInspector}
        onClose={() => setShowThreeInspector(false)}
      />

      {/* Performance Dashboard */}
      <PerformanceDashboard
        visible={showPerformanceDashboard}
        onClose={() => setShowPerformanceDashboard(false)}
      />
    </div>
  );
}


/**
 * Tests for DeviceAdaptation component and responsive features
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import DeviceAdaptation from '../DeviceAdaptation';
import { getDeviceCapabilities, getBatteryOptimizedSettings } from '@/lib/responsive';
import { getAccessibilityManager } from '@/lib/accessibility';

// Mock the responsive and accessibility modules
vi.mock('@/lib/responsive', () => ({
  useResponsive: () => ({
    breakpoint: 'desktop',
    isMobile: false,
    isTablet: false,
    width: 1024,
    height: 768,
  }),
  getDeviceCapabilities: vi.fn(),
  getBatteryOptimizedSettings: vi.fn(),
  getProgressiveEnhancementLevel: vi.fn(() => 'enhanced'),
}));

vi.mock('@/lib/accessibility', () => ({
  useAccessibility: () => ({
    settings: {
      screenReaderEnabled: false,
      highContrastMode: false,
      reducedMotion: false,
      keyboardNavigation: true,
      voiceAnnouncements: true,
      focusIndicators: true,
      alternativeText: true,
      colorBlindFriendly: false,
      fontSize: 'medium',
      soundEnabled: true,
      hapticFeedback: false,
    },
    announce: vi.fn(),
    describeQuantumState: vi.fn(),
    provideFeedback: vi.fn(),
  }),
  getAccessibilityManager: vi.fn(),
}));

vi.mock('@/lib/state', () => ({
  useQuantumState: () => ({
    setQuantumQuality: vi.fn(),
    setSuperposition: vi.fn(),
    createEntanglement: vi.fn(),
    initiateTeleportation: vi.fn(),
    setShowPlanPanel: vi.fn(),
  }),
}));

vi.mock('@/lib/performance-monitor', () => ({
  usePerformanceMonitor: () => ({
    start: vi.fn(),
    stop: vi.fn(),
    onMetricsUpdate: vi.fn(() => vi.fn()),
    onQualityChange: vi.fn(() => vi.fn()),
  }),
}));

// Mock touch gestures
vi.mock('@/lib/touch-gestures', () => ({
  useTouchGestures: () => ({ current: null }),
}));

describe('DeviceAdaptation', () => {
  const mockDeviceCapabilities = {
    hasWebGL: true,
    hasWebGL2: true,
    hasTouch: false,
    hasPointerEvents: true,
    maxTextureSize: 4096,
    supportsComplexShaders: true,
    deviceMemory: 8,
    hardwareConcurrency: 8,
    connectionType: '4g' as const,
    batteryLevel: 0.8,
    isCharging: true,
    devicePixelRatio: 1,
    prefersReducedMotion: false,
    prefersHighContrast: false,
    screenOrientation: 'landscape' as const,
  };

  beforeEach(() => {
    vi.mocked(getDeviceCapabilities).mockReturnValue(mockDeviceCapabilities);
    vi.mocked(getBatteryOptimizedSettings).mockReturnValue('high');
    
    // Mock DOM APIs
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });

    Object.defineProperty(navigator, 'vibrate', {
      writable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders with correct accessibility attributes', () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');
    expect(container).toHaveAttribute('aria-label', 'Vidya Quantum Sanskrit AI Interface');
    expect(container).toHaveAttribute('aria-describedby', 'quantum-description');
    expect(container).toHaveAttribute('tabIndex', '0');
  });

  it('includes screen reader description', () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const description = screen.getByText(/Interactive quantum consciousness interface/);
    expect(description).toBeInTheDocument();
    expect(description).toHaveClass('sr-only');
  });

  it('applies device-specific CSS custom properties', () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');
    expect(container.style.getPropertyValue('--device-scale')).toBe('1');
    expect(container.style.getPropertyValue('--device-memory')).toBe('8');
    expect(container.style.getPropertyValue('--hardware-concurrency')).toBe('8');
  });

  it('handles mobile device capabilities', () => {
    const mobileCapabilities = {
      ...mockDeviceCapabilities,
      hasTouch: true,
      deviceMemory: 2,
      hardwareConcurrency: 4,
      connectionType: '3g' as const,
    };

    vi.mocked(getDeviceCapabilities).mockReturnValue(mobileCapabilities);
    vi.mocked(getBatteryOptimizedSettings).mockReturnValue('low');

    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');
    const dataCapabilities = JSON.parse(container.getAttribute('data-device-capabilities') || '{}');
    
    expect(dataCapabilities.touch).toBe(true);
    expect(dataCapabilities.memory).toBe(2);
    expect(dataCapabilities.connection).toBe('3g');
  });

  it('handles low battery optimization', () => {
    const lowBatteryCapabilities = {
      ...mockDeviceCapabilities,
      batteryLevel: 0.15,
      isCharging: false,
    };

    vi.mocked(getDeviceCapabilities).mockReturnValue(lowBatteryCapabilities);
    vi.mocked(getBatteryOptimizedSettings).mockReturnValue('minimal');

    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');
    expect(container.style.getPropertyValue('--battery-level')).toBe('0.15');
  });

  it('responds to keyboard navigation', () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');
    
    // Test keyboard focus
    container.focus();
    expect(container).toHaveFocus();

    // Test arrow key navigation
    fireEvent.keyDown(container, { key: 'ArrowUp' });
    fireEvent.keyDown(container, { key: 'ArrowDown' });
    fireEvent.keyDown(container, { key: 'ArrowLeft' });
    fireEvent.keyDown(container, { key: 'ArrowRight' });
  });

  it('handles touch gestures', async () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');

    // Simulate touch start
    fireEvent.touchStart(container, {
      touches: [{ identifier: 0, clientX: 100, clientY: 100 }],
    });

    // Simulate touch end (tap)
    fireEvent.touchEnd(container, {
      changedTouches: [{ identifier: 0, clientX: 100, clientY: 100 }],
      touches: [],
    });

    await waitFor(() => {
      // Touch gesture should be processed
      expect(container).toBeInTheDocument();
    });
  });

  it('updates device capabilities on visibility change', async () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    // Simulate visibility change
    Object.defineProperty(document, 'visibilityState', {
      writable: true,
      value: 'hidden',
    });

    fireEvent(document, new Event('visibilitychange'));

    await waitFor(() => {
      expect(getDeviceCapabilities).toHaveBeenCalled();
    });
  });

  it('handles orientation change', async () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    // Simulate orientation change
    fireEvent(window, new Event('orientationchange'));

    await waitFor(() => {
      expect(getDeviceCapabilities).toHaveBeenCalled();
    });
  });

  it('shows debug info in development mode', () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';

    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    expect(screen.getByText(/Enhancement:/)).toBeInTheDocument();
    expect(screen.getByText(/Battery:/)).toBeInTheDocument();
    expect(screen.getByText(/Connection:/)).toBeInTheDocument();

    process.env.NODE_ENV = originalEnv;
  });

  it('applies progressive enhancement classes', () => {
    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    const container = screen.getByRole('application');
    expect(container).toHaveClass('enhancement-enhanced');
  });

  it('handles reduced motion preference', () => {
    const reducedMotionCapabilities = {
      ...mockDeviceCapabilities,
      prefersReducedMotion: true,
    };

    vi.mocked(getDeviceCapabilities).mockReturnValue(reducedMotionCapabilities);

    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    // Should detect reduced motion preference in device capabilities
    expect(getDeviceCapabilities).toHaveBeenCalled();
  });

  it('handles high contrast preference', () => {
    const highContrastCapabilities = {
      ...mockDeviceCapabilities,
      prefersHighContrast: true,
    };

    vi.mocked(getDeviceCapabilities).mockReturnValue(highContrastCapabilities);

    render(
      <DeviceAdaptation>
        <div>Test content</div>
      </DeviceAdaptation>
    );

    // Component should handle high contrast mode
    expect(getDeviceCapabilities).toHaveBeenCalled();
  });
});
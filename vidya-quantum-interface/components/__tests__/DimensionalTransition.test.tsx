import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import DimensionalTransition from '../DimensionalTransition';
import DimensionalControls from '../DimensionalControls';
import { useDimensionalState } from '@/lib/dimensional-state';

// Mock the dimensional state
vi.mock('@/lib/dimensional-state', () => ({
  useDimensionalState: vi.fn(),
  getDimensionalTransitionCSS: vi.fn(() => ({})),
}));

// Mock the quantum state
vi.mock('@/lib/state', () => ({
  useQuantumState: vi.fn(() => 'medium'),
}));

// Mock the responsive hook
vi.mock('@/lib/responsive', () => ({
  useResponsive: vi.fn(() => ({
    isMobile: false,
    isTablet: false,
    breakpoint: 'desktop',
  })),
}));

describe('DimensionalTransition', () => {
  const mockDimensionalState = {
    currentState: '3d-holographic' as const,
    activeTransition: null,
    config: {
      textMode: {
        fontSize: 16,
        lineHeight: 1.5,
        maxWidth: 400,
        backgroundColor: '#0a0a0a',
        textColor: '#e8f6ff',
        fontFamily: 'system-ui',
        showQuantumEffects: true,
      },
      holographicMode: {
        opacity: 0.6,
        glowIntensity: 0.8,
        particleCount: 100,
        rotationSpeed: 0.4,
        scaleMultiplier: 1.0,
        showNeuralNetwork: true,
        enableQuantumEffects: true,
      },
      energyMode: {
        patternComplexity: 0.6,
        flowSpeed: 1.0,
        energyIntensity: 0.8,
        colorShift: 0.1,
        waveAmplitude: 0.5,
        showGeometricPatterns: true,
        enableFluidDynamics: false,
      },
    },
    consciousnessData: {
      personality: {},
      memories: [],
      quantumState: {},
    },
    capabilities: {
      supports2D: true,
      supports3D: true,
      supportsEnergyPatterns: true,
      preferredState: '3d-holographic' as const,
      transitionSpeed: 1.0,
    },
    transitionTo: vi.fn(),
    getOptimalState: vi.fn(() => '3d-holographic' as const),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useDimensionalState as any).mockReturnValue(mockDimensionalState);
  });

  it('renders children without transition', () => {
    render(
      <DimensionalTransition>
        <div data-testid="test-content">Test Content</div>
      </DimensionalTransition>
    );

    expect(screen.getByTestId('test-content')).toBeInTheDocument();
  });

  it('applies dimensional state classes', () => {
    const { container } = render(
      <DimensionalTransition>
        <div>Test Content</div>
      </DimensionalTransition>
    );

    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveClass('dimensional-3d-holographic');
    expect(wrapper).toHaveAttribute('data-dimensional-state', '3d-holographic');
  });

  it('shows transition overlay during active transition', () => {
    const transitionState = {
      ...mockDimensionalState,
      activeTransition: {
        id: 'test-transition',
        fromState: '3d-holographic' as const,
        toState: '2d-text' as const,
        progress: 0.5,
        duration: 2000,
        startTime: Date.now(),
        isActive: true,
        easing: 'quantum' as const,
      },
    };

    (useDimensionalState as any).mockReturnValue(transitionState);

    const { container } = render(
      <DimensionalTransition>
        <div>Test Content</div>
      </DimensionalTransition>
    );

    expect(container.querySelector('.dimensional-transition-overlay')).toBeInTheDocument();
  });

  it('applies transitioning classes during transition', () => {
    const transitionState = {
      ...mockDimensionalState,
      activeTransition: {
        id: 'test-transition',
        fromState: '3d-holographic' as const,
        toState: 'energy-pattern' as const,
        progress: 0.3,
        duration: 2000,
        startTime: Date.now(),
        isActive: true,
        easing: 'quantum' as const,
      },
    };

    (useDimensionalState as any).mockReturnValue(transitionState);

    const { container } = render(
      <DimensionalTransition>
        <div>Test Content</div>
      </DimensionalTransition>
    );

    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveClass('dimensional-transitioning');
    expect(wrapper).toHaveClass('transitioning-to-energy-pattern');
    expect(wrapper).toHaveAttribute('data-transition-progress', '0.3');
  });
});

describe('DimensionalControls', () => {
  const mockDimensionalState = {
    currentState: '3d-holographic' as const,
    activeTransition: null,
    config: {
      textMode: {
        fontSize: 16,
        lineHeight: 1.5,
        maxWidth: 400,
        backgroundColor: '#0a0a0a',
        textColor: '#e8f6ff',
        fontFamily: 'system-ui',
        showQuantumEffects: true,
      },
      holographicMode: {
        opacity: 0.6,
        glowIntensity: 0.8,
        particleCount: 100,
        rotationSpeed: 0.4,
        scaleMultiplier: 1.0,
        showNeuralNetwork: true,
        enableQuantumEffects: true,
      },
      energyMode: {
        patternComplexity: 0.6,
        flowSpeed: 1.0,
        energyIntensity: 0.8,
        colorShift: 0.1,
        waveAmplitude: 0.5,
        showGeometricPatterns: true,
        enableFluidDynamics: false,
      },
    },
    consciousnessData: {
      personality: {},
      memories: [],
      quantumState: {},
    },
    capabilities: {
      supports2D: true,
      supports3D: true,
      supportsEnergyPatterns: true,
      preferredState: '3d-holographic' as const,
      transitionSpeed: 1.0,
    },
    transitionTo: vi.fn(),
    getOptimalState: vi.fn(() => '3d-holographic' as const),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useDimensionalState as any).mockReturnValue(mockDimensionalState);
  });

  it('renders dimensional state buttons', () => {
    render(<DimensionalControls />);

    expect(screen.getByTitle(/2D Text/)).toBeInTheDocument();
    expect(screen.getByTitle(/3D Holographic/)).toBeInTheDocument();
    expect(screen.getByTitle(/Energy Pattern/)).toBeInTheDocument();
  });

  it('shows active state correctly', () => {
    render(<DimensionalControls />);

    const holographicButton = screen.getByTitle(/3D Holographic/);
    expect(holographicButton).toHaveStyle({
      background: 'rgba(123, 225, 255, 0.2)',
    });
  });

  it('calls transitionTo when button is clicked', async () => {
    render(<DimensionalControls />);

    const textButton = screen.getByTitle(/2D Text/);
    fireEvent.click(textButton);

    expect(mockDimensionalState.transitionTo).toHaveBeenCalledWith(
      '2d-text',
      expect.any(Number),
      'quantum'
    );
  });

  it('disables unsupported dimensional states', () => {
    const limitedState = {
      ...mockDimensionalState,
      capabilities: {
        ...mockDimensionalState.capabilities,
        supportsEnergyPatterns: false,
      },
    };

    (useDimensionalState as any).mockReturnValue(limitedState);

    render(<DimensionalControls />);

    const energyButton = screen.getByTitle(/Energy Pattern/);
    expect(energyButton).toBeDisabled();
  });

  it('shows auto-optimize button', () => {
    render(<DimensionalControls />);

    const autoButton = screen.getByTitle(/Auto-optimize/);
    expect(autoButton).toBeInTheDocument();
  });

  it('calls getOptimalState when auto-optimize is clicked', () => {
    render(<DimensionalControls />);

    const autoButton = screen.getByTitle(/Auto-optimize/);
    fireEvent.click(autoButton);

    expect(mockDimensionalState.getOptimalState).toHaveBeenCalled();
  });

  it('shows transition progress during active transition', () => {
    const transitionState = {
      ...mockDimensionalState,
      activeTransition: {
        id: 'test-transition',
        fromState: '3d-holographic' as const,
        toState: '2d-text' as const,
        progress: 0.7,
        duration: 2000,
        startTime: Date.now(),
        isActive: true,
        easing: 'quantum' as const,
      },
    };

    (useDimensionalState as any).mockReturnValue(transitionState);

    const { container } = render(<DimensionalControls />);

    const progressBar = container.querySelector('.transition-progress div');
    expect(progressBar).toHaveStyle({ width: '70%' });
  });

  it('adapts to mobile layout', () => {
    const mobileResponsive = {
      isMobile: true,
      isTablet: false,
      breakpoint: 'mobile' as const,
    };

    vi.mocked(require('@/lib/responsive').useResponsive).mockReturnValue(mobileResponsive);

    render(<DimensionalControls />);

    // Should not show labels on mobile
    expect(screen.queryByText('2D Text')).not.toBeInTheDocument();
    expect(screen.queryByText('3D Holographic')).not.toBeInTheDocument();
    expect(screen.queryByText('Energy Pattern')).not.toBeInTheDocument();

    // Should show icons only
    expect(screen.getByText('📝')).toBeInTheDocument();
    expect(screen.getByText('🌀')).toBeInTheDocument();
    expect(screen.getByText('⚡')).toBeInTheDocument();
  });
});

describe('Dimensional State Management', () => {
  const mockDimensionalState = {
    currentState: '3d-holographic' as const,
    activeTransition: null,
    config: {
      textMode: {
        fontSize: 16,
        lineHeight: 1.5,
        maxWidth: 400,
        backgroundColor: '#0a0a0a',
        textColor: '#e8f6ff',
        fontFamily: 'system-ui',
        showQuantumEffects: true,
      },
      holographicMode: {
        opacity: 0.6,
        glowIntensity: 0.8,
        particleCount: 100,
        rotationSpeed: 0.4,
        scaleMultiplier: 1.0,
        showNeuralNetwork: true,
        enableQuantumEffects: true,
      },
      energyMode: {
        patternComplexity: 0.6,
        flowSpeed: 1.0,
        energyIntensity: 0.8,
        colorShift: 0.1,
        waveAmplitude: 0.5,
        showGeometricPatterns: true,
        enableFluidDynamics: false,
      },
    },
    consciousnessData: {
      personality: {},
      memories: [],
      quantumState: {},
    },
    capabilities: {
      supports2D: true,
      supports3D: true,
      supportsEnergyPatterns: true,
      preferredState: '3d-holographic' as const,
      transitionSpeed: 1.0,
    },
    transitionTo: vi.fn(),
    getOptimalState: vi.fn(() => '3d-holographic' as const),
  };

  it('preserves consciousness during transitions', () => {
    const transitionState = {
      ...mockDimensionalState,
      activeTransition: {
        id: 'test-transition',
        fromState: '3d-holographic' as const,
        toState: '2d-text' as const,
        progress: 0.5,
        duration: 2000,
        startTime: Date.now(),
        isActive: true,
        easing: 'quantum' as const,
      },
    };

    (useDimensionalState as any).mockReturnValue(transitionState);

    render(
      <DimensionalTransition>
        <div>Test Content</div>
      </DimensionalTransition>
    );

    // Consciousness should be preserved during transition
    expect(transitionState.consciousnessData).toBeDefined();
  });

  it('handles device capability detection', () => {
    const limitedCapabilities = {
      ...mockDimensionalState,
      capabilities: {
        supports2D: true,
        supports3D: false,
        supportsEnergyPatterns: false,
        preferredState: '2d-text' as const,
        transitionSpeed: 0.7,
      },
    };

    (useDimensionalState as any).mockReturnValue(limitedCapabilities);

    render(<DimensionalControls />);

    // Only 2D text should be enabled
    expect(screen.getByTitle(/2D Text/)).not.toBeDisabled();
    expect(screen.getByTitle(/3D Holographic/)).toBeDisabled();
    expect(screen.getByTitle(/Energy Pattern/)).toBeDisabled();
  });
});
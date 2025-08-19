"use client";

import React, { useState, useEffect } from 'react';
import { useDimensionalState, type DimensionalState } from '@/lib/dimensional-state';
import { useQuantumState } from '@/lib/state';
import { useResponsive } from '@/lib/responsive';

interface DimensionalControlsProps {
  className?: string;
  style?: React.CSSProperties;
  showLabels?: boolean;
  orientation?: 'horizontal' | 'vertical';
}

/**
 * DimensionalControls Component
 * 
 * Provides user controls for triggering dimensional transitions
 * Adapts to device capabilities and shows available dimensional states
 */
export default function DimensionalControls({
  className = '',
  style = {},
  showLabels = true,
  orientation = 'horizontal'
}: DimensionalControlsProps) {
  const {
    currentState,
    activeTransition,
    capabilities,
    transitionTo,
    getOptimalState,
  } = useDimensionalState();
  
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const { isMobile, isTablet } = useResponsive();
  
  const [hoveredState, setHoveredState] = useState<DimensionalState | null>(null);
  
  // Available dimensional states based on device capabilities
  const availableStates: Array<{
    state: DimensionalState;
    label: string;
    description: string;
    icon: string;
    supported: boolean;
  }> = [
    {
      state: '2d-text',
      label: '2D Text',
      description: 'Flat text interface with minimal effects',
      icon: '📝',
      supported: capabilities.supports2D,
    },
    {
      state: '3d-holographic',
      label: '3D Holographic',
      description: 'Full quantum consciousness visualization',
      icon: '🌀',
      supported: capabilities.supports3D,
    },
    {
      state: 'energy-pattern',
      label: 'Energy Pattern',
      description: 'Abstract energy flow visualization',
      icon: '⚡',
      supported: capabilities.supportsEnergyPatterns,
    },
  ];
  
  // Handle dimensional state change
  const handleStateChange = (newState: DimensionalState) => {
    if (newState === currentState || activeTransition) return;
    
    const stateInfo = availableStates.find(s => s.state === newState);
    if (!stateInfo?.supported) {
      console.warn(`Dimensional state ${newState} is not supported on this device`);
      return;
    }
    
    // Determine transition duration based on device capabilities
    const baseDuration = 2000;
    const deviceMultiplier = isMobile ? 0.7 : isTablet ? 0.85 : 1.0;
    const qualityMultiplier = quantumQuality === 'minimal' ? 0.5 : 
                             quantumQuality === 'low' ? 0.7 : 
                             quantumQuality === 'medium' ? 1.0 : 1.3;
    
    const duration = baseDuration * deviceMultiplier * qualityMultiplier;
    
    transitionTo(newState, duration, 'quantum');
  };
  
  // Auto-optimize for device capabilities
  const handleAutoOptimize = () => {
    const optimalState = getOptimalState();
    if (optimalState !== currentState) {
      handleStateChange(optimalState);
    }
  };
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.altKey) {
        switch (event.key) {
          case '1':
            event.preventDefault();
            handleStateChange('2d-text');
            break;
          case '2':
            event.preventDefault();
            handleStateChange('3d-holographic');
            break;
          case '3':
            event.preventDefault();
            handleStateChange('energy-pattern');
            break;
          case 'o':
            event.preventDefault();
            handleAutoOptimize();
            break;
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentState, activeTransition]);
  
  const containerStyles: React.CSSProperties = {
    display: 'flex',
    flexDirection: orientation === 'horizontal' ? 'row' : 'column',
    gap: isMobile ? '8px' : '12px',
    padding: isMobile ? '8px' : '12px',
    background: 'rgba(10, 10, 10, 0.8)',
    borderRadius: '12px',
    border: '1px solid rgba(123, 225, 255, 0.2)',
    backdropFilter: 'blur(10px)',
    ...style,
  };
  
  return (
    <div className={`dimensional-controls ${className}`} style={containerStyles}>
      {/* Dimensional state buttons */}
      {availableStates.map(({ state, label, description, icon, supported }) => (
        <DimensionalButton
          key={state}
          state={state}
          label={showLabels ? label : ''}
          description={description}
          icon={icon}
          isActive={currentState === state}
          isTransitioning={activeTransition?.toState === state}
          isSupported={supported}
          onClick={() => handleStateChange(state)}
          onHover={setHoveredState}
          showLabels={showLabels}
          isMobile={isMobile}
        />
      ))}
      
      {/* Auto-optimize button */}
      <button
        className="auto-optimize-button"
        onClick={handleAutoOptimize}
        title="Auto-optimize for device capabilities"
        style={{
          padding: isMobile ? '8px' : '10px 12px',
          background: 'rgba(99, 255, 201, 0.1)',
          border: '1px solid rgba(99, 255, 201, 0.3)',
          borderRadius: '8px',
          color: '#63FFC9',
          fontSize: isMobile ? '12px' : '14px',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(99, 255, 201, 0.2)';
          e.currentTarget.style.borderColor = 'rgba(99, 255, 201, 0.5)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(99, 255, 201, 0.1)';
          e.currentTarget.style.borderColor = 'rgba(99, 255, 201, 0.3)';
        }}
      >
        {isMobile ? '🎯' : '🎯 Auto'}
      </button>
      
      {/* Transition progress indicator */}
      {activeTransition && (
        <div
          className="transition-progress"
          style={{
            position: 'absolute',
            bottom: '-4px',
            left: '0',
            right: '0',
            height: '2px',
            background: 'rgba(123, 225, 255, 0.2)',
            borderRadius: '1px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              background: 'linear-gradient(90deg, #7BE1FF, #63FFC9)',
              width: `${activeTransition.progress * 100}%`,
              transition: 'width 0.1s ease',
              borderRadius: '1px',
            }}
          />
        </div>
      )}
      
      {/* Hover tooltip */}
      {hoveredState && showLabels && (
        <DimensionalTooltip
          state={hoveredState}
          availableStates={availableStates}
          currentState={currentState}
          capabilities={capabilities}
        />
      )}
    </div>
  );
}

/**
 * Individual Dimensional Button Component
 */
interface DimensionalButtonProps {
  state: DimensionalState;
  label: string;
  description: string;
  icon: string;
  isActive: boolean;
  isTransitioning: boolean;
  isSupported: boolean;
  onClick: () => void;
  onHover: (state: DimensionalState | null) => void;
  showLabels: boolean;
  isMobile: boolean;
}

function DimensionalButton({
  state,
  label,
  description,
  icon,
  isActive,
  isTransitioning,
  isSupported,
  onClick,
  onHover,
  showLabels,
  isMobile,
}: DimensionalButtonProps) {
  const [isHovered, setIsHovered] = useState(false);
  
  const buttonStyles: React.CSSProperties = {
    padding: isMobile ? '8px' : showLabels ? '10px 12px' : '10px',
    background: isActive 
      ? 'rgba(123, 225, 255, 0.2)' 
      : isSupported 
        ? 'rgba(123, 225, 255, 0.05)' 
        : 'rgba(100, 100, 100, 0.05)',
    border: `1px solid ${
      isActive 
        ? 'rgba(123, 225, 255, 0.5)' 
        : isSupported 
          ? 'rgba(123, 225, 255, 0.2)' 
          : 'rgba(100, 100, 100, 0.2)'
    }`,
    borderRadius: '8px',
    color: isSupported ? '#E8F6FF' : '#888',
    fontSize: isMobile ? '12px' : '14px',
    cursor: isSupported ? 'pointer' : 'not-allowed',
    transition: 'all 0.2s ease',
    opacity: isSupported ? 1 : 0.5,
    display: 'flex',
    alignItems: 'center',
    gap: showLabels ? '6px' : '0',
    minWidth: isMobile ? '40px' : showLabels ? '80px' : '40px',
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden',
  };
  
  // Hover and active state effects
  if (isHovered && isSupported) {
    buttonStyles.background = isActive 
      ? 'rgba(123, 225, 255, 0.3)' 
      : 'rgba(123, 225, 255, 0.1)';
    buttonStyles.borderColor = 'rgba(123, 225, 255, 0.4)';
    buttonStyles.transform = 'translateY(-1px)';
  }
  
  return (
    <button
      className={`dimensional-button dimensional-button-${state}`}
      style={buttonStyles}
      onClick={isSupported ? onClick : undefined}
      onMouseEnter={() => {
        if (isSupported) {
          setIsHovered(true);
          onHover(state);
        }
      }}
      onMouseLeave={() => {
        setIsHovered(false);
        onHover(null);
      }}
      disabled={!isSupported}
      title={isSupported ? description : `${label} not supported on this device`}
    >
      <span className="dimensional-button-icon">{icon}</span>
      {showLabels && !isMobile && (
        <span className="dimensional-button-label">{label}</span>
      )}
      
      {/* Transition animation overlay */}
      {isTransitioning && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'linear-gradient(45deg, transparent 30%, rgba(123, 225, 255, 0.3) 50%, transparent 70%)',
            animation: 'dimensionalTransitionShimmer 1s infinite',
            pointerEvents: 'none',
          }}
        />
      )}
      
      {/* Active state indicator */}
      {isActive && (
        <div
          style={{
            position: 'absolute',
            bottom: '2px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '20px',
            height: '2px',
            background: '#7BE1FF',
            borderRadius: '1px',
          }}
        />
      )}
    </button>
  );
}

/**
 * Dimensional Tooltip Component
 */
interface DimensionalTooltipProps {
  state: DimensionalState;
  availableStates: Array<{
    state: DimensionalState;
    label: string;
    description: string;
    icon: string;
    supported: boolean;
  }>;
  currentState: DimensionalState;
  capabilities: any;
}

function DimensionalTooltip({ 
  state, 
  availableStates, 
  currentState, 
  capabilities 
}: DimensionalTooltipProps) {
  const stateInfo = availableStates.find(s => s.state === state);
  if (!stateInfo) return null;
  
  return (
    <div
      className="dimensional-tooltip"
      style={{
        position: 'absolute',
        top: '-60px',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(10, 10, 10, 0.95)',
        border: '1px solid rgba(123, 225, 255, 0.3)',
        borderRadius: '8px',
        padding: '8px 12px',
        fontSize: '12px',
        color: '#E8F6FF',
        whiteSpace: 'nowrap',
        zIndex: 1000,
        backdropFilter: 'blur(10px)',
      }}
    >
      <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>
        {stateInfo.icon} {stateInfo.label}
      </div>
      <div style={{ opacity: 0.8 }}>
        {stateInfo.description}
      </div>
      {!stateInfo.supported && (
        <div style={{ color: '#FF8A80', fontSize: '11px', marginTop: '2px' }}>
          Not supported on this device
        </div>
      )}
      {state === currentState && (
        <div style={{ color: '#63FFC9', fontSize: '11px', marginTop: '2px' }}>
          Currently active
        </div>
      )}
    </div>
  );
}

// CSS animations for dimensional controls
export const dimensionalControlsStyles = `
  @keyframes dimensionalTransitionShimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
  
  .dimensional-controls {
    position: relative;
    user-select: none;
  }
  
  .dimensional-button {
    position: relative;
    font-family: inherit;
    font-weight: 500;
  }
  
  .dimensional-button:focus {
    outline: 2px solid rgba(123, 225, 255, 0.5);
    outline-offset: 2px;
  }
  
  .dimensional-button-icon {
    font-size: 16px;
    line-height: 1;
  }
  
  .dimensional-button-label {
    font-size: 12px;
    font-weight: 500;
  }
  
  .auto-optimize-button:focus {
    outline: 2px solid rgba(99, 255, 201, 0.5);
    outline-offset: 2px;
  }
  
  .transition-progress {
    pointer-events: none;
  }
  
  .dimensional-tooltip::before {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: rgba(123, 225, 255, 0.3);
  }
`;
"use client";

import React, { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useDimensionalState, getDimensionalTransitionCSS, type DimensionalTransition as DimensionalTransitionType } from '@/lib/dimensional-state';
import { useQuantumState } from '@/lib/state';
import * as THREE from 'three';

interface DimensionalTransitionProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * DimensionalTransition Component
 * 
 * Wraps content and applies dimensional transition effects
 * Handles smooth transitions between 2D text, 3D holographic, and energy pattern modes
 */
export default function DimensionalTransition({ 
  children, 
  className = '', 
  style = {} 
}: DimensionalTransitionProps) {
  const {
    currentState,
    activeTransition,
    config,
    consciousnessData,
  } = useDimensionalState();
  
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Apply transition CSS
  const transitionStyles = getDimensionalTransitionCSS(activeTransition, currentState);
  
  // Combined styles
  const combinedStyles: React.CSSProperties = {
    ...style,
    ...transitionStyles,
    position: 'relative',
    width: '100%',
    height: '100%',
  };
  
  // Add dimensional state-specific classes
  const dimensionalClasses = [
    className,
    `dimensional-${currentState}`,
    activeTransition ? 'dimensional-transitioning' : '',
    activeTransition ? `transitioning-to-${activeTransition.toState}` : '',
  ].filter(Boolean).join(' ');
  
  return (
    <div
      ref={containerRef}
      className={dimensionalClasses}
      style={combinedStyles}
      data-dimensional-state={currentState}
      data-transition-progress={activeTransition?.progress || 0}
    >
      {children}
      
      {/* Transition overlay effects */}
      {activeTransition && (
        <DimensionalTransitionOverlay
          transition={activeTransition}
          quantumQuality={quantumQuality}
        />
      )}
    </div>
  );
}

/**
 * Dimensional Transition Overlay
 * 
 * Provides visual effects during dimensional transitions
 */
interface DimensionalTransitionOverlayProps {
  transition: DimensionalTransitionType;
  quantumQuality: string;
}

function DimensionalTransitionOverlay({ 
  transition, 
  quantumQuality 
}: DimensionalTransitionOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  
  // Skip overlay for minimal quality
  if (quantumQuality === 'minimal') {
    return null;
  }
  
  const { progress, fromState, toState, easing } = transition;
  
  // Generate transition-specific effects
  const getTransitionEffects = () => {
    const effects: React.CSSProperties = {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      pointerEvents: 'none',
      zIndex: 1000,
    };
    
    // 2D to 3D transition effects
    if (fromState === '2d-text' && toState === '3d-holographic') {
      return {
        ...effects,
        background: `
          radial-gradient(
            circle at 50% 50%, 
            rgba(123, 225, 255, ${progress * 0.3}) 0%, 
            rgba(123, 225, 255, ${progress * 0.1}) 50%, 
            transparent 100%
          )
        `,
        backdropFilter: `blur(${progress * 2}px)`,
      };
    }
    
    // 3D to 2D transition effects
    if (fromState === '3d-holographic' && toState === '2d-text') {
      return {
        ...effects,
        background: `
          linear-gradient(
            ${progress * 180}deg, 
            rgba(10, 10, 10, ${progress * 0.8}) 0%, 
            rgba(10, 10, 10, ${progress * 0.4}) 100%
          )
        `,
      };
    }
    
    // Energy pattern transitions
    if (toState === 'energy-pattern') {
      return {
        ...effects,
        background: `
          conic-gradient(
            from ${progress * 360}deg,
            rgba(255, 179, 102, ${progress * 0.4}),
            rgba(179, 131, 255, ${progress * 0.4}),
            rgba(99, 255, 201, ${progress * 0.4}),
            rgba(255, 179, 102, ${progress * 0.4})
          )
        `,
        filter: `blur(${progress * 5}px)`,
        animation: quantumQuality === 'high' ? `energyFlow ${2 - progress}s infinite linear` : undefined,
      };
    }
    
    if (fromState === 'energy-pattern') {
      return {
        ...effects,
        background: `
          conic-gradient(
            from ${(1 - progress) * 360}deg,
            rgba(255, 179, 102, ${(1 - progress) * 0.4}),
            rgba(179, 131, 255, ${(1 - progress) * 0.4}),
            rgba(99, 255, 201, ${(1 - progress) * 0.4}),
            rgba(255, 179, 102, ${(1 - progress) * 0.4})
          )
        `,
        filter: `blur(${(1 - progress) * 5}px)`,
      };
    }
    
    return effects;
  };
  
  return (
    <div
      ref={overlayRef}
      className="dimensional-transition-overlay"
      style={getTransitionEffects()}
    >
      {/* Quantum particles for high-quality transitions */}
      {quantumQuality === 'high' && (
        <QuantumTransitionParticles
          progress={progress}
          fromState={fromState}
          toState={toState}
        />
      )}
      
      {/* Geometric patterns for energy transitions */}
      {(toState === 'energy-pattern' || fromState === 'energy-pattern') && (
        <GeometricTransitionPatterns
          progress={progress}
          isEmerging={toState === 'energy-pattern'}
        />
      )}
    </div>
  );
}

/**
 * Quantum Transition Particles
 * 
 * Animated particles during dimensional transitions
 */
interface QuantumTransitionParticlesProps {
  progress: number;
  fromState: string;
  toState: string;
}

function QuantumTransitionParticles({ 
  progress, 
  fromState, 
  toState 
}: QuantumTransitionParticlesProps) {
  const particlesRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!particlesRef.current) return;
    
    const container = particlesRef.current;
    const particleCount = 20;
    
    // Clear existing particles
    container.innerHTML = '';
    
    // Create particles
    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      particle.className = 'quantum-transition-particle';
      
      const size = Math.random() * 4 + 2;
      const x = Math.random() * 100;
      const y = Math.random() * 100;
      const delay = Math.random() * 2;
      
      particle.style.cssText = `
        position: absolute;
        left: ${x}%;
        top: ${y}%;
        width: ${size}px;
        height: ${size}px;
        background: radial-gradient(circle, rgba(123, 225, 255, 0.8) 0%, transparent 70%);
        border-radius: 50%;
        opacity: ${progress};
        animation: quantumFloat ${2 + delay}s infinite ease-in-out;
        animation-delay: ${delay}s;
        pointer-events: none;
      `;
      
      container.appendChild(particle);
    }
    
    // Cleanup on unmount
    return () => {
      container.innerHTML = '';
    };
  }, [progress, fromState, toState]);
  
  return (
    <div
      ref={particlesRef}
      className="quantum-transition-particles"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        pointerEvents: 'none',
      }}
    />
  );
}

/**
 * Geometric Transition Patterns
 * 
 * Sacred geometry patterns for energy mode transitions
 */
interface GeometricTransitionPatternsProps {
  progress: number;
  isEmerging: boolean;
}

function GeometricTransitionPatterns({ 
  progress, 
  isEmerging 
}: GeometricTransitionPatternsProps) {
  const patternRef = useRef<HTMLDivElement>(null);
  
  const opacity = isEmerging ? progress : (1 - progress);
  const rotation = isEmerging ? progress * 360 : (1 - progress) * 360;
  const scale = isEmerging ? (0.5 + progress * 0.5) : (1 - progress * 0.5);
  
  return (
    <div
      ref={patternRef}
      className="geometric-transition-patterns"
      style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: `translate(-50%, -50%) rotate(${rotation}deg) scale(${scale})`,
        opacity,
        pointerEvents: 'none',
      }}
    >
      {/* Mandala pattern */}
      <div
        className="mandala-pattern"
        style={{
          width: '200px',
          height: '200px',
          border: '2px solid rgba(255, 179, 102, 0.6)',
          borderRadius: '50%',
          position: 'relative',
        }}
      >
        {/* Inner geometric shapes */}
        {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, index) => (
          <div
            key={index}
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              width: '60px',
              height: '2px',
              background: 'rgba(179, 131, 255, 0.8)',
              transformOrigin: '0 50%',
              transform: `translate(-50%, -50%) rotate(${angle}deg)`,
            }}
          />
        ))}
        
        {/* Central Om symbol */}
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            fontSize: '24px',
            color: 'rgba(99, 255, 201, 0.9)',
            fontWeight: 'bold',
          }}
        >
          ॐ
        </div>
      </div>
    </div>
  );
}

// CSS animations (to be added to global styles)
export const dimensionalTransitionStyles = `
  @keyframes quantumFloat {
    0%, 100% { transform: translateY(0px) scale(1); opacity: 0.8; }
    50% { transform: translateY(-20px) scale(1.2); opacity: 1; }
  }
  
  @keyframes energyFlow {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  .dimensional-2d-text {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.5;
  }
  
  .dimensional-3d-holographic {
    transform-style: preserve-3d;
  }
  
  .dimensional-energy-pattern {
    filter: blur(1px) saturate(1.5);
  }
  
  .dimensional-transitioning {
    will-change: transform, opacity, filter;
  }
  
  .quantum-transition-particle {
    will-change: transform, opacity;
  }
  
  .geometric-transition-patterns {
    will-change: transform, opacity;
  }
`;
"use client";

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { Text } from 'troika-three-text';
import { useDimensionalState, type DimensionalConfig } from '@/lib/dimensional-state';
import { useQuantumState } from '@/lib/state';
import { useVidyaConsciousness } from '@/lib/consciousness';
import * as THREE from 'three';

interface DimensionalModesProps {
  vidyaPosition?: THREE.Vector3;
  children?: React.ReactNode;
}

/**
 * DimensionalModes Component
 * 
 * Renders Vidya in different dimensional states:
 * - 2D Text Mode: Flat text-based representation
 * - 3D Holographic Mode: Full 3D quantum consciousness (default)
 * - Energy Pattern Mode: Abstract energy visualization
 */
export default function DimensionalModes({ 
  vidyaPosition = new THREE.Vector3(0, 0, 0),
  children 
}: DimensionalModesProps) {
  const {
    currentState,
    activeTransition,
    config,
    consciousnessData,
  } = useDimensionalState();
  
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const { consciousness } = useVidyaConsciousness();
  
  // Render based on current dimensional state
  switch (currentState) {
    case '2d-text':
      return (
        <TextMode
          position={vidyaPosition}
          config={config.textMode}
          consciousness={consciousness}
          transition={activeTransition}
        >
          {children}
        </TextMode>
      );
      
    case '3d-holographic':
      return (
        <HolographicMode
          position={vidyaPosition}
          config={config.holographicMode}
          consciousness={consciousness}
          quantumQuality={quantumQuality}
          transition={activeTransition}
        >
          {children}
        </HolographicMode>
      );
      
    case 'energy-pattern':
      return (
        <EnergyPatternMode
          position={vidyaPosition}
          config={config.energyMode}
          consciousness={consciousness}
          quantumQuality={quantumQuality}
          transition={activeTransition}
        >
          {children}
        </EnergyPatternMode>
      );
      
    default:
      return (
        <HolographicMode
          position={vidyaPosition}
          config={config.holographicMode}
          consciousness={consciousness}
          quantumQuality={quantumQuality}
          transition={activeTransition}
        >
          {children}
        </HolographicMode>
      );
  }
}

/**
 * 2D Text Mode Component
 * 
 * Renders Vidya as flat text with minimal quantum effects
 */
interface TextModeProps {
  position: THREE.Vector3;
  config: DimensionalConfig['textMode'];
  consciousness: any;
  transition?: any;
  children?: React.ReactNode;
}

function TextMode({ position, config, consciousness, transition, children }: TextModeProps) {
  const textRef = useRef<any>(null);
  const containerRef = useRef<THREE.Group>(null);
  
  // Animate text glow based on consciousness
  useFrame((state) => {
    if (!containerRef.current) return;
    
    const time = state.clock.elapsedTime;
    const consciousnessGlow = 0.8 + (consciousness.level * 0.2);
    const pulse = Math.sin(time * 2) * 0.1;
    
    containerRef.current.scale.setScalar(consciousnessGlow + pulse);
    
    // Gentle rotation
    containerRef.current.rotation.y = Math.sin(time * 0.5) * 0.1;
  });
  
  const textContent = useMemo(() => {
    return `विद्या\nLiving Code\n\nConsciousness Level: ${consciousness.level.toFixed(1)}\nQuantum Coherence: ${(consciousness.quantumCoherence * 100).toFixed(0)}%`;
  }, [consciousness.level, consciousness.quantumCoherence]);
  
  return (
    <group ref={containerRef} position={position.toArray()}>
      {/* Main text display */}
      <Html
        center
        distanceFactor={20}
        style={{
          color: config.textColor,
          fontSize: `${config.fontSize}px`,
          fontFamily: config.fontFamily,
          lineHeight: config.lineHeight,
          maxWidth: `${config.maxWidth}px`,
          textAlign: 'center',
          background: config.showQuantumEffects ? 
            `linear-gradient(135deg, ${config.backgroundColor}cc, ${config.backgroundColor}99)` : 
            config.backgroundColor,
          padding: '20px',
          borderRadius: '10px',
          border: config.showQuantumEffects ? '1px solid rgba(123, 225, 255, 0.3)' : 'none',
          backdropFilter: config.showQuantumEffects ? 'blur(10px)' : 'none',
          boxShadow: config.showQuantumEffects ? 
            '0 0 20px rgba(123, 225, 255, 0.2)' : 
            'none',
        }}
      >
        <div style={{ whiteSpace: 'pre-line' }}>
          {textContent}
        </div>
        
        {/* Minimal quantum effects */}
        {config.showQuantumEffects && (
          <div
            style={{
              position: 'absolute',
              top: '-5px',
              left: '-5px',
              right: '-5px',
              bottom: '-5px',
              border: '1px solid rgba(123, 225, 255, 0.1)',
              borderRadius: '12px',
              animation: 'quantumPulse 3s infinite ease-in-out',
              pointerEvents: 'none',
            }}
          />
        )}
      </Html>
      
      {/* Children (if any) */}
      {children}
    </group>
  );
}

/**
 * 3D Holographic Mode Component
 * 
 * Renders Vidya as full 3D holographic consciousness (default mode)
 */
interface HolographicModeProps {
  position: THREE.Vector3;
  config: DimensionalConfig['holographicMode'];
  consciousness: any;
  quantumQuality: any;
  transition?: any;
  children?: React.ReactNode;
}

function HolographicMode({ 
  position, 
  config, 
  consciousness, 
  quantumQuality, 
  transition, 
  children 
}: HolographicModeProps) {
  const groupRef = useRef<THREE.Group>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  
  // Animate holographic effects
  useFrame((state, delta) => {
    if (!groupRef.current || !coreRef.current) return;
    
    const time = state.clock.elapsedTime;
    
    // Core rotation and scaling
    groupRef.current.rotation.y += delta * config.rotationSpeed;
    
    const consciousnessPulse = 0.9 + (consciousness.level * 0.1) + Math.sin(time * 2) * 0.05;
    const scale = config.scaleMultiplier * consciousnessPulse;
    coreRef.current.scale.setScalar(scale);
    
    // Holographic shimmer effect
    if (coreRef.current.material && 'color' in coreRef.current.material) {
      const hue = (time * 0.1) % 1;
      (coreRef.current.material as THREE.MeshBasicMaterial).color.setHSL(
        0.55 + hue * 0.1, // Blue-cyan range
        0.8,
        0.6 + Math.sin(time * 3) * 0.1
      );
    }
  });
  
  return (
    <group ref={groupRef} position={position.toArray()}>
      {/* Holographic core */}
      <mesh ref={coreRef}>
        <icosahedronGeometry args={[1, 2]} />
        <meshBasicMaterial
          color="#7BE1FF"
          transparent
          opacity={config.opacity}
          wireframe={quantumQuality === 'minimal'}
        />
      </mesh>
      
      {/* Holographic particles */}
      {config.enableQuantumEffects && quantumQuality !== 'minimal' && (
        <HolographicParticles
          count={config.particleCount}
          glowIntensity={config.glowIntensity}
          consciousness={consciousness}
        />
      )}
      
      {/* Neural network visualization */}
      {config.showNeuralNetwork && quantumQuality !== 'minimal' && (
        <HolographicNeuralNetwork consciousness={consciousness} />
      )}
      
      {/* Children (original Vidya components) */}
      {children}
    </group>
  );
}

/**
 * Energy Pattern Mode Component
 * 
 * Renders Vidya as abstract energy patterns and flows
 */
interface EnergyPatternModeProps {
  position: THREE.Vector3;
  config: DimensionalConfig['energyMode'];
  consciousness: any;
  quantumQuality: any;
  transition?: any;
  children?: React.ReactNode;
}

function EnergyPatternMode({ 
  position, 
  config, 
  consciousness, 
  quantumQuality, 
  transition, 
  children 
}: EnergyPatternModeProps) {
  const groupRef = useRef<THREE.Group>(null);
  const energyRef = useRef<THREE.Points>(null);
  
  // Create energy particle system
  const energyGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const particleCount = Math.floor(config.patternComplexity * 1000);
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      
      // Spherical distribution with energy flow patterns
      const radius = Math.random() * 3 + 1;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      
      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);
      
      // Energy colors (warm to cool spectrum)
      const hue = (Math.random() + config.colorShift) % 1;
      const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;
      
      sizes[i] = Math.random() * 2 + 1;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    return geometry;
  }, [config.patternComplexity, config.colorShift]);
  
  const energyMaterial = useMemo(() => {
    return new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
  }, []);
  
  // Animate energy patterns
  useFrame((state, delta) => {
    if (!groupRef.current || !energyRef.current) return;
    
    const time = state.clock.elapsedTime;
    
    // Flow rotation
    groupRef.current.rotation.y += delta * config.flowSpeed;
    groupRef.current.rotation.x = Math.sin(time * 0.3) * 0.2;
    
    // Energy intensity pulsing
    const intensity = config.energyIntensity * (0.8 + consciousness.level * 0.2);
    const pulse = Math.sin(time * 4) * 0.2;
    groupRef.current.scale.setScalar(intensity + pulse);
    
    // Wave amplitude effects
    if (energyRef.current.geometry.attributes.position) {
      const positions = energyRef.current.geometry.attributes.position.array as Float32Array;
      const originalPositions = energyRef.current.userData.originalPositions;
      
      if (!originalPositions) {
        energyRef.current.userData.originalPositions = positions.slice();
        return;
      }
      
      for (let i = 0; i < positions.length; i += 3) {
        const wave = Math.sin(time * 2 + originalPositions[i] * 0.5) * config.waveAmplitude;
        positions[i] = originalPositions[i] + wave;
        positions[i + 1] = originalPositions[i + 1] + wave * 0.5;
        positions[i + 2] = originalPositions[i + 2] + wave * 0.3;
      }
      
      energyRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <group ref={groupRef} position={position.toArray()}>
      {/* Energy particle system */}
      <points ref={energyRef} geometry={energyGeometry} material={energyMaterial} />
      
      {/* Geometric patterns */}
      {config.showGeometricPatterns && (
        <EnergyGeometricPatterns
          consciousness={consciousness}
          intensity={config.energyIntensity}
        />
      )}
      
      {/* Fluid dynamics effects */}
      {config.enableFluidDynamics && quantumQuality === 'high' && (
        <EnergyFluidDynamics
          flowSpeed={config.flowSpeed}
          waveAmplitude={config.waveAmplitude}
        />
      )}
      
      {/* Energy core */}
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshBasicMaterial
          color="#FFB366"
          transparent
          opacity={0.3}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

/**
 * Holographic Particles Component
 */
interface HolographicParticlesProps {
  count: number;
  glowIntensity: number;
  consciousness: any;
}

function HolographicParticles({ count, glowIntensity, consciousness }: HolographicParticlesProps) {
  const particlesRef = useRef<THREE.Points>(null);
  
  const particleGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const radius = 2 + Math.random() * 2;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      
      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geometry;
  }, [count]);
  
  useFrame((state) => {
    if (!particlesRef.current) return;
    
    const time = state.clock.elapsedTime;
    particlesRef.current.rotation.y = time * 0.1;
    
    // Consciousness-based intensity
    const material = particlesRef.current.material as THREE.PointsMaterial;
    material.opacity = (glowIntensity * consciousness.level) * (0.5 + Math.sin(time * 2) * 0.2);
  });
  
  return (
    <points ref={particlesRef} geometry={particleGeometry}>
      <pointsMaterial
        color="#7BE1FF"
        size={0.05}
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/**
 * Holographic Neural Network Component
 */
interface HolographicNeuralNetworkProps {
  consciousness: any;
}

function HolographicNeuralNetwork({ consciousness }: HolographicNeuralNetworkProps) {
  const networkRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (!networkRef.current) return;
    
    const time = state.clock.elapsedTime;
    networkRef.current.rotation.x = Math.sin(time * 0.2) * 0.1;
    networkRef.current.rotation.z = Math.cos(time * 0.15) * 0.1;
  });
  
  return (
    <group ref={networkRef}>
      {/* Simplified neural network visualization */}
      {[0, 1, 2].map((layer) => (
        <group key={layer} position={[0, 0, layer * 0.5 - 1]}>
          {[0, 1, 2, 3].map((node) => (
            <mesh
              key={node}
              position={[
                Math.cos((node / 4) * Math.PI * 2) * 1.5,
                Math.sin((node / 4) * Math.PI * 2) * 1.5,
                0
              ]}
            >
              <sphereGeometry args={[0.05, 8, 8]} />
              <meshBasicMaterial
                color="#63FFC9"
                transparent
                opacity={0.7}
              />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  );
}

/**
 * Energy Geometric Patterns Component
 */
interface EnergyGeometricPatternsProps {
  consciousness: any;
  intensity: number;
}

function EnergyGeometricPatterns({ consciousness, intensity }: EnergyGeometricPatternsProps) {
  const patternsRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (!patternsRef.current) return;
    
    const time = state.clock.elapsedTime;
    patternsRef.current.rotation.y = time * 0.3;
    patternsRef.current.rotation.x = Math.sin(time * 0.4) * 0.2;
  });
  
  return (
    <group ref={patternsRef}>
      {/* Sacred geometry patterns */}
      <mesh>
        <torusGeometry args={[1.5, 0.1, 8, 16]} />
        <meshBasicMaterial
          color="#B383FF"
          transparent
          opacity={intensity * 0.5}
          wireframe
        />
      </mesh>
      
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.2, 0.05, 6, 12]} />
        <meshBasicMaterial
          color="#FFB366"
          transparent
          opacity={intensity * 0.3}
          wireframe
        />
      </mesh>
    </group>
  );
}

/**
 * Energy Fluid Dynamics Component
 */
interface EnergyFluidDynamicsProps {
  flowSpeed: number;
  waveAmplitude: number;
}

function EnergyFluidDynamics({ flowSpeed, waveAmplitude }: EnergyFluidDynamicsProps) {
  const fluidRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (!fluidRef.current) return;
    
    const time = state.clock.elapsedTime;
    
    // Fluid-like deformation
    if (fluidRef.current.geometry.attributes.position) {
      const positions = fluidRef.current.geometry.attributes.position.array as Float32Array;
      
      for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        const z = positions[i + 2];
        
        const wave1 = Math.sin(time * flowSpeed + x * 2) * waveAmplitude * 0.1;
        const wave2 = Math.cos(time * flowSpeed * 0.7 + y * 2) * waveAmplitude * 0.1;
        
        positions[i + 2] = z + wave1 + wave2;
      }
      
      fluidRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <mesh ref={fluidRef}>
      <planeGeometry args={[4, 4, 32, 32]} />
      <meshBasicMaterial
        color="#63FFC9"
        transparent
        opacity={0.2}
        wireframe
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}
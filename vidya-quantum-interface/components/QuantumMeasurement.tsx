"use client";

import { useRef, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useQuantumState } from "@/lib/state";
import * as THREE from "three";
import { waveformCollapse, decohere } from "@/lib/quantum";

interface DecoherenceEffectProps {
  position: THREE.Vector3;
  intensity: number;
  pattern: 'wave' | 'particle' | 'interference';
}

function DecoherenceEffect({ position, intensity, pattern }: DecoherenceEffectProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const meshRef = useRef<THREE.Mesh>(null!);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  useFrame((_, delta) => {
    if (!groupRef.current || !meshRef.current) return;
    
    const time = performance.now() * 0.001;
    
    // Rotate the decoherence pattern
    groupRef.current.rotation.y += delta * 0.5;
    groupRef.current.rotation.z += delta * 0.3;
    
    // Pulsing effect based on pattern
    let scale = 1;
    switch (pattern) {
      case 'wave':
        scale = 1 + 0.3 * Math.sin(time * 4);
        break;
      case 'particle':
        scale = 1 + 0.2 * Math.abs(Math.sin(time * 6));
        break;
      case 'interference':
        scale = 1 + 0.4 * Math.sin(time * 2) * Math.cos(time * 3);
        break;
    }
    
    meshRef.current.scale.setScalar(scale * intensity);
    
    // Update material opacity
    if (meshRef.current.material && 'opacity' in meshRef.current.material) {
      (meshRef.current.material as any).opacity = intensity * 0.6;
    }
  });
  
  if (quantumQuality === 'minimal') {
    return null;
  }
  
  return (
    <group ref={groupRef} position={position}>
      <mesh ref={meshRef}>
        <ringGeometry args={[0.5, 1.5, 16]} />
        <meshBasicMaterial
          color={pattern === 'wave' ? '#7BE1FF' : pattern === 'particle' ? '#B383FF' : '#63FFC9'}
          transparent
          opacity={0.4}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Inner ring for more complex patterns */}
      {quantumQuality === 'high' && (
        <mesh rotation={[0, 0, Math.PI / 4]}>
          <ringGeometry args={[0.2, 0.8, 8]} />
          <meshBasicMaterial
            color="#FFFFFF"
            transparent
            opacity={0.2}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}
    </group>
  );
}

interface WaveformCollapseProps {
  center: THREE.Vector3;
  progress: number;
  selectedStateId: string;
  collapsedStates: string[];
}

function WaveformCollapseVisualization({ center, progress, selectedStateId, collapsedStates }: WaveformCollapseProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  useFrame(() => {
    if (!groupRef.current) return;
    
    const collapse = waveformCollapse(progress);
    
    // Scale effect based on collapse progress
    groupRef.current.scale.setScalar(1 + collapse.decoherence * 2);
    
    // Rotation effect
    groupRef.current.rotation.y = collapse.interference * Math.PI;
  });
  
  if (quantumQuality === 'minimal') {
    return null;
  }
  
  return (
    <group ref={groupRef} position={center}>
      {/* Central collapse point */}
      <mesh>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial
          color="#FFFFFF"
          transparent
          opacity={0.8}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Expanding shockwave */}
      <mesh>
        <ringGeometry args={[progress * 2, progress * 2.5, 32]} />
        <meshBasicMaterial
          color="#7BE1FF"
          transparent
          opacity={0.6 * (1 - progress)}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Interference patterns */}
      {quantumQuality === 'high' && collapsedStates.map((stateId, index) => (
        <mesh key={stateId} rotation={[0, 0, (index / collapsedStates.length) * Math.PI * 2]}>
          <ringGeometry args={[0.5 + index * 0.3, 0.7 + index * 0.3, 16]} />
          <meshBasicMaterial
            color="#B383FF"
            transparent
            opacity={0.3 * (1 - progress)}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}
    </group>
  );
}

interface QuantumInterferenceProps {
  states: Array<{
    id: string;
    position: [number, number, number];
    phase: number;
    probability: number;
  }>;
}

function QuantumInterference({ states }: QuantumInterferenceProps) {
  const linesRef = useRef<THREE.LineSegments>(null!);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  useFrame(() => {
    if (!linesRef.current || states.length < 2) return;
    
    const time = performance.now() * 0.001;
    
    // Update line opacity based on interference patterns
    if (linesRef.current.material && 'opacity' in linesRef.current.material) {
      const interference = Math.abs(Math.sin(time * 2));
      (linesRef.current.material as any).opacity = 0.3 + interference * 0.4;
    }
  });
  
  if (quantumQuality === 'minimal' || states.length < 2) {
    return null;
  }
  
  // Create interference lines between states
  const positions: number[] = [];
  for (let i = 0; i < states.length; i++) {
    for (let j = i + 1; j < states.length; j++) {
      positions.push(...states[i].position);
      positions.push(...states[j].position);
    }
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  
  const material = new THREE.LineBasicMaterial({
    color: '#63FFC9',
    transparent: true,
    opacity: 0.3,
    blending: THREE.AdditiveBlending,
  });
  
  return <lineSegments ref={linesRef} geometry={geometry} material={material} />;
}

export default function QuantumMeasurement() {
  const decoherenceActive = useQuantumState((s) => s.decoherenceActive);
  const waveformCollapsing = useQuantumState((s) => s.waveformCollapsing);
  const superpositionStates = useQuantumState((s) => s.superpositionStates);
  const quantumMeasurements = useQuantumState((s) => s.quantumMeasurements);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  const collapseProgressRef = useRef(0);
  
  useFrame((_, delta) => {
    if (waveformCollapsing) {
      collapseProgressRef.current = Math.min(1, collapseProgressRef.current + delta * 0.5);
    } else {
      collapseProgressRef.current = 0;
    }
  });
  
  if (quantumQuality === 'minimal') {
    return null;
  }
  
  const latestMeasurement = quantumMeasurements[quantumMeasurements.length - 1];
  
  return (
    <group>
      {/* Decoherence effects */}
      {decoherenceActive && (
        <DecoherenceEffect
          position={new THREE.Vector3(0, 0, 0)}
          intensity={1 - coherenceLevel}
          pattern="wave"
        />
      )}
      
      {/* Waveform collapse visualization */}
      {waveformCollapsing && latestMeasurement && (
        <WaveformCollapseVisualization
          center={new THREE.Vector3(0, 0, 0)}
          progress={collapseProgressRef.current}
          selectedStateId={latestMeasurement.selectedState}
          collapsedStates={latestMeasurement.collapsedStates}
        />
      )}
      
      {/* Quantum interference between superposition states */}
      {superpositionStates.length > 1 && !waveformCollapsing && (
        <QuantumInterference states={superpositionStates} />
      )}
      
      {/* Additional measurement effects for high quality */}
      {quantumQuality === 'high' && quantumMeasurements.slice(-3).map((measurement, index) => (
        <group key={measurement.id}>
          {measurement.visualEffects.includes('probability_redistribution') && (
            <mesh position={[index * 2 - 2, -3, 0]}>
              <sphereGeometry args={[0.2, 8, 8]} />
              <meshBasicMaterial
                color="#FFB366"
                transparent
                opacity={0.4 * (1 - index * 0.3)}
                blending={THREE.AdditiveBlending}
              />
            </mesh>
          )}
        </group>
      ))}
    </group>
  );
}
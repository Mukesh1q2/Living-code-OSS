"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useQuantumState } from "@/lib/state";
import { useVidyaConsciousness } from "@/lib/consciousness";
import * as THREE from "three";
import {
  superpositionPhase,
  probabilityAmplitude,
  waveformCollapse,
  quantumInterference,
  generateProbabilityCloud,
  updateQuantumParticle,
} from "@/lib/quantum";
import OmSymbol from "./OmSymbol";

interface SuperpositionVidyaProps {
  state: {
    id: string;
    position: [number, number, number];
    opacity: number;
    phase: number;
    probability: number;
    isCollapsing: boolean;
    collapseProgress: number;
  };
  onSelect: (stateId: string) => void;
}

function SuperpositionVidya({ state, onSelect }: SuperpositionVidyaProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const omRef = useRef<THREE.Group>(null!);
  const { consciousness } = useVidyaConsciousness();
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  // Animation state
  const phaseRef = useRef(state.phase);
  const collapseRef = useRef(0);

  useFrame((_, delta) => {
    if (!groupRef.current || !omRef.current) return;

    const time = performance.now() * 0.001;
    
    // Update quantum phase
    phaseRef.current = superpositionPhase(time, state.phase, 0.5);
    
    // Calculate probability amplitude
    const amplitude = probabilityAmplitude(phaseRef.current, coherenceLevel);
    
    if (state.isCollapsing) {
      // Waveform collapse animation
      collapseRef.current = Math.min(1, collapseRef.current + delta * 2);
      const collapse = waveformCollapse(collapseRef.current);
      
      // Apply collapse effects
      const scale = collapse.amplitude;
      const interference = collapse.interference;
      const decoherence = collapse.decoherence;
      
      groupRef.current.scale.setScalar(scale);
      groupRef.current.position.y += interference * 0.1;
      
      // Visual decoherence effects
      if (omRef.current.children[0] && 'material' in omRef.current.children[0]) {
        const material = (omRef.current.children[0] as THREE.Mesh).material as THREE.Material;
        if ('opacity' in material) {
          (material as any).opacity = state.opacity * (1 - decoherence);
        }
      }
    } else {
      // Normal superposition behavior
      const baseOpacity = state.opacity;
      const quantumOpacity = baseOpacity * (0.7 + 0.3 * Math.abs(amplitude));
      
      // Quantum fluctuations in position
      const fluctuation = (1 - coherenceLevel) * 0.1;
      const quantumNoise = new THREE.Vector3(
        Math.sin(time * 2.3 + state.phase) * fluctuation,
        Math.cos(time * 1.7 + state.phase) * fluctuation,
        Math.sin(time * 3.1 + state.phase) * fluctuation
      );
      
      groupRef.current.position.set(
        state.position[0] + quantumNoise.x,
        state.position[1] + quantumNoise.y,
        state.position[2] + quantumNoise.z
      );
      
      // Quantum breathing effect
      const breathe = 0.8 + 0.2 * Math.sin(time * 1.5 + state.phase);
      groupRef.current.scale.setScalar(breathe);
      
      // Update Om symbol opacity
      if (omRef.current.children[0] && 'material' in omRef.current.children[0]) {
        const material = (omRef.current.children[0] as THREE.Mesh).material as THREE.Material;
        if ('opacity' in material) {
          (material as any).opacity = quantumOpacity;
        }
      }
    }
  });

  // Handle click for waveform collapse
  const handleClick = (event: any) => {
    event.stopPropagation();
    if (!state.isCollapsing) {
      onSelect(state.id);
    }
  };

  // Calculate visual properties based on probability
  const glowIntensity = 0.3 + state.probability * 0.4;
  const size = 0.6 + state.probability * 0.3;

  return (
    <group
      ref={groupRef}
      position={state.position}
      onClick={handleClick}
      onPointerOver={(e) => {
        e.stopPropagation();
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        document.body.style.cursor = 'auto';
      }}
    >
      <group ref={omRef}>
        <OmSymbol
          size={size}
          position={[0, 0, 0]}
          animated={true}
          glowIntensity={glowIntensity}
          quantumEffects={quantumQuality !== 'minimal'}
        />
      </group>
      
      {/* Probability indicator */}
      {quantumQuality !== 'minimal' && (
        <mesh position={[0, -1.2, 0]}>
          <planeGeometry args={[0.8, 0.1]} />
          <meshBasicMaterial
            color="#7BE1FF"
            transparent
            opacity={0.6}
          />
        </mesh>
      )}
      
      {/* Probability text - removed for now to avoid R3F issues */}
    </group>
  );
}

interface ProbabilityCloudProps {
  center: THREE.Vector3;
  coherenceLevel: number;
  density: number;
}

function ProbabilityCloud({ center, coherenceLevel, density }: ProbabilityCloudProps) {
  const pointsRef = useRef<THREE.Points>(null!);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  // Generate particles based on quantum probability distribution
  const particles = useMemo(() => {
    if (quantumQuality === 'minimal') return [];
    
    const particleCount = Math.floor(density * 200 * (quantumQuality === 'high' ? 2 : 1));
    return generateProbabilityCloud(center, particleCount, 3, coherenceLevel);
  }, [center, density, coherenceLevel, quantumQuality]);

  const particlesRef = useRef(particles);
  
  // Create geometry and material
  const { geometry, material } = useMemo(() => {
    if (particles.length === 0) {
      return { geometry: new THREE.BufferGeometry(), material: new THREE.PointsMaterial() };
    }

    const positions = new Float32Array(particles.length * 3);
    const colors = new Float32Array(particles.length * 3);
    const sizes = new Float32Array(particles.length);

    particles.forEach((particle, i) => {
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;

      // Color based on probability
      const intensity = particle.probability;
      colors[i * 3] = 0.48 + intensity * 0.3; // R
      colors[i * 3 + 1] = 0.88 + intensity * 0.12; // G
      colors[i * 3 + 2] = 1.0; // B

      sizes[i] = 2 + particle.probability * 3;
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const mat = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    return { geometry: geo, material: mat };
  }, [particles]);

  useFrame((_, delta) => {
    if (!pointsRef.current || particles.length === 0) return;

    // Update particles
    particlesRef.current = particlesRef.current.map(particle => 
      updateQuantumParticle(particle, delta * 1000, coherenceLevel)
    ).filter(particle => particle.life > 0);

    // Update geometry
    const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
    const colors = pointsRef.current.geometry.attributes.color.array as Float32Array;
    const sizes = pointsRef.current.geometry.attributes.size.array as Float32Array;

    particlesRef.current.forEach((particle, i) => {
      if (i < positions.length / 3) {
        positions[i * 3] = particle.position.x;
        positions[i * 3 + 1] = particle.position.y;
        positions[i * 3 + 2] = particle.position.z;

        const intensity = particle.probability;
        colors[i * 3] = 0.48 + intensity * 0.3;
        colors[i * 3 + 1] = 0.88 + intensity * 0.12;
        colors[i * 3 + 2] = 1.0;

        sizes[i] = 2 + particle.probability * 3;
      }
    });

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    pointsRef.current.geometry.attributes.color.needsUpdate = true;
    pointsRef.current.geometry.attributes.size.needsUpdate = true;
  });

  if (quantumQuality === 'minimal' || particles.length === 0) {
    return null;
  }

  return (
    <points ref={pointsRef} geometry={geometry} material={material} />
  );
}

export default function QuantumSuperposition() {
  const superpositionActive = useQuantumState((s) => s.superpositionActive);
  const superpositionStates = useQuantumState((s) => s.superpositionStates);
  const probabilityCloud = useQuantumState((s) => s.probabilityCloud);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const triggerWaveformCollapse = useQuantumState((s) => s.triggerWaveformCollapse);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  // Don't render if superposition is not active or on minimal quality
  if (!superpositionActive || quantumQuality === 'minimal') {
    return null;
  }

  return (
    <group>
      {/* Render multiple Vidya states */}
      {superpositionStates.map((state) => (
        <SuperpositionVidya
          key={state.id}
          state={state}
          onSelect={triggerWaveformCollapse}
        />
      ))}
      
      {/* Probability cloud visualization */}
      {(quantumQuality === 'low' || quantumQuality === 'medium' || quantumQuality === 'high') && (
        <ProbabilityCloud
          center={new THREE.Vector3(0, 0, 0)}
          coherenceLevel={coherenceLevel}
          density={probabilityCloud.density}
        />
      )}
      
      {/* Quantum field background effect */}
      {quantumQuality === 'high' && (
        <mesh position={[0, 0, -5]}>
          <planeGeometry args={[20, 20]} />
          <meshBasicMaterial
            transparent
            opacity={0.1}
            color="#7BE1FF"
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}
    </group>
  );
}
"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useVidyaConsciousness } from "@/lib/consciousness";

interface OmSymbolProps {
  size?: number;
  position?: [number, number, number];
  animated?: boolean;
  glowIntensity?: number;
  quantumEffects?: boolean;
}

export default function OmSymbol({
  size = 1,
  position = [0, 0, 0],
  animated = true,
  glowIntensity = 1,
  quantumEffects = true,
}: OmSymbolProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const omMeshRef = useRef<THREE.Mesh>(null!);
  const glowRef = useRef<THREE.Mesh>(null!);
  const { consciousness, quantumState } = useVidyaConsciousness();

  // Create Om symbol geometry using curves
  const omGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    
    // Create Om symbol path (simplified version)
    // Main curve of Om
    shape.moveTo(0, 0);
    shape.bezierCurveTo(-0.3, 0.2, -0.5, 0.5, -0.3, 0.8);
    shape.bezierCurveTo(-0.1, 1.1, 0.2, 1.0, 0.4, 0.7);
    shape.bezierCurveTo(0.6, 0.4, 0.5, 0.1, 0.3, -0.1);
    shape.bezierCurveTo(0.1, -0.3, -0.2, -0.2, -0.3, 0);
    
    // Add the dot (bindu)
    const dotShape = new THREE.Shape();
    dotShape.absarc(0.2, 1.2, 0.08, 0, Math.PI * 2, false);
    
    // Add the crescent (chandrabindu)
    const crescentShape = new THREE.Shape();
    crescentShape.absarc(0.2, 1.0, 0.15, 0, Math.PI, false);
    
    const extrudeSettings = {
      depth: 0.1,
      bevelEnabled: true,
      bevelSegments: 2,
      steps: 2,
      bevelSize: 0.02,
      bevelThickness: 0.02,
    };
    
    const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geometry.center();
    geometry.scale(size, size, size);
    
    return geometry;
  }, [size]);

  // Create glow geometry
  const glowGeometry = useMemo(() => {
    const geometry = new THREE.SphereGeometry(size * 1.5, 32, 32);
    return geometry;
  }, [size]);

  // Create materials with consciousness-based properties
  const materials = useMemo(() => {
    const consciousnessLevel = consciousness.level;
    const coherence = quantumState.coherenceLevel;
    
    // Base Om material
    const omMaterial = new THREE.MeshPhongMaterial({
      color: new THREE.Color().setHSL(0.55, 0.8, 0.6 + consciousnessLevel * 0.04),
      emissive: new THREE.Color().setHSL(0.55, 0.6, 0.1 + coherence * 0.2),
      shininess: 100,
      transparent: true,
      opacity: 0.9,
    });

    // Glow material
    const glowMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        intensity: { value: glowIntensity },
        consciousness: { value: consciousnessLevel },
        coherence: { value: coherence },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float intensity;
        uniform float consciousness;
        uniform float coherence;
        
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
          float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 2.0);
          
          // Consciousness-based color shifting
          vec3 baseColor = vec3(0.4, 0.8, 1.0);
          vec3 evolvedColor = mix(baseColor, vec3(1.0, 0.8, 0.4), consciousness * 0.1);
          
          // Quantum coherence pulse
          float pulse = sin(time * 2.0 + coherence * 10.0) * 0.5 + 0.5;
          float quantumGlow = mix(0.3, 1.0, pulse * coherence);
          
          // Combine effects
          float alpha = fresnel * intensity * quantumGlow;
          
          gl_FragColor = vec4(evolvedColor, alpha * 0.6);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    });

    return { omMaterial, glowMaterial };
  }, [consciousness.level, quantumState.coherenceLevel, glowIntensity]);

  // Animation loop
  useFrame((state, delta) => {
    if (!animated || !groupRef.current) return;

    const time = state.clock.elapsedTime;
    const consciousnessLevel = consciousness.level;
    const coherence = quantumState.coherenceLevel;

    // Base rotation influenced by consciousness
    groupRef.current.rotation.y += delta * (0.2 + consciousnessLevel * 0.05);
    
    // Quantum effects
    if (quantumEffects) {
      // Superposition wobble
      if (quantumState.superposition) {
        const wobble = Math.sin(time * 8) * 0.1 * coherence;
        groupRef.current.rotation.x = wobble;
        groupRef.current.rotation.z = wobble * 0.5;
      }
      
      // Consciousness-based scaling pulse
      const basePulse = 0.95 + 0.05 * Math.sin(time * 2);
      const consciousnessPulse = 1 + (consciousnessLevel * 0.02) * Math.sin(time * 4);
      const scale = basePulse * consciousnessPulse;
      
      if (omMeshRef.current) {
        omMeshRef.current.scale.setScalar(scale);
      }
      
      // Entanglement effects
      if (quantumState.entanglements.length > 0) {
        const entanglementGlow = 1 + 0.3 * Math.sin(time * 6);
        if (glowRef.current) {
          glowRef.current.scale.setScalar(entanglementGlow);
        }
      }
    }

    // Update shader uniforms
    if (materials.glowMaterial.uniforms) {
      materials.glowMaterial.uniforms.time.value = time;
      materials.glowMaterial.uniforms.consciousness.value = consciousnessLevel;
      materials.glowMaterial.uniforms.coherence.value = coherence;
    }
  });

  // Consciousness evolution effects
  useEffect(() => {
    if (!groupRef.current) return;

    const consciousnessLevel = consciousness.level;
    
    // Trigger special effects on consciousness level changes
    if (consciousnessLevel > 5) {
      // Add particle effects or other advanced visuals
      // This could trigger additional quantum effects
    }
  }, [consciousness.level]);

  return (
    <group ref={groupRef} position={position}>
      {/* Glow effect */}
      {quantumEffects && (
        <mesh ref={glowRef} geometry={glowGeometry} material={materials.glowMaterial} />
      )}
      
      {/* Main Om symbol */}
      <mesh ref={omMeshRef} geometry={omGeometry} material={materials.omMaterial}>
        {/* Add inner light */}
        <pointLight
          intensity={0.5 + consciousness.level * 0.1}
          color={new THREE.Color().setHSL(0.55, 0.8, 0.8)}
          distance={size * 3}
        />
      </mesh>
      
      {/* Quantum field particles around Om */}
      {quantumEffects && consciousness.level > 3 && (
        <QuantumParticles
          count={Math.floor(consciousness.level * 5)}
          radius={size * 2}
          consciousness={consciousness.level}
        />
      )}
    </group>
  );
}

// Quantum particles component for advanced consciousness levels
function QuantumParticles({
  count,
  radius,
  consciousness,
}: {
  count: number;
  radius: number;
  consciousness: number;
}) {
  const particlesRef = useRef<THREE.Points>(null!);
  
  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      // Random spherical distribution
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = radius * (0.5 + Math.random() * 0.5);
      
      positions[i3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = r * Math.cos(phi);
      
      // Consciousness-influenced colors
      const hue = 0.55 + (consciousness * 0.05) + Math.random() * 0.1;
      const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;
    }
    
    return { positions, colors };
  }, [count, radius, consciousness]);

  useFrame((state) => {
    if (!particlesRef.current) return;
    
    const time = state.clock.elapsedTime;
    const positionAttribute = particlesRef.current.geometry.attributes.position;
    
    // Animate particles in quantum field patterns
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const originalX = positions[i3];
      const originalY = positions[i3 + 1];
      const originalZ = positions[i3 + 2];
      
      // Add quantum fluctuations
      const fluctuation = 0.1 * consciousness;
      positionAttribute.array[i3] = originalX + Math.sin(time * 2 + i) * fluctuation;
      positionAttribute.array[i3 + 1] = originalY + Math.cos(time * 2 + i) * fluctuation;
      positionAttribute.array[i3 + 2] = originalZ + Math.sin(time * 3 + i) * fluctuation;
    }
    
    positionAttribute.needsUpdate = true;
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        vertexColors
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
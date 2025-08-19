"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useQuantumState } from "@/lib/state";
import * as THREE from "three";
import { getQuantumShaderSettings } from "./QuantumShaderManager";

// Inline particle shaders
const quantumParticlesVert = `
uniform float uTime;
uniform float uQuantumEnergy;
uniform float uFieldStrength;
uniform vec3 uEnergyCenter;
uniform float uParticleSize;
uniform float uFlowSpeed;
uniform bool uCollapsing;
uniform float uCollapseProgress;
uniform float uPerformanceLevel;

attribute vec3 position;
attribute vec3 velocity;
attribute float particleId;
attribute float lifespan;
attribute float energy;

varying float vParticleId;
varying float vLifespan;
varying float vEnergy;
varying vec3 vVelocity;
varying float vDistanceToCenter;

void main() {
  vParticleId = particleId;
  vLifespan = lifespan;
  vEnergy = energy;
  vVelocity = velocity;
  
  vec3 currentPos = position;
  vDistanceToCenter = length(currentPos - uEnergyCenter);
  
  float sizeMultiplier = mix(0.5, 2.0, energy);
  sizeMultiplier *= mix(1.0, 0.3, vDistanceToCenter * 0.1);
  
  if (uCollapsing) {
    sizeMultiplier *= mix(1.0, 0.1, uCollapseProgress);
  }
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(currentPos, 1.0);
  gl_PointSize = uParticleSize * sizeMultiplier;
}
`;

const quantumParticlesFrag = `
precision highp float;

uniform float uTime;
uniform float uQuantumEnergy;
uniform vec3 uQuantumColors[4];
uniform float uPerformanceLevel;
uniform bool uCollapsing;
uniform float uCollapseProgress;

varying float vParticleId;
varying float vLifespan;
varying float vEnergy;
varying vec3 vVelocity;
varying float vDistanceToCenter;

float circleShape(vec2 coord) {
  return 1.0 - smoothstep(0.3, 0.5, length(coord));
}

vec3 energyColor(float energy, float distanceToCenter) {
  vec3 lowEnergyColor = uQuantumColors[0];
  vec3 medEnergyColor = uQuantumColors[1];
  vec3 highEnergyColor = uQuantumColors[2];
  vec3 maxEnergyColor = uQuantumColors[3];
  
  vec3 color;
  if (energy < 0.33) {
    color = mix(lowEnergyColor, medEnergyColor, energy * 3.0);
  } else if (energy < 0.66) {
    color = mix(medEnergyColor, highEnergyColor, (energy - 0.33) * 3.0);
  } else {
    color = mix(highEnergyColor, maxEnergyColor, (energy - 0.66) * 3.0);
  }
  
  float distanceFactor = exp(-distanceToCenter * 0.1);
  color = mix(color * 0.5, color, distanceFactor);
  
  return color;
}

float lifespanAlpha(float lifespan) {
  float fadeIn = smoothstep(0.0, 0.1, lifespan);
  float fadeOut = smoothstep(1.0, 0.8, lifespan);
  
  return fadeIn * fadeOut;
}

void main() {
  vec2 coord = gl_PointCoord - 0.5;
  
  float shape = circleShape(coord);
  if (shape < 0.01) discard;
  
  vec3 color = energyColor(vEnergy, vDistanceToCenter);
  
  float alpha = shape * lifespanAlpha(vLifespan);
  alpha *= mix(0.3, 1.0, vEnergy);
  alpha *= mix(0.5, 1.0, uQuantumEnergy);
  
  float distanceFade = exp(-vDistanceToCenter * 0.05);
  alpha *= mix(0.2, 1.0, distanceFade);
  
  gl_FragColor = vec4(color, alpha);
}
`;

interface QuantumParticle {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  id: number;
  lifespan: number;
  energy: number;
  maxLifespan: number;
}

export default function QuantumParticleSystem() {
  const pointsRef = useRef<THREE.Points>(null!);
  const materialRef = useRef<THREE.ShaderMaterial>(null!);
  const particlesRef = useRef<QuantumParticle[]>([]);
  const geometryRef = useRef<THREE.BufferGeometry>(null!);
  
  // Quantum state
  const quantumEnergy = useQuantumState((s) => s.coherenceLevel);
  const fieldStrength = useQuantumState((s) => s.probabilityCloud.density);
  const energyCenter = [0, 0, 0] as [number, number, number];
  const waveformCollapsing = useQuantumState((s) => s.waveformCollapsing);
  const collapseProgress = useQuantumState((s) => 
    s.superpositionStates.length > 0 
      ? Math.max(...s.superpositionStates.map(state => state.collapseProgress))
      : 0
  );
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  // Performance settings
  const shaderSettings = useMemo(() => getQuantumShaderSettings(), []);
  
  // Don't render particles if performance is too low or quality is minimal
  if (!shaderSettings.useParticles || quantumQuality === 'minimal') {
    return null;
  }

  const particleCount = shaderSettings.particleCount;

  // Initialize particles
  const initializeParticles = useMemo(() => {
    const particles: QuantumParticle[] = [];
    
    for (let i = 0; i < particleCount; i++) {
      // Create particles in a sphere around the energy center
      const radius = Math.random() * 8 + 2;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      
      const x = energyCenter[0] + radius * Math.sin(phi) * Math.cos(theta);
      const y = energyCenter[1] + radius * Math.cos(phi);
      const z = energyCenter[2] + radius * Math.sin(phi) * Math.sin(theta);
      
      // Initial velocity with some randomness
      const velocityMagnitude = Math.random() * 0.5 + 0.1;
      const velocityTheta = Math.random() * Math.PI * 2;
      const velocityPhi = Math.random() * Math.PI;
      
      const particle: QuantumParticle = {
        position: new THREE.Vector3(x, y, z),
        velocity: new THREE.Vector3(
          velocityMagnitude * Math.sin(velocityPhi) * Math.cos(velocityTheta),
          velocityMagnitude * Math.cos(velocityPhi),
          velocityMagnitude * Math.sin(velocityPhi) * Math.sin(velocityTheta)
        ),
        id: i,
        lifespan: Math.random(),
        energy: Math.random(),
        maxLifespan: Math.random() * 10 + 5, // 5-15 seconds
      };
      
      particles.push(particle);
    }
    
    particlesRef.current = particles;
    return particles;
  }, [particleCount, energyCenter]);

  // Create geometry and material
  const { geometry, material } = useMemo(() => {
    const particles = initializeParticles;
    
    // Create buffer geometry
    const geo = new THREE.BufferGeometry();
    
    // Position attribute
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const particleIds = new Float32Array(particleCount);
    const lifespans = new Float32Array(particleCount);
    const energies = new Float32Array(particleCount);
    
    particles.forEach((particle, i) => {
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;
      
      velocities[i * 3] = particle.velocity.x;
      velocities[i * 3 + 1] = particle.velocity.y;
      velocities[i * 3 + 2] = particle.velocity.z;
      
      particleIds[i] = particle.id;
      lifespans[i] = particle.lifespan;
      energies[i] = particle.energy;
    });
    
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
    geo.setAttribute('particleId', new THREE.BufferAttribute(particleIds, 1));
    geo.setAttribute('lifespan', new THREE.BufferAttribute(lifespans, 1));
    geo.setAttribute('energy', new THREE.BufferAttribute(energies, 1));
    
    // Create shader material
    const mat = new THREE.ShaderMaterial({
      vertexShader: quantumParticlesVert,
      fragmentShader: quantumParticlesFrag,
      uniforms: {
        uTime: { value: 0 },
        uQuantumEnergy: { value: quantumEnergy },
        uFieldStrength: { value: fieldStrength },
        uEnergyCenter: { value: new THREE.Vector3(...energyCenter) },
        uParticleSize: { value: shaderSettings.performanceLevel * 6 + 2 },
        uFlowSpeed: { value: 1.0 },
        uCollapsing: { value: false },
        uCollapseProgress: { value: 0 },
        uPerformanceLevel: { value: shaderSettings.performanceLevel },
        uQuantumColors: { value: [
          new THREE.Color(0x0066ff), // Blue
          new THREE.Color(0x00ccff), // Cyan
          new THREE.Color(0xffffff), // White
          new THREE.Color(0xffaa00), // Gold
        ]},
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    
    return { geometry: geo, material: mat };
  }, [initializeParticles, particleCount, quantumEnergy, fieldStrength, energyCenter, shaderSettings]);

  // Particle physics update
  const updateParticles = (delta: number) => {
    const particles = particlesRef.current;
    const positions = geometryRef.current.attributes.position as THREE.BufferAttribute;
    const velocities = geometryRef.current.attributes.velocity as THREE.BufferAttribute;
    const lifespans = geometryRef.current.attributes.lifespan as THREE.BufferAttribute;
    const energiesAttr = geometryRef.current.attributes.energy as THREE.BufferAttribute;
    
    const centerVec = new THREE.Vector3(...energyCenter);
    
    particles.forEach((particle, i) => {
      // Update lifespan
      particle.lifespan -= delta / particle.maxLifespan;
      
      // Respawn particle if it died
      if (particle.lifespan <= 0) {
        // Reset particle near energy center
        const radius = Math.random() * 3 + 1;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        
        particle.position.set(
          centerVec.x + radius * Math.sin(phi) * Math.cos(theta),
          centerVec.y + radius * Math.cos(phi),
          centerVec.z + radius * Math.sin(phi) * Math.sin(theta)
        );
        
        particle.lifespan = 1.0;
        particle.energy = Math.random();
        particle.maxLifespan = Math.random() * 10 + 5;
        
        // Reset velocity
        const velocityMagnitude = Math.random() * 0.5 + 0.1;
        const velocityTheta = Math.random() * Math.PI * 2;
        const velocityPhi = Math.random() * Math.PI;
        
        particle.velocity.set(
          velocityMagnitude * Math.sin(velocityPhi) * Math.cos(velocityTheta),
          velocityMagnitude * Math.cos(velocityPhi),
          velocityMagnitude * Math.sin(velocityPhi) * Math.sin(velocityTheta)
        );
      }
      
      // Apply quantum forces (simplified CPU version)
      if (shaderSettings.performanceLevel >= 0.5) {
        const toCenter = centerVec.clone().sub(particle.position);
        const distToCenter = toCenter.length();
        
        // Attraction to center
        const centerForce = toCenter.normalize().multiplyScalar(quantumEnergy * 0.1 / (distToCenter + 0.1));
        
        // Orbital motion
        const up = new THREE.Vector3(0, 1, 0);
        const tangent = new THREE.Vector3().crossVectors(toCenter, up).normalize();
        const orbitalForce = tangent.multiplyScalar(fieldStrength * 0.05);
        
        // Apply forces
        particle.velocity.add(centerForce.multiplyScalar(delta));
        particle.velocity.add(orbitalForce.multiplyScalar(delta));
        
        // Damping
        particle.velocity.multiplyScalar(0.99);
      }
      
      // Update position
      particle.position.add(particle.velocity.clone().multiplyScalar(delta));
      
      // Update buffer attributes
      positions.setXYZ(i, particle.position.x, particle.position.y, particle.position.z);
      velocities.setXYZ(i, particle.velocity.x, particle.velocity.y, particle.velocity.z);
      lifespans.setX(i, particle.lifespan);
      energiesAttr.setX(i, particle.energy);
    });
    
    // Mark attributes as needing update
    positions.needsUpdate = true;
    velocities.needsUpdate = true;
    lifespans.needsUpdate = true;
    energiesAttr.needsUpdate = true;
  };

  // Animation loop
  useFrame((_, delta) => {
    if (!materialRef.current || !geometryRef.current) return;
    
    const time = performance.now() * 0.001;
    
    // Update shader uniforms
    materialRef.current.uniforms.uTime.value = time;
    materialRef.current.uniforms.uQuantumEnergy.value = quantumEnergy;
    materialRef.current.uniforms.uFieldStrength.value = fieldStrength;
    materialRef.current.uniforms.uEnergyCenter.value.set(...energyCenter);
    materialRef.current.uniforms.uCollapsing.value = waveformCollapsing;
    materialRef.current.uniforms.uCollapseProgress.value = collapseProgress;
    
    // Update particle physics
    updateParticles(delta);
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry ref={geometryRef} attach="geometry" {...geometry} />
      <shaderMaterial
        ref={materialRef}
        attach="material"
        {...material}
      />
    </points>
  );
}
"use client";

import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { useQuantumState } from "@/lib/state";
import * as THREE from "three";
import { getQuantumShaderSettings } from "./QuantumShaderManager";

// Import advanced shader code
import { readFileSync } from 'fs';
import { join } from 'path';

// Load shaders at build time
const quantumFieldAdvancedVert = `
// Advanced Quantum Field Vertex Shader
uniform float uTime;
uniform float uCoherence;
uniform float uQuantumEnergy;
uniform float uFieldStrength;
uniform vec3 uEnergyCenter;
uniform bool uCollapsing;
uniform float uCollapseProgress;
uniform float uPerformanceLevel;

attribute vec3 position;
attribute vec2 uv;
attribute vec3 normal;

varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying float vQuantumIntensity;
varying float vEnergyFlow;

vec3 quantumWaveDistortion(vec3 pos, float intensity) {
  vec3 distorted = pos;
  
  float wave1 = sin(pos.x * 3.14159 + uTime * 2.0) * intensity;
  float wave2 = cos(pos.z * 2.71828 + uTime * 1.5) * intensity;
  float wave3 = sin(length(pos.xz) * 1.41421 - uTime * 1.8) * intensity;
  
  float coherenceModulation = mix(0.5, 1.0, uCoherence);
  
  distorted.y += (wave1 + wave2 + wave3) * 0.1 * coherenceModulation;
  
  vec3 toCenter = pos - uEnergyCenter;
  float distToCenter = length(toCenter);
  float energyInfluence = exp(-distToCenter * 0.5) * uQuantumEnergy;
  
  float radialWave = sin(distToCenter * 5.0 - uTime * 3.0) * energyInfluence * 0.15;
  distorted.y += radialWave;
  
  return distorted;
}

void main() {
  vUv = uv;
  vNormal = normal;
  
  vec3 distortedPosition = quantumWaveDistortion(position, 1.0);
  
  vPosition = distortedPosition;
  vWorldPosition = (modelMatrix * vec4(distortedPosition, 1.0)).xyz;
  
  float distToCenter = length(distortedPosition - uEnergyCenter);
  vQuantumIntensity = exp(-distToCenter * 0.3) * uQuantumEnergy * uCoherence;
  
  vec3 flowDirection = normalize(uEnergyCenter - distortedPosition);
  vEnergyFlow = dot(flowDirection, normal) * uFieldStrength;
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(distortedPosition, 1.0);
}
`;

const quantumFieldAdvancedFrag = `
precision highp float;

uniform float uTime;
uniform vec2 uResolution;
uniform float uCoherence;
uniform float uQuantumEnergy;
uniform float uFieldStrength;
uniform vec3 uEnergyCenter;
uniform float uSuperpositionCount;
uniform vec3 uStatePositions[8];
uniform float uStateProbabilities[8];
uniform bool uCollapsing;
uniform float uCollapseProgress;
uniform float uPerformanceLevel;
uniform vec3 uQuantumColors[4];

varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying float vQuantumIntensity;
varying float vEnergyFlow;

float quantumInterference(vec2 pos) {
  float totalWave = 0.0;
  float totalAmplitude = 0.0;
  
  for (int i = 0; i < 8; i++) {
    if (float(i) >= uSuperpositionCount) break;
    
    vec2 diff = pos - uStatePositions[i].xy;
    float dist = length(diff);
    
    float phase = float(i) * 1.5707963;
    float frequency = 8.0 + float(i) * 2.0;
    float wave = sin(dist * frequency - uTime * 3.0 + phase);
    
    float envelope = exp(-dist * dist * 1.5) * uStateProbabilities[i];
    
    totalWave += wave * envelope;
    totalAmplitude += envelope;
  }
  
  return totalAmplitude > 0.0 ? totalWave / totalAmplitude : 0.0;
}

float probabilityDensity(vec2 pos) {
  float density = 0.0;
  
  for (int i = 0; i < 8; i++) {
    if (float(i) >= uSuperpositionCount) break;
    
    vec2 diff = pos - uStatePositions[i].xy;
    float dist = length(diff);
    
    float gaussian1 = exp(-dist * dist * 3.0) * uStateProbabilities[i];
    float gaussian2 = exp(-dist * dist * 0.8) * uStateProbabilities[i] * 0.3;
    
    density += gaussian1 + gaussian2;
  }
  
  return density;
}

vec3 collapseEffect(vec2 pos, vec3 baseColor) {
  if (!uCollapsing || uCollapseProgress <= 0.0) return baseColor;
  
  float distToCenter = length(pos - uEnergyCenter.xy);
  
  float ringFreq = 15.0;
  float ringSpeed = 30.0;
  float ring = sin(distToCenter * ringFreq - uCollapseProgress * ringSpeed);
  ring = smoothstep(-0.3, 0.3, ring);
  
  vec3 collapseColor = vec3(1.0, 0.2, 0.2);
  float collapseIntensity = ring * 0.6;
  collapseIntensity *= exp(-uCollapseProgress * 2.0);
  
  return mix(baseColor, collapseColor, collapseIntensity);
}

void main() {
  vec2 pos = vPosition.xz;
  
  vec3 fieldColor = uQuantumColors[0] * 0.3;
  
  if (uPerformanceLevel >= 0.5) {
    float interference = quantumInterference(pos);
    float density = probabilityDensity(pos);
    
    fieldColor += uQuantumColors[1] * interference * 0.4;
    fieldColor += uQuantumColors[2] * density * 0.3;
  }
  
  fieldColor = collapseEffect(pos, fieldColor);
  
  float coherenceIntensity = mix(0.4, 1.0, uCoherence);
  fieldColor *= coherenceIntensity;
  
  float alpha = (vQuantumIntensity + length(fieldColor) * 0.3) * 0.7;
  alpha = clamp(alpha, 0.0, 0.9);
  
  fieldColor *= uFieldStrength;
  
  gl_FragColor = vec4(fieldColor, alpha);
}
`;

// Fallback shaders for low-end devices
const fallbackVertexShader = `
uniform float uTime;
uniform float uCoherence;
attribute vec3 position;
attribute vec2 uv;
varying vec2 vUv;
varying vec3 vPosition;

void main() {
  vUv = uv;
  vec3 pos = position;
  pos.y += sin(pos.x + uTime) * 0.1 * (1.0 - uCoherence);
  vPosition = pos;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

const fallbackFragmentShader = `
precision mediump float;
uniform float uTime;
uniform float uQuantumEnergy;
uniform vec3 uQuantumColors[4];
varying vec2 vUv;
varying vec3 vPosition;

void main() {
  vec2 center = vec2(0.0);
  float dist = length(vPosition.xz - center);
  float intensity = exp(-dist * 0.5) * uQuantumEnergy;
  
  vec3 color = mix(uQuantumColors[0], uQuantumColors[1], sin(uTime) * 0.5 + 0.5);
  float alpha = intensity * 0.6;
  
  gl_FragColor = vec4(color * intensity, alpha);
}
`;

export default function QuantumField() {
  const meshRef = useRef<THREE.Mesh>(null!);
  const materialRef = useRef<THREE.ShaderMaterial>(null!);
  
  const superpositionActive = useQuantumState((s) => s.superpositionActive);
  const superpositionStates = useQuantumState((s) => s.superpositionStates);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const waveformCollapsing = useQuantumState((s) => s.waveformCollapsing);
  const collapseProgress = useQuantumState((s) => 
    s.superpositionStates.length > 0 
      ? Math.max(...s.superpositionStates.map(state => state.collapseProgress))
      : 0
  );
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const quantumEnergy = useQuantumState((s) => s.coherenceLevel); // Use coherence as energy proxy
  const fieldStrength = useQuantumState((s) => s.probabilityCloud.density); // Use density as field strength
  const energyCenter = [0, 0, 0] as [number, number, number]; // Fixed center for now

  // Get performance settings
  const shaderSettings = useMemo(() => getQuantumShaderSettings(), []);

  // Create advanced shader material
  const shaderMaterial = useMemo(() => {
    const useAdvanced = shaderSettings.useAdvancedField && quantumQuality !== 'minimal';
    const statePositions = new Array(8).fill(0).map(() => new THREE.Vector3());
    const stateProbabilities = new Array(8).fill(0);
    
    // Quantum color palette
    const quantumColors = [
      new THREE.Color(0x0066ff), // Blue
      new THREE.Color(0x00ccff), // Cyan
      new THREE.Color(0xffffff), // White
      new THREE.Color(0xffaa00), // Gold
    ];

    return new THREE.ShaderMaterial({
      vertexShader: useAdvanced ? quantumFieldAdvancedVert : fallbackVertexShader,
      fragmentShader: useAdvanced ? quantumFieldAdvancedFrag : fallbackFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        uCoherence: { value: coherenceLevel },
        uQuantumEnergy: { value: quantumEnergy },
        uFieldStrength: { value: fieldStrength },
        uEnergyCenter: { value: new THREE.Vector3(...energyCenter) },
        uSuperpositionCount: { value: 0 },
        uStatePositions: { value: statePositions },
        uStateProbabilities: { value: stateProbabilities },
        uCollapsing: { value: false },
        uCollapseProgress: { value: 0 },
        uPerformanceLevel: { value: shaderSettings.performanceLevel },
        
        // Lighting uniforms (for advanced shader)
        uLightPosition: { value: new THREE.Vector3(5, 10, 5) },
        uLightColor: { value: new THREE.Color(0xffffff) },
        uLightIntensity: { value: 1.0 },
        uAmbientColor: { value: new THREE.Color(0x404040) },
        
        // Quantum colors
        uQuantumColors: { value: quantumColors },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
  }, [shaderSettings, quantumQuality, coherenceLevel, quantumEnergy, fieldStrength, energyCenter]);

  useFrame((_, delta) => {
    if (!materialRef.current) return;

    const time = performance.now() * 0.001;
    materialRef.current.uniforms.uTime.value = time;
    materialRef.current.uniforms.uCoherence.value = coherenceLevel;
    materialRef.current.uniforms.uQuantumEnergy.value = quantumEnergy;
    materialRef.current.uniforms.uFieldStrength.value = fieldStrength;
    materialRef.current.uniforms.uCollapsing.value = waveformCollapsing;
    materialRef.current.uniforms.uCollapseProgress.value = collapseProgress;
    
    // Update energy center
    materialRef.current.uniforms.uEnergyCenter.value.set(...energyCenter);

    // Update superposition state data
    if (superpositionStates.length > 0) {
      materialRef.current.uniforms.uSuperpositionCount.value = superpositionStates.length;
      
      superpositionStates.forEach((state, index) => {
        if (index < 8) {
          materialRef.current.uniforms.uStatePositions.value[index].set(
            state.position[0],
            state.position[1],
            state.position[2]
          );
          materialRef.current.uniforms.uStateProbabilities.value[index] = state.probability;
        }
      });
    } else {
      materialRef.current.uniforms.uSuperpositionCount.value = 0;
    }
  });

  // Don't render field if superposition is not active or on minimal quality
  if (!superpositionActive || quantumQuality === 'minimal') {
    return null;
  }

  // Adjust geometry resolution based on performance
  const geometryArgs: [number, number, number, number] = [
    20, 20, 
    shaderSettings.fieldResolution, 
    shaderSettings.fieldResolution
  ];

  return (
    <mesh ref={meshRef} position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={geometryArgs} />
      <shaderMaterial
        ref={materialRef}
        attach="material"
        vertexShader={shaderMaterial.vertexShader}
        fragmentShader={shaderMaterial.fragmentShader}
        uniforms={shaderMaterial.uniforms}
        transparent={shaderMaterial.transparent}
        blending={shaderMaterial.blending}
        side={shaderMaterial.side}
      />
    </mesh>
  );
}
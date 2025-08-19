"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useQuantumState } from "@/lib/state";
import * as THREE from "three";

// Inline shader code for build compatibility
const quantumFieldAdvancedVert = `
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

const quantumLightingVert = `
uniform float uTime;
uniform float uQuantumEnergy;
uniform vec3 uEnergyCenter;
uniform float uCoherence;
uniform bool uCollapsing;
uniform float uCollapseProgress;

uniform vec3 uLightPositions[4];
uniform vec3 uLightColors[4];
uniform float uLightIntensities[4];
uniform int uActiveLights;

attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec3 vViewPosition;

varying vec3 vLightDirections[4];
varying float vLightDistances[4];
varying float vQuantumLightModulation[4];

float quantumLightInfluence(vec3 worldPos, vec3 lightPos) {
  float distToCenter = length(worldPos - uEnergyCenter);
  float centerInfluence = exp(-distToCenter * 0.2) * uQuantumEnergy;
  
  float distToLight = length(worldPos - lightPos);
  float lightFalloff = 1.0 / (1.0 + distToLight * distToLight * 0.1);
  
  float coherenceEffect = mix(0.7, 1.0, uCoherence);
  
  float fluctuation = sin(uTime * 2.0 + distToCenter * 0.5) * 0.1 + 0.9;
  fluctuation = mix(1.0, fluctuation, 1.0 - uCoherence);
  
  return centerInfluence * lightFalloff * coherenceEffect * fluctuation;
}

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  
  vec4 viewPos = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = viewPos.xyz;
  
  for (int i = 0; i < 4; i++) {
    if (i >= uActiveLights) break;
    
    vec3 lightDir = uLightPositions[i] - vWorldPosition;
    vLightDistances[i] = length(lightDir);
    vLightDirections[i] = normalize(lightDir);
    
    vQuantumLightModulation[i] = quantumLightInfluence(vWorldPosition, uLightPositions[i]);
  }
  
  gl_Position = projectionMatrix * viewPos;
}
`;

const quantumLightingFrag = `
precision highp float;

uniform float uTime;
uniform float uQuantumEnergy;
uniform vec3 uEnergyCenter;
uniform float uCoherence;
uniform bool uCollapsing;
uniform float uCollapseProgress;

uniform vec3 uLightPositions[4];
uniform vec3 uLightColors[4];
uniform float uLightIntensities[4];
uniform int uActiveLights;

uniform vec3 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uQuantumReflectance;

uniform vec3 uQuantumColors[4];
uniform float uFieldStrength;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec3 vViewPosition;
varying vec3 vLightDirections[4];
varying float vLightDistances[4];
varying float vQuantumLightModulation[4];

vec3 quantumFieldLighting(vec3 worldPos, vec3 normal) {
  vec3 fieldColor = vec3(0.0);
  
  float distToCenter = length(worldPos - uEnergyCenter);
  float fieldIntensity = exp(-distToCenter * 0.3) * uQuantumEnergy * uFieldStrength;
  
  vec3 coherentColor = mix(uQuantumColors[0], uQuantumColors[1], uCoherence);
  vec3 incoherentColor = mix(uQuantumColors[2], uQuantumColors[3], sin(uTime * 3.0) * 0.5 + 0.5);
  vec3 quantumColor = mix(incoherentColor, coherentColor, uCoherence);
  
  vec3 fieldDirection = normalize(uEnergyCenter - worldPos);
  float fieldAlignment = dot(normal, fieldDirection) * 0.5 + 0.5;
  
  float fluctuation = sin(uTime * 2.0 + distToCenter * 0.8) * 0.2 + 0.8;
  fluctuation = mix(0.5, fluctuation, uCoherence);
  
  fieldColor = quantumColor * fieldIntensity * fieldAlignment * fluctuation;
  
  return fieldColor;
}

vec3 collapseLighting(vec3 baseColor, vec3 worldPos) {
  if (!uCollapsing || uCollapseProgress <= 0.0) return baseColor;
  
  float distToCenter = length(worldPos - uEnergyCenter);
  
  float collapseWave = sin(distToCenter * 10.0 - uCollapseProgress * 30.0);
  collapseWave = smoothstep(-0.5, 0.5, collapseWave);
  
  vec3 collapseColor = vec3(1.0, 0.2, 0.2) * 2.0;
  
  float collapseIntensity = collapseWave * uCollapseProgress * exp(-distToCenter * 0.5);
  
  return baseColor + collapseColor * collapseIntensity;
}

void main() {
  vec3 normal = normalize(vNormal);
  vec3 viewDir = normalize(-vViewPosition);
  
  vec3 albedo = uBaseColor;
  
  vec3 lighting = vec3(0.0);
  
  for (int i = 0; i < 4; i++) {
    if (i >= uActiveLights) break;
    
    vec3 lightDir = vLightDirections[i];
    float distance = vLightDistances[i];
    
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = uLightColors[i] * uLightIntensities[i] * attenuation;
    
    radiance *= vQuantumLightModulation[i];
    
    float quantumReflectance = mix(1.0, uQuantumReflectance, uQuantumEnergy);
    radiance *= quantumReflectance;
    
    float NdotL = max(dot(normal, lightDir), 0.0);
    lighting += albedo * radiance * NdotL;
  }
  
  vec3 fieldLighting = quantumFieldLighting(vWorldPosition, normal);
  lighting += fieldLighting;
  
  vec3 ambient = uQuantumColors[0] * 0.1 * uQuantumEnergy;
  lighting += ambient * albedo;
  
  lighting = collapseLighting(lighting, vWorldPosition);
  
  float coherenceModulation = mix(0.7, 1.0, uCoherence);
  lighting *= coherenceModulation;
  
  lighting = lighting / (lighting + vec3(1.0));
  lighting = pow(lighting, vec3(1.0/2.2));
  
  gl_FragColor = vec4(lighting, 1.0);
}
`;

interface QuantumShaderManagerProps {
  children?: React.ReactNode;
}

// Performance detection utility
const detectPerformanceLevel = (): number => {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  
  if (!gl) return 0.0; // No WebGL support
  
  // Check WebGL capabilities
  const renderer = gl.getParameter(gl.RENDERER);
  const vendor = gl.getParameter(gl.VENDOR);
  
  // Basic performance heuristics
  let performanceScore = 0.5; // Default medium
  
  // Check for high-end GPU indicators
  if (renderer && (renderer.includes('RTX') || renderer.includes('GTX') || 
      renderer.includes('Radeon RX') || renderer.includes('Apple M'))) {
    performanceScore = 1.0;
  }
  
  // Check for integrated graphics
  if (renderer && renderer.includes('Intel') && !renderer.includes('Arc')) {
    performanceScore = 0.3;
  }
  
  // Check WebGL extensions
  const extensions = gl.getSupportedExtensions();
  if (extensions && extensions.length > 50) {
    performanceScore = Math.min(1.0, performanceScore + 0.2);
  }
  
  // Memory considerations
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (maxTextureSize >= 8192) {
    performanceScore = Math.min(1.0, performanceScore + 0.1);
  }
  
  return Math.max(0.0, Math.min(1.0, performanceScore));
};

// Fallback shader for low-end devices
const fallbackVertexShader = `
uniform float uTime;
attribute vec3 position;
attribute vec2 uv;
varying vec2 vUv;

void main() {
  vUv = uv;
  vec3 pos = position;
  pos.y += sin(pos.x + uTime) * 0.1;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

const fallbackFragmentShader = `
precision mediump float;
uniform float uTime;
uniform vec3 uQuantumColors[4];
varying vec2 vUv;

void main() {
  vec2 center = vec2(0.5);
  float dist = length(vUv - center);
  float intensity = exp(-dist * 2.0);
  
  vec3 color = mix(uQuantumColors[0], uQuantumColors[1], sin(uTime) * 0.5 + 0.5);
  gl_FragColor = vec4(color * intensity, intensity * 0.5);
}
`;

export default function QuantumShaderManager({ children }: QuantumShaderManagerProps) {
  const performanceLevel = useRef<number>(0.5);
  const shaderMaterials = useRef<Map<string, THREE.ShaderMaterial>>(new Map());
  
  // Quantum state
  const quantumEnergy = useQuantumState((s) => s.coherenceLevel);
  const fieldStrength = useQuantumState((s) => s.probabilityCloud.density);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const energyCenter = [0, 0, 0] as [number, number, number];
  const superpositionStates = useQuantumState((s) => s.superpositionStates);
  const waveformCollapsing = useQuantumState((s) => s.waveformCollapsing);
  const collapseProgress = useQuantumState((s) => 
    s.superpositionStates.length > 0 
      ? Math.max(...s.superpositionStates.map(state => state.collapseProgress))
      : 0
  );
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  // Detect performance level on mount
  useEffect(() => {
    performanceLevel.current = detectPerformanceLevel();
    console.log(`Quantum Shader Performance Level: ${performanceLevel.current.toFixed(2)}`);
  }, []);

  // Quantum color palette
  const quantumColors = useMemo(() => [
    new THREE.Color(0x0066ff), // Blue
    new THREE.Color(0x00ccff), // Cyan
    new THREE.Color(0xffffff), // White
    new THREE.Color(0xffaa00), // Gold
  ], []);

  // Create advanced quantum field material
  const createQuantumFieldMaterial = useMemo(() => {
    const useAdvanced = performanceLevel.current >= 0.5 && quantumQuality !== 'minimal';
    
    const material = new THREE.ShaderMaterial({
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
        uStatePositions: { value: new Array(8).fill(0).map(() => new THREE.Vector3()) },
        uStateProbabilities: { value: new Array(8).fill(0) },
        uCollapsing: { value: false },
        uCollapseProgress: { value: 0 },
        uPerformanceLevel: { value: performanceLevel.current },
        
        // Lighting uniforms
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

    shaderMaterials.current.set('quantumField', material);
    return material;
  }, [quantumColors, coherenceLevel, quantumEnergy, fieldStrength, energyCenter, quantumQuality]);

  // Create quantum particle material
  const createQuantumParticleMaterial = useMemo(() => {
    const useAdvanced = performanceLevel.current >= 0.3 && quantumQuality !== 'minimal';
    
    if (!useAdvanced) return null; // Skip particles on low-end devices
    
    const material = new THREE.ShaderMaterial({
      vertexShader: quantumParticlesVert,
      fragmentShader: quantumParticlesFrag,
      uniforms: {
        uTime: { value: 0 },
        uQuantumEnergy: { value: quantumEnergy },
        uFieldStrength: { value: fieldStrength },
        uEnergyCenter: { value: new THREE.Vector3(...energyCenter) },
        uParticleSize: { value: 4.0 },
        uFlowSpeed: { value: 1.0 },
        uCollapsing: { value: false },
        uCollapseProgress: { value: 0 },
        uPerformanceLevel: { value: performanceLevel.current },
        uQuantumColors: { value: quantumColors },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    shaderMaterials.current.set('quantumParticles', material);
    return material;
  }, [quantumColors, quantumEnergy, fieldStrength, energyCenter, quantumQuality]);

  // Create quantum lighting material
  const createQuantumLightingMaterial = useMemo(() => {
    const useAdvanced = performanceLevel.current >= 0.7 && quantumQuality === 'high';
    
    if (!useAdvanced) return null; // Skip advanced lighting on lower-end devices
    
    const material = new THREE.ShaderMaterial({
      vertexShader: quantumLightingVert,
      fragmentShader: quantumLightingFrag,
      uniforms: {
        uTime: { value: 0 },
        uQuantumEnergy: { value: quantumEnergy },
        uEnergyCenter: { value: new THREE.Vector3(...energyCenter) },
        uCoherence: { value: coherenceLevel },
        uCollapsing: { value: false },
        uCollapseProgress: { value: 0 },
        
        // Light arrays
        uLightPositions: { value: [
          new THREE.Vector3(5, 10, 5),
          new THREE.Vector3(-5, 10, -5),
          new THREE.Vector3(0, 15, 0),
          new THREE.Vector3(0, 5, 10),
        ]},
        uLightColors: { value: [
          new THREE.Color(0xffffff),
          new THREE.Color(0x00aaff),
          new THREE.Color(0xffaa00),
          new THREE.Color(0xff00aa),
        ]},
        uLightIntensities: { value: [1.0, 0.8, 0.6, 0.4] },
        uActiveLights: { value: 2 },
        
        // Material properties
        uBaseColor: { value: new THREE.Color(0x888888) },
        uMetallic: { value: 0.1 },
        uRoughness: { value: 0.8 },
        uQuantumReflectance: { value: 1.5 },
        
        // Quantum properties
        uQuantumColors: { value: quantumColors },
        uFieldStrength: { value: fieldStrength },
      },
    });

    shaderMaterials.current.set('quantumLighting', material);
    return material;
  }, [quantumColors, quantumEnergy, energyCenter, coherenceLevel, fieldStrength, quantumQuality]);

  // Update shader uniforms
  useFrame((_, delta) => {
    const time = performance.now() * 0.001;
    
    shaderMaterials.current.forEach((material, key) => {
      if (!material.uniforms) return;
      
      // Common uniforms
      if (material.uniforms.uTime) material.uniforms.uTime.value = time;
      if (material.uniforms.uCoherence) material.uniforms.uCoherence.value = coherenceLevel;
      if (material.uniforms.uQuantumEnergy) material.uniforms.uQuantumEnergy.value = quantumEnergy;
      if (material.uniforms.uFieldStrength) material.uniforms.uFieldStrength.value = fieldStrength;
      if (material.uniforms.uCollapsing) material.uniforms.uCollapsing.value = waveformCollapsing;
      if (material.uniforms.uCollapseProgress) material.uniforms.uCollapseProgress.value = collapseProgress;
      
      // Update energy center
      if (material.uniforms.uEnergyCenter) {
        material.uniforms.uEnergyCenter.value.set(...energyCenter);
      }
      
      // Update superposition states for field shader
      if (key === 'quantumField' && material.uniforms.uSuperpositionCount) {
        material.uniforms.uSuperpositionCount.value = superpositionStates.length;
        
        superpositionStates.forEach((state, index) => {
          if (index < 8 && material.uniforms.uStatePositions && material.uniforms.uStateProbabilities) {
            material.uniforms.uStatePositions.value[index].set(...state.position);
            material.uniforms.uStateProbabilities.value[index] = state.probability;
          }
        });
      }
    });
  });

  // Provide shader materials to children via context or props
  return (
    <>
      {children}
    </>
  );
}

// Export materials for use in other components
export const useQuantumShaderMaterials = () => {
  const materials = useRef<Map<string, THREE.ShaderMaterial>>(new Map());
  return materials.current;
};

// Utility function to get performance-appropriate shader settings
export const getQuantumShaderSettings = () => {
  const performanceLevel = detectPerformanceLevel();
  
  return {
    performanceLevel,
    useAdvancedField: performanceLevel >= 0.5,
    useParticles: performanceLevel >= 0.3,
    useAdvancedLighting: performanceLevel >= 0.7,
    particleCount: Math.floor(performanceLevel * 1000 + 100),
    fieldResolution: Math.floor(performanceLevel * 128 + 32),
    maxLights: Math.floor(performanceLevel * 4 + 1),
  };
};
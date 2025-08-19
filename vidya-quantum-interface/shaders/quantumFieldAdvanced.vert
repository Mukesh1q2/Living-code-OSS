// Advanced Quantum Field Vertex Shader
// Implements sophisticated quantum field distortions with performance optimizations

uniform float uTime;
uniform float uCoherence;
uniform float uQuantumEnergy;
uniform float uFieldStrength;
uniform vec3 uEnergyCenter;
uniform bool uCollapsing;
uniform float uCollapseProgress;
uniform float uPerformanceLevel; // 0.0 = minimal, 1.0 = maximum

attribute vec3 position;
attribute vec2 uv;
attribute vec3 normal;

varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying float vQuantumIntensity;
varying float vEnergyFlow;

// Quantum field wave functions
vec3 quantumWaveDistortion(vec3 pos, float intensity) {
  vec3 distorted = pos;
  
  // Primary quantum oscillations
  float wave1 = sin(pos.x * 3.14159 + uTime * 2.0) * intensity;
  float wave2 = cos(pos.z * 2.71828 + uTime * 1.5) * intensity;
  float wave3 = sin(length(pos.xz) * 1.41421 - uTime * 1.8) * intensity;
  
  // Coherence-based amplitude modulation
  float coherenceModulation = mix(0.5, 1.0, uCoherence);
  
  distorted.y += (wave1 + wave2 + wave3) * 0.1 * coherenceModulation;
  
  // Energy center influence
  vec3 toCenter = pos - uEnergyCenter;
  float distToCenter = length(toCenter);
  float energyInfluence = exp(-distToCenter * 0.5) * uQuantumEnergy;
  
  // Radial energy waves
  float radialWave = sin(distToCenter * 5.0 - uTime * 3.0) * energyInfluence * 0.15;
  distorted.y += radialWave;
  
  // Tangential energy flow
  float tangentialFlow = cos(atan(toCenter.z, toCenter.x) * 4.0 + uTime * 2.5) * energyInfluence * 0.08;
  distorted.x += tangentialFlow * sin(uTime * 0.8);
  distorted.z += tangentialFlow * cos(uTime * 0.8);
  
  return distorted;
}

// Collapse distortion effects
vec3 collapseDistortion(vec3 pos, float progress) {
  if (!uCollapsing || progress <= 0.0) return pos;
  
  vec3 distorted = pos;
  float distToCenter = length(pos - uEnergyCenter);
  
  // Implosion effect
  float implosionStrength = progress * progress; // Quadratic for dramatic effect
  vec3 toCenter = normalize(uEnergyCenter - pos);
  distorted += toCenter * implosionStrength * 0.3;
  
  // Spiral collapse
  float angle = atan(pos.z - uEnergyCenter.z, pos.x - uEnergyCenter.x);
  float spiralOffset = progress * 6.28318; // 2π
  float spiralRadius = distToCenter * (1.0 - progress * 0.8);
  
  distorted.x = uEnergyCenter.x + cos(angle + spiralOffset) * spiralRadius;
  distorted.z = uEnergyCenter.z + sin(angle + spiralOffset) * spiralRadius;
  
  // Vertical compression
  distorted.y *= (1.0 - progress * 0.5);
  
  return distorted;
}

// Performance-adaptive distortion
vec3 adaptiveDistortion(vec3 pos) {
  if (uPerformanceLevel < 0.3) {
    // Minimal distortion for low-end devices
    float simpleWave = sin(pos.x + uTime) * 0.05;
    return pos + vec3(0.0, simpleWave, 0.0);
  } else if (uPerformanceLevel < 0.7) {
    // Medium distortion
    return quantumWaveDistortion(pos, 0.5);
  } else {
    // Full distortion
    return quantumWaveDistortion(pos, 1.0);
  }
}

void main() {
  vUv = uv;
  vNormal = normal;
  
  // Apply performance-adaptive quantum distortions
  vec3 distortedPosition = adaptiveDistortion(position);
  
  // Apply collapse effects
  distortedPosition = collapseDistortion(distortedPosition, uCollapseProgress);
  
  vPosition = distortedPosition;
  vWorldPosition = (modelMatrix * vec4(distortedPosition, 1.0)).xyz;
  
  // Calculate quantum intensity for fragment shader
  float distToCenter = length(distortedPosition - uEnergyCenter);
  vQuantumIntensity = exp(-distToCenter * 0.3) * uQuantumEnergy * uCoherence;
  
  // Calculate energy flow direction and strength
  vec3 flowDirection = normalize(uEnergyCenter - distortedPosition);
  vEnergyFlow = dot(flowDirection, normal) * uFieldStrength;
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(distortedPosition, 1.0);
}
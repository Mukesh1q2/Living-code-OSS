// Quantum Dynamic Lighting Vertex Shader
// Handles vertex transformations for quantum-responsive lighting

uniform float uTime;
uniform float uQuantumEnergy;
uniform vec3 uEnergyCenter;
uniform float uCoherence;
uniform bool uCollapsing;
uniform float uCollapseProgress;

// Light uniforms
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

// Light-specific varyings
varying vec3 vLightDirections[4];
varying float vLightDistances[4];
varying float vQuantumLightModulation[4];

// Quantum field influence on lighting
float quantumLightInfluence(vec3 worldPos, vec3 lightPos) {
  // Distance from quantum energy center affects light behavior
  float distToCenter = length(worldPos - uEnergyCenter);
  float centerInfluence = exp(-distToCenter * 0.2) * uQuantumEnergy;
  
  // Distance from light source
  float distToLight = length(worldPos - lightPos);
  float lightFalloff = 1.0 / (1.0 + distToLight * distToLight * 0.1);
  
  // Quantum coherence affects light stability
  float coherenceEffect = mix(0.7, 1.0, uCoherence);
  
  // Time-based quantum fluctuations
  float fluctuation = sin(uTime * 2.0 + distToCenter * 0.5) * 0.1 + 0.9;
  fluctuation = mix(1.0, fluctuation, 1.0 - uCoherence);
  
  return centerInfluence * lightFalloff * coherenceEffect * fluctuation;
}

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  
  // Calculate world position
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  
  // Calculate view position
  vec4 viewPos = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = viewPos.xyz;
  
  // Calculate light directions and quantum modulations
  for (int i = 0; i < 4; i++) {
    if (i >= uActiveLights) break;
    
    vec3 lightDir = uLightPositions[i] - vWorldPosition;
    vLightDistances[i] = length(lightDir);
    vLightDirections[i] = normalize(lightDir);
    
    // Calculate quantum influence on this light
    vQuantumLightModulation[i] = quantumLightInfluence(vWorldPosition, uLightPositions[i]);
  }
  
  gl_Position = projectionMatrix * viewPos;
}
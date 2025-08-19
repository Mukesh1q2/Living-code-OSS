// Quantum Dynamic Lighting Fragment Shader
// Implements quantum-responsive lighting with energy field interactions

precision highp float;

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

// Material properties
uniform vec3 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uQuantumReflectance;

// Quantum field properties
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

// PBR lighting functions
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  
  float num = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = 3.14159265 * denom * denom;
  
  return num / denom;
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0);
  float k = (r * r) / 8.0;
  
  float num = NdotV;
  float denom = NdotV * (1.0 - k) + k;
  
  return num / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);
  
  return ggx1 * ggx2;
}

// Quantum field lighting effects
vec3 quantumFieldLighting(vec3 worldPos, vec3 normal) {
  vec3 fieldColor = vec3(0.0);
  
  // Distance-based quantum field intensity
  float distToCenter = length(worldPos - uEnergyCenter);
  float fieldIntensity = exp(-distToCenter * 0.3) * uQuantumEnergy * uFieldStrength;
  
  // Quantum field color based on coherence
  vec3 coherentColor = mix(uQuantumColors[0], uQuantumColors[1], uCoherence);
  vec3 incoherentColor = mix(uQuantumColors[2], uQuantumColors[3], sin(uTime * 3.0) * 0.5 + 0.5);
  vec3 quantumColor = mix(incoherentColor, coherentColor, uCoherence);
  
  // Field direction influence
  vec3 fieldDirection = normalize(uEnergyCenter - worldPos);
  float fieldAlignment = dot(normal, fieldDirection) * 0.5 + 0.5;
  
  // Quantum fluctuations
  float fluctuation = sin(uTime * 2.0 + distToCenter * 0.8) * 0.2 + 0.8;
  fluctuation = mix(0.5, fluctuation, uCoherence);
  
  fieldColor = quantumColor * fieldIntensity * fieldAlignment * fluctuation;
  
  return fieldColor;
}

// Collapse lighting effects
vec3 collapseLighting(vec3 baseColor, vec3 worldPos) {
  if (!uCollapsing || uCollapseProgress <= 0.0) return baseColor;
  
  float distToCenter = length(worldPos - uEnergyCenter);
  
  // Collapse wave effect
  float collapseWave = sin(distToCenter * 10.0 - uCollapseProgress * 30.0);
  collapseWave = smoothstep(-0.5, 0.5, collapseWave);
  
  // Collapse color
  vec3 collapseColor = vec3(1.0, 0.2, 0.2) * 2.0; // Bright red
  
  // Intensity based on collapse progress and distance
  float collapseIntensity = collapseWave * uCollapseProgress * exp(-distToCenter * 0.5);
  
  return baseColor + collapseColor * collapseIntensity;
}

// Main PBR lighting calculation with quantum effects
vec3 calculateLighting(vec3 albedo, vec3 normal, vec3 viewDir) {
  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, uMetallic);
  
  vec3 Lo = vec3(0.0);
  
  // Calculate lighting for each active light
  for (int i = 0; i < 4; i++) {
    if (i >= uActiveLights) break;
    
    vec3 lightDir = vLightDirections[i];
    vec3 halfwayDir = normalize(viewDir + lightDir);
    float distance = vLightDistances[i];
    
    // Light attenuation with quantum modulation
    float attenuation = 1.0 / (distance * distance);
    vec3 radiance = uLightColors[i] * uLightIntensities[i] * attenuation;
    
    // Apply quantum modulation
    radiance *= vQuantumLightModulation[i];
    
    // Quantum-enhanced reflectance
    float quantumReflectance = mix(1.0, uQuantumReflectance, uQuantumEnergy);
    radiance *= quantumReflectance;
    
    // PBR calculations
    float NDF = distributionGGX(normal, halfwayDir, uRoughness);
    float G = geometrySmith(normal, viewDir, lightDir, uRoughness);
    vec3 F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), F0);
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - uMetallic;
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0) + 0.001;
    vec3 specular = numerator / denominator;
    
    float NdotL = max(dot(normal, lightDir), 0.0);
    Lo += (kD * albedo / 3.14159265 + specular) * radiance * NdotL;
  }
  
  return Lo;
}

void main() {
  vec3 normal = normalize(vNormal);
  vec3 viewDir = normalize(-vViewPosition);
  
  // Base material color
  vec3 albedo = uBaseColor;
  
  // Calculate PBR lighting with quantum effects
  vec3 lighting = calculateLighting(albedo, normal, viewDir);
  
  // Add quantum field lighting
  vec3 fieldLighting = quantumFieldLighting(vWorldPosition, normal);
  lighting += fieldLighting;
  
  // Add ambient quantum glow
  vec3 ambient = uQuantumColors[0] * 0.1 * uQuantumEnergy;
  lighting += ambient * albedo;
  
  // Apply collapse effects
  lighting = collapseLighting(lighting, vWorldPosition);
  
  // Coherence-based final modulation
  float coherenceModulation = mix(0.7, 1.0, uCoherence);
  lighting *= coherenceModulation;
  
  // Tone mapping and gamma correction
  lighting = lighting / (lighting + vec3(1.0));
  lighting = pow(lighting, vec3(1.0/2.2));
  
  gl_FragColor = vec4(lighting, 1.0);
}
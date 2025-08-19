// Advanced Quantum Field Fragment Shader
// Implements sophisticated quantum field visualization with dynamic lighting

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

// Lighting uniforms
uniform vec3 uLightPosition;
uniform vec3 uLightColor;
uniform float uLightIntensity;
uniform vec3 uAmbientColor;

// Quantum state colors
uniform vec3 uQuantumColors[4];

varying vec2 vUv;
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying float vQuantumIntensity;
varying float vEnergyFlow;

// Noise functions for quantum fluctuations
float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 st) {
  vec2 i = floor(st);
  vec2 f = fract(st);
  
  float a = random(i);
  float b = random(i + vec2(1.0, 0.0));
  float c = random(i + vec2(0.0, 1.0));
  float d = random(i + vec2(1.0, 1.0));
  
  vec2 u = f * f * (3.0 - 2.0 * f);
  
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Fractal noise for complex quantum patterns
float fbm(vec2 st) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 0.0;
  
  for (int i = 0; i < 4; i++) {
    value += amplitude * noise(st);
    st *= 2.0;
    amplitude *= 0.5;
  }
  return value;
}

// Quantum wave interference calculation
float quantumInterference(vec2 pos) {
  float totalWave = 0.0;
  float totalAmplitude = 0.0;
  
  for (int i = 0; i < 8; i++) {
    if (float(i) >= uSuperpositionCount) break;
    
    vec2 diff = pos - uStatePositions[i].xy;
    float dist = length(diff);
    
    // Wave equation with quantum phase
    float phase = float(i) * 1.5707963; // π/2 phase differences
    float frequency = 8.0 + float(i) * 2.0; // Varying frequencies
    float wave = sin(dist * frequency - uTime * 3.0 + phase);
    
    // Gaussian envelope with probability weighting
    float envelope = exp(-dist * dist * 1.5) * uStateProbabilities[i];
    
    totalWave += wave * envelope;
    totalAmplitude += envelope;
  }
  
  return totalAmplitude > 0.0 ? totalWave / totalAmplitude : 0.0;
}

// Probability density field
float probabilityDensity(vec2 pos) {
  float density = 0.0;
  
  for (int i = 0; i < 8; i++) {
    if (float(i) >= uSuperpositionCount) break;
    
    vec2 diff = pos - uStatePositions[i].xy;
    float dist = length(diff);
    
    // Multi-scale Gaussian for rich probability clouds
    float gaussian1 = exp(-dist * dist * 3.0) * uStateProbabilities[i];
    float gaussian2 = exp(-dist * dist * 0.8) * uStateProbabilities[i] * 0.3;
    
    density += gaussian1 + gaussian2;
  }
  
  return density;
}

// Energy flow visualization
vec3 energyFlowColor(vec2 pos) {
  vec3 toCenter = vec3(uEnergyCenter.xy - pos, 0.0);
  float distToCenter = length(toCenter.xy);
  
  // Radial energy streams
  float angle = atan(toCenter.y, toCenter.x);
  float streamPattern = sin(angle * 6.0 + uTime * 2.0) * 0.5 + 0.5;
  
  // Distance-based energy intensity
  float energyIntensity = exp(-distToCenter * 0.4) * uQuantumEnergy;
  
  // Flowing energy color
  vec3 energyColor = mix(
    vec3(0.0, 0.8, 1.0), // Blue
    vec3(1.0, 0.6, 0.0), // Orange
    streamPattern
  );
  
  return energyColor * energyIntensity * vEnergyFlow;
}

// Dynamic quantum lighting
vec3 quantumLighting(vec3 baseColor, vec3 normal, vec3 worldPos) {
  // Standard Phong lighting
  vec3 lightDir = normalize(uLightPosition - worldPos);
  vec3 viewDir = normalize(cameraPosition - worldPos);
  vec3 reflectDir = reflect(-lightDir, normal);
  
  // Ambient
  vec3 ambient = uAmbientColor * baseColor;
  
  // Diffuse with quantum modulation
  float diff = max(dot(normal, lightDir), 0.0);
  float quantumModulation = 0.8 + 0.2 * sin(uTime * 2.0 + worldPos.x * 0.5);
  vec3 diffuse = uLightColor * diff * baseColor * quantumModulation;
  
  // Specular with quantum shimmer
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
  float shimmer = 0.7 + 0.3 * sin(uTime * 4.0 + worldPos.z * 0.3);
  vec3 specular = uLightColor * spec * shimmer;
  
  // Quantum energy glow
  float energyGlow = vQuantumIntensity * uLightIntensity;
  vec3 quantumGlow = uQuantumColors[0] * energyGlow;
  
  return ambient + diffuse + specular + quantumGlow;
}

// Waveform collapse visual effect
vec3 collapseEffect(vec2 pos, vec3 baseColor) {
  if (!uCollapsing || uCollapseProgress <= 0.0) return baseColor;
  
  float distToCenter = length(pos - uEnergyCenter.xy);
  
  // Expanding collapse rings
  float ringFreq = 15.0;
  float ringSpeed = 30.0;
  float ring = sin(distToCenter * ringFreq - uCollapseProgress * ringSpeed);
  ring = smoothstep(-0.3, 0.3, ring);
  
  // Decoherence noise
  vec2 noiseCoord = pos * 10.0 + uTime * 5.0;
  float decoherenceNoise = fbm(noiseCoord) * (1.0 - uCoherence) * uCollapseProgress;
  
  // Collapse color mixing
  vec3 collapseColor = vec3(1.0, 0.2, 0.2); // Red collapse
  float collapseIntensity = ring * 0.6 + decoherenceNoise * 0.4;
  collapseIntensity *= exp(-uCollapseProgress * 2.0); // Fade over time
  
  return mix(baseColor, collapseColor, collapseIntensity);
}

// Performance-adaptive rendering
vec3 adaptiveQuantumField(vec2 pos) {
  vec3 fieldColor = vec3(0.0);
  
  if (uPerformanceLevel < 0.3) {
    // Minimal quality - simple gradient
    float dist = length(pos - uEnergyCenter.xy);
    float intensity = exp(-dist * 0.5) * uQuantumEnergy;
    fieldColor = uQuantumColors[0] * intensity;
  } else if (uPerformanceLevel < 0.7) {
    // Medium quality - basic interference
    float interference = quantumInterference(pos) * 0.5;
    float density = probabilityDensity(pos) * 0.3;
    
    fieldColor = uQuantumColors[0] * (interference + density);
    fieldColor += energyFlowColor(pos) * 0.5;
  } else {
    // High quality - full effects
    float interference = quantumInterference(pos);
    float density = probabilityDensity(pos);
    
    // Base quantum field
    fieldColor = uQuantumColors[0] * 0.3;
    
    // Add interference patterns
    fieldColor += uQuantumColors[1] * interference * 0.4;
    
    // Add probability density
    fieldColor += uQuantumColors[2] * density * 0.3;
    
    // Add energy flow
    fieldColor += energyFlowColor(pos);
    
    // Add quantum noise
    vec2 noiseCoord = pos * 5.0 + uTime * 0.5;
    float quantumNoise = fbm(noiseCoord) * 0.1;
    fieldColor += uQuantumColors[3] * quantumNoise;
  }
  
  return fieldColor;
}

void main() {
  vec2 pos = vPosition.xz;
  
  // Calculate base quantum field color
  vec3 fieldColor = adaptiveQuantumField(pos);
  
  // Apply dynamic lighting (only for medium+ quality)
  if (uPerformanceLevel >= 0.3) {
    fieldColor = quantumLighting(fieldColor, vNormal, vWorldPosition);
  }
  
  // Apply collapse effects
  fieldColor = collapseEffect(pos, fieldColor);
  
  // Coherence-based intensity modulation
  float coherenceIntensity = mix(0.4, 1.0, uCoherence);
  fieldColor *= coherenceIntensity;
  
  // Calculate alpha with quantum intensity
  float alpha = (vQuantumIntensity + length(fieldColor) * 0.3) * 0.7;
  alpha = clamp(alpha, 0.0, 0.9);
  
  // Apply field strength
  fieldColor *= uFieldStrength;
  
  gl_FragColor = vec4(fieldColor, alpha);
}
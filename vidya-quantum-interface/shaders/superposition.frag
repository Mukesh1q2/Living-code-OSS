precision highp float;

uniform float uTime;
uniform vec2 uResolution;
uniform float uCoherence;
uniform float uSuperpositionCount;
uniform vec3 uStatePositions[8];
uniform float uStateProbabilities[8];
uniform bool uCollapsing;
uniform float uCollapseProgress;

// Quantum wave function visualization
float quantumWave(vec2 pos, vec3 center, float probability, float phase) {
  vec2 diff = pos - center.xy;
  float dist = length(diff);
  
  // Wave equation with quantum properties
  float wave = sin(dist * 10.0 - uTime * 2.0 + phase) * probability;
  wave *= exp(-dist * dist * 0.5); // Gaussian envelope
  
  return wave;
}

// Interference pattern between quantum states
float quantumInterference(vec2 pos) {
  float totalWave = 0.0;
  
  for (int i = 0; i < 8; i++) {
    if (i >= int(uSuperpositionCount)) break;
    
    float phase = float(i) * 1.57; // π/2 phase difference
    float wave = quantumWave(pos, uStatePositions[i], uStateProbabilities[i], phase);
    totalWave += wave;
  }
  
  return totalWave;
}

// Probability density visualization
float probabilityDensity(vec2 pos) {
  float density = 0.0;
  
  for (int i = 0; i < 8; i++) {
    if (i >= int(uSuperpositionCount)) break;
    
    vec2 diff = pos - uStatePositions[i].xy;
    float dist = length(diff);
    float gaussian = exp(-dist * dist * 2.0) * uStateProbabilities[i];
    density += gaussian;
  }
  
  return density;
}

// Waveform collapse effect
vec3 collapseEffect(vec2 pos, vec3 baseColor) {
  if (!uCollapsing) return baseColor;
  
  // Create expanding rings from collapse center
  float dist = length(pos);
  float ring = sin(dist * 20.0 - uCollapseProgress * 50.0);
  ring *= exp(-uCollapseProgress * 3.0); // Fade out over time
  
  // Decoherence noise
  float noise = fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
  noise *= (1.0 - uCoherence) * uCollapseProgress;
  
  vec3 collapseColor = vec3(1.0, 0.3, 0.3); // Red for collapse
  return mix(baseColor, collapseColor, ring * 0.5 + noise * 0.3);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 pos = (uv - 0.5) * 10.0; // Scale to world coordinates
  
  // Calculate quantum interference
  float interference = quantumInterference(pos);
  
  // Calculate probability density
  float density = probabilityDensity(pos);
  
  // Base quantum field color
  vec3 fieldColor = vec3(0.0, 0.4, 0.8); // Blue base
  
  // Add interference patterns
  fieldColor += vec3(0.3, 0.6, 1.0) * interference * 0.5;
  
  // Add probability density glow
  fieldColor += vec3(0.8, 0.9, 1.0) * density * 0.3;
  
  // Coherence effects
  float coherenceGlow = uCoherence * 0.2;
  fieldColor += vec3(coherenceGlow, coherenceGlow * 0.8, coherenceGlow);
  
  // Apply collapse effects
  fieldColor = collapseEffect(pos, fieldColor);
  
  // Final alpha based on field intensity
  float alpha = (interference * 0.3 + density * 0.4 + coherenceGlow) * 0.6;
  alpha = clamp(alpha, 0.0, 0.8);
  
  gl_FragColor = vec4(fieldColor, alpha);
}
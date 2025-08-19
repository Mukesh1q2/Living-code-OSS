// Quantum Particle System Fragment Shader
// Renders individual quantum energy particles with glow effects

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

// Particle shape functions
float circleShape(vec2 coord) {
  return 1.0 - smoothstep(0.3, 0.5, length(coord));
}

float starShape(vec2 coord, float points) {
  float angle = atan(coord.y, coord.x);
  float radius = length(coord);
  
  float starPattern = sin(angle * points) * 0.3 + 0.7;
  return (1.0 - smoothstep(0.2, 0.5, radius)) * starPattern;
}

float quantumShape(vec2 coord) {
  float radius = length(coord);
  
  // Quantum probability cloud shape
  float core = exp(-radius * radius * 8.0);
  float halo = exp(-radius * radius * 2.0) * 0.3;
  
  // Add quantum fluctuations
  float angle = atan(coord.y, coord.x);
  float fluctuation = sin(angle * 4.0 + uTime * 3.0 + vParticleId) * 0.1 + 0.9;
  
  return (core + halo) * fluctuation;
}

// Energy-based color calculation
vec3 energyColor(float energy, float distanceToCenter) {
  // Base color from energy level
  vec3 lowEnergyColor = uQuantumColors[0];   // Blue
  vec3 medEnergyColor = uQuantumColors[1];   // Cyan
  vec3 highEnergyColor = uQuantumColors[2];  // White/Yellow
  vec3 maxEnergyColor = uQuantumColors[3];   // Gold/Orange
  
  vec3 color;
  if (energy < 0.33) {
    color = mix(lowEnergyColor, medEnergyColor, energy * 3.0);
  } else if (energy < 0.66) {
    color = mix(medEnergyColor, highEnergyColor, (energy - 0.33) * 3.0);
  } else {
    color = mix(highEnergyColor, maxEnergyColor, (energy - 0.66) * 3.0);
  }
  
  // Distance-based color shift
  float distanceFactor = exp(-distanceToCenter * 0.1);
  color = mix(color * 0.5, color, distanceFactor);
  
  return color;
}

// Velocity-based effects
vec3 velocityEffects(vec3 baseColor, vec3 velocity) {
  float speed = length(velocity);
  
  // Motion blur effect (simulated)
  float motionIntensity = min(speed * 0.5, 1.0);
  vec3 motionColor = mix(baseColor, vec3(1.0, 0.8, 0.6), motionIntensity * 0.3);
  
  return motionColor;
}

// Lifespan effects
float lifespanAlpha(float lifespan) {
  // Fade in and out based on particle lifespan
  float fadeIn = smoothstep(0.0, 0.1, lifespan);
  float fadeOut = smoothstep(1.0, 0.8, lifespan);
  
  return fadeIn * fadeOut;
}

// Collapse effects
vec3 collapseEffects(vec3 baseColor, float progress) {
  if (!uCollapsing || progress <= 0.0) return baseColor;
  
  // Color shift during collapse
  vec3 collapseColor = vec3(1.0, 0.3, 0.3); // Red
  float collapseIntensity = progress * progress;
  
  // Flickering effect
  float flicker = sin(uTime * 20.0 + vParticleId * 10.0) * 0.5 + 0.5;
  flicker = mix(1.0, flicker, progress);
  
  vec3 finalColor = mix(baseColor, collapseColor, collapseIntensity * 0.7);
  return finalColor * flicker;
}

void main() {
  vec2 coord = gl_PointCoord - 0.5;
  
  // Choose particle shape based on performance level
  float shape;
  if (uPerformanceLevel < 0.3) {
    shape = circleShape(coord);
  } else if (uPerformanceLevel < 0.7) {
    // Mix between circle and star based on energy
    float starAmount = smoothstep(0.5, 1.0, vEnergy);
    shape = mix(circleShape(coord), starShape(coord, 5.0), starAmount);
  } else {
    // Full quantum shape
    shape = quantumShape(coord);
  }
  
  if (shape < 0.01) discard;
  
  // Calculate base color from energy and distance
  vec3 color = energyColor(vEnergy, vDistanceToCenter);
  
  // Apply velocity effects (medium+ quality)
  if (uPerformanceLevel >= 0.5) {
    color = velocityEffects(color, vVelocity);
  }
  
  // Apply collapse effects
  color = collapseEffects(color, uCollapseProgress);
  
  // Calculate alpha
  float alpha = shape * lifespanAlpha(vLifespan);
  
  // Energy-based intensity
  alpha *= mix(0.3, 1.0, vEnergy);
  
  // Quantum energy modulation
  alpha *= mix(0.5, 1.0, uQuantumEnergy);
  
  // Distance-based fade
  float distanceFade = exp(-vDistanceToCenter * 0.05);
  alpha *= mix(0.2, 1.0, distanceFade);
  
  gl_FragColor = vec4(color, alpha);
}
// Quantum Particle System Vertex Shader
// Handles individual quantum particles with energy flow dynamics

uniform float uTime;
uniform float uQuantumEnergy;
uniform float uFieldStrength;
uniform vec3 uEnergyCenter;
uniform float uParticleSize;
uniform float uFlowSpeed;
uniform bool uCollapsing;
uniform float uCollapseProgress;
uniform float uPerformanceLevel;

// Particle attributes
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

// Quantum particle physics
vec3 quantumForces(vec3 pos, vec3 vel, float id) {
  vec3 forces = vec3(0.0);
  
  // Attraction to energy center
  vec3 toCenter = uEnergyCenter - pos;
  float distToCenter = length(toCenter);
  vec3 centerForce = normalize(toCenter) * uQuantumEnergy * 0.1 / (distToCenter + 0.1);
  forces += centerForce;
  
  // Quantum orbital motion
  vec3 tangent = cross(toCenter, vec3(0.0, 1.0, 0.0));
  if (length(tangent) > 0.0) {
    tangent = normalize(tangent);
    float orbitalStrength = uFieldStrength * 0.05;
    forces += tangent * orbitalStrength;
  }
  
  // Particle-specific quantum fluctuations
  float phase = id * 6.28318 + uTime * 2.0;
  vec3 fluctuation = vec3(
    sin(phase) * 0.02,
    cos(phase * 1.3) * 0.02,
    sin(phase * 0.7) * 0.02
  );
  forces += fluctuation * uQuantumEnergy;
  
  // Velocity damping
  forces -= vel * 0.1;
  
  return forces;
}

// Collapse behavior
vec3 collapseForces(vec3 pos, float progress) {
  if (!uCollapsing || progress <= 0.0) return vec3(0.0);
  
  vec3 toCenter = uEnergyCenter - pos;
  float distToCenter = length(toCenter);
  
  // Strong attraction during collapse
  vec3 collapseForce = normalize(toCenter) * progress * progress * 2.0;
  
  // Spiral motion during collapse
  vec3 tangent = cross(toCenter, vec3(0.0, 1.0, 0.0));
  if (length(tangent) > 0.0) {
    tangent = normalize(tangent);
    float spiralStrength = progress * 0.5;
    collapseForce += tangent * spiralStrength;
  }
  
  return collapseForce;
}

void main() {
  vParticleId = particleId;
  vLifespan = lifespan;
  vEnergy = energy;
  vVelocity = velocity;
  
  // Calculate current particle position with physics
  vec3 currentPos = position;
  
  if (uPerformanceLevel >= 0.5) {
    // Apply quantum forces (medium+ quality)
    vec3 forces = quantumForces(currentPos, velocity, particleId);
    forces += collapseForces(currentPos, uCollapseProgress);
    
    // Simple Euler integration
    vec3 newVelocity = velocity + forces * 0.016; // Assume ~60fps
    currentPos += newVelocity * uFlowSpeed * 0.016;
  } else {
    // Simple movement for low-end devices
    vec3 toCenter = normalize(uEnergyCenter - currentPos);
    currentPos += toCenter * uFlowSpeed * 0.01;
  }
  
  vDistanceToCenter = length(currentPos - uEnergyCenter);
  
  // Calculate particle size based on energy and distance
  float sizeMultiplier = mix(0.5, 2.0, energy);
  sizeMultiplier *= mix(1.0, 0.3, vDistanceToCenter * 0.1);
  
  // Collapse size effect
  if (uCollapsing) {
    sizeMultiplier *= mix(1.0, 0.1, uCollapseProgress);
  }
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(currentPos, 1.0);
  gl_PointSize = uParticleSize * sizeMultiplier;
}
uniform float uTime;
uniform float uCoherence;
uniform bool uCollapsing;
uniform float uCollapseProgress;

attribute vec3 position;
attribute vec2 uv;

varying vec2 vUv;
varying vec3 vPosition;

// Quantum field distortion
vec3 quantumDistortion(vec3 pos) {
  vec3 distorted = pos;
  
  // Coherence-based wave distortion
  float wave1 = sin(pos.x * 2.0 + uTime * 1.5) * (1.0 - uCoherence) * 0.1;
  float wave2 = cos(pos.z * 1.5 + uTime * 1.2) * (1.0 - uCoherence) * 0.1;
  
  distorted.y += wave1 + wave2;
  
  // Collapse distortion
  if (uCollapsing) {
    float collapseWave = sin(length(pos.xz) * 5.0 - uTime * 10.0) * uCollapseProgress * 0.2;
    distorted.y += collapseWave;
  }
  
  return distorted;
}

void main() {
  vUv = uv;
  
  // Apply quantum distortions
  vec3 distortedPosition = quantumDistortion(position);
  vPosition = distortedPosition;
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(distortedPosition, 1.0);
}
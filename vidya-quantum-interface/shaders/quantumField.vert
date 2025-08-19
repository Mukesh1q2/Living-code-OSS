uniform float uTime;
attribute vec3 position;

void main() {
  vec3 pos = position;
  pos.z += sin(pos.x * 0.3 + uTime * 0.6) * 0.15;
  pos.y += cos(pos.z * 0.25 + uTime * 0.5) * 0.15;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
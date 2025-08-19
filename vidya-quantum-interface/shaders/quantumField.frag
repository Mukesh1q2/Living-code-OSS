precision highp float;
uniform float uTime;
uniform vec2 uResolution;

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 p = uv - 0.5;
  float r = length(p);
  float angle = atan(p.y, p.x);
  
  float waves = sin(10.0 * r - uTime * 1.2) + sin(8.0 * angle + uTime * 0.9);
  float glow = smoothstep(0.35, 0.0, r) * 0.6 + 0.4 * smoothstep(0.9, 0.2, abs(waves));
  
  vec3 color = mix(vec3(0.03, 0.08, 0.18), vec3(0.0, 0.6, 1.0), glow);
  color += 0.2 * vec3(0.8, 0.3, 1.0) * (0.5 + 0.5 * sin(uTime * 0.7 + r * 10.0));
  
  gl_FragColor = vec4(color, 1.0);
}
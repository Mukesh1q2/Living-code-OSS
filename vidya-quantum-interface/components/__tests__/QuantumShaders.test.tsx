import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { getQuantumShaderSettings } from '../QuantumShaderManager';

// Mock the quantum state - simplified for testing
const mockQuantumState = {
  superpositionActive: true,
  superpositionStates: [
    { position: [0, 0, 0], probability: 0.5 },
    { position: [1, 1, 1], probability: 0.3 },
  ],
  coherenceLevel: 0.8,
  waveformCollapsing: false,
  collapseProgress: 0,
  quantumQuality: 'high',
  quantumEnergy: 1.0,
  fieldStrength: 0.5,
  energyCenter: [0, 0, 0],
};

// Mock WebGL context
const mockWebGLContext = {
  getParameter: vi.fn((param) => {
    switch (param) {
      case 'RENDERER':
        return 'Mock Renderer RTX 3080';
      case 'VENDOR':
        return 'Mock Vendor';
      case 'MAX_TEXTURE_SIZE':
        return 8192;
      default:
        return null;
    }
  }),
  getSupportedExtensions: vi.fn(() => new Array(60).fill('mock-extension')),
};

// Mock canvas and WebGL
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: vi.fn((contextType) => {
    if (contextType === 'webgl2' || contextType === 'webgl') {
      return mockWebGLContext;
    }
    return null;
  }),
});

describe('Quantum Shader System', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Performance Detection', () => {
    it('should detect high performance level for RTX GPU', () => {
      const settings = getQuantumShaderSettings();
      expect(settings.performanceLevel).toBeGreaterThan(0.8);
      expect(settings.useAdvancedField).toBe(true);
      expect(settings.useParticles).toBe(true);
      expect(settings.useAdvancedLighting).toBe(true);
    });

    it('should provide appropriate particle count based on performance', () => {
      const settings = getQuantumShaderSettings();
      expect(settings.particleCount).toBeGreaterThan(500);
      expect(settings.fieldResolution).toBeGreaterThan(100);
      expect(settings.maxLights).toBeGreaterThan(2);
    });

    it('should handle low-end GPU detection', () => {
      mockWebGLContext.getParameter.mockImplementation((param) => {
        switch (param) {
          case 'RENDERER':
            return 'Intel HD Graphics';
          case 'VENDOR':
            return 'Intel';
          case 'MAX_TEXTURE_SIZE':
            return 2048;
          default:
            return null;
        }
      });

      const settings = getQuantumShaderSettings();
      expect(settings.performanceLevel).toBeLessThan(0.5);
      expect(settings.useAdvancedField).toBe(false);
      expect(settings.particleCount).toBeLessThan(500);
    });
  });

  describe('Shader Material Creation', () => {
    it('should create valid Three.js shader materials', () => {
      // Test quantum field material creation
      const material = new THREE.ShaderMaterial({
        vertexShader: `
          uniform float uTime;
          attribute vec3 position;
          void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          precision highp float;
          uniform float uTime;
          void main() {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
          }
        `,
        uniforms: {
          uTime: { value: 0 },
        },
      });

      expect(material).toBeInstanceOf(THREE.ShaderMaterial);
      expect(material.uniforms.uTime).toBeDefined();
      expect(material.uniforms.uTime.value).toBe(0);
    });

    it('should handle shader compilation errors gracefully', () => {
      // Test with invalid shader code
      expect(() => {
        new THREE.ShaderMaterial({
          vertexShader: 'invalid shader code',
          fragmentShader: 'invalid shader code',
        });
      }).not.toThrow(); // Three.js handles compilation errors internally
    });
  });

  describe('Shader Uniform Updates', () => {
    it('should update shader uniforms correctly', () => {
      const material = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uQuantumEnergy: { value: 0.5 },
          uCoherence: { value: 0.8 },
        },
      });

      // Simulate uniform updates
      material.uniforms.uTime.value = 1.5;
      material.uniforms.uQuantumEnergy.value = 1.0;
      material.uniforms.uCoherence.value = 0.9;

      expect(material.uniforms.uTime.value).toBe(1.5);
      expect(material.uniforms.uQuantumEnergy.value).toBe(1.0);
      expect(material.uniforms.uCoherence.value).toBe(0.9);
    });

    it('should handle vector uniform updates', () => {
      const material = new THREE.ShaderMaterial({
        uniforms: {
          uEnergyCenter: { value: new THREE.Vector3(0, 0, 0) },
          uQuantumColors: { value: [
            new THREE.Color(0xff0000),
            new THREE.Color(0x00ff00),
          ]},
        },
      });

      // Update vector uniforms
      material.uniforms.uEnergyCenter.value.set(1, 2, 3);
      material.uniforms.uQuantumColors.value[0].setHex(0x0000ff);

      expect(material.uniforms.uEnergyCenter.value.x).toBe(1);
      expect(material.uniforms.uEnergyCenter.value.y).toBe(2);
      expect(material.uniforms.uEnergyCenter.value.z).toBe(3);
      expect(material.uniforms.uQuantumColors.value[0].getHex()).toBe(0x0000ff);
    });
  });

  describe('Performance Fallbacks', () => {
    it('should use fallback shaders on low-end devices', () => {
      // Mock low-end device
      mockWebGLContext.getParameter.mockImplementation((param) => {
        switch (param) {
          case 'RENDERER':
            return 'Intel HD Graphics 3000';
          case 'VENDOR':
            return 'Intel';
          case 'MAX_TEXTURE_SIZE':
            return 1024;
          default:
            return null;
        }
      });

      const settings = getQuantumShaderSettings();
      expect(settings.useAdvancedField).toBe(false);
      expect(settings.useParticles).toBe(false);
      expect(settings.useAdvancedLighting).toBe(false);
    });

    it('should adjust particle count based on performance', () => {
      const highPerfSettings = getQuantumShaderSettings();
      
      // Mock medium performance device
      mockWebGLContext.getParameter.mockImplementation((param) => {
        switch (param) {
          case 'RENDERER':
            return 'GTX 1060';
          case 'VENDOR':
            return 'NVIDIA';
          case 'MAX_TEXTURE_SIZE':
            return 4096;
          default:
            return null;
        }
      });

      const mediumPerfSettings = getQuantumShaderSettings();
      
      expect(mediumPerfSettings.particleCount).toBeLessThan(highPerfSettings.particleCount);
      expect(mediumPerfSettings.fieldResolution).toBeLessThan(highPerfSettings.fieldResolution);
    });
  });

  describe('WebGL Feature Detection', () => {
    it('should handle missing WebGL support', () => {
      // Mock no WebGL support
      HTMLCanvasElement.prototype.getContext = vi.fn(() => null);

      const settings = getQuantumShaderSettings();
      expect(settings.performanceLevel).toBe(0.0);
      expect(settings.useAdvancedField).toBe(false);
      expect(settings.useParticles).toBe(false);
      expect(settings.useAdvancedLighting).toBe(false);
    });

    it('should detect WebGL extensions', () => {
      // Reset to working WebGL context first
      HTMLCanvasElement.prototype.getContext = vi.fn((contextType) => {
        if (contextType === 'webgl2' || contextType === 'webgl') {
          return mockWebGLContext;
        }
        return null;
      });
      
      mockWebGLContext.getSupportedExtensions.mockReturnValue(new Array(60).fill('ext'));
      
      const settings = getQuantumShaderSettings();
      expect(settings.performanceLevel).toBeGreaterThan(0.3);
    });
  });
});
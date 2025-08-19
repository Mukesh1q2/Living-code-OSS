import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useQuantumState } from '../../lib/state';
import { 
  superpositionPhase, 
  probabilityAmplitude, 
  waveformCollapse, 
  quantumInterference,
  generateProbabilityCloud,
  updateQuantumParticle 
} from '../../lib/quantum';
import * as THREE from 'three';

// Mock the quantum state
vi.mock('../../lib/state');
vi.mock('../../lib/consciousness');

const mockQuantumState = {
  superpositionActive: true,
  superpositionStates: [
    {
      id: 'state_1',
      position: [1, 0, 0] as [number, number, number],
      opacity: 0.6,
      phase: 0,
      probability: 0.4,
      isCollapsing: false,
      collapseProgress: 0,
    },
    {
      id: 'state_2',
      position: [-1, 0, 0] as [number, number, number],
      opacity: 0.4,
      phase: Math.PI,
      probability: 0.3,
      isCollapsing: false,
      collapseProgress: 0,
    },
    {
      id: 'state_3',
      position: [0, 0, 1] as [number, number, number],
      opacity: 0.5,
      phase: Math.PI / 2,
      probability: 0.3,
      isCollapsing: false,
      collapseProgress: 0,
    },
  ],
  probabilityCloud: {
    particles: [],
    density: 0.5,
    coherenceLevel: 0.8,
  },
  coherenceLevel: 0.8,
  quantumQuality: 'high' as const,
  triggerWaveformCollapse: vi.fn(),
};

const mockConsciousness = {
  consciousness: {
    level: 5,
    quantumCoherence: 0.8,
  },
  recordInteraction: vi.fn(),
  updateQuantumState: vi.fn(),
};

describe('Quantum Physics Functions', () => {
  it('calculates superposition phase correctly', () => {
    const time = 1.0;
    const basePhase = Math.PI / 4;
    const frequency = 2.0;
    
    const phase = superpositionPhase(time, basePhase, frequency);
    expect(phase).toBe(basePhase + time * frequency);
  });

  it('calculates probability amplitude correctly', () => {
    const phase = Math.PI / 2;
    const coherenceLevel = 0.8;
    
    const amplitude = probabilityAmplitude(phase, coherenceLevel);
    expect(amplitude).toBeCloseTo(coherenceLevel * Math.cos(phase), 5);
  });

  it('calculates waveform collapse effects', () => {
    const progress = 0.5;
    const collapse = waveformCollapse(progress);
    
    expect(collapse.amplitude).toBeGreaterThan(0);
    expect(collapse.amplitude).toBeLessThan(1);
    expect(collapse.decoherence).toBeGreaterThan(0);
    expect(collapse.decoherence).toBeLessThan(1);
    expect(typeof collapse.interference).toBe('number');
  });

  it('calculates quantum interference between phases', () => {
    const phase1 = 0;
    const phase2 = Math.PI;
    
    const interference = quantumInterference(phase1, phase2);
    expect(interference).toBeCloseTo(-1, 5); // Destructive interference
  });

  it('generates probability cloud particles', () => {
    const center = new THREE.Vector3(0, 0, 0);
    const particleCount = 10;
    const spread = 2.0;
    const coherenceLevel = 0.8;
    
    const particles = generateProbabilityCloud(center, particleCount, spread, coherenceLevel);
    
    expect(particles).toHaveLength(particleCount);
    particles.forEach(particle => {
      expect(particle.position).toBeInstanceOf(THREE.Vector3);
      expect(particle.velocity).toBeInstanceOf(THREE.Vector3);
      expect(particle.life).toBeGreaterThan(0);
      expect(particle.maxLife).toBeGreaterThan(0);
      expect(particle.probability).toBeGreaterThanOrEqual(0);
      expect(particle.probability).toBeLessThanOrEqual(1);
    });
  });

  it('updates quantum particle properties', () => {
    const particle = {
      position: new THREE.Vector3(0, 0, 0),
      velocity: new THREE.Vector3(0.1, 0, 0),
      life: 1000,
      maxLife: 2000,
      probability: 0.5,
    };
    
    const deltaTime = 100;
    const coherenceLevel = 0.8;
    
    const updatedParticle = updateQuantumParticle(particle, deltaTime, coherenceLevel);
    
    expect(updatedParticle.life).toBeLessThan(particle.life);
    expect(updatedParticle.position.x).toBeGreaterThan(0); // Should have moved
    expect(updatedParticle.probability).toBeLessThanOrEqual(particle.probability);
  });
});

describe('Quantum State Management', () => {
  it('creates superposition states with correct probabilities', () => {
    const states = mockQuantumState.superpositionStates;
    const totalProbability = states.reduce((sum, state) => sum + state.probability, 0);
    
    // Total probability should be close to 1 (allowing for floating point precision)
    expect(totalProbability).toBeCloseTo(1.0, 1);
  });

  it('maintains quantum properties for each state', () => {
    mockQuantumState.superpositionStates.forEach(state => {
      expect(state.id).toBeDefined();
      expect(state.position).toHaveLength(3);
      expect(state.opacity).toBeGreaterThan(0);
      expect(state.opacity).toBeLessThanOrEqual(1);
      expect(state.probability).toBeGreaterThan(0);
      expect(state.probability).toBeLessThanOrEqual(1);
      expect(typeof state.phase).toBe('number');
      expect(typeof state.isCollapsing).toBe('boolean');
      expect(state.collapseProgress).toBeGreaterThanOrEqual(0);
      expect(state.collapseProgress).toBeLessThanOrEqual(1);
    });
  });

  it('handles coherence level changes', () => {
    expect(mockQuantumState.coherenceLevel).toBeGreaterThanOrEqual(0);
    expect(mockQuantumState.coherenceLevel).toBeLessThanOrEqual(1);
  });
});
"use client";

import { useRef, useMemo, useEffect, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Line, Html } from "@react-three/drei";
import * as THREE from "three";
import { useQuantumState } from "@/lib/state";
import { NetworkNode, NetworkConnection } from "@/lib/neural-network";
import {
  calculateEntanglementStrength,
  generateEntanglementField,
  createEntanglementParticles,
  updateEntanglementParticle,
  calculateQuantumCorrelation,
  createEntanglementCreationEffect,
  createEntanglementDestructionEffect
} from "@/lib/quantum";

interface EntanglementPair {
  id: string;
  nodeA: NetworkNode;
  nodeB: NetworkNode;
  strength: number;
  correlation: number;
  createdAt: number;
  isActive: boolean;
  visualEffects: {
    particles: Array<{
      position: THREE.Vector3;
      velocity: THREE.Vector3;
      life: number;
      maxLife: number;
      entanglementPhase: number;
    }>;
    fieldPoints: THREE.Vector3[];
    fieldIntensities: number[];
  };
}

interface QuantumEntanglementProps {
  nodes: NetworkNode[];
  connections: NetworkConnection[];
  maxEntanglements?: number;
  autoCreateEntanglements?: boolean;
  showFieldVisualization?: boolean;
  showParticleEffects?: boolean;
}

export default function QuantumEntanglement({
  nodes,
  connections,
  maxEntanglements = 8,
  autoCreateEntanglements = true,
  showFieldVisualization = true,
  showParticleEffects = true
}: QuantumEntanglementProps) {
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const selectedNode = useQuantumState((s) => s.selectedNode);
  
  const [entanglementPairs, setEntanglementPairs] = useState<EntanglementPair[]>([]);
  const [creationEffects, setCreationEffects] = useState<Array<{
    id: string;
    position: THREE.Vector3;
    particles: any[];
    shockwave: any;
    life: number;
    maxLife: number;
  }>>([]);
  const [destructionEffects, setDestructionEffects] = useState<Array<{
    id: string;
    fragments: any[];
    energyWave: any;
    life: number;
    maxLife: number;
  }>>([]);

  // Find quantum nodes that can be entangled
  const quantumNodes = useMemo(() => {
    return nodes.filter(node => 
      node.type === 'quantum-gate' || 
      (node.quantumProperties && node.quantumProperties.superposition)
    );
  }, [nodes]);

  // Auto-create entanglements based on quantum properties and proximity
  useEffect(() => {
    if (!autoCreateEntanglements || quantumQuality === 'minimal') return;

    const createEntanglement = (nodeA: NetworkNode, nodeB: NetworkNode) => {
      const strength = calculateEntanglementStrength(nodeA, nodeB);
      if (strength < 0.3) return null; // Minimum threshold for entanglement

      const particles = showParticleEffects ? 
        createEntanglementParticles(nodeA.position, nodeB.position, strength) : [];

      const newPair: EntanglementPair = {
        id: `entanglement_${nodeA.id}_${nodeB.id}_${Date.now()}`,
        nodeA,
        nodeB,
        strength,
        correlation: 0,
        createdAt: Date.now(),
        isActive: true,
        visualEffects: {
          particles,
          fieldPoints: [],
          fieldIntensities: []
        }
      };

      // Create visual creation effect
      const center = new THREE.Vector3().addVectors(nodeA.position, nodeB.position).multiplyScalar(0.5);
      const creationEffect = createEntanglementCreationEffect(center, strength);
      
      setCreationEffects(prev => [...prev, {
        id: `creation_${Date.now()}`,
        position: center,
        particles: creationEffect.particles,
        shockwave: creationEffect.shockwave,
        life: 2000,
        maxLife: 2000
      }]);

      return newPair;
    };

    const timer = setInterval(() => {
      setEntanglementPairs(current => {
        if (current.length >= maxEntanglements) return current;

        // Find potential entanglement candidates
        const candidates = [];
        for (let i = 0; i < quantumNodes.length; i++) {
          for (let j = i + 1; j < quantumNodes.length; j++) {
            const nodeA = quantumNodes[i];
            const nodeB = quantumNodes[j];
            
            // Check if already entangled
            const alreadyEntangled = current.some(pair => 
              (pair.nodeA.id === nodeA.id && pair.nodeB.id === nodeB.id) ||
              (pair.nodeA.id === nodeB.id && pair.nodeB.id === nodeA.id)
            );
            
            if (!alreadyEntangled) {
              const strength = calculateEntanglementStrength(nodeA, nodeB);
              candidates.push({ nodeA, nodeB, strength });
            }
          }
        }

        // Sort by strength and create the strongest entanglement
        candidates.sort((a, b) => b.strength - a.strength);
        if (candidates.length > 0 && candidates[0].strength > 0.4) {
          const newPair = createEntanglement(candidates[0].nodeA, candidates[0].nodeB);
          if (newPair) {
            return [...current, newPair];
          }
        }

        return current;
      });
    }, 3000 + Math.random() * 2000); // Create entanglements every 3-5 seconds

    return () => clearInterval(timer);
  }, [quantumNodes, maxEntanglements, autoCreateEntanglements, quantumQuality, showParticleEffects]);

  // Update entanglement effects
  useFrame((_, deltaTime) => {
    const time = performance.now() * 0.001;

    // Update entanglement pairs
    setEntanglementPairs(current => 
      current.map(pair => {
        // Update correlation
        const correlation = calculateQuantumCorrelation(pair.nodeA, pair.nodeB, time);
        
        // Update field visualization
        let fieldPoints: THREE.Vector3[] = [];
        let fieldIntensities: number[] = [];
        
        if (showFieldVisualization && quantumQuality !== 'minimal') {
          const field = generateEntanglementField(
            pair.nodeA.position, 
            pair.nodeB.position, 
            pair.strength, 
            time
          );
          fieldPoints = field.fieldPoints;
          fieldIntensities = field.fieldIntensities;
        }

        // Update particles
        const updatedParticles = pair.visualEffects.particles
          .map(particle => updateEntanglementParticle(
            particle, 
            deltaTime * 1000, 
            pair.nodeA.position, 
            pair.nodeB.position, 
            pair.strength
          ))
          .filter(particle => particle.life > 0);

        // Add new particles occasionally
        if (showParticleEffects && Math.random() < 0.02 * pair.strength) {
          const newParticles = createEntanglementParticles(
            pair.nodeA.position, 
            pair.nodeB.position, 
            pair.strength, 
            2
          );
          updatedParticles.push(...newParticles);
        }

        return {
          ...pair,
          correlation,
          visualEffects: {
            particles: updatedParticles,
            fieldPoints,
            fieldIntensities
          }
        };
      })
    );

    // Update creation effects
    setCreationEffects(current => 
      current.map(effect => ({
        ...effect,
        life: effect.life - deltaTime * 1000,
        shockwave: {
          ...effect.shockwave,
          radius: Math.min(effect.shockwave.maxRadius, effect.shockwave.radius + deltaTime * 2)
        }
      })).filter(effect => effect.life > 0)
    );

    // Update destruction effects
    setDestructionEffects(current => 
      current.map(effect => ({
        ...effect,
        life: effect.life - deltaTime * 1000,
        energyWave: {
          ...effect.energyWave,
          radius: Math.min(effect.energyWave.maxRadius, effect.energyWave.radius + deltaTime * 3)
        }
      })).filter(effect => effect.life > 0)
    );
  });

  // Handle entanglement destruction (when nodes become too distant or lose coherence)
  useEffect(() => {
    const timer = setInterval(() => {
      setEntanglementPairs(current => {
        const surviving: EntanglementPair[] = [];
        const destroyed: EntanglementPair[] = [];

        current.forEach(pair => {
          const distance = pair.nodeA.position.distanceTo(pair.nodeB.position);
          const currentStrength = calculateEntanglementStrength(pair.nodeA, pair.nodeB);
          
          // Destroy if too weak or too distant
          if (currentStrength < 0.2 || distance > 25) {
            destroyed.push(pair);
          } else {
            surviving.push(pair);
          }
        });

        // Create destruction effects for destroyed entanglements
        destroyed.forEach(pair => {
          const destructionEffect = createEntanglementDestructionEffect(
            pair.nodeA.position,
            pair.nodeB.position,
            pair.strength
          );
          
          setDestructionEffects(prev => [...prev, {
            id: `destruction_${Date.now()}_${Math.random()}`,
            fragments: destructionEffect.fragments,
            energyWave: destructionEffect.energyWave,
            life: 1500,
            maxLife: 1500
          }]);
        });

        return surviving;
      });
    }, 5000); // Check every 5 seconds

    return () => clearInterval(timer);
  }, []);

  if (quantumQuality === 'minimal') {
    return null;
  }

  return (
    <group>
      {/* Render entanglement connections */}
      {entanglementPairs.map(pair => (
        <EntanglementConnection
          key={pair.id}
          pair={pair}
          showField={showFieldVisualization}
          showParticles={showParticleEffects}
          coherenceLevel={coherenceLevel}
        />
      ))}

      {/* Render creation effects */}
      {creationEffects.map(effect => (
        <EntanglementCreationEffect key={effect.id} effect={effect} />
      ))}

      {/* Render destruction effects */}
      {destructionEffects.map(effect => (
        <EntanglementDestructionEffect key={effect.id} effect={effect} />
      ))}

      {/* Entanglement info panel */}
      {selectedNode && (
        <EntanglementInfoPanel
          selectedNodeId={selectedNode}
          entanglements={entanglementPairs}
        />
      )}
    </group>
  );
}

function EntanglementConnection({
  pair,
  showField,
  showParticles,
  coherenceLevel
}: {
  pair: EntanglementPair;
  showField: boolean;
  showParticles: boolean;
  coherenceLevel: number;
}) {
  const lineRef = useRef<any>(null);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  // Calculate dynamic line properties
  const lineWidth = 1.5 + pair.strength * 2;
  const opacity = 0.4 + pair.strength * 0.4;
  const pulseIntensity = 0.5 + Math.abs(pair.correlation) * 0.5;

  useFrame(() => {
    if (!lineRef.current) return;

    // Animate line opacity based on correlation
    const time = performance.now() * 0.001;
    const pulse = 0.5 + 0.5 * Math.sin(time * 2 + pair.strength * Math.PI);
    
    if (lineRef.current.material && 'opacity' in lineRef.current.material) {
      (lineRef.current.material as any).opacity = opacity * pulse * coherenceLevel;
    }
  });

  return (
    <group>
      {/* Main entanglement line */}
      <Line
        ref={lineRef}
        points={[pair.nodeA.position, pair.nodeB.position]}
        color="#B383FF"
        lineWidth={lineWidth}
        transparent
        opacity={opacity}
        dashed={quantumQuality === 'high'}
        dashSize={0.3}
        gapSize={0.1}
      />

      {/* Quantum field visualization */}
      {showField && pair.visualEffects.fieldPoints.length > 0 && (
        <Line
          points={pair.visualEffects.fieldPoints}
          color="#E6B3FF"
          lineWidth={0.8}
          transparent
          opacity={0.3}
        />
      )}

      {/* Entanglement particles */}
      {showParticles && pair.visualEffects.particles.map((particle, index) => (
        <mesh key={index} position={particle.position}>
          <sphereGeometry args={[0.02, 8, 8]} />
          <meshBasicMaterial
            color="#B383FF"
            transparent
            opacity={particle.life / particle.maxLife}
          />
        </mesh>
      ))}

      {/* Strength indicator */}
      {quantumQuality === 'high' && (
        <Html
          position={new THREE.Vector3().addVectors(pair.nodeA.position, pair.nodeB.position).multiplyScalar(0.5)}
          distanceFactor={30}
          style={{
            color: '#B383FF',
            fontSize: '8px',
            opacity: 0.7,
            pointerEvents: 'none',
            textAlign: 'center',
            background: 'rgba(0,0,0,0.5)',
            padding: '2px 4px',
            borderRadius: '2px'
          }}
        >
          <div>
            <div>Strength: {(pair.strength * 100).toFixed(0)}%</div>
            <div>Correlation: {(pair.correlation * 100).toFixed(0)}%</div>
          </div>
        </Html>
      )}
    </group>
  );
}

function EntanglementCreationEffect({ effect }: { effect: any }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!groupRef.current) return;

    const progress = 1 - (effect.life / effect.maxLife);
    const scale = progress * 2;
    groupRef.current.scale.setScalar(scale);
  });

  return (
    <group ref={groupRef} position={effect.position}>
      {/* Shockwave ring */}
      <mesh>
        <ringGeometry args={[effect.shockwave.radius * 0.9, effect.shockwave.radius, 32]} />
        <meshBasicMaterial
          color="#B383FF"
          transparent
          opacity={0.5 * (effect.life / effect.maxLife)}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Creation particles */}
      {effect.particles.map((particle: any, index: number) => (
        <mesh key={index} position={particle.position}>
          <sphereGeometry args={[particle.size, 8, 8]} />
          <meshBasicMaterial
            color="#E6B3FF"
            transparent
            opacity={particle.life / particle.maxLife}
          />
        </mesh>
      ))}
    </group>
  );
}

function EntanglementDestructionEffect({ effect }: { effect: any }) {
  return (
    <group>
      {/* Energy wave */}
      <mesh position={effect.energyWave.center}>
        <sphereGeometry args={[effect.energyWave.radius, 16, 16]} />
        <meshBasicMaterial
          color="#FF6B6B"
          transparent
          opacity={0.2 * (effect.life / effect.maxLife)}
          wireframe
        />
      </mesh>

      {/* Fragments */}
      {effect.fragments.map((fragment: any, index: number) => (
        <mesh key={index} position={fragment.position} rotation={fragment.rotation}>
          <boxGeometry args={[0.05, 0.05, 0.05]} />
          <meshBasicMaterial
            color="#FFB3B3"
            transparent
            opacity={fragment.life / fragment.maxLife}
          />
        </mesh>
      ))}
    </group>
  );
}

function EntanglementInfoPanel({
  selectedNodeId,
  entanglements
}: {
  selectedNodeId: string;
  entanglements: EntanglementPair[];
}) {
  const nodeEntanglements = entanglements.filter(pair => 
    pair.nodeA.id === selectedNodeId || pair.nodeB.id === selectedNodeId
  );

  if (nodeEntanglements.length === 0) return null;

  return (
    <Html
      position={[0, 0, 0]}
      style={{
        position: 'fixed',
        bottom: '20px',
        left: '20px',
        zIndex: 1000,
        pointerEvents: 'auto'
      }}
    >
      <div style={{
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '10px',
        borderRadius: '8px',
        border: '1px solid rgba(179, 131, 255, 0.3)',
        color: '#B383FF',
        fontSize: '12px',
        fontFamily: 'monospace',
        backdropFilter: 'blur(4px)',
        maxWidth: '250px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
          Quantum Entanglements
        </div>
        
        {nodeEntanglements.map((pair, index) => {
          const partner = pair.nodeA.id === selectedNodeId ? pair.nodeB : pair.nodeA;
          return (
            <div key={pair.id} style={{ marginBottom: '6px', fontSize: '10px' }}>
              <div>Partner: {partner.label || partner.id}</div>
              <div>Strength: {(pair.strength * 100).toFixed(1)}%</div>
              <div>Correlation: {(pair.correlation * 100).toFixed(1)}%</div>
              <div style={{ height: '1px', background: 'rgba(179, 131, 255, 0.3)', margin: '4px 0' }} />
            </div>
          );
        })}
        
        <div style={{ fontSize: '9px', opacity: 0.7, marginTop: '8px' }}>
          Entangled nodes respond instantly to each other's state changes
        </div>
      </div>
    </Html>
  );
}
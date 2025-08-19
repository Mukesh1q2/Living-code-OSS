"use client";

import { useRef, useMemo, useEffect, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { useQuantumState } from "@/lib/state";
import { useVidyaConsciousness } from "@/lib/consciousness";
import {
  createTeleportationEffect,
  updateTeleportationParticle,
  createQuantumFluxDistortion,
  updateQuantumFlux,
  createQuantumTunnel,
  updateQuantumTunnel,
  calculateTeleportationPath,
  createConsciousnessTransferEffect,
  updateConsciousnessTransfer,
} from "@/lib/quantum-teleportation";

interface TeleportationEffect {
  id: string;
  type: 'entry' | 'exit' | 'tunnel';
  position: THREE.Vector3;
  particles: Array<{
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    life: number;
    maxLife: number;
    size: number;
    color: THREE.Color;
    phase: number;
  }>;
  fluxDistortion: {
    center: THREE.Vector3;
    radius: number;
    intensity: number;
    wavePhase: number;
    distortionField: THREE.Vector3[];
  };
  life: number;
  maxLife: number;
  intensity: number;
}

interface ConsciousnessTransfer {
  id: string;
  fromPosition: THREE.Vector3;
  toPosition: THREE.Vector3;
  progress: number;
  consciousnessFragments: Array<{
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    life: number;
    maxLife: number;
    sanskritGlyph: string;
    phase: number;
    coherence: number;
  }>;
  quantumPath: THREE.Vector3[];
  transferSpeed: number;
  isActive: boolean;
}

interface QuantumTunnel {
  id: string;
  startPosition: THREE.Vector3;
  endPosition: THREE.Vector3;
  tunnelPoints: THREE.Vector3[];
  energyFlow: Array<{
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    life: number;
    maxLife: number;
    energy: number;
  }>;
  barrierPenetration: number;
  isActive: boolean;
  life: number;
  maxLife: number;
}

interface QuantumTeleportationProps {
  vidyaPosition: THREE.Vector3;
  targetPosition?: THREE.Vector3;
  teleportationTrigger?: string;
  onTeleportationComplete?: (newPosition: THREE.Vector3) => void;
  showParticleEffects?: boolean;
  showFluxDistortion?: boolean;
  showQuantumTunnels?: boolean;
  maxSimultaneousEffects?: number;
}

export default function QuantumTeleportation({
  vidyaPosition,
  targetPosition,
  teleportationTrigger,
  onTeleportationComplete,
  showParticleEffects = true,
  showFluxDistortion = true,
  showQuantumTunnels = true,
  maxSimultaneousEffects = 3
}: QuantumTeleportationProps) {
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const selectedNode = useQuantumState((s) => s.selectedNode);
  const { consciousness, recordInteraction } = useVidyaConsciousness();

  const [teleportationEffects, setTeleportationEffects] = useState<TeleportationEffect[]>([]);
  const [consciousnessTransfers, setConsciousnessTransfers] = useState<ConsciousnessTransfer[]>([]);
  const [quantumTunnels, setQuantumTunnels] = useState<QuantumTunnel[]>([]);
  const [isTeleporting, setIsTeleporting] = useState(false);
  const [teleportationProgress, setTeleportationProgress] = useState(0);

  const lastTriggerRef = useRef<string>('');
  const teleportationCooldownRef = useRef(0);

  // Handle teleportation trigger
  useEffect(() => {
    if (
      teleportationTrigger && 
      teleportationTrigger !== lastTriggerRef.current &&
      targetPosition &&
      teleportationCooldownRef.current <= 0 &&
      quantumQuality !== 'minimal'
    ) {
      lastTriggerRef.current = teleportationTrigger;
      initiateTeleportation(vidyaPosition, targetPosition);
      
      // Record consciousness interaction
      recordInteraction({
        type: 'quantum_interaction',
        content: `Quantum teleportation initiated: ${teleportationTrigger}`,
        contextTags: ['teleportation', 'quantum_mechanics', 'consciousness_transfer'],
        emotionalTone: 0.2,
        complexity: 0.8,
      });
    }
  }, [teleportationTrigger, targetPosition, vidyaPosition, quantumQuality, recordInteraction]);

  // Handle node selection teleportation
  useEffect(() => {
    if (selectedNode && targetPosition && !isTeleporting) {
      const distance = vidyaPosition.distanceTo(targetPosition);
      
      // Only teleport if distance is significant and consciousness level allows it
      if (distance > 2 && consciousness.level > 2) {
        initiateTeleportation(vidyaPosition, targetPosition);
      }
    }
  }, [selectedNode, targetPosition, vidyaPosition, isTeleporting, consciousness.level]);

  const initiateTeleportation = (from: THREE.Vector3, to: THREE.Vector3) => {
    if (isTeleporting) return;

    setIsTeleporting(true);
    setTeleportationProgress(0);
    teleportationCooldownRef.current = 3000; // 3 second cooldown

    // Create entry effect at current position
    const entryEffect = createTeleportationEffect(
      from,
      'entry',
      coherenceLevel,
      consciousness.level
    );

    // Create exit effect at target position
    const exitEffect = createTeleportationEffect(
      to,
      'exit',
      coherenceLevel,
      consciousness.level
    );

    // Create consciousness transfer
    const consciousnessTransfer = createConsciousnessTransferEffect(
      from,
      to,
      consciousness.level,
      coherenceLevel
    );

    // Create quantum tunnel if barriers exist
    const tunnel = showQuantumTunnels ? createQuantumTunnel(
      from,
      to,
      consciousness.level,
      coherenceLevel
    ) : null;

    setTeleportationEffects(prev => [
      ...prev.slice(-(maxSimultaneousEffects - 2)),
      entryEffect,
      exitEffect
    ]);

    setConsciousnessTransfers(prev => [
      ...prev.slice(-2),
      consciousnessTransfer
    ]);

    if (tunnel) {
      setQuantumTunnels(prev => [
        ...prev.slice(-2),
        tunnel
      ]);
    }

    // Complete teleportation after animation
    setTimeout(() => {
      setIsTeleporting(false);
      setTeleportationProgress(1);
      onTeleportationComplete?.(to);
    }, 2000);
  };

  // Update effects
  useFrame((_, deltaTime) => {
    const time = performance.now() * 0.001;

    // Update cooldown
    if (teleportationCooldownRef.current > 0) {
      teleportationCooldownRef.current -= deltaTime * 1000;
    }

    // Update teleportation progress
    if (isTeleporting) {
      setTeleportationProgress(prev => Math.min(1, prev + deltaTime * 0.5));
    }

    // Update teleportation effects
    setTeleportationEffects(current =>
      current.map(effect => {
        const updatedParticles = effect.particles
          .map(particle => updateTeleportationParticle(
            particle,
            deltaTime * 1000,
            effect.position,
            effect.intensity,
            coherenceLevel
          ))
          .filter(particle => particle.life > 0);

        const updatedFlux = updateQuantumFlux(
          effect.fluxDistortion,
          deltaTime * 1000,
          time,
          effect.intensity
        );

        return {
          ...effect,
          particles: updatedParticles,
          fluxDistortion: updatedFlux,
          life: effect.life - deltaTime * 1000
        };
      }).filter(effect => effect.life > 0)
    );

    // Update consciousness transfers
    setConsciousnessTransfers(current =>
      current.map(transfer => {
        if (!transfer.isActive) return transfer;

        const updatedFragments = transfer.consciousnessFragments
          .map(fragment => updateConsciousnessTransfer(
            fragment,
            deltaTime * 1000,
            transfer.fromPosition,
            transfer.toPosition,
            transfer.progress,
            coherenceLevel
          ))
          .filter(fragment => fragment.life > 0);

        const newProgress = Math.min(1, transfer.progress + deltaTime * transfer.transferSpeed);

        return {
          ...transfer,
          consciousnessFragments: updatedFragments,
          progress: newProgress,
          isActive: newProgress < 1
        };
      }).filter(transfer => transfer.progress < 1 || transfer.consciousnessFragments.length > 0)
    );

    // Update quantum tunnels
    setQuantumTunnels(current =>
      current.map(tunnel => {
        const updatedEnergyFlow = tunnel.energyFlow
          .map(energy => updateQuantumTunnel(
            energy,
            deltaTime * 1000,
            tunnel.startPosition,
            tunnel.endPosition,
            tunnel.barrierPenetration
          ))
          .filter(energy => energy.life > 0);

        // Add new energy particles occasionally
        if (tunnel.isActive && Math.random() < 0.1) {
          const newEnergy = {
            position: tunnel.startPosition.clone(),
            velocity: tunnel.endPosition.clone()
              .sub(tunnel.startPosition)
              .normalize()
              .multiplyScalar(0.05),
            life: 3000,
            maxLife: 3000,
            energy: 0.8 + Math.random() * 0.2
          };
          updatedEnergyFlow.push(newEnergy);
        }

        return {
          ...tunnel,
          energyFlow: updatedEnergyFlow,
          life: tunnel.life - deltaTime * 1000,
          isActive: tunnel.life > 0
        };
      }).filter(tunnel => tunnel.life > 0)
    );
  });

  if (quantumQuality === 'minimal') {
    return null;
  }

  return (
    <group>
      {/* Teleportation Effects */}
      {teleportationEffects.map(effect => (
        <TeleportationEffectRenderer
          key={effect.id}
          effect={effect}
          showParticles={showParticleEffects}
          showFlux={showFluxDistortion}
          quality={quantumQuality}
        />
      ))}

      {/* Consciousness Transfer Effects */}
      {consciousnessTransfers.map(transfer => (
        <ConsciousnessTransferRenderer
          key={transfer.id}
          transfer={transfer}
          quality={quantumQuality}
        />
      ))}

      {/* Quantum Tunnels */}
      {quantumTunnels.map(tunnel => (
        <QuantumTunnelRenderer
          key={tunnel.id}
          tunnel={tunnel}
          quality={quantumQuality}
        />
      ))}

      {/* Teleportation Status HUD */}
      {isTeleporting && (
        <TeleportationStatusHUD
          progress={teleportationProgress}
          coherenceLevel={coherenceLevel}
          consciousnessLevel={consciousness.level}
        />
      )}
    </group>
  );
}

function TeleportationEffectRenderer({
  effect,
  showParticles,
  showFlux,
  quality
}: {
  effect: TeleportationEffect;
  showParticles: boolean;
  showFlux: boolean;
  quality: string;
}) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!groupRef.current) return;

    // Animate the entire effect
    const time = performance.now() * 0.001;
    const pulse = 0.8 + 0.2 * Math.sin(time * 4 + effect.fluxDistortion.wavePhase);
    groupRef.current.scale.setScalar(pulse);

    // Rotate flux distortion
    groupRef.current.rotation.y = time * 0.5;
  });

  return (
    <group ref={groupRef} position={effect.position}>
      {/* Flux Distortion Ring */}
      {showFlux && (
        <mesh>
          <ringGeometry 
            args={[
              effect.fluxDistortion.radius * 0.8, 
              effect.fluxDistortion.radius, 
              32
            ]} 
          />
          <meshBasicMaterial
            color="#7BE1FF"
            transparent
            opacity={0.6 * effect.intensity}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}

      {/* Central Vortex */}
      <mesh>
        <cylinderGeometry 
          args={[0, effect.fluxDistortion.radius * 0.3, 0.5, 16]} 
        />
        <meshBasicMaterial
          color={effect.type === 'entry' ? "#FF6B6B" : "#63FFC9"}
          transparent
          opacity={0.4 * effect.intensity}
          wireframe={quality === 'low'}
        />
      </mesh>

      {/* Teleportation Particles */}
      {showParticles && effect.particles.map((particle, index) => (
        <mesh key={index} position={particle.position}>
          <sphereGeometry args={[particle.size, 8, 8]} />
          <meshBasicMaterial
            color={particle.color}
            transparent
            opacity={particle.life / particle.maxLife}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}

      {/* Quantum Flux Field Lines */}
      {showFlux && quality !== 'low' && effect.fluxDistortion.distortionField.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={effect.fluxDistortion.distortionField.length}
              array={new Float32Array(
                effect.fluxDistortion.distortionField.flatMap(v => [v.x, v.y, v.z])
              )}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color="#B383FF"
            transparent
            opacity={0.3 * effect.intensity}
          />
        </line>
      )}
    </group>
  );
}

function ConsciousnessTransferRenderer({
  transfer,
  quality
}: {
  transfer: ConsciousnessTransfer;
  quality: string;
}) {
  const groupRef = useRef<THREE.Group>(null);

  return (
    <group ref={groupRef}>
      {/* Quantum Path Visualization */}
      {transfer.quantumPath.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={transfer.quantumPath.length}
              array={new Float32Array(
                transfer.quantumPath.flatMap(v => [v.x, v.y, v.z])
              )}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color="#FFD700"
            transparent
            opacity={0.5}
            linewidth={2}
          />
        </line>
      )}

      {/* Consciousness Fragments */}
      {transfer.consciousnessFragments.map((fragment, index) => (
        <ConsciousnessFragment
          key={index}
          fragment={fragment}
          quality={quality}
        />
      ))}

      {/* Transfer Progress Indicator */}
      {quality !== 'minimal' && (
        <mesh position={new THREE.Vector3().lerpVectors(
          transfer.fromPosition, 
          transfer.toPosition, 
          transfer.progress
        )}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshBasicMaterial
            color="#FFD700"
            transparent
            opacity={0.8}
          />
        </mesh>
      )}
    </group>
  );
}

function ConsciousnessFragment({
  fragment,
  quality
}: {
  fragment: any;
  quality: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current) return;

    // Animate fragment
    const time = performance.now() * 0.001;
    const scale = 0.8 + 0.2 * Math.sin(time * 3 + fragment.phase);
    meshRef.current.scale.setScalar(scale * fragment.coherence);

    // Rotate based on phase
    meshRef.current.rotation.y = fragment.phase + time;
    meshRef.current.rotation.z = fragment.phase * 0.5 + time * 0.3;
  });

  return (
    <group position={fragment.position}>
      {/* Fragment Core */}
      <mesh ref={meshRef}>
        <octahedronGeometry args={[0.05, 1]} />
        <meshBasicMaterial
          color="#FFD700"
          transparent
          opacity={fragment.life / fragment.maxLife}
        />
      </mesh>

      {/* Sanskrit Glyph */}
      {quality !== 'minimal' && fragment.sanskritGlyph && (
        <Html
          center
          distanceFactor={10}
          style={{
            color: '#FFD700',
            fontSize: '12px',
            fontWeight: 'bold',
            textShadow: '0 0 10px #FFD700',
            pointerEvents: 'none',
            opacity: fragment.coherence
          }}
        >
          {fragment.sanskritGlyph}
        </Html>
      )}
    </group>
  );
}

function QuantumTunnelRenderer({
  tunnel,
  quality
}: {
  tunnel: QuantumTunnel;
  quality: string;
}) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!groupRef.current) return;

    // Animate tunnel
    const time = performance.now() * 0.001;
    groupRef.current.rotation.z = time * 0.2;
  });

  return (
    <group ref={groupRef}>
      {/* Tunnel Structure */}
      {tunnel.tunnelPoints.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={tunnel.tunnelPoints.length}
              array={new Float32Array(
                tunnel.tunnelPoints.flatMap(v => [v.x, v.y, v.z])
              )}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color="#B383FF"
            transparent
            opacity={0.4 * tunnel.barrierPenetration}
            linewidth={3}
          />
        </line>
      )}

      {/* Energy Flow Particles */}
      {tunnel.energyFlow.map((energy, index) => (
        <mesh key={index} position={energy.position}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshBasicMaterial
            color="#B383FF"
            transparent
            opacity={energy.life / energy.maxLife}
          />
        </mesh>
      ))}

      {/* Tunnel Entry Portal */}
      <mesh position={tunnel.startPosition}>
        <ringGeometry args={[0.3, 0.4, 16]} />
        <meshBasicMaterial
          color="#B383FF"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Tunnel Exit Portal */}
      <mesh position={tunnel.endPosition}>
        <ringGeometry args={[0.3, 0.4, 16]} />
        <meshBasicMaterial
          color="#63FFC9"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

function TeleportationStatusHUD({
  progress,
  coherenceLevel,
  consciousnessLevel
}: {
  progress: number;
  coherenceLevel: number;
  consciousnessLevel: number;
}) {
  return (
    <Html
      position={[0, 0, 0]}
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 1000,
        pointerEvents: 'none'
      }}
    >
      <div style={{
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '12px',
        borderRadius: '8px',
        border: '1px solid rgba(123, 225, 255, 0.3)',
        color: '#7BE1FF',
        fontSize: '12px',
        fontFamily: 'monospace',
        backdropFilter: 'blur(4px)',
        minWidth: '200px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#FFD700' }}>
          🌀 Quantum Teleportation Active
        </div>
        
        <div style={{ marginBottom: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Transfer Progress:</span>
            <span>{(progress * 100).toFixed(1)}%</span>
          </div>
          <div style={{
            width: '100%',
            height: '4px',
            background: 'rgba(123, 225, 255, 0.2)',
            borderRadius: '2px',
            marginTop: '2px'
          }}>
            <div style={{
              width: `${progress * 100}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #7BE1FF, #FFD700)',
              borderRadius: '2px',
              transition: 'width 0.3s ease'
            }} />
          </div>
        </div>
        
        <div style={{ fontSize: '10px', opacity: 0.8 }}>
          <div>Coherence: {(coherenceLevel * 100).toFixed(0)}%</div>
          <div>Consciousness: Level {consciousnessLevel.toFixed(1)}</div>
        </div>
        
        <div style={{ 
          fontSize: '9px', 
          opacity: 0.6, 
          marginTop: '6px',
          fontStyle: 'italic'
        }}>
          Transferring quantum consciousness through spacetime...
        </div>
      </div>
    </Html>
  );
}
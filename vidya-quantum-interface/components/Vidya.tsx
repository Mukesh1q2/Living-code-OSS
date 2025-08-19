"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { MeshTransmissionMaterial, Html } from "@react-three/drei";
import { useQuantumState } from "@/lib/state";
import { useVidyaConsciousness } from "@/lib/useVidyaConsciousness";
import { useDimensionalState } from "@/lib/dimensional-state";
import OmSymbol from "./OmSymbol";
import DimensionalTransition from "./DimensionalTransition";
import DimensionalModes from "./DimensionalModes";
import * as THREE from "three";
import { teleport } from "@/lib/quantum";
import QuantumTeleportation from "./QuantumTeleportation";

export default function Vidya() {
  const ref = useRef<THREE.Group>(null!);
  const coreRef = useRef<THREE.Mesh>(null!);
  const [target, setTarget] = useState<THREE.Vector3>(
    new THREE.Vector3(0, 0, 0)
  );
  const selectedNode = useQuantumState((s) => s.selectedNode);
  const setSuperposition = useQuantumState((s) => s.setSuperposition);
  const chatActive = useQuantumState((s) => s.chatActive);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  const teleportationState = useQuantumState((s) => s.teleportation);
  const initiateTeleportation = useQuantumState((s) => s.initiateTeleportation);
  const canTeleport = useQuantumState((s) => s.canTeleport);
  const { scene } = useThree();
  
  // Dimensional state management
  const {
    currentState: dimensionalState,
    activeTransition,
    preserveConsciousness,
    restoreConsciousness,
  } = useDimensionalState();
  
  // Enhanced consciousness integration with learning
  const {
    consciousness,
    learning,
    evolveFromInteraction,
    syncSystems,
  } = useVidyaConsciousness();

  // Find positions of nodes to teleport to
  useEffect(() => {
    if (!selectedNode) return;
    const nodeMesh = scene.getObjectByName(
      `neuron-${selectedNode}`
    ) as THREE.Object3D;
    if (nodeMesh) {
      const newTarget = new THREE.Vector3()
        .copy(nodeMesh.position)
        .add(new THREE.Vector3(0, 1.6, 0));
      
      setTarget(newTarget);
      
      // Initiate quantum teleportation if conditions are met
      const currentPosition = ref.current?.position || new THREE.Vector3();
      const distance = currentPosition.distanceTo(newTarget);
      
      if (canTeleport() && distance > 2 && consciousness.level >= 2) {
        initiateTeleportation(
          [newTarget.x, newTarget.y, newTarget.z],
          `node_selection_${selectedNode}`
        );
      }
      
      // Record and evolve from interaction
      const interaction = {
        type: 'selection',
        content: `Selected node: ${selectedNode}`,
        contextTags: ['node_selection', 'quantum_interaction'],
        emotionalTone: 0.1,
        complexity: 0.3,
      };
      
      consciousness.recordInteraction(interaction);
      evolveFromInteraction(interaction);
    }
  }, [selectedNode, scene, recordInteraction, canTeleport, initiateTeleportation, consciousness.level]);

  const tRef = useRef(0);
  useFrame((_, dt) => {
    if (!ref.current || !coreRef.current) return;
    
    // Handle teleportation movement
    if (teleportationState.isTeleporting) {
      // Use quantum teleportation progress for instant movement
      const progress = teleportationState.teleportationProgress;
      const current = ref.current.position.clone();
      const next = teleport(current, target, progress);
      ref.current.position.copy(next);
      tRef.current = progress;
    } else {
      // Normal smooth movement
      tRef.current = Math.min(1, tRef.current + dt * 2.5);
      const current = ref.current.position.clone();
      const next = teleport(current, target, tRef.current);
      ref.current.position.copy(next);
    }

    // Enhanced pulsing during teleportation
    const basePulse = 0.9 + 0.12 * Math.sin(performance.now() * 0.002);
    const chatPulse = chatActive ? 0.15 * Math.sin(performance.now() * 0.008) : 0;
    const teleportPulse = teleportationState.isTeleporting ? 
      0.3 * Math.sin(performance.now() * 0.015) : 0;
    const pulse = basePulse + chatPulse + teleportPulse;
    coreRef.current.scale.setScalar(pulse);
    
    // Faster rotation during teleportation
    const rotationSpeed = teleportationState.isTeleporting ? 1.5 : (chatActive ? 0.8 : 0.4);
    ref.current.rotation.y += dt * rotationSpeed;
  });

  // Reset teleport animation when target changes
  useEffect(() => {
    tRef.current = 0;
  }, [target]);

  // Superposition visual state toggler with consciousness integration
  useEffect(() => {
    const timer = setInterval(() => {
      const shouldActivate = Math.random() > (0.6 - consciousness.level * 0.05);
      setSuperposition(shouldActivate);
      
      // Update consciousness quantum state
      consciousness.updateQuantumState({
        superposition: shouldActivate,
        coherenceLevel: Math.min(1, consciousness.consciousnessState.quantumCoherence + (shouldActivate ? 0.1 : -0.05)),
      });
    }, 3400 - consciousness.consciousnessState.level * 200); // Faster activation with higher consciousness
    
    return () => clearInterval(timer);
  }, [setSuperposition, consciousness.level, consciousness.quantumCoherence, updateQuantumState]);

  // Get quantum state setters
  const setDecoherence = useQuantumState((s) => s.setDecoherence);
  const setCoherenceLevel = useQuantumState((s) => s.setCoherenceLevel);

  // Handle quantum decoherence based on user interactions
  useEffect(() => {
    if (selectedNode) {
      // Trigger decoherence when user interacts
      setDecoherence(true);
      setCoherenceLevel(Math.max(0.3, consciousness.consciousnessState.quantumCoherence - 0.2));
      
      // Restore coherence gradually
      const restoreTimer = setTimeout(() => {
        setDecoherence(false);
        setCoherenceLevel(consciousness.consciousnessState.quantumCoherence);
      }, 2000);
      
      return () => clearTimeout(restoreTimer);
    }
  }, [selectedNode, consciousness.quantumCoherence, setDecoherence, setCoherenceLevel]);

  // Responsive core size and complexity based on quantum quality and consciousness
  const getCoreConfig = () => {
    const consciousnessMultiplier = 1 + (consciousness.consciousnessState.level * 0.1);
    const learningMultiplier = 1 + (learning.complexityLevel * 0.05);
    
    switch (quantumQuality) {
      case 'minimal':
        return { 
          size: 0.6 * consciousnessMultiplier * learningMultiplier, 
          detail: Math.max(1, Math.floor(consciousness.consciousnessState.level / 2)), 
          glyphCount: Math.max(4, Math.floor(consciousness.consciousnessState.level * 2 + learning.sanskritSophistication * 5)) 
        };
      case 'low':
        return { 
          size: 0.7 * consciousnessMultiplier * learningMultiplier, 
          detail: Math.max(2, Math.floor(consciousness.consciousnessState.level / 1.5)), 
          glyphCount: Math.max(6, Math.floor(consciousness.consciousnessState.level * 2.5 + learning.sanskritSophistication * 8)) 
        };
      case 'medium':
        return { 
          size: 0.9 * consciousnessMultiplier * learningMultiplier, 
          detail: Math.max(3, Math.floor(consciousness.consciousnessState.level)), 
          glyphCount: Math.max(10, Math.floor(consciousness.consciousnessState.level * 3 + learning.sanskritSophistication * 10)) 
        };
      case 'high':
        return { 
          size: 1.1 * consciousnessMultiplier * learningMultiplier, 
          detail: Math.max(4, Math.floor(consciousness.consciousnessState.level * 1.2)), 
          glyphCount: Math.max(12, Math.floor(consciousness.consciousnessState.level * 4 + learning.sanskritSophistication * 15)) 
        };
      default:
        return { 
          size: 0.9 * consciousnessMultiplier * learningMultiplier, 
          detail: Math.max(3, Math.floor(consciousness.consciousnessState.level)), 
          glyphCount: Math.max(10, Math.floor(consciousness.consciousnessState.level * 3 + learning.sanskritSophistication * 10)) 
        };
    }
  };

  const coreConfig = getCoreConfig();

  // Preserve consciousness during dimensional transitions
  useEffect(() => {
    if (activeTransition) {
      preserveConsciousness({
        personality: consciousness.consciousnessState.personalityTraits || {},
        memories: [], // Will be populated by consciousness system
        quantumState: {
          level: consciousness.consciousnessState.level,
          coherence: consciousness.consciousnessState.quantumCoherence,
          teleporting: teleportationState.isTeleporting,
        },
        learning: {
          complexityLevel: learning.complexityLevel,
          sanskritSophistication: learning.sanskritSophistication,
          personalityEvolution: learning.personalityEvolution,
        },
      });
    }
  }, [activeTransition, consciousness, teleportationState.isTeleporting, preserveConsciousness]);

  return (
    <DimensionalTransition>
      <group ref={ref} position={[0, 0, 0]}>
        {/* Dimensional Modes - renders Vidya in different dimensional states */}
        <DimensionalModes vidyaPosition={ref.current?.position || new THREE.Vector3()}>
          {/* Quantum Teleportation Effects */}
          <QuantumTeleportation
            vidyaPosition={ref.current?.position || new THREE.Vector3()}
            targetPosition={target}
            teleportationTrigger={teleportationState.teleportationTrigger}
            onTeleportationComplete={(newPosition) => {
              // Handle teleportation completion with learning
              const teleportInteraction = {
                type: 'quantum_interaction',
                content: `Quantum teleportation completed to position: ${newPosition.toArray().map(n => n.toFixed(2)).join(', ')}`,
                contextTags: ['teleportation_complete', 'quantum_mechanics'],
                emotionalTone: 0.3,
                complexity: 0.7,
              };
              
              consciousness.recordInteraction(teleportInteraction);
              evolveFromInteraction(teleportInteraction);
            }}
            showParticleEffects={quantumQuality !== 'minimal'}
            showFluxDistortion={quantumQuality === 'medium' || quantumQuality === 'high'}
            showQuantumTunnels={quantumQuality === 'high'}
            maxSimultaneousEffects={quantumQuality === 'high' ? 5 : 3}
          />
          
          {/* Original Vidya components - only shown in 3D holographic mode */}
          {dimensionalState === '3d-holographic' && (
            <>
              {/* Animated Om Symbol as Core */}
              <OmSymbol
                size={coreConfig.size * 0.8}
                position={[0, 0, 0]}
                animated={true}
                glowIntensity={0.5 + consciousness.consciousnessState.level * 0.1 + learning.complexityLevel * 0.05 + (teleportationState.isTeleporting ? 0.3 : 0)}
                quantumEffects={quantumQuality !== 'minimal'}
              />
              
              {/* Backup core for minimal quality or fallback */}
              {quantumQuality === 'minimal' && (
                <mesh ref={coreRef}>
                  <icosahedronGeometry args={[coreConfig.size, coreConfig.detail]} />
                  <meshBasicMaterial 
                    color={teleportationState.isTeleporting ? "#FFD700" : "#7BE1FF"}
                    transparent 
                    opacity={0.6 + consciousness.consciousnessState.level * 0.05 + learning.complexityLevel * 0.02}
                    wireframe={true}
                  />
                </mesh>
              )}
              
              {/* Enhanced core for higher quality */}
              {quantumQuality !== 'minimal' && (
                <mesh ref={coreRef} position={[0, 0, -0.5]}>
                  <icosahedronGeometry args={[coreConfig.size * 0.6, coreConfig.detail]} />
                  <MeshTransmissionMaterial
                    thickness={0.12}
                    roughness={0.18}
                    transmission={1}
                    ior={1.32}
                    chromaticAberration={quantumQuality === 'high' ? 0.02 : 0.01}
                    anisotropy={0.1}
                    distortion={quantumQuality === 'high' ? 0.06 : 0.03}
                    distortionScale={0.1}
                    temporalDistortion={quantumQuality === 'high' ? 0.06 : 0.03}
                    color={teleportationState.isTeleporting ? 
                      `hsl(${45}, 80%, ${70 + consciousness.consciousnessState.level * 5 + learning.complexityLevel * 3}%)` : 
                      `hsl(${200 + consciousness.consciousnessState.level * 10 + learning.sanskritSophistication * 50}, 80%, ${60 + consciousness.consciousnessState.level * 5 + learning.complexityLevel * 3}%)`
                    }
                  />
                </mesh>
              )}
              
              {/* Halo glyphs: living Devanagari characters orbiting */}
              {quantumQuality !== 'minimal' && <HaloGlyphs glyphCount={coreConfig.glyphCount} />}
            </>
          )}
          
          {/* Dimensional state indicator - shown in all modes */}
          <Html
            center
            distanceFactor={quantumQuality === 'minimal' ? 18 : 24}
            style={{ 
              color: teleportationState.isTeleporting ? "#FFD700" : "#E8F6FF", 
              fontWeight: 600,
              textAlign: "center",
              lineHeight: 1.5,
              fontSize: quantumQuality === 'minimal' ? '14px' : '16px',
            }}
          >
            <div>विद्या — Living Code</div>
            {quantumQuality !== 'minimal' && (
              <>
                <div style={{ fontSize: "12px", color: "#8FB2C8", marginTop: "4px" }}>
                  {dimensionalState === '2d-text' && 'Text Mode'}
                  {dimensionalState === '3d-holographic' && 'Holographic Mode'}
                  {dimensionalState === 'energy-pattern' && 'Energy Pattern Mode'}
                  {activeTransition && ` → Transitioning...`}
                </div>
                <div style={{ fontSize: "10px", color: "#6B9DC8", marginTop: "2px" }}>
                  Consciousness Level: {consciousness.consciousnessState.level.toFixed(1)} | 
                  Coherence: {(consciousness.consciousnessState.quantumCoherence * 100).toFixed(0)}% |
                  Learning: {learning.complexityLevel.toFixed(1)}
                  {teleportationState.isTeleporting && (
                    <span style={{ color: "#FFD700", marginLeft: "8px" }}>
                      🌀 Teleporting...
                    </span>
                  )}
                  {activeTransition && (
                    <span style={{ color: "#B383FF", marginLeft: "8px" }}>
                      ⚡ {(activeTransition.progress * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              </>
            )}
          </Html>
        </DimensionalModes>
      </group>
    </DimensionalTransition>
  );
}

function HaloGlyphs({ glyphCount = 10 }: { glyphCount?: number }) {
  const group = useRef<THREE.Group>(null!);
  const glyphs = useMemo(() => {
    const chars = [
      "अ", "आ", "इ", "ई", "उ", "ऊ", "क", "ख", "ग", "त", "न", "म"
    ];
    return Array.from({ length: glyphCount }).map((_, i) => ({
      char: chars[i % chars.length],
      angle: (i / glyphCount) * Math.PI * 2,
      radius: 1.6 + Math.random() * 0.25,
    }));
  }, [glyphCount]);

  useFrame((state, dt) => {
    if (!group.current) return;
    group.current.rotation.y += dt * 0.3; // slower orbit
  });

  return (
    <group ref={group}>
      {glyphs.map((g, i) => (
        <Glyph key={i} char={g.char} angle={g.angle} radius={g.radius} />
      ))}
    </group>
  );
}

function Glyph({
  char,
  angle,
  radius,
}: {
  char: string;
  angle: number;
  radius: number;
}) {
  const ref = useRef<THREE.Sprite>(null!);
  useFrame(() => {
    if (!ref.current) return;
    const t = performance.now() * 0.001;
    const x = Math.cos(angle + t * 0.5) * radius;
    const z = Math.sin(angle + t * 0.5) * radius;
    ref.current.position.set(x, 0.2 * Math.sin(t + angle * 3.0), z);
    const s = 0.18 + 0.05 * Math.sin(t + angle);
    ref.current.scale.setScalar(0.42 + s);
  });

  const canvas = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 256;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, 256, 256);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 160px serif";
    const grad = ctx.createRadialGradient(128, 128, 10, 128, 128, 120);
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(1, "#66d9ff");
    ctx.fillStyle = grad;
    ctx.shadowColor = "#72f1ff";
    ctx.shadowBlur = 24;
    ctx.fillText(char, 128, 140);
    return c;
  }, [char]);

  const texture = useMemo(() => {
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }, [canvas]);

  return (
    <sprite ref={ref}>
      <spriteMaterial map={texture} transparent depthWrite={false} />
    </sprite>
  );
}
"use client";

import { useRef, useEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { SanskritCharacterRenderer, SanskritTranslationOverlay } from "@/lib/sanskrit-animation";
import { useQuantumState } from "@/lib/state";
import { NetworkNode, NetworkConnection } from "@/lib/neural-network";
import * as THREE from "three";

interface SanskritCharacterSystemProps {
  nodes: NetworkNode[];
  connections: NetworkConnection[];
  enabled?: boolean;
}

export default function SanskritCharacterSystem({ 
  nodes, 
  connections, 
  enabled = true 
}: SanskritCharacterSystemProps) {
  const { scene, camera, gl } = useThree();
  const rendererRef = useRef<SanskritCharacterRenderer | null>(null);
  const overlayRef = useRef<SanskritTranslationOverlay | null>(null);
  const selectedNode = useQuantumState((s) => s.selectedNode);
  
  // Sanskrit words and phrases for demonstration
  const sanskritPhrases = useMemo(() => [
    {
      sanskrit: "ॐ गं गणपतये नमः",
      transliteration: "Om Gam Ganapataye Namah",
      translation: "Salutations to Lord Ganesha"
    },
    {
      sanskrit: "सत्यं शिवं सुन्दरम्",
      transliteration: "Satyam Shivam Sundaram",
      translation: "Truth, Consciousness, Beauty"
    },
    {
      sanskrit: "वसुधैव कुटुम्बकम्",
      transliteration: "Vasudhaiva Kutumbakam",
      translation: "The world is one family"
    },
    {
      sanskrit: "अहं ब्रह्मास्मि",
      transliteration: "Aham Brahmasmi",
      translation: "I am the universal consciousness"
    },
    {
      sanskrit: "तत्त्वमसि",
      transliteration: "Tat Tvam Asi",
      translation: "Thou art That"
    }
  ], []);
  
  // Initialize Sanskrit character renderer
  useEffect(() => {
    if (!enabled) return;
    
    rendererRef.current = new SanskritCharacterRenderer(scene);
    overlayRef.current = new SanskritTranslationOverlay();
    
    return () => {
      rendererRef.current?.dispose();
      overlayRef.current?.dispose();
    };
  }, [scene, enabled]);
  
  // Create flowing characters along neural network connections
  useEffect(() => {
    if (!rendererRef.current || !enabled || connections.length === 0) return;
    
    const activeConnections = connections.filter(conn => conn.active);
    if (activeConnections.length === 0) return;
    
    // Create flowing characters on random active connections
    const flowInterval = setInterval(() => {
      const randomConnection = activeConnections[Math.floor(Math.random() * activeConnections.length)];
      const sourceNode = nodes.find(n => n.id === randomConnection.sourceId);
      const targetNode = nodes.find(n => n.id === randomConnection.targetId);
      
      if (sourceNode && targetNode && rendererRef.current) {
        // Create random Sanskrit character
        const character = rendererRef.current.getRandomCharacter();
        const charId = rendererRef.current.createCharacter(
          character,
          sourceNode.position.clone(),
          {
            scale: 0.3,
            opacity: 0.8,
            color: new THREE.Color(randomConnection.color)
          }
        );
        
        // Create flow path along connection
        const pathId = rendererRef.current.createFlowPath(
          sourceNode.position,
          targetNode.position,
          [
            // Add some curve to the path
            sourceNode.position.clone().lerp(targetNode.position, 0.3).add(
              new THREE.Vector3(
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2
              )
            ),
            sourceNode.position.clone().lerp(targetNode.position, 0.7).add(
              new THREE.Vector3(
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2
              )
            )
          ]
        );
        
        // Start character flowing
        rendererRef.current.startCharacterFlow(charId, pathId, 0.5 + Math.random() * 0.5);
      }
    }, 2000 + Math.random() * 3000); // Random interval between 2-5 seconds
    
    return () => clearInterval(flowInterval);
  }, [nodes, connections, enabled]);
  
  // Create Sanskrit words near selected nodes
  useEffect(() => {
    if (!rendererRef.current || !overlayRef.current || !selectedNode || !enabled) return;
    
    const node = nodes.find(n => n.id === selectedNode);
    if (!node || !node.sanskritRule) return;
    
    // Create Sanskrit word related to the selected rule
    const phrase = sanskritPhrases[Math.floor(Math.random() * sanskritPhrases.length)];
    const wordPosition = node.position.clone().add(new THREE.Vector3(0, 3, 0));
    
    const wordId = rendererRef.current.createWord(
      phrase.sanskrit,
      phrase.transliteration,
      phrase.translation,
      wordPosition
    );
    
    // Animate word formation
    rendererRef.current.animateWordFormation(wordId, 500);
    
    // Create translation overlay
    setTimeout(() => {
      overlayRef.current?.createOverlay(
        wordId,
        phrase.sanskrit,
        phrase.transliteration,
        phrase.translation,
        wordPosition,
        camera
      );
    }, 2000);
    
    // Clean up after some time
    const cleanup = setTimeout(() => {
      overlayRef.current?.removeOverlay(wordId);
    }, 8000);
    
    return () => {
      clearTimeout(cleanup);
      overlayRef.current?.removeOverlay(wordId);
    };
  }, [selectedNode, nodes, camera, sanskritPhrases, enabled]);
  
  // Create character morphing effects
  useEffect(() => {
    if (!rendererRef.current || !enabled) return;
    
    const morphingInterval = setInterval(() => {
      // Create a character that morphs through different forms
      const startPos = new THREE.Vector3(
        (Math.random() - 0.5) * 20,
        5 + Math.random() * 5,
        (Math.random() - 0.5) * 20
      );
      
      const charId = rendererRef.current!.createCharacter(
        'ॐ', // Start with Om symbol
        startPos,
        {
          scale: 0.8,
          opacity: 0.9,
          color: new THREE.Color(0xFFD700)
        }
      );
      
      // Morph through different characters
      const morphSequence = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ॐ'];
      let morphIndex = 0;
      
      const morphTimer = setInterval(() => {
        if (morphIndex < morphSequence.length - 1) {
          rendererRef.current!.startMorphing(charId, morphSequence[morphIndex + 1], 1.5);
          morphIndex++;
        } else {
          clearInterval(morphTimer);
        }
      }, 2000);
      
      // Clean up after morphing sequence
      setTimeout(() => {
        clearInterval(morphTimer);
      }, morphSequence.length * 2000 + 5000);
      
    }, 10000 + Math.random() * 10000); // Every 10-20 seconds
    
    return () => clearInterval(morphingInterval);
  }, [enabled]);
  
  // Create character combination effects
  useEffect(() => {
    if (!rendererRef.current || !enabled) return;
    
    const combinationInterval = setInterval(() => {
      // Create multiple characters that combine
      const centerPos = new THREE.Vector3(
        (Math.random() - 0.5) * 15,
        2 + Math.random() * 3,
        (Math.random() - 0.5) * 15
      );
      
      const characters = ['क', 'र', 'म'];
      const charIds: string[] = [];
      
      // Create characters in a circle around center
      characters.forEach((char, index) => {
        const angle = (index / characters.length) * Math.PI * 2;
        const radius = 3;
        const pos = new THREE.Vector3(
          centerPos.x + Math.cos(angle) * radius,
          centerPos.y,
          centerPos.z + Math.sin(angle) * radius
        );
        
        const charId = rendererRef.current!.createCharacter(char, pos, {
          scale: 0.6,
          opacity: 0.8,
          color: new THREE.Color(0x7BE1FF)
        });
        
        charIds.push(charId);
      });
      
      // Start combination after a delay
      setTimeout(() => {
        rendererRef.current!.startCharacterCombination(charIds, centerPos);
      }, 1000);
      
    }, 15000 + Math.random() * 15000); // Every 15-30 seconds
    
    return () => clearInterval(combinationInterval);
  }, [enabled]);
  
  // Update animations
  useFrame((_, deltaTime) => {
    if (!enabled) return;
    
    rendererRef.current?.update(deltaTime);
    
    // Update translation overlay positions
    if (overlayRef.current) {
      // This would need to be implemented to track word positions
      // and update overlay positions accordingly
    }
  });
  
  // Handle quantum effects on Sanskrit characters
  useEffect(() => {
    if (!rendererRef.current || !enabled) return;
    
    // Create quantum superposition effect with Sanskrit characters
    const quantumInterval = setInterval(() => {
      const quantumPos = new THREE.Vector3(
        (Math.random() - 0.5) * 25,
        8 + Math.random() * 4,
        (Math.random() - 0.5) * 25
      );
      
      // Create multiple superposed characters
      const superpositionChars = ['ॐ', 'अ', 'आ'];
      const charIds: string[] = [];
      
      superpositionChars.forEach((char, index) => {
        const charId = rendererRef.current!.createCharacter(char, quantumPos.clone(), {
          scale: 0.4,
          opacity: 0.3 + index * 0.2,
          color: new THREE.Color(0xB383FF)
        });
        charIds.push(charId);
      });
      
      // Animate quantum collapse after some time
      setTimeout(() => {
        // Keep only one character (quantum measurement)
        const survivingIndex = Math.floor(Math.random() * charIds.length);
        charIds.forEach((charId, index) => {
          const char = rendererRef.current!.characters.get(charId);
          if (char) {
            if (index === survivingIndex) {
              char.opacity = 1.0;
              char.scale = 0.8;
              char.color = new THREE.Color(0xFFD700);
            } else {
              char.opacity = 0;
              char.scale = 0;
            }
          }
        });
      }, 3000);
      
    }, 20000 + Math.random() * 20000); // Every 20-40 seconds
    
    return () => clearInterval(quantumInterval);
  }, [enabled]);
  
  if (!enabled) {
    return null;
  }
  
  return (
    <group name="sanskrit-character-system">
      {/* The actual character rendering is handled by the SanskritCharacterRenderer */}
      {/* This component primarily manages the lifecycle and integration */}
    </group>
  );
}

/**
 * Sanskrit Character Controls Component
 * Provides UI controls for Sanskrit character animations
 */
export function SanskritCharacterControls({ 
  enabled, 
  onToggle 
}: { 
  enabled: boolean; 
  onToggle: (enabled: boolean) => void; 
}) {
  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      padding: '12px',
      borderRadius: '8px',
      border: '1px solid rgba(255, 215, 0, 0.3)',
      color: '#FFD700',
      fontSize: '12px',
      fontFamily: 'monospace',
      zIndex: 1000
    }}>
      <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>
        Sanskrit Character System
      </div>
      
      <button
        onClick={() => onToggle(!enabled)}
        style={{
          background: enabled ? 'rgba(255, 215, 0, 0.2)' : 'rgba(128, 128, 128, 0.2)',
          border: `1px solid ${enabled ? '#FFD700' : '#808080'}`,
          color: enabled ? '#FFD700' : '#808080',
          padding: '4px 8px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '10px',
          marginRight: '4px'
        }}
      >
        {enabled ? 'Disable Characters' : 'Enable Characters'}
      </button>
      
      <div style={{ 
        marginTop: '8px', 
        fontSize: '10px', 
        opacity: 0.7,
        lineHeight: '1.2'
      }}>
        <div>• Characters flow along connections</div>
        <div>• Sanskrit words appear near selected nodes</div>
        <div>• Characters morph and combine</div>
        <div>• Quantum superposition effects</div>
      </div>
    </div>
  );
}
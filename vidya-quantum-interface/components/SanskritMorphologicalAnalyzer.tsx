"use client";

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { useFrame, useThree, extend } from "@react-three/fiber";
import { useQuantumWebSocket } from "@/lib/useQuantumWebSocket";
import { useWebSocket } from "./WebSocketProvider";
import { useWebSocketMessages } from "@/lib/websocket";
import { useQuantumState } from "@/lib/state";
import { NetworkNode, NetworkConnection } from "@/lib/neural-network";
import { Text } from "@react-three/drei";
import * as THREE from "three";

// Extend Three.js with custom materials
extend({ THREE });

interface MorphologicalData {
  word: string;
  root: string;
  suffixes: string[];
  prefixes: string[];
  morphemes: MorphemeData[];
  etymologicalConnections: EtymologicalConnection[];
  paniniRules: PaniniRuleApplication[];
  grammaticalRelationships: GrammaticalRelationship[];
}

interface MorphemeData {
  id: string;
  text: string;
  type: 'root' | 'suffix' | 'prefix' | 'inflection';
  meaning?: string;
  position: { start: number; end: number };
  confidence: number;
  quantumProperties: {
    superposition: boolean;
    entanglements: string[];
    probability: number;
  };
  visualizationData: {
    color: string;
    size: number;
    position: { x: number; y: number; z: number };
    animation: string;
  };
}

interface EtymologicalConnection {
  id: string;
  fromMorpheme: string;
  toMorpheme: string;
  relationship: string;
  strength: number;
  historicalPath: string[];
}

interface PaniniRuleApplication {
  id: string;
  ruleNumber: string;
  ruleName: string;
  description: string;
  appliedTo: string[];
  result: string;
  confidence: number;
  mandalaGeometry: {
    center: { x: number; y: number; z: number };
    radius: number;
    segments: number;
    color: string;
    pattern: string;
  };
}

interface GrammaticalRelationship {
  id: string;
  type: 'dependency' | 'agreement' | 'modification' | 'coordination';
  source: string;
  target: string;
  label: string;
  strength: number;
}

interface SanskritMorphologicalAnalyzerProps {
  text: string;
  onAnalysisUpdate?: (data: MorphologicalData) => void;
  enableRealTimeUpdates?: boolean;
  showEtymologicalConnections?: boolean;
  showPaniniRules?: boolean;
  showMorphemeFlow?: boolean;
  interactionEnabled?: boolean;
}
export 
default function SanskritMorphologicalAnalyzer({
  text,
  onAnalysisUpdate,
  enableRealTimeUpdates = true,
  showEtymologicalConnections = true,
  showPaniniRules = true,
  showMorphemeFlow = true,
  interactionEnabled = true
}: SanskritMorphologicalAnalyzerProps) {
  const { scene, camera } = useThree();
  const { sendMessage } = useWebSocket();
  const { messageHistory } = useWebSocketMessages();
  const quantumWebSocket = useQuantumWebSocket();
  const quantumState = useQuantumState();
  
  // Get last message from history
  const lastMessage = messageHistory.length > 0 ? messageHistory[messageHistory.length - 1] : null;
  
  // State management
  const [morphologicalData, setMorphologicalData] = useState<MorphologicalData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedMorpheme, setSelectedMorpheme] = useState<string | null>(null);
  const [hoveredElement, setHoveredElement] = useState<string | null>(null);
  const [animationProgress, setAnimationProgress] = useState(0);
  
  // Refs for 3D objects
  const morphemeGroupRef = useRef<THREE.Group>(null);
  const connectionGroupRef = useRef<THREE.Group>(null);
  const mandalaGroupRef = useRef<THREE.Group>(null);
  const flowParticlesRef = useRef<THREE.Points>(null);
  
  // Memoized analysis request
  const analysisRequest = useMemo(() => ({
    text,
    enableVisualization: true,
    enableEtymology: showEtymologicalConnections,
    enablePaniniRules: showPaniniRules,
    enableMorphemeFlow: showMorphemeFlow
  }), [text, showEtymologicalConnections, showPaniniRules, showMorphemeFlow]);

  // Real-time morphological analysis
  useEffect(() => {
    if (!text || !enableRealTimeUpdates) return;

    const performAnalysis = async () => {
      setIsAnalyzing(true);
      
      try {
        // Send analysis request via WebSocket for real-time updates
        sendMessage({
          type: 'morphological_analysis',
          data: analysisRequest
        });
      } catch (error) {
        console.error('Error starting morphological analysis:', error);
        setIsAnalyzing(false);
      }
    };

    performAnalysis();
  }, [text, analysisRequest, sendMessage, enableRealTimeUpdates]);

  // Handle WebSocket messages for real-time updates
  useEffect(() => {
    if (!lastMessage) return;

    const message = lastMessage;
    
    switch (message.type) {
      case 'morphological_analysis_update':
        handleAnalysisUpdate(message.payload);
        break;
      case 'morphological_analysis_complete':
        handleAnalysisComplete(message.payload);
        break;
      case 'morpheme_flow_update':
        handleMorphemeFlowUpdate(message.payload);
        break;
      case 'panini_rule_applied':
        handlePaniniRuleApplication(message.payload);
        break;
    }
  }, [lastMessage]);

  const handleAnalysisUpdate = useCallback((data: Partial<MorphologicalData>) => {
    setMorphologicalData(prev => prev ? { ...prev, ...data } : data as MorphologicalData);
  }, []);

  const handleAnalysisComplete = useCallback((data: MorphologicalData) => {
    setMorphologicalData(data);
    setIsAnalyzing(false);
    onAnalysisUpdate?.(data);
  }, [onAnalysisUpdate]);

  const handleMorphemeFlowUpdate = useCallback((data: any) => {
    // Update morpheme flow animation
    setAnimationProgress(data.progress || 0);
  }, []);

  const handlePaniniRuleApplication = useCallback((data: PaniniRuleApplication) => {
    // Animate Pāṇini rule application with mandala visualization
    if (mandalaGroupRef.current) {
      animatePaniniRuleMandala(data);
    }
  }, []);

  // Morpheme interaction handlers
  const handleMorphemeClick = useCallback((morphemeId: string) => {
    if (!interactionEnabled) return;
    
    setSelectedMorpheme(morphemeId);
    
    // Send interaction event
    sendMessage({
      type: 'morpheme_interaction',
      data: {
        morphemeId,
        action: 'select',
        timestamp: Date.now()
      }
    });
  }, [interactionEnabled, sendMessage]);

  const handleMorphemeHover = useCallback((morphemeId: string | null) => {
    if (!interactionEnabled) return;
    
    setHoveredElement(morphemeId);
    
    if (morphemeId) {
      sendMessage({
        type: 'morpheme_interaction',
        data: {
          morphemeId,
          action: 'hover',
          timestamp: Date.now()
        }
      });
    }
  }, [interactionEnabled, sendMessage]);

  // Animation functions
  const animatePaniniRuleMandala = useCallback((ruleData: PaniniRuleApplication) => {
    if (!mandalaGroupRef.current) return;

    const mandala = createPaniniRuleMandala(ruleData);
    mandalaGroupRef.current.add(mandala);

    // Animate mandala appearance
    const startTime = Date.now();
    const duration = 2000; // 2 seconds

    const animateMandala = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      mandala.scale.setScalar(progress);
      mandala.rotation.z = progress * Math.PI * 2;
      
      if (progress < 1) {
        requestAnimationFrame(animateMandala);
      } else {
        // Fade out after display
        setTimeout(() => {
          mandalaGroupRef.current?.remove(mandala);
        }, 3000);
      }
    };

    animateMandala();
  }, []);

  const createPaniniRuleMandala = useCallback((ruleData: PaniniRuleApplication) => {
    const group = new THREE.Group();
    const { center, radius, segments, color, pattern } = ruleData.mandalaGeometry;

    // Create mandala geometry based on Pāṇini rule pattern
    const geometry = new THREE.RingGeometry(radius * 0.5, radius, segments);
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide
    });

    const mandala = new THREE.Mesh(geometry, material);
    mandala.position.set(center.x, center.y, center.z);

    // Add Sanskrit text for rule number
    group.add(mandala);
    group.position.set(center.x, center.y, center.z);

    return group;
  }, []);

  // Frame animation
  useFrame((state, delta) => {
    if (!morphologicalData) return;

    // Animate morpheme flow
    if (showMorphemeFlow && flowParticlesRef.current) {
      animateMorphemeFlow(delta);
    }

    // Animate etymological connections
    if (showEtymologicalConnections && connectionGroupRef.current) {
      animateEtymologicalConnections(delta);
    }

    // Update quantum effects
    updateQuantumEffects(delta);
  });

  const animateMorphemeFlow = useCallback((delta: number) => {
    if (!flowParticlesRef.current || !morphologicalData) return;

    const positions = flowParticlesRef.current.geometry.attributes.position;
    const colors = flowParticlesRef.current.geometry.attributes.color;

    // Animate particles flowing through neural network pathways
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);

      // Update particle positions along morpheme pathways
      positions.setX(i, x + delta * 0.5);
      
      // Update colors based on morpheme properties
      const morpheme = morphologicalData.morphemes[i % morphologicalData.morphemes.length];
      if (morpheme) {
        const color = new THREE.Color(morpheme.visualizationData.color);
        colors.setXYZ(i, color.r, color.g, color.b);
      }
    }

    positions.needsUpdate = true;
    colors.needsUpdate = true;
  }, [morphologicalData]);

  const animateEtymologicalConnections = useCallback((delta: number) => {
    if (!connectionGroupRef.current || !morphologicalData) return;

    connectionGroupRef.current.children.forEach((connection, index) => {
      if (connection instanceof THREE.Line) {
        const material = connection.material as THREE.LineBasicMaterial;
        
        // Animate connection opacity based on strength
        const etymConnection = morphologicalData.etymologicalConnections[index];
        if (etymConnection) {
          material.opacity = 0.3 + 0.4 * Math.sin(Date.now() * 0.001 + index) * etymConnection.strength;
        }
      }
    });
  }, [morphologicalData]);

  const updateQuantumEffects = useCallback((delta: number) => {
    if (!morphologicalData) return;

    // Update quantum superposition effects for morphemes
    morphologicalData.morphemes.forEach((morpheme, index) => {
      if (morpheme.quantumProperties.superposition) {
        // Apply quantum fluctuation effects
        const element = morphemeGroupRef.current?.children[index];
        if (element) {
          element.position.y += Math.sin(Date.now() * 0.005 + index) * 0.01;
        }
      }
    });
  }, [morphologicalData]);

  // Render morphemes
  const renderMorphemes = useMemo(() => {
    if (!morphologicalData) return null;

    return morphologicalData.morphemes.map((morpheme, index) => (
      <group key={morpheme.id} position={[
        morpheme.visualizationData.position.x,
        morpheme.visualizationData.position.y,
        morpheme.visualizationData.position.z
      ]}>
        {/* Morpheme sphere */}
        <mesh
          onClick={() => handleMorphemeClick(morpheme.id)}
          onPointerOver={() => handleMorphemeHover(morpheme.id)}
          onPointerOut={() => handleMorphemeHover(null)}
        >
          <sphereGeometry args={[morpheme.visualizationData.size, 16, 16]} />
          <meshStandardMaterial
            color={morpheme.visualizationData.color}
            transparent
            opacity={selectedMorpheme === morpheme.id ? 1.0 : 0.8}
            emissive={hoveredElement === morpheme.id ? morpheme.visualizationData.color : '#000000'}
            emissiveIntensity={hoveredElement === morpheme.id ? 0.3 : 0}
          />
        </mesh>

        {/* Sanskrit text */}
        <Text
          position={[0, morpheme.visualizationData.size + 0.3, 0]}
          fontSize={0.2}
          color={morpheme.visualizationData.color}
          anchorX="center"
          anchorY="middle"
        >
          {morpheme.text}
        </Text>

        {/* Morpheme type label */}
        <Text
          position={[0, -morpheme.visualizationData.size - 0.3, 0]}
          fontSize={0.15}
          color="#888888"
          anchorX="center"
          anchorY="middle"
        >
          {morpheme.type}
        </Text>

        {/* Quantum superposition effect */}
        {morpheme.quantumProperties.superposition && (
          <mesh>
            <sphereGeometry args={[morpheme.visualizationData.size * 1.5, 16, 16]} />
            <meshBasicMaterial
              color={morpheme.visualizationData.color}
              transparent
              opacity={0.2}
              wireframe
            />
          </mesh>
        )}
      </group>
    ));
  }, [morphologicalData, selectedMorpheme, hoveredElement, handleMorphemeClick, handleMorphemeHover]);

  // Render etymological connections
  const renderEtymologicalConnections = useMemo(() => {
    if (!morphologicalData || !showEtymologicalConnections) return null;

    return morphologicalData.etymologicalConnections.map((connection) => {
      const fromMorpheme = morphologicalData.morphemes.find(m => m.id === connection.fromMorpheme);
      const toMorpheme = morphologicalData.morphemes.find(m => m.id === connection.toMorpheme);

      if (!fromMorpheme || !toMorpheme) return null;

      const points = [
        new THREE.Vector3(
          fromMorpheme.visualizationData.position.x,
          fromMorpheme.visualizationData.position.y,
          fromMorpheme.visualizationData.position.z
        ),
        new THREE.Vector3(
          toMorpheme.visualizationData.position.x,
          toMorpheme.visualizationData.position.y,
          toMorpheme.visualizationData.position.z
        )
      ];

      const geometry = new THREE.BufferGeometry().setFromPoints(points);

      return (
        <primitive key={connection.id} object={new THREE.Line(geometry, new THREE.LineBasicMaterial({
          color: "#4a90e2",
          transparent: true,
          opacity: connection.strength * 0.6,
          linewidth: connection.strength * 3
        }))} />
      );
    });
  }, [morphologicalData, showEtymologicalConnections]);

  // Render Pāṇini rule mandalas
  const renderPaniniRuleMandalas = useMemo(() => {
    if (!morphologicalData || !showPaniniRules) return null;

    return morphologicalData.paniniRules.map((rule) => (
      <group key={rule.id} position={[
        rule.mandalaGeometry.center.x,
        rule.mandalaGeometry.center.y,
        rule.mandalaGeometry.center.z
      ]}>
        {/* Mandala ring */}
        <mesh>
          <ringGeometry args={[
            rule.mandalaGeometry.radius * 0.5,
            rule.mandalaGeometry.radius,
            rule.mandalaGeometry.segments
          ]} />
          <meshBasicMaterial
            color={rule.mandalaGeometry.color}
            transparent
            opacity={0.7}
            side={THREE.DoubleSide}
          />
        </mesh>

        {/* Rule number text */}
        <Text
          position={[0, 0, 0.1]}
          fontSize={0.15}
          color={rule.mandalaGeometry.color}
          anchorX="center"
          anchorY="middle"
        >
          {rule.ruleNumber}
        </Text>

        {/* Rule name */}
        <Text
          position={[0, -rule.mandalaGeometry.radius - 0.3, 0]}
          fontSize={0.1}
          color="#888888"
          anchorX="center"
          anchorY="middle"
        >
          {rule.ruleName}
        </Text>
      </group>
    ));
  }, [morphologicalData, showPaniniRules]);

  // Render morpheme flow particles
  const renderMorphemeFlow = useMemo(() => {
    if (!morphologicalData || !showMorphemeFlow) return null;

    const particleCount = 100;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      // Distribute particles along morpheme pathways
      const morphemeIndex = i % morphologicalData.morphemes.length;
      const morpheme = morphologicalData.morphemes[morphemeIndex];

      positions[i * 3] = morpheme.visualizationData.position.x + (Math.random() - 0.5) * 2;
      positions[i * 3 + 1] = morpheme.visualizationData.position.y + (Math.random() - 0.5) * 2;
      positions[i * 3 + 2] = morpheme.visualizationData.position.z + (Math.random() - 0.5) * 2;

      const color = new THREE.Color(morpheme.visualizationData.color);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    return (
      <points ref={flowParticlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            array={positions}
            count={particleCount}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            array={colors}
            count={particleCount}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.05}
          vertexColors
          transparent
          opacity={0.8}
        />
      </points>
    );
  }, [morphologicalData, showMorphemeFlow]);

  return (
    <group>
      {/* Morpheme visualization */}
      <group ref={morphemeGroupRef}>
        {renderMorphemes}
      </group>

      {/* Etymological connections */}
      <group ref={connectionGroupRef}>
        {renderEtymologicalConnections}
      </group>

      {/* Pāṇini rule mandalas */}
      <group ref={mandalaGroupRef}>
        {renderPaniniRuleMandalas}
      </group>

      {/* Morpheme flow particles */}
      {renderMorphemeFlow}

      {/* Analysis status indicator */}
      {isAnalyzing && (
        <Text
          position={[0, 3, 0]}
          fontSize={0.3}
          color="#4a90e2"
          anchorX="center"
          anchorY="middle"
        >
          Analyzing Sanskrit morphology...
        </Text>
      )}
    </group>
  );
}
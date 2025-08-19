"use client";

import { useMemo, useRef, useEffect, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { NeuralNetworkGenerator, NeuralNetworkAnimator, NetworkNode, NetworkConnection, NodeType } from "@/lib/neural-network";
import { NeuralNetworkCameraControls } from "@/lib/camera-controls";
import { NetworkInteractionManager, NetworkSection } from "@/lib/network-interactions";
import { useQuantumState } from "@/lib/state";
import { useResponsive } from "@/lib/responsive";
import { Line, Html } from "@react-three/drei";
import * as THREE from "three";
import SanskritCharacterSystem from "./SanskritCharacterSystem";
import NetworkTooltip from "./NetworkTooltip";
import NetworkNavigationPanel from "./NetworkNavigationPanel";
import QuantumEntanglement from "./QuantumEntanglement";

interface QuantumNetworkProps {
  sanskritEnabled?: boolean;
}

export default function QuantumNetwork({ sanskritEnabled = true }: QuantumNetworkProps) {
  const { camera, gl } = useThree();
  const { isMobile, isTablet } = useResponsive();
  const cameraControlsRef = useRef<NeuralNetworkCameraControls | null>(null);
  const interactionManagerRef = useRef<NetworkInteractionManager | null>(null);
  const selectedNode = useQuantumState((s) => s.selectedNode);
  const setSelectedNode = useQuantumState((s) => s.setSelectedNode);
  
  // Interaction state
  const [hoveredNode, setHoveredNode] = useState<NetworkNode | null>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [currentSection, setCurrentSection] = useState<NetworkSection | null>(null);
  const [sections, setSections] = useState<NetworkSection[]>([]);
  const [showNavigationPanel, setShowNavigationPanel] = useState(!isMobile);
  
  const { nodes, connections, animator } = useMemo(() => {
    const generator = new NeuralNetworkGenerator({
      nodeCount: 120,
      maxConnections: 4,
      connectionDistance: 8,
      quantumEntanglementProbability: 0.15,
      sanskritRuleRatio: 0.4,
      neuralUnitRatio: 0.4,
      quantumGateRatio: 0.2
    });
    
    const { nodes, connections } = generator.generateNetwork();
    const animator = new NeuralNetworkAnimator(nodes, connections);
    
    return { nodes, connections, animator };
  }, []);

  // Initialize camera controls and interaction manager
  useEffect(() => {
    if (gl.domElement && camera) {
      // Initialize camera controls
      cameraControlsRef.current = new NeuralNetworkCameraControls(camera, gl.domElement, {
        minDistance: 5,
        maxDistance: 50,
        enableAutoRotate: false,
        autoRotateSpeed: 0.5,
        dampingFactor: 0.1,
        focusTransitionSpeed: 2.0,
        explorationSpeed: 1.0
      });
      
      // Initialize interaction manager
      interactionManagerRef.current = new NetworkInteractionManager(
        nodes,
        connections,
        animator,
        camera,
        gl.domElement,
        {
          hoverDelay: isMobile ? 500 : 300,
          touchSensitivity: 1.2,
          enableHapticFeedback: true
        }
      );
      
      // Set up event handlers
      interactionManagerRef.current.setEventHandlers({
        onNodeSelect: (node) => {
          setSelectedNode(node?.id);
          if (node && cameraControlsRef.current) {
            cameraControlsRef.current.focusOnNode(node, true);
          }
        },
        onNodeHover: (node) => {
          setHoveredNode(node);
          setTooltipVisible(!!node);
        },
        onSectionChange: (section) => {
          setCurrentSection(section);
        }
      });
      
      // Get sections
      setSections(interactionManagerRef.current.getSections());
      
      return () => {
        cameraControlsRef.current?.dispose();
        interactionManagerRef.current?.dispose();
      };
    }
  }, [camera, gl.domElement, nodes, connections, animator, isMobile, setSelectedNode]);

  // Focus on selected node
  useEffect(() => {
    if (cameraControlsRef.current && selectedNode) {
      const node = nodes.find(n => n.id === selectedNode);
      if (node) {
        cameraControlsRef.current.focusOnNode(node, true);
      }
    }
  }, [selectedNode, nodes]);

  // Update animation and camera controls
  useFrame((_, deltaTime) => {
    animator.update(deltaTime);
    cameraControlsRef.current?.update(deltaTime);
  });

  const handleSectionSelect = (sectionId: string) => {
    interactionManagerRef.current?.navigateToSection(sectionId);
  };

  return (
    <group>
      {/* Network connections */}
      <NetworkConnections connections={connections} nodes={nodes} />
      
      {/* Network nodes */}
      <NetworkNodes nodes={nodes} animator={animator} />
      
      {/* Sanskrit character animation system */}
      <SanskritCharacterSystem 
        nodes={nodes} 
        connections={connections} 
        enabled={sanskritEnabled}
      />
      
      {/* Quantum entanglement visualization */}
      <QuantumEntanglement
        nodes={nodes}
        connections={connections}
        maxEntanglements={8}
        autoCreateEntanglements={true}
        showFieldVisualization={true}
        showParticleEffects={true}
      />
      
      {/* Interactive tooltip */}
      <NetworkTooltip
        node={hoveredNode}
        visible={tooltipVisible}
        onClose={isMobile ? () => setTooltipVisible(false) : undefined}
      />
      
      {/* Network navigation panel */}
      <NetworkNavigationPanel
        sections={sections}
        currentSection={currentSection}
        onSectionSelect={handleSectionSelect}
        onClose={() => setShowNavigationPanel(false)}
        visible={showNavigationPanel}
      />
      
      {/* Network exploration controls */}
      <NetworkExplorationControls 
        cameraControls={cameraControlsRef.current} 
        nodes={nodes}
        onToggleNavigation={() => setShowNavigationPanel(!showNavigationPanel)}
        showNavigation={showNavigationPanel}
      />
    </group>
  );
}

function NetworkNodes({ nodes, animator }: { nodes: NetworkNode[]; animator: NeuralNetworkAnimator }) {
  return (
    <group>
      {nodes.map((node) => (
        <EnhancedNeuron key={node.id} node={node} animator={animator} />
      ))}
    </group>
  );
}

function EnhancedNeuron({ node, animator }: { node: NetworkNode; animator: NeuralNetworkAnimator }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const setSelectedNode = useQuantumState((s) => s.setSelectedNode);
  const setShowPlanPanel = useQuantumState((s) => s.setShowPlanPanel);
  const selectedNode = useQuantumState((s) => s.selectedNode);

  // Update visual properties based on node state
  useFrame(() => {
    if (!meshRef.current) return;
    
    // Apply scale
    meshRef.current.scale.setScalar(node.scale);
    
    // Apply rotation
    meshRef.current.rotation.y += node.rotationSpeed * 0.016; // Assuming 60fps
    
    // Update material properties
    const material = meshRef.current.material as THREE.MeshStandardMaterial;
    material.emissiveIntensity = node.emissiveIntensity;
    
    // Update color based on selection/highlighting
    if (node.selected) {
      material.emissive.setHex(0xFFFFFF);
      material.emissiveIntensity = Math.max(material.emissiveIntensity, 0.5);
    } else if (node.highlighted) {
      material.emissive.setHex(0xFFFF00);
      material.emissiveIntensity = Math.max(material.emissiveIntensity, 0.3);
    } else {
      material.emissive.setHex(parseInt(node.color.replace('#', '0x')));
    }
  });

  const handleClick = (e: any) => {
    e.stopPropagation();
    setSelectedNode(node.id);
    setShowPlanPanel(true);
    animator.selectNode(node.id);
  };

  // Get geometry based on node type
  const getGeometry = () => {
    switch (node.type) {
      case NodeType.SANSKRIT_RULE:
        return <octahedronGeometry args={[1, 2]} />;
      case NodeType.NEURAL_UNIT:
        return <icosahedronGeometry args={[1, 1]} />;
      case NodeType.QUANTUM_GATE:
        return <dodecahedronGeometry args={[1, 0]} />;
      default:
        return <sphereGeometry args={[1, 16, 16]} />;
    }
  };

  // Get material properties based on node type and state
  const getMaterialProps = () => {
    const baseProps = {
      color: node.color,
      emissive: node.color,
      metalness: 0.2,
      roughness: 0.35,
      transparent: true,
      opacity: node.selected ? 1.0 : (node.highlighted ? 0.9 : 0.8)
    };

    if (node.type === NodeType.QUANTUM_GATE && node.quantumProperties?.superposition) {
      return {
        ...baseProps,
        metalness: 0.8,
        roughness: 0.1,
        opacity: 0.7
      };
    }

    return baseProps;
  };

  return (
    <mesh
      ref={meshRef}
      position={node.position}
      name={`neuron-${node.id}`}
      onClick={handleClick}
      onPointerOver={(e: any) => {
        e.stopPropagation();
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={(e: any) => {
        e.stopPropagation();
        document.body.style.cursor = 'default';
      }}
    >
      {getGeometry()}
      <meshStandardMaterial {...getMaterialProps()} />
      
      {/* Node label */}
      {node.label && (
        <Html
          distanceFactor={20}
          style={{ 
            color: node.selected ? "#ffffff" : "#9bd3ff", 
            fontSize: node.selected ? 14 : 12, 
            opacity: node.selected ? 1.0 : 0.9,
            fontWeight: node.selected ? 'bold' : 'normal',
            textShadow: '0 0 4px rgba(0,0,0,0.8)',
            pointerEvents: 'none'
          }}
        >
          {node.label}
        </Html>
      )}
      
      {/* Sanskrit rule details for selected nodes */}
      {node.selected && node.sanskritRule && (
        <Html
          distanceFactor={15}
          position={[0, -2, 0]}
          style={{ 
            color: "#FFD700", 
            fontSize: 10, 
            opacity: 0.8,
            textAlign: 'center',
            pointerEvents: 'none',
            background: 'rgba(0,0,0,0.7)',
            padding: '4px 8px',
            borderRadius: '4px',
            border: '1px solid rgba(255,215,0,0.3)'
          }}
        >
          <div>
            <div>{node.sanskritRule.description}</div>
            {node.sanskritRule.paniiniSutra && (
              <div style={{ fontSize: 8, opacity: 0.7 }}>
                Sūtra: {node.sanskritRule.paniiniSutra}
              </div>
            )}
          </div>
        </Html>
      )}
    </mesh>
  );
}

function NetworkConnections({ connections, nodes }: { connections: NetworkConnection[]; nodes: NetworkNode[] }) {
  const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  
  return (
    <group>
      {connections.map((connection) => {
        const sourceNode = nodeMap.get(connection.sourceId);
        const targetNode = nodeMap.get(connection.targetId);
        
        if (!sourceNode || !targetNode) return null;
        
        return (
          <ConnectionLine
            key={connection.id}
            connection={connection}
            sourceNode={sourceNode}
            targetNode={targetNode}
          />
        );
      })}
    </group>
  );
}

function ConnectionLine({ 
  connection, 
  sourceNode, 
  targetNode 
}: { 
  connection: NetworkConnection; 
  sourceNode: NetworkNode; 
  targetNode: NetworkNode; 
}) {
  const dataFlowRef = useRef<THREE.Mesh>(null);
  
  // Create data flow particle
  useFrame(() => {
    if (!dataFlowRef.current || !connection.active) return;
    
    // Interpolate position along the line
    const t = connection.dataFlowProgress;
    const position = new THREE.Vector3().lerpVectors(sourceNode.position, targetNode.position, t);
    dataFlowRef.current.position.copy(position);
    
    // Update particle visibility and scale based on connection activity
    const scale = connection.active ? 0.1 + 0.05 * Math.sin(Date.now() * 0.01) : 0;
    dataFlowRef.current.scale.setScalar(scale);
  });
  
  return (
    <group>
      {/* Connection line */}
      <Line
        points={[sourceNode.position, targetNode.position]}
        color={connection.color}
        lineWidth={connection.width}
        transparent
        opacity={connection.opacity}
      />
      
      {/* Data flow particle */}
      {connection.active && (
        <mesh ref={dataFlowRef}>
          <sphereGeometry args={[1, 8, 8]} />
          <meshBasicMaterial 
            color={connection.isEntangled ? "#B383FF" : "#4A9EFF"}
            transparent
            opacity={0.8}
          />
        </mesh>
      )}
      
      {/* Quantum entanglement effect */}
      {connection.isEntangled && (
        <Line
          points={[sourceNode.position, targetNode.position]}
          color="#B383FF"
          lineWidth={connection.width * 1.5}
          transparent
          opacity={0.3}
          dashed
          dashSize={0.5}
          gapSize={0.3}
        />
      )}
    </group>
  );
}

function NetworkExplorationControls({ 
  cameraControls, 
  nodes,
  onToggleNavigation,
  showNavigation
}: { 
  cameraControls: NeuralNetworkCameraControls | null; 
  nodes: NetworkNode[];
  onToggleNavigation: () => void;
  showNavigation: boolean;
}) {
  const { isMobile } = useResponsive();
  const explorationActive = cameraControls?.isExploring ?? false;
  
  return (
    <Html
      position={[0, 0, 0]}
      style={{
        position: 'fixed',
        top: isMobile ? '10px' : '20px',
        right: isMobile ? '10px' : '20px',
        zIndex: 1000,
        pointerEvents: 'auto'
      }}
    >
      <div style={{
        background: 'rgba(0, 0, 0, 0.8)',
        padding: isMobile ? '8px' : '10px',
        borderRadius: '8px',
        border: '1px solid rgba(123, 225, 255, 0.3)',
        color: '#7BE1FF',
        fontSize: isMobile ? '11px' : '12px',
        fontFamily: 'monospace',
        backdropFilter: 'blur(4px)'
      }}>
        <div style={{ 
          marginBottom: '8px', 
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <span>Network Controls</span>
          <button
            onClick={onToggleNavigation}
            style={{
              background: showNavigation ? 'rgba(123, 225, 255, 0.2)' : 'rgba(123, 225, 255, 0.1)',
              border: '1px solid rgba(123, 225, 255, 0.3)',
              color: '#7BE1FF',
              padding: '2px 6px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '9px'
            }}
          >
            {showNavigation ? 'Hide Nav' : 'Show Nav'}
          </button>
        </div>
        
        <div style={{ 
          display: 'flex', 
          gap: '4px', 
          marginBottom: '8px',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => {
              if (explorationActive) {
                cameraControls?.stopExploration();
              } else {
                cameraControls?.startExploration(nodes);
              }
            }}
            style={{
              background: explorationActive ? 'rgba(255, 100, 100, 0.2)' : 'rgba(123, 225, 255, 0.2)',
              border: `1px solid ${explorationActive ? '#ff6464' : '#7BE1FF'}`,
              color: explorationActive ? '#ff6464' : '#7BE1FF',
              padding: '4px 8px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '10px'
            }}
          >
            {explorationActive ? 'Stop Auto' : 'Auto Explore'}
          </button>
          
          <button
            onClick={() => {
              // Focus on a random active node
              const activeNodes = nodes.filter(n => n.active);
              if (activeNodes.length > 0) {
                const randomNode = activeNodes[Math.floor(Math.random() * activeNodes.length)];
                cameraControls?.focusOnNode(randomNode, true);
              }
            }}
            style={{
              background: 'rgba(255, 215, 0, 0.2)',
              border: '1px solid #FFD700',
              color: '#FFD700',
              padding: '4px 8px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '10px'
            }}
          >
            Random
          </button>
        </div>
        
        <div style={{ 
          fontSize: '9px', 
          opacity: 0.7,
          lineHeight: '1.2'
        }}>
          {isMobile ? (
            <>
              <div>• Tap nodes: Select</div>
              <div>• Double tap: Focus</div>
              <div>• Hold: Tooltip</div>
              <div>• Pinch: Zoom</div>
            </>
          ) : (
            <>
              <div>• Click nodes: Select</div>
              <div>• Double-click: Focus</div>
              <div>• Hover: Tooltip</div>
              <div>• Tab/Arrows: Navigate</div>
            </>
          )}
        </div>
      </div>
    </Html>
  );
}
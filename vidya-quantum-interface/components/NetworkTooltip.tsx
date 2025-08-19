"use client";

import { useState, useEffect } from "react";
import { Html } from "@react-three/drei";
import { NetworkNode, NodeType } from "@/lib/neural-network";
import { useResponsive } from "@/lib/responsive";
import * as THREE from "three";

interface NetworkTooltipProps {
  node: NetworkNode | null;
  position?: THREE.Vector3;
  visible: boolean;
  onClose?: () => void;
}

export default function NetworkTooltip({ 
  node, 
  position, 
  visible, 
  onClose 
}: NetworkTooltipProps) {
  const { isMobile, isTablet } = useResponsive();
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    if (!visible) {
      setShowDetails(false);
    }
  }, [visible]);

  if (!node || !visible) return null;

  const getTooltipContent = () => {
    switch (node.type) {
      case NodeType.SANSKRIT_RULE:
        return (
          <SanskritRuleTooltip 
            node={node} 
            showDetails={showDetails}
            onToggleDetails={() => setShowDetails(!showDetails)}
          />
        );
      case NodeType.NEURAL_UNIT:
        return (
          <NeuralUnitTooltip 
            node={node} 
            showDetails={showDetails}
            onToggleDetails={() => setShowDetails(!showDetails)}
          />
        );
      case NodeType.QUANTUM_GATE:
        return (
          <QuantumGateTooltip 
            node={node} 
            showDetails={showDetails}
            onToggleDetails={() => setShowDetails(!showDetails)}
          />
        );
      default:
        return <BasicTooltip node={node} />;
    }
  };

  const getTooltipStyles = () => {
    const baseStyles = {
      background: 'rgba(0, 0, 0, 0.9)',
      border: '1px solid rgba(123, 225, 255, 0.4)',
      borderRadius: '8px',
      padding: isMobile ? '8px' : '12px',
      color: '#7BE1FF',
      fontSize: isMobile ? '11px' : '12px',
      fontFamily: 'monospace',
      maxWidth: isMobile ? '200px' : '280px',
      minWidth: isMobile ? '150px' : '200px',
      backdropFilter: 'blur(4px)',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.6)',
      pointerEvents: 'auto' as const,
      zIndex: 1000,
    };

    // Add type-specific styling
    switch (node.type) {
      case NodeType.SANSKRIT_RULE:
        return {
          ...baseStyles,
          borderColor: 'rgba(255, 215, 0, 0.4)',
          background: 'rgba(20, 15, 0, 0.95)',
        };
      case NodeType.QUANTUM_GATE:
        return {
          ...baseStyles,
          borderColor: 'rgba(179, 131, 255, 0.4)',
          background: 'rgba(15, 10, 20, 0.95)',
        };
      default:
        return baseStyles;
    }
  };

  return (
    <Html
      position={position || node.position}
      distanceFactor={isMobile ? 25 : 20}
      style={getTooltipStyles()}
      occlude={false}
      transform={false}
      sprite={false}
    >
      <div style={{ position: 'relative' }}>
        {/* Close button for mobile */}
        {(isMobile || isTablet) && onClose && (
          <button
            onClick={onClose}
            style={{
              position: 'absolute',
              top: '-4px',
              right: '-4px',
              background: 'rgba(255, 100, 100, 0.8)',
              border: 'none',
              borderRadius: '50%',
              width: '20px',
              height: '20px',
              color: 'white',
              fontSize: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            ×
          </button>
        )}
        
        {getTooltipContent()}
      </div>
    </Html>
  );
}

function SanskritRuleTooltip({ 
  node, 
  showDetails, 
  onToggleDetails 
}: { 
  node: NetworkNode; 
  showDetails: boolean; 
  onToggleDetails: () => void; 
}) {
  const { sanskritRule } = node;
  if (!sanskritRule) return <BasicTooltip node={node} />;

  return (
    <div>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: '8px'
      }}>
        <div style={{ 
          color: '#FFD700', 
          fontWeight: 'bold',
          fontSize: '14px'
        }}>
          {sanskritRule.name}
        </div>
        <div style={{
          background: 'rgba(255, 215, 0, 0.2)',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '10px',
          color: '#FFD700',
          textTransform: 'uppercase'
        }}>
          {sanskritRule.category}
        </div>
      </div>
      
      <div style={{ 
        marginBottom: '8px',
        lineHeight: '1.4',
        opacity: 0.9
      }}>
        {sanskritRule.description}
      </div>
      
      {sanskritRule.paniiniSutra && (
        <div style={{ 
          fontSize: '10px', 
          opacity: 0.7,
          marginBottom: '8px',
          fontStyle: 'italic'
        }}>
          Pāṇini Sūtra: {sanskritRule.paniiniSutra}
        </div>
      )}
      
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '10px',
        opacity: 0.6
      }}>
        <div>
          Status: {node.active ? 'Active' : 'Inactive'}
        </div>
        <button
          onClick={onToggleDetails}
          style={{
            background: 'rgba(255, 215, 0, 0.2)',
            border: '1px solid rgba(255, 215, 0, 0.3)',
            color: '#FFD700',
            padding: '2px 6px',
            borderRadius: '3px',
            fontSize: '9px',
            cursor: 'pointer'
          }}
        >
          {showDetails ? 'Less' : 'More'}
        </button>
      </div>
      
      {showDetails && (
        <div style={{ 
          marginTop: '8px',
          padding: '8px',
          background: 'rgba(255, 215, 0, 0.1)',
          borderRadius: '4px',
          fontSize: '10px',
          lineHeight: '1.3'
        }}>
          <div><strong>Node ID:</strong> {node.id}</div>
          <div><strong>Position:</strong> ({node.position.x.toFixed(1)}, {node.position.y.toFixed(1)}, {node.position.z.toFixed(1)})</div>
          <div><strong>Scale:</strong> {node.scale.toFixed(2)}</div>
          <div><strong>Pulse Speed:</strong> {node.pulseSpeed.toFixed(2)}</div>
          {node.selected && <div style={{ color: '#FFD700' }}><strong>Selected</strong></div>}
          {node.highlighted && <div style={{ color: '#FFFF00' }}><strong>Highlighted</strong></div>}
        </div>
      )}
    </div>
  );
}

function NeuralUnitTooltip({ 
  node, 
  showDetails, 
  onToggleDetails 
}: { 
  node: NetworkNode; 
  showDetails: boolean; 
  onToggleDetails: () => void; 
}) {
  return (
    <div>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: '8px'
      }}>
        <div style={{ 
          color: '#7BE1FF', 
          fontWeight: 'bold',
          fontSize: '14px'
        }}>
          Neural Unit
        </div>
        <div style={{
          background: 'rgba(123, 225, 255, 0.2)',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '10px',
          color: '#7BE1FF',
          textTransform: 'uppercase'
        }}>
          Processing
        </div>
      </div>
      
      <div style={{ 
        marginBottom: '8px',
        lineHeight: '1.4',
        opacity: 0.9
      }}>
        Neural processing unit for pattern recognition and linguistic analysis
      </div>
      
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '10px',
        opacity: 0.6
      }}>
        <div>
          Activity: {node.active ? 'Processing' : 'Idle'}
        </div>
        <button
          onClick={onToggleDetails}
          style={{
            background: 'rgba(123, 225, 255, 0.2)',
            border: '1px solid rgba(123, 225, 255, 0.3)',
            color: '#7BE1FF',
            padding: '2px 6px',
            borderRadius: '3px',
            fontSize: '9px',
            cursor: 'pointer'
          }}
        >
          {showDetails ? 'Less' : 'More'}
        </button>
      </div>
      
      {showDetails && (
        <div style={{ 
          marginTop: '8px',
          padding: '8px',
          background: 'rgba(123, 225, 255, 0.1)',
          borderRadius: '4px',
          fontSize: '10px',
          lineHeight: '1.3'
        }}>
          <div><strong>Node ID:</strong> {node.id}</div>
          <div><strong>Activation Level:</strong> {(node.emissiveIntensity * 100).toFixed(0)}%</div>
          <div><strong>Processing Speed:</strong> {node.pulseSpeed.toFixed(2)}x</div>
          <div><strong>Rotation:</strong> {node.rotationSpeed.toFixed(3)} rad/s</div>
          {node.selected && <div style={{ color: '#7BE1FF' }}><strong>Selected</strong></div>}
          {node.highlighted && <div style={{ color: '#FFFF00' }}><strong>Connected</strong></div>}
        </div>
      )}
    </div>
  );
}

function QuantumGateTooltip({ 
  node, 
  showDetails, 
  onToggleDetails 
}: { 
  node: NetworkNode; 
  showDetails: boolean; 
  onToggleDetails: () => void; 
}) {
  const { quantumProperties } = node;
  
  return (
    <div>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: '8px'
      }}>
        <div style={{ 
          color: '#B383FF', 
          fontWeight: 'bold',
          fontSize: '14px'
        }}>
          Quantum Gate
        </div>
        <div style={{
          background: 'rgba(179, 131, 255, 0.2)',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '10px',
          color: '#B383FF',
          textTransform: 'uppercase'
        }}>
          {quantumProperties?.quantumState || 'Coherent'}
        </div>
      </div>
      
      <div style={{ 
        marginBottom: '8px',
        lineHeight: '1.4',
        opacity: 0.9
      }}>
        Quantum processing gate for superposition and entanglement operations
      </div>
      
      {quantumProperties && (
        <div style={{ 
          marginBottom: '8px',
          fontSize: '10px',
          lineHeight: '1.3'
        }}>
          {quantumProperties.superposition && (
            <div style={{ color: '#B383FF' }}>⚡ Superposition Active</div>
          )}
          {quantumProperties.entangledWith.length > 0 && (
            <div style={{ color: '#FF6B9D' }}>
              🔗 Entangled with {quantumProperties.entangledWith.length} node(s)
            </div>
          )}
          <div style={{ opacity: 0.7 }}>
            Coherence: {((quantumProperties.coherenceLevel || 0) * 100).toFixed(0)}%
          </div>
        </div>
      )}
      
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '10px',
        opacity: 0.6
      }}>
        <div>
          Status: {node.active ? 'Quantum Active' : 'Decoherent'}
        </div>
        <button
          onClick={onToggleDetails}
          style={{
            background: 'rgba(179, 131, 255, 0.2)',
            border: '1px solid rgba(179, 131, 255, 0.3)',
            color: '#B383FF',
            padding: '2px 6px',
            borderRadius: '3px',
            fontSize: '9px',
            cursor: 'pointer'
          }}
        >
          {showDetails ? 'Less' : 'More'}
        </button>
      </div>
      
      {showDetails && (
        <div style={{ 
          marginTop: '8px',
          padding: '8px',
          background: 'rgba(179, 131, 255, 0.1)',
          borderRadius: '4px',
          fontSize: '10px',
          lineHeight: '1.3'
        }}>
          <div><strong>Node ID:</strong> {node.id}</div>
          <div><strong>Quantum State:</strong> {quantumProperties?.quantumState || 'Unknown'}</div>
          {quantumProperties?.entangledWith.map((partnerId, index) => (
            <div key={partnerId}>
              <strong>Entangled #{index + 1}:</strong> {partnerId}
            </div>
          ))}
          {node.selected && <div style={{ color: '#B383FF' }}><strong>Selected</strong></div>}
          {node.highlighted && <div style={{ color: '#FFFF00' }}><strong>Connected</strong></div>}
        </div>
      )}
    </div>
  );
}

function BasicTooltip({ node }: { node: NetworkNode }) {
  return (
    <div>
      <div style={{ 
        color: '#7BE1FF', 
        fontWeight: 'bold',
        marginBottom: '4px'
      }}>
        {node.label || `Node ${node.id}`}
      </div>
      <div style={{ fontSize: '10px', opacity: 0.7 }}>
        Type: {node.type}
      </div>
      <div style={{ fontSize: '10px', opacity: 0.7 }}>
        Status: {node.active ? 'Active' : 'Inactive'}
      </div>
    </div>
  );
}
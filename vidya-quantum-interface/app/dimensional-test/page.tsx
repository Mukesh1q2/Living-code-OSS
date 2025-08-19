"use client";

import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import DimensionalControls from '@/components/DimensionalControls';
import DimensionalModes from '@/components/DimensionalModes';
import DimensionalTransition from '@/components/DimensionalTransition';
import { useDimensionalState } from '@/lib/dimensional-state';
import { useQuantumState } from '@/lib/state';
import * as THREE from 'three';

export default function DimensionalTestPage() {
  const { currentState, activeTransition } = useDimensionalState();
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  // Mock consciousness data for testing
  const mockConsciousness = {
    level: 3.5,
    quantumCoherence: 0.8,
    personalityTraits: {
      curiosity: 0.8,
      wisdom: 0.7,
      playfulness: 0.6,
    },
  };

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      background: 'linear-gradient(135deg, #05070B 0%, #0A0E1A 100%)',
      position: 'relative',
    }}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 5], fov: 75 }}
        style={{ width: '100%', height: '100%' }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        
        <DimensionalModes
          vidyaPosition={new THREE.Vector3(0, 0, 0)}
        />
        
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
      
      {/* UI Overlay */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        pointerEvents: 'none',
        zIndex: 10,
      }}>
        {/* Dimensional Controls */}
        <div style={{ 
          position: 'absolute', 
          top: '20px', 
          right: '20px',
          pointerEvents: 'auto',
        }}>
          <DimensionalControls showLabels={true} />
        </div>
        
        {/* Status Display */}
        <div style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          background: 'rgba(10, 10, 10, 0.8)',
          border: '1px solid rgba(123, 225, 255, 0.2)',
          borderRadius: '12px',
          padding: '16px',
          color: '#E8F6FF',
          fontFamily: 'system-ui, sans-serif',
          fontSize: '14px',
          backdropFilter: 'blur(10px)',
          pointerEvents: 'auto',
        }}>
          <h3 style={{ margin: '0 0 12px 0', color: '#7BE1FF' }}>
            Dimensional State Test
          </h3>
          
          <div style={{ marginBottom: '8px' }}>
            <strong>Current State:</strong> {currentState}
          </div>
          
          <div style={{ marginBottom: '8px' }}>
            <strong>Quantum Quality:</strong> {quantumQuality}
          </div>
          
          {activeTransition && (
            <div style={{ marginBottom: '8px' }}>
              <strong>Transition:</strong> {activeTransition.fromState} → {activeTransition.toState}
              <div style={{ 
                width: '100px', 
                height: '4px', 
                background: 'rgba(123, 225, 255, 0.2)', 
                borderRadius: '2px',
                marginTop: '4px',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${activeTransition.progress * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #7BE1FF, #63FFC9)',
                  borderRadius: '2px',
                  transition: 'width 0.1s ease',
                }} />
              </div>
            </div>
          )}
          
          <div style={{ marginBottom: '8px' }}>
            <strong>Consciousness Level:</strong> {mockConsciousness.level.toFixed(1)}
          </div>
          
          <div>
            <strong>Quantum Coherence:</strong> {(mockConsciousness.quantumCoherence * 100).toFixed(0)}%
          </div>
        </div>
        
        {/* Instructions */}
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          background: 'rgba(10, 10, 10, 0.8)',
          border: '1px solid rgba(99, 255, 201, 0.2)',
          borderRadius: '12px',
          padding: '16px',
          color: '#E8F6FF',
          fontFamily: 'system-ui, sans-serif',
          fontSize: '12px',
          backdropFilter: 'blur(10px)',
          maxWidth: '300px',
          pointerEvents: 'auto',
        }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#63FFC9' }}>
            Keyboard Shortcuts
          </h4>
          <div>Alt + 1: 2D Text Mode</div>
          <div>Alt + 2: 3D Holographic Mode</div>
          <div>Alt + 3: Energy Pattern Mode</div>
          <div>Alt + O: Auto-optimize</div>
          <div style={{ marginTop: '8px', opacity: 0.7 }}>
            Use mouse to orbit around Vidya
          </div>
        </div>
      </div>
    </div>
  );
}
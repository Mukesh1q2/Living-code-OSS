"use client";

import React from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import QuantumNetwork from "./QuantumNetwork";
import Vidya from "./Vidya";
import EntangledInstances from "./EntangledInstances";
import SuperpositionOverlay from "./SuperpositionOverlay";
import QuantumField from "./QuantumField";
import QuantumParticleSystem from "./QuantumParticleSystem";
import QuantumShaderManager from "./QuantumShaderManager";
// import QuantumSuperposition from "./QuantumSuperposition";
// import QuantumMeasurement from "./QuantumMeasurement";
import { useResponsive } from "@/lib/responsive";
import { useQuantumState } from "@/lib/state";

interface QuantumCanvasProps {
  sanskritEnabled?: boolean;
}

export default function QuantumCanvas({ sanskritEnabled = true }: QuantumCanvasProps) {
  const { isMobile, isTablet, width, height } = useResponsive();
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  // Responsive camera settings
  const getCameraConfig = () => {
    if (isMobile) {
      return { position: [6, 10, 18] as [number, number, number], fov: 65 };
    }
    if (isTablet) {
      return { position: [5, 9, 16] as [number, number, number], fov: 60 };
    }
    return { position: [4, 8, 15] as [number, number, number], fov: 55 };
  };

  // Responsive interaction sensitivity
  const getInteractionSensitivity = () => {
    if (isMobile) return { x: 0.04, y: 0.02 }; // Reduced for touch
    if (isTablet) return { x: 0.06, y: 0.03 };
    return { x: 0.08, y: 0.04 }; // Full desktop sensitivity
  };

  const cameraConfig = getCameraConfig();
  const sensitivity = getInteractionSensitivity();

  // Handle quantum-specific events from touch gestures
  React.useEffect(() => {
    const handleQuantumCollapse = () => {
      // Trigger quantum collapse from keyboard or touch
      document.dispatchEvent(new CustomEvent('quantum-superposition-collapse'));
    };

    const handleNeuralNetworkNavigate = (event: CustomEvent) => {
      const { direction, velocity } = event.detail;
      // Navigate through neural network based on gesture
      const cam = (window as any).__r3f?.root?.store.getState().camera;
      if (cam) {
        const moveAmount = 2;
        switch (direction) {
          case 'up':
            cam.position.y += moveAmount;
            break;
          case 'down':
            cam.position.y -= moveAmount;
            break;
          case 'left':
            cam.position.x -= moveAmount;
            break;
          case 'right':
            cam.position.x += moveAmount;
            break;
        }
      }
    };

    const handleDimensionalShift = (event: CustomEvent) => {
      const { scale } = event.detail;
      // Adjust camera FOV based on dimensional shift
      const cam = (window as any).__r3f?.root?.store.getState().camera;
      if (cam) {
        cam.fov = Math.max(30, Math.min(120, cam.fov * scale));
        cam.updateProjectionMatrix();
      }
    };

    document.addEventListener('quantum-collapse-keyboard', handleQuantumCollapse);
    document.addEventListener('neural-network-navigate', handleNeuralNetworkNavigate as EventListener);
    document.addEventListener('dimensional-shift', handleDimensionalShift as EventListener);

    return () => {
      document.removeEventListener('quantum-collapse-keyboard', handleQuantumCollapse);
      document.removeEventListener('neural-network-navigate', handleNeuralNetworkNavigate as EventListener);
      document.removeEventListener('dimensional-shift', handleDimensionalShift as EventListener);
    };
  }, []);

  return (
    <div 
      style={{ position: "fixed", inset: 0, zIndex: 0 }}
      onPointerMove={(e) => {
        // Skip interaction on minimal quality or very small screens
        if (quantumQuality === 'minimal' || width < 400) return;
        
        const el = e.currentTarget as HTMLDivElement;
        const x = (e.clientX / el.clientWidth) - 0.5;
        const y = (e.clientY / el.clientHeight) - 0.5;
        const cam = (window as any).__r3f?.root?.store.getState().camera;
        if (cam) {
          cam.rotation.y = THREE.MathUtils.lerp(cam.rotation.y, x * sensitivity.x, 0.08);
          cam.rotation.x = THREE.MathUtils.lerp(cam.rotation.x, -y * sensitivity.y, 0.08);
        }
      }}
      // Enhanced touch gesture support
      onTouchMove={(e) => {
        // Only handle single-finger camera movement if not handled by gesture system
        if (e.touches.length !== 1) return;
        
        const touch = e.touches[0];
        const el = e.currentTarget as HTMLDivElement;
        const x = (touch.clientX / el.clientWidth) - 0.5;
        const y = (touch.clientY / el.clientHeight) - 0.5;
        const cam = (window as any).__r3f?.root?.store.getState().camera;
        if (cam) {
          const dampening = isMobile ? 0.3 : 0.5; // More dampening on mobile
          cam.rotation.y = THREE.MathUtils.lerp(cam.rotation.y, x * sensitivity.x * dampening, 0.1);
          cam.rotation.x = THREE.MathUtils.lerp(cam.rotation.x, -y * sensitivity.y * dampening, 0.1);
        }
      }}
      // Accessibility support
      role="img"
      aria-label="Interactive quantum neural network visualization"
      tabIndex={0}
      onKeyDown={(e) => {
        // Keyboard navigation for accessibility
        const cam = (window as any).__r3f?.root?.store.getState().camera;
        if (!cam) return;

        const moveAmount = 1;
        switch (e.key) {
          case 'ArrowUp':
            e.preventDefault();
            cam.position.y += moveAmount;
            break;
          case 'ArrowDown':
            e.preventDefault();
            cam.position.y -= moveAmount;
            break;
          case 'ArrowLeft':
            e.preventDefault();
            cam.position.x -= moveAmount;
            break;
          case 'ArrowRight':
            e.preventDefault();
            cam.position.x += moveAmount;
            break;
          case '+':
          case '=':
            e.preventDefault();
            cam.position.z = Math.max(5, cam.position.z - 2);
            break;
          case '-':
            e.preventDefault();
            cam.position.z = Math.min(50, cam.position.z + 2);
            break;
        }
      }}
    >
      <Canvas 
        camera={cameraConfig}
        dpr={quantumQuality === 'high' ? [1, 2] : quantumQuality === 'medium' ? [0.75, 1.5] : [0.5, 1]} 
        performance={{ 
          min: isMobile ? 0.2 : isTablet ? 0.4 : 0.5,
          max: quantumQuality === 'high' ? 1 : quantumQuality === 'medium' ? 0.8 : 0.6
        }}
        gl={{
          antialias: quantumQuality !== 'minimal',
          alpha: true,
          powerPreference: isMobile ? 'low-power' : 'high-performance',
          failIfMajorPerformanceCaveat: false,
        }}
        frameloop={quantumQuality === 'minimal' ? 'demand' : 'always'}
      >
        <color attach="background" args={["#05070B"]} />
        
        {/* Responsive lighting based on quality */}
        <ambientLight intensity={quantumQuality === 'minimal' ? 0.6 : 0.4} />
        {quantumQuality !== 'minimal' && (
          <>
            <pointLight position={[10, 10, 10]} intensity={0.8} color={"#7BE1FF"} />
            <pointLight position={[-5, 5, -5]} intensity={0.4} color={"#B383FF"} />
          </>
        )}
        
        {/* Quantum shader management and effects */}
        <QuantumShaderManager>
          <QuantumNetwork sanskritEnabled={sanskritEnabled} />
          <Vidya />
          {quantumQuality !== 'minimal' && <EntangledInstances />}
          
          {/* Advanced quantum field effects */}
          <QuantumField />
          {quantumQuality !== 'minimal' && <QuantumParticleSystem />}
        </QuantumShaderManager>
      </Canvas>
      
      {/* Overlay components */}
      <SuperpositionOverlay />
    </div>
  );
}
import React, { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Text, Sphere } from '@react-three/drei'
import * as THREE from 'three'

interface VidyaConsciousnessProps {
  isConnected: boolean
}

const VidyaConsciousness: React.FC<VidyaConsciousnessProps> = ({ isConnected }) => {
  const meshRef = useRef<THREE.Mesh>(null)
  const textRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  const [clicked, setClicked] = useState(false)

  // Animation loop
  useFrame((state, delta) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.2
      
      // Rotation based on connection status
      if (isConnected) {
        meshRef.current.rotation.y += delta * 0.5
        meshRef.current.rotation.x += delta * 0.2
      } else {
        meshRef.current.rotation.y += delta * 0.1
      }
    }
    
    if (textRef.current) {
      // Om symbol gentle rotation
      textRef.current.rotation.z += delta * 0.3
    }
  })

  return (
    <group position={[0, 0, 0]}>
      {/* Core consciousness sphere */}
      <Sphere
        ref={meshRef}
        args={[1, 32, 32]}
        scale={clicked ? 1.2 : hovered ? 1.1 : 1}
        onClick={() => setClicked(!clicked)}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={isConnected ? "#4a90e2" : "#666666"}
          transparent
          opacity={0.7}
          emissive={isConnected ? "#1a4480" : "#333333"}
          emissiveIntensity={hovered ? 0.5 : 0.2}
        />
      </Sphere>
      
      {/* Om symbol at the center */}
      <Text
        ref={textRef}
        position={[0, 0, 0.1]}
        fontSize={1.5}
        color={isConnected ? "#ffffff" : "#888888"}
        anchorX="center"
        anchorY="middle"
      >
        ॐ
      </Text>
      
      {/* Status indicator */}
      <Text
        position={[0, -2, 0]}
        fontSize={0.3}
        color={isConnected ? "#00ff00" : "#ff6666"}
        anchorX="center"
        anchorY="middle"
      >
        {isConnected ? "Vidya Consciousness Active" : "Connecting to Vidya..."}
      </Text>
      
      {/* Quantum field particles */}
      {isConnected && (
        <group>
          {Array.from({ length: 20 }, (_, i) => (
            <Sphere
              key={i}
              args={[0.02, 8, 8]}
              position={[
                Math.sin(i * 0.5) * 3,
                Math.cos(i * 0.3) * 2,
                Math.sin(i * 0.7) * 2
              ]}
            >
              <meshBasicMaterial
                color="#4a90e2"
                transparent
                opacity={0.6}
              />
            </Sphere>
          ))}
        </group>
      )}
    </group>
  )
}

export default VidyaConsciousness
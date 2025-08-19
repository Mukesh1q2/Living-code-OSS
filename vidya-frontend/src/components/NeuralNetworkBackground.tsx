import React, { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line, Sphere } from '@react-three/drei'
import * as THREE from 'three'
import { NetworkNode, NetworkConnection } from '../types/shared'

const NeuralNetworkBackground: React.FC = () => {
  const groupRef = useRef<THREE.Group>(null)
  
  // Generate neural network nodes
  const nodes = useMemo<NetworkNode[]>(() => {
    const nodeArray: NetworkNode[] = []
    
    for (let i = 0; i < 50; i++) {
      nodeArray.push({
        id: `node-${i}`,
        position: {
          x: (Math.random() - 0.5) * 20,
          y: (Math.random() - 0.5) * 20,
          z: (Math.random() - 0.5) * 20
        },
        type: Math.random() > 0.7 ? 'sanskrit-rule' : 'neural-unit',
        activationLevel: Math.random(),
        quantumProperties: {
          superpositionStates: [],
          entanglementPartners: [],
          coherenceTime: Math.random() * 1000
        }
      })
    }
    
    return nodeArray
  }, [])
  
  // Generate connections between nearby nodes
  const connections = useMemo<NetworkConnection[]>(() => {
    const connectionArray: NetworkConnection[] = []
    
    nodes.forEach((node, i) => {
      // Connect to 2-4 nearby nodes
      const nearbyNodes = nodes
        .filter((otherNode, j) => {
          if (i === j) return false
          const distance = Math.sqrt(
            Math.pow(node.position.x - otherNode.position.x, 2) +
            Math.pow(node.position.y - otherNode.position.y, 2) +
            Math.pow(node.position.z - otherNode.position.z, 2)
          )
          return distance < 8
        })
        .slice(0, Math.floor(Math.random() * 3) + 2)
      
      nearbyNodes.forEach(nearbyNode => {
        connectionArray.push({
          id: `connection-${node.id}-${nearbyNode.id}`,
          fromNodeId: node.id,
          toNodeId: nearbyNode.id,
          strength: Math.random(),
          isQuantumEntangled: Math.random() > 0.8
        })
      })
    })
    
    return connectionArray
  }, [nodes])

  // Animation
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05
      groupRef.current.rotation.x += delta * 0.02
    }
  })

  return (
    <group ref={groupRef} position={[0, 0, -5]}>
      {/* Render nodes */}
      {nodes.map((node) => (
        <Sphere
          key={node.id}
          args={[0.1, 8, 8]}
          position={[node.position.x, node.position.y, node.position.z]}
        >
          <meshBasicMaterial
            color={node.type === 'sanskrit-rule' ? '#ff6b35' : '#4a90e2'}
            transparent
            opacity={0.3 + node.activationLevel * 0.4}
          />
        </Sphere>
      ))}
      
      {/* Render connections */}
      {connections.map((connection) => {
        const fromNode = nodes.find(n => n.id === connection.fromNodeId)
        const toNode = nodes.find(n => n.id === connection.toNodeId)
        
        if (!fromNode || !toNode) return null
        
        return (
          <Line
            key={connection.id}
            points={[
              [fromNode.position.x, fromNode.position.y, fromNode.position.z],
              [toNode.position.x, toNode.position.y, toNode.position.z]
            ]}
            color={connection.isQuantumEntangled ? '#ff00ff' : '#ffffff'}
            lineWidth={connection.strength * 2}
            transparent
            opacity={0.2 + connection.strength * 0.3}
          />
        )
      })}
    </group>
  )
}

export default NeuralNetworkBackground
import { Suspense, useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stats } from '@react-three/drei'
import VidyaConsciousness from './components/VidyaConsciousness'
import NeuralNetworkBackground from './components/NeuralNetworkBackground'
import WebSocketManager from './services/WebSocketManager'
import { PerformanceMetrics } from './types/shared'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [performanceMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    memoryUsage: 0,
    renderTime: 0,
    networkLatency: 0
  })
  const [isDevelopmentMode] = useState(import.meta.env.MODE === 'development')

  useEffect(() => {
    // Initialize WebSocket connection
    const wsManager = new WebSocketManager('ws://localhost:8000/ws')
    
    wsManager.onConnect = () => {
      setIsConnected(true)
      console.log('Connected to Vidya backend')
    }
    
    wsManager.onDisconnect = () => {
      setIsConnected(false)
      console.log('Disconnected from Vidya backend')
    }
    
    wsManager.connect()
    
    return () => {
      wsManager.disconnect()
    }
  }, [])

  return (
    <div className="quantum-interface">
      {isDevelopmentMode && (
        <div className="dev-info">
          <div>Connection: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}</div>
          <div>FPS: {performanceMetrics.fps.toFixed(1)}</div>
          <div>Memory: {performanceMetrics.memoryUsage.toFixed(1)}MB</div>
          <div>Latency: {performanceMetrics.networkLatency.toFixed(0)}ms</div>
        </div>
      )}
      
      <Suspense fallback={<div className="loading">Initializing Vidya Consciousness...</div>}>
        <Canvas
          className="neural-network-background"
          camera={{ position: [0, 0, 10], fov: 75 }}
          gl={{ antialias: true, alpha: true }}
          onCreated={({ gl }) => {
            gl.setClearColor('#0f0f23', 0)
          }}
        >
          {isDevelopmentMode && <Stats />}
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
          
          {/* Lighting setup */}
          <ambientLight intensity={0.3} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          <pointLight position={[-10, -10, -10]} intensity={0.4} color="#4a90e2" />
          
          {/* Neural Network Background */}
          <NeuralNetworkBackground />
          
          {/* Vidya Consciousness Core */}
          <VidyaConsciousness isConnected={isConnected} />
        </Canvas>
      </Suspense>
    </div>
  )
}

export default App
"use client";

import { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { usePerformanceOptimization, useAdaptiveParticles, useAdaptiveShaders } from '@/lib/usePerformanceOptimization';
import * as THREE from 'three';

// Example component that demonstrates performance optimization
export default function PerformanceExample() {
    const performance = usePerformanceOptimization({
        enableAutoQualityAdjustment: true,
        enableMemoryManagement: true,
        enableProfiling: true,
        targetFPS: 60,
        maxMemoryMB: 128,
    });

    const [showStats, setShowStats] = useState(false);

    return (
        <div style={{ width: '100%', height: '400px', position: 'relative' }}>
            {/* Performance Stats Overlay */}
            {showStats && (
                <div style={{
                    position: 'absolute',
                    top: 10,
                    left: 10,
                    zIndex: 1000,
                    background: 'rgba(0,0,0,0.8)',
                    color: '#7BE1FF',
                    padding: 12,
                    borderRadius: 8,
                    fontSize: '12px',
                    fontFamily: 'monospace',
                }}>
                    <div>FPS: {performance.metrics?.fps || 0}</div>
                    <div>Memory: {performance.memoryUsage?.estimatedMB.toFixed(1) || 0}MB</div>
                    <div>Quality: {performance.currentQuality}</div>
                    <div>Adaptations: {performance.adaptationCount}</div>
                    {performance.isOptimizing && <div style={{ color: '#63FFC9' }}>⚡ Optimizing...</div>}
                </div>
            )}

            {/* Controls */}
            <div style={{
                position: 'absolute',
                top: 10,
                right: 10,
                zIndex: 1000,
                display: 'flex',
                gap: 8,
            }}>
                <button
                    onClick={() => setShowStats(!showStats)}
                    style={{
                        background: 'rgba(123,225,255,0.2)',
                        border: '1px solid rgba(123,225,255,0.3)',
                        color: '#7BE1FF',
                        borderRadius: 4,
                        padding: '4px 8px',
                        cursor: 'pointer',
                        fontSize: '10px',
                    }}
                >
                    {showStats ? 'Hide Stats' : 'Show Stats'}
                </button>

                <button
                    onClick={() => performance.cleanupMemory()}
                    style={{
                        background: 'rgba(255,123,213,0.2)',
                        border: '1px solid rgba(255,123,213,0.3)',
                        color: '#FF7BD5',
                        borderRadius: 4,
                        padding: '4px 8px',
                        cursor: 'pointer',
                        fontSize: '10px',
                    }}
                >
                    Cleanup Memory
                </button>

                <button
                    onClick={() => performance.exportPerformanceData()}
                    style={{
                        background: 'rgba(99,255,201,0.2)',
                        border: '1px solid rgba(99,255,201,0.3)',
                        color: '#63FFC9',
                        borderRadius: 4,
                        padding: '4px 8px',
                        cursor: 'pointer',
                        fontSize: '10px',
                    }}
                >
                    Export Data
                </button>
            </div>

            {/* 3D Scene */}
            <Canvas>
                <AdaptiveScene performance={performance} />
            </Canvas>
        </div>
    );
}

// Adaptive 3D scene that responds to performance optimization
function AdaptiveScene({ performance }: { performance: any }) {
    const particleCount = useAdaptiveParticles(1000);
    const shaderComplexity = useAdaptiveShaders();

    useEffect(() => {
        // Start profiling the scene render
        performance.startProfile('Scene Render');

        return () => {
            performance.endProfile();
        };
    }, [performance]);

    return (
        <>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />

            {/* Adaptive particle system */}
            <AdaptiveParticles count={particleCount} complexity={shaderComplexity} performance={performance} />

            {/* Adaptive geometry */}
            <AdaptiveGeometry complexity={shaderComplexity} performance={performance} />
        </>
    );
}

// Particle system that adapts to performance
function AdaptiveParticles({
    count,
    complexity,
    performance
}: {
    count: number;
    complexity: string;
    performance: any;
}) {
    const [particles, setParticles] = useState<THREE.Vector3[]>([]);

    useEffect(() => {
        performance.profile('Generate Particles', () => {
            const newParticles = [];
            for (let i = 0; i < count; i++) {
                newParticles.push(new THREE.Vector3(
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 10
                ));
            }
            setParticles(newParticles);
        });
    }, [count, performance]);

    // Register particles for memory management
    useEffect(() => {
        const ids: string[] = [];
        particles.forEach(particle => {
            const id = performance.registerObject(particle, 'object');
            ids.push(id);
        });

        return () => {
            ids.forEach(id => performance.disposeObject(id));
        };
    }, [particles, performance]);

    return (
        <group>
            {particles.map((position, index) => (
                <mesh key={index} position={position}>
                    <sphereGeometry args={[0.05, complexity === 'high' ? 16 : complexity === 'medium' ? 8 : 4]} />
                    <meshBasicMaterial
                        color={complexity === 'high' ? '#7BE1FF' : complexity === 'medium' ? '#B383FF' : '#63FFC9'}
                    />
                </mesh>
            ))}
        </group>
    );
}

// Geometry that adapts complexity based on performance
function AdaptiveGeometry({
    complexity,
    performance
}: {
    complexity: string;
    performance: any;
}) {
    const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

    useEffect(() => {
        performance.profile('Create Geometry', () => {
            let newGeometry: THREE.BufferGeometry;

            switch (complexity) {
                case 'high':
                    newGeometry = new THREE.IcosahedronGeometry(2, 3);
                    break;
                case 'medium':
                    newGeometry = new THREE.IcosahedronGeometry(2, 2);
                    break;
                case 'low':
                    newGeometry = new THREE.IcosahedronGeometry(2, 1);
                    break;
                default:
                    newGeometry = new THREE.BoxGeometry(2, 2, 2);
            }

            setGeometry(newGeometry);
        });
    }, [complexity, performance]);

    // Register geometry for memory management
    useEffect(() => {
        if (geometry) {
            const id = performance.registerObject(geometry, 'geometry');
            return () => performance.disposeObject(id);
        }
    }, [geometry, performance]);

    if (!geometry) return null;

    return (
        <mesh geometry={geometry}>
            <meshStandardMaterial
                color="#7BE1FF"
                wireframe={complexity === 'minimal'}
            />
        </mesh>
    );
}
"use client";

import { useMemo, useRef } from "react";
import { useQuantumState } from "@/lib/state";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";

function EntangledInstance({
  offset = [0, 0, 0],
  color = "#ff7bf2",
}: {
  offset?: [number, number, number];
  color?: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  useFrame((_, dt) => {
    if (!meshRef.current) return;
    meshRef.current.rotation.y += dt * 0.6;
  });
  return (
    <mesh ref={meshRef} position={offset}>
      <icosahedronGeometry args={[0.6, 2]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        metalness={0.2}
        roughness={0.35}
      />
    </mesh>
  );
}

export default function EntangledInstances() {
  const count = useQuantumState((s) => s.vidyaInstances);
  const instances = useMemo(
    () => Array.from({ length: Math.max(0, count - 1) }),
    [count]
  );

  return (
    <group>
      {instances.map((_, i) => {
        const angle = (i / Math.max(1, instances.length)) * Math.PI * 2;
        const r = 3.6;
        return (
          <EntangledInstance
            key={i}
            offset={[Math.cos(angle) * r, 0.5, Math.sin(angle) * r]}
          />
        );
      })}
    </group>
  );
}
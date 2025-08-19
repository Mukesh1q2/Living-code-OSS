"use client";

import { useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface ThreeInspectorProps {
  visible: boolean;
  onClose: () => void;
}

interface SceneObject {
  id: string;
  name: string;
  type: string;
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
  children: number;
}

export default function ThreeInspector({ visible, onClose }: ThreeInspectorProps) {
  const { scene, camera, gl } = useThree();
  const [sceneObjects, setSceneObjects] = useState<SceneObject[]>([]);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!visible) return;

    const updateSceneObjects = () => {
      const objects: SceneObject[] = [];
      
      scene.traverse((object) => {
        objects.push({
          id: object.uuid,
          name: object.name || object.type,
          type: object.type,
          position: [object.position.x, object.position.y, object.position.z],
          rotation: [object.rotation.x, object.rotation.y, object.rotation.z],
          scale: [object.scale.x, object.scale.y, object.scale.z],
          visible: object.visible,
          children: object.children.length,
        });
      });

      setSceneObjects(objects);
    };

    updateSceneObjects();
    
    // Auto-refresh every 2 seconds when visible
    const interval = setInterval(updateSceneObjects, 2000);
    return () => clearInterval(interval);
  }, [scene, visible, refreshKey]);

  const getObjectFromScene = (id: string): THREE.Object3D | null => {
    let foundObject: THREE.Object3D | null = null;
    scene.traverse((object) => {
      if (object.uuid === id) {
        foundObject = object;
      }
    });
    return foundObject;
  };

  const handleObjectClick = (objectId: string) => {
    setSelectedObject(objectId === selectedObject ? null : objectId);
    
    // Highlight object in scene
    const object = getObjectFromScene(objectId);
    if (object && object instanceof THREE.Mesh) {
      // Add a temporary wireframe overlay
      const wireframe = new THREE.WireframeGeometry(object.geometry);
      const line = new THREE.LineSegments(wireframe);
      line.material = new THREE.LineBasicMaterial({ color: 0xff0000 });
      line.position.copy(object.position);
      line.rotation.copy(object.rotation);
      line.scale.copy(object.scale);
      line.name = 'inspector-highlight';
      
      // Remove existing highlights
      scene.children.forEach(child => {
        if (child.name === 'inspector-highlight') {
          scene.remove(child);
        }
      });
      
      scene.add(line);
      
      // Remove highlight after 3 seconds
      setTimeout(() => {
        scene.remove(line);
      }, 3000);
    }
  };

  if (!visible) return null;

  const selectedObj = selectedObject ? sceneObjects.find(obj => obj.id === selectedObject) : null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 120,
        right: 16,
        width: 350,
        height: 400,
        zIndex: 9999,
        background: 'rgba(0,0,0,0.95)',
        border: '1px solid rgba(123,225,255,0.3)',
        borderRadius: 8,
        color: '#E8F6FF',
        fontSize: '12px',
        fontFamily: 'monospace',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        padding: 12,
        borderBottom: '1px solid rgba(123,225,255,0.2)'
      }}>
        <h3 style={{ margin: 0, color: '#7BE1FF' }}>Three.js Inspector</h3>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={() => setRefreshKey(prev => prev + 1)}
            style={{
              background: 'rgba(123,225,255,0.1)',
              border: '1px solid rgba(123,225,255,0.3)',
              color: '#7BE1FF',
              borderRadius: 4,
              padding: '2px 6px',
              cursor: 'pointer',
              fontSize: '10px',
            }}
          >
            Refresh
          </button>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: '1px solid rgba(123,225,255,0.3)',
              color: '#7BE1FF',
              borderRadius: 4,
              padding: '2px 6px',
              cursor: 'pointer',
            }}
          >
            ×
          </button>
        </div>
      </div>

      {/* Scene Info */}
      <div style={{ padding: 12, borderBottom: '1px solid rgba(123,225,255,0.1)' }}>
        <div>Objects: <span style={{ color: '#63FFC9' }}>{sceneObjects.length}</span></div>
        <div>Renderer: <span style={{ color: '#63FFC9' }}>{gl.info.render.triangles} triangles</span></div>
        <div>Memory: <span style={{ color: '#63FFC9' }}>{gl.info.memory.geometries}G / {gl.info.memory.textures}T</span></div>
      </div>

      {/* Object List */}
      <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
        {sceneObjects.map((obj) => (
          <div
            key={obj.id}
            onClick={() => handleObjectClick(obj.id)}
            style={{
              padding: '4px 8px',
              margin: '2px 0',
              borderRadius: 4,
              cursor: 'pointer',
              background: selectedObject === obj.id ? 'rgba(123,225,255,0.2)' : 'transparent',
              border: selectedObject === obj.id ? '1px solid rgba(123,225,255,0.4)' : '1px solid transparent',
            }}
          >
            <div style={{ fontWeight: 'bold', color: obj.visible ? '#E8F6FF' : '#8FB2C8' }}>
              {obj.name} ({obj.type})
            </div>
            <div style={{ fontSize: '10px', color: '#8FB2C8' }}>
              Children: {obj.children}
            </div>
          </div>
        ))}
      </div>

      {/* Selected Object Details */}
      {selectedObj && (
        <div style={{ 
          padding: 12, 
          borderTop: '1px solid rgba(123,225,255,0.2)',
          background: 'rgba(123,225,255,0.05)'
        }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#B383FF' }}>Selected: {selectedObj.name}</h4>
          <div style={{ fontSize: '10px', lineHeight: 1.4 }}>
            <div>Position: ({selectedObj.position.map(v => v.toFixed(2)).join(', ')})</div>
            <div>Rotation: ({selectedObj.rotation.map(v => v.toFixed(2)).join(', ')})</div>
            <div>Scale: ({selectedObj.scale.map(v => v.toFixed(2)).join(', ')})</div>
            <div>Visible: <span style={{ color: selectedObj.visible ? '#63FFC9' : '#FF7BD5' }}>
              {selectedObj.visible ? 'Yes' : 'No'}
            </span></div>
          </div>
        </div>
      )}
    </div>
  );
}
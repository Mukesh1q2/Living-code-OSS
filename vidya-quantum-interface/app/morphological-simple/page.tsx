"use client";

import { Suspense, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import SanskritMorphologicalAnalyzer from "@/components/SanskritMorphologicalAnalyzer";
import WebSocketProvider from "@/components/WebSocketProvider";

interface MorphologicalData {
  word: string;
  root: string;
  suffixes: string[];
  prefixes: string[];
  morphemes: any[];
  etymologicalConnections: any[];
  paniniRules: any[];
  grammaticalRelationships: any[];
}

export default function SimpleMorphologicalTestPage() {
  const [inputText, setInputText] = useState("गच्छति");
  const [analysisData, setAnalysisData] = useState<MorphologicalData | null>(null);

  const handleAnalysisUpdate = (data: MorphologicalData) => {
    setAnalysisData(data);
    console.log("Morphological analysis updated:", data);
  };

  const testWords = [
    "गच्छति",      // gacchati - he/she goes
    "करोति",       // karoti - he/she does
    "रामस्य",      // ramasya - of Rama
  ];

  return (
    <WebSocketProvider>
      <div className="min-h-screen bg-black">
        {/* Simple Header */}
        <div className="p-4 bg-gray-900 text-white">
          <h1 className="text-2xl font-bold mb-4">
            Sanskrit Morphological Analysis Test
          </h1>

          {/* Simple Controls */}
          <div className="flex gap-4 items-center">
            <label className="text-white">Text:</label>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white"
              placeholder="Enter Sanskrit text..."
            />
            
            <select
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white"
            >
              {testWords.map((word) => (
                <option key={word} value={word}>
                  {word}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* 3D Visualization */}
        <div className="h-screen">
          <Canvas
            camera={{ position: [0, 0, 10], fov: 75 }}
            style={{ height: "80vh" }}
          >
            <Suspense fallback={null}>
              {/* Basic Lighting */}
              <ambientLight intensity={0.4} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4a90e2" />

              {/* Sanskrit Morphological Analyzer */}
              <SanskritMorphologicalAnalyzer
                text={inputText}
                onAnalysisUpdate={handleAnalysisUpdate}
                enableRealTimeUpdates={true}
                showEtymologicalConnections={true}
                showPaniniRules={true}
                showMorphemeFlow={true}
                interactionEnabled={true}
              />

              {/* Camera Controls */}
              <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                minDistance={5}
                maxDistance={50}
              />
            </Suspense>
          </Canvas>

          {/* Simple Results Display */}
          {analysisData && (
            <div className="absolute top-20 right-4 w-80 bg-gray-900 bg-opacity-90 rounded p-4 text-white max-h-96 overflow-y-auto">
              <h3 className="text-lg font-bold mb-3">Analysis Results</h3>
              
              <div className="space-y-2 text-sm">
                <div>
                  <strong>Word:</strong> {analysisData.word}
                </div>
                <div>
                  <strong>Root:</strong> {analysisData.root}
                </div>
                {analysisData.suffixes.length > 0 && (
                  <div>
                    <strong>Suffixes:</strong> {analysisData.suffixes.join(", ")}
                  </div>
                )}
                {analysisData.prefixes.length > 0 && (
                  <div>
                    <strong>Prefixes:</strong> {analysisData.prefixes.join(", ")}
                  </div>
                )}
                <div>
                  <strong>Morphemes:</strong> {analysisData.morphemes.length}
                </div>
                <div>
                  <strong>Etymology:</strong> {analysisData.etymologicalConnections.length} connections
                </div>
                <div>
                  <strong>Pāṇini Rules:</strong> {analysisData.paniniRules.length} applied
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </WebSocketProvider>
  );
}
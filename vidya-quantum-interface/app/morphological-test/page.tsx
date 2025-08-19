"use client";

import { Suspense, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stats } from "@react-three/drei";
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

export default function MorphologicalTestPage() {
  const [inputText, setInputText] = useState("गच्छति");
  const [analysisData, setAnalysisData] = useState<MorphologicalData | null>(null);
  const [showEtymology, setShowEtymology] = useState(true);
  const [showPaniniRules, setShowPaniniRules] = useState(true);
  const [showMorphemeFlow, setShowMorphemeFlow] = useState(true);
  const [interactionEnabled, setInteractionEnabled] = useState(true);

  const handleAnalysisUpdate = (data: MorphologicalData) => {
    setAnalysisData(data);
    console.log("Morphological analysis updated:", data);
  };

  const testWords = [
    "गच्छति",      // gacchati - he/she goes
    "करोति",       // karoti - he/she does
    "रामस्य",      // ramasya - of Rama
    "देवपुत्र",     // devaputra - son of god
    "प्रगच्छति",    // pragacchati - he/she goes forward
    "महाराज",      // maharaja - great king
  ];

  return (
    <WebSocketProvider>
      <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
        {/* Header */}
        <div className="p-6 bg-black/20 backdrop-blur-sm">
          <h1 className="text-3xl font-bold text-white mb-4">
            Sanskrit Morphological Analysis Visualization
          </h1>
          <p className="text-gray-300 mb-6">
            Real-time visualization of Sanskrit word decomposition, etymological connections, 
            and Pāṇini rule applications with interactive exploration.
          </p>

          {/* Controls */}
          <div className="flex flex-wrap gap-4 items-center">
            {/* Text Input */}
            <div className="flex items-center gap-2">
              <label className="text-white font-medium">Sanskrit Text:</label>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter Sanskrit text..."
              />
            </div>

            {/* Test Words */}
            <div className="flex items-center gap-2">
              <label className="text-white font-medium">Quick Test:</label>
              <select
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {testWords.map((word) => (
                  <option key={word} value={word} className="bg-gray-800">
                    {word}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="flex flex-wrap gap-4 mt-4">
            <label className="flex items-center gap-2 text-white">
              <input
                type="checkbox"
                checked={showEtymology}
                onChange={(e) => setShowEtymology(e.target.checked)}
                className="rounded"
              />
              Etymological Connections
            </label>
            <label className="flex items-center gap-2 text-white">
              <input
                type="checkbox"
                checked={showPaniniRules}
                onChange={(e) => setShowPaniniRules(e.target.checked)}
                className="rounded"
              />
              Pāṇini Rule Mandalas
            </label>
            <label className="flex items-center gap-2 text-white">
              <input
                type="checkbox"
                checked={showMorphemeFlow}
                onChange={(e) => setShowMorphemeFlow(e.target.checked)}
                className="rounded"
              />
              Morpheme Flow Animation
            </label>
            <label className="flex items-center gap-2 text-white">
              <input
                type="checkbox"
                checked={interactionEnabled}
                onChange={(e) => setInteractionEnabled(e.target.checked)}
                className="rounded"
              />
              Interactive Exploration
            </label>
          </div>
        </div>

        {/* Main Visualization */}
        <div className="flex-1 relative">
          <Canvas
            camera={{ position: [0, 0, 10], fov: 75 }}
            style={{ height: "70vh" }}
          >
            <Suspense fallback={null}>
              {/* Lighting */}
              <ambientLight intensity={0.4} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4a90e2" />

              {/* Sanskrit Morphological Analyzer */}
              <SanskritMorphologicalAnalyzer
                text={inputText}
                onAnalysisUpdate={handleAnalysisUpdate}
                enableRealTimeUpdates={true}
                showEtymologicalConnections={showEtymology}
                showPaniniRules={showPaniniRules}
                showMorphemeFlow={showMorphemeFlow}
                interactionEnabled={interactionEnabled}
              />

              {/* Controls */}
              <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                minDistance={5}
                maxDistance={50}
              />

              {/* Performance Stats */}
              <Stats />
            </Suspense>
          </Canvas>

          {/* Analysis Results Panel */}
          {analysisData && (
            <div className="absolute top-4 right-4 w-80 bg-black/80 backdrop-blur-sm rounded-lg p-4 text-white max-h-96 overflow-y-auto">
              <h3 className="text-lg font-bold mb-3">Analysis Results</h3>
              
              <div className="space-y-3">
                <div>
                  <h4 className="font-semibold text-blue-300">Word:</h4>
                  <p className="text-sm">{analysisData.word}</p>
                </div>

                <div>
                  <h4 className="font-semibold text-green-300">Root:</h4>
                  <p className="text-sm">{analysisData.root}</p>
                </div>

                {analysisData.suffixes.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-yellow-300">Suffixes:</h4>
                    <p className="text-sm">{analysisData.suffixes.join(", ")}</p>
                  </div>
                )}

                {analysisData.prefixes.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-purple-300">Prefixes:</h4>
                    <p className="text-sm">{analysisData.prefixes.join(", ")}</p>
                  </div>
                )}

                <div>
                  <h4 className="font-semibold text-pink-300">Morphemes:</h4>
                  <div className="text-sm space-y-1">
                    {analysisData.morphemes.map((morpheme, index) => (
                      <div key={morpheme.id} className="flex justify-between">
                        <span>{morpheme.text}</span>
                        <span className="text-gray-400">({morpheme.type})</span>
                      </div>
                    ))}
                  </div>
                </div>

                {analysisData.etymologicalConnections.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-cyan-300">Etymology:</h4>
                    <p className="text-sm">{analysisData.etymologicalConnections.length} connections</p>
                  </div>
                )}

                {analysisData.paniniRules.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-orange-300">Pāṇini Rules:</h4>
                    <div className="text-sm space-y-1">
                      {analysisData.paniniRules.map((rule, index) => (
                        <div key={rule.id}>
                          {rule.ruleNumber}: {rule.ruleName}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="p-6 bg-black/20 backdrop-blur-sm">
          <h3 className="text-xl font-bold text-white mb-3">Instructions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm text-gray-300">
            <div>
              <h4 className="font-semibold text-blue-300 mb-2">Morpheme Visualization</h4>
              <p>Colored spheres represent different morphemes. Click to select and explore details.</p>
            </div>
            <div>
              <h4 className="font-semibold text-green-300 mb-2">Etymological Connections</h4>
              <p>Lines connect morphemes with historical relationships. Thickness indicates strength.</p>
            </div>
            <div>
              <h4 className="font-semibold text-yellow-300 mb-2">Pāṇini Rule Mandalas</h4>
              <p>Geometric mandalas appear when Sanskrit grammar rules are applied to the text.</p>
            </div>
            <div>
              <h4 className="font-semibold text-purple-300 mb-2">Interactive Controls</h4>
              <p>Use mouse to orbit, zoom, and pan. Hover over elements for additional information.</p>
            </div>
          </div>
        </div>
      </div>
    </WebSocketProvider>
  );
}
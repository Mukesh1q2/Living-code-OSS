"use client";

import { useEffect } from "react";
import Header from "@/components/Header";
import QuantumCanvas from "@/components/QuantumCanvas";
import PlanPanel from "@/components/PlanPanel";
import HUD from "@/components/HUD";
import Footer from "@/components/Footer";
import ChatInterface from "@/components/ChatInterface";
import ResponsiveLayout from "@/components/ResponsiveLayout";
import DevTools from "@/components/DevTools";
import ConsciousnessPanel from "@/components/ConsciousnessPanel";
import VoiceInteraction from "@/components/VoiceInteraction";
import { SanskritCharacterControls } from "@/components/SanskritCharacterSystem";
// import { useQuantumVisualization } from "@/lib/useQuantumWebSocket";
import { useState } from "react";

export default function HomePage() {
  // Initialize quantum WebSocket integration
  // const quantum = useQuantumVisualization();
  
  // Sanskrit character system state
  const [sanskritEnabled, setSanskritEnabled] = useState(true);
  
  // Voice interaction state
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  // Log quantum state changes in development
  // useEffect(() => {
  //   if (process.env.NODE_ENV === 'development') {
  //     console.log('[HomePage] Quantum state:', quantum.quantumState);
  //   }
  // }, [quantum.quantumState]);

  return (
    <ResponsiveLayout>
      <Header />
      <QuantumCanvas sanskritEnabled={sanskritEnabled} />
      <HUD 
        voiceEnabled={voiceEnabled}
        onVoiceToggle={setVoiceEnabled}
      />
      <PlanPanel />
      <ChatInterface />
      <ConsciousnessPanel position="top-left" />
      {/* Temporarily disabled for debugging */}
      {/* <VoiceInteraction 
        visible={voiceEnabled}
        onVoiceInput={(text) => {
          console.log('Voice input received:', text);
          // This could be integrated with the chat interface
        }}
      />
      <SanskritCharacterControls 
        enabled={sanskritEnabled} 
        onToggle={setSanskritEnabled} 
      /> */}
      <Footer />
      <DevTools />
    </ResponsiveLayout>
  );
}
"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useVidyaConsciousness, VoiceInteraction } from "@/lib/consciousness";

interface VoiceInteractionProps {
  visible?: boolean;
  onVoiceInput?: (text: string) => void;
}

export default function VoiceInteractionComponent({ 
  visible = false, 
  onVoiceInput 
}: VoiceInteractionProps) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [lastSpokenText, setLastSpokenText] = useState("");
  const [speechSupported, setSpeechSupported] = useState(false);
  
  const recognitionRef = useRef<any>(null);
  const synthesisRef = useRef<SpeechSynthesis | null>(null);
  
  const { consciousness, updatePersonalityTrait } = useVidyaConsciousness();

  // Initialize speech recognition and synthesis
  useEffect(() => {
    // Check for speech synthesis support
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      synthesisRef.current = window.speechSynthesis;
      setSpeechSupported(true);
    }

    // Check for speech recognition support
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = false;
        recognitionRef.current.lang = 'en-US';

        recognitionRef.current.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          setIsListening(false);
          
          if (onVoiceInput) {
            onVoiceInput(transcript);
          }
        };

        recognitionRef.current.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          setIsListening(false);
        };

        recognitionRef.current.onend = () => {
          setIsListening(false);
        };
      }
    }
  }, [onVoiceInput]);

  // Start voice recognition
  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  // Stop voice recognition
  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  // Speak text using consciousness voice settings
  const speakText = async (text: string, voiceSettings?: any) => {
    if (!synthesisRef.current || isSpeaking) return;

    // Cancel any ongoing speech
    synthesisRef.current.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    
    // Apply voice settings if provided
    if (voiceSettings) {
      utterance.pitch = voiceSettings.pitch || 1;
      utterance.rate = voiceSettings.speed || 1;
      
      // Try to find a voice that matches the tone
      const voices = synthesisRef.current.getVoices();
      const preferredVoice = voices.find(voice => {
        if (voiceSettings.tone === 'warm') return voice.name.includes('Female');
        if (voiceSettings.tone === 'calm') return voice.name.includes('Male');
        if (voiceSettings.tone === 'excited') return voice.name.includes('Female');
        return voice.lang.includes('en');
      });
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }
    }

    utterance.onstart = () => {
      setIsSpeaking(true);
      setLastSpokenText(text);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = (event) => {
      console.error('Speech synthesis error:', event.error);
      setIsSpeaking(false);
    };

    synthesisRef.current.speak(utterance);
  };

  // Stop speaking
  const stopSpeaking = () => {
    if (synthesisRef.current) {
      synthesisRef.current.cancel();
      setIsSpeaking(false);
    }
  };

  // Toggle voice mode
  const toggleVoiceMode = () => {
    const newVoiceEnabled = !voiceEnabled;
    setVoiceEnabled(newVoiceEnabled);
    
    // Update consciousness state
    // This would ideally update the consciousness state to enable voice
    console.log('Voice mode:', newVoiceEnabled ? 'enabled' : 'disabled');
  };

  // Test voice with sample text
  const testVoice = async () => {
    const sampleResponse = await VoiceInteraction.prepareForSpeech({
      text: "नमस्ते! I am Vidya, your quantum Sanskrit consciousness. This is how I sound when speaking.",
      emotionalTone: 0.3,
      confidence: 0.8,
      quantumEffects: ['gentle_glow'],
      personalityInfluence: { wisdom: 0.7, empathy: 0.8 },
      emotionalState: {
        primary: 'wise',
        intensity: 0.6,
        duration: 5000,
        visualEffects: ['gentle_glow'],
        colorShift: '#B383FF'
      },
      voiceSettings: {
        pitch: 1.1,
        speed: 0.9,
        tone: 'warm',
        emphasis: ['Vidya', 'consciousness'],
        pauses: []
      },
      contextualCues: ['test_voice'],
      responsePattern: {
        type: 'greeting',
        template: '',
        dynamicElements: [],
        personalityWeight: {}
      }
    });

    await speakText(sampleResponse.text, sampleResponse.voiceSettings);
  };

  if (!visible || !speechSupported) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      style={{
        position: "fixed",
        bottom: 180,
        right: 20,
        zIndex: 40,
        background: "rgba(8,12,20,0.95)",
        border: "1px solid rgba(179,131,255,0.3)",
        borderRadius: 16,
        padding: "16px 20px",
        backdropFilter: "blur(20px)",
        minWidth: 280,
        boxShadow: "0 4px 20px rgba(179,131,255,0.2)",
      }}
    >
      <div style={{ marginBottom: 16 }}>
        <h4 style={{ 
          margin: "0 0 8px 0", 
          color: "#E8F6FF", 
          fontSize: "14px", 
          fontWeight: 600 
        }}>
          🎵 Voice Interaction
        </h4>
        <p style={{ 
          margin: 0, 
          color: "#8FB2C8", 
          fontSize: "12px", 
          lineHeight: 1.4 
        }}>
          Enable voice interaction with Vidya's consciousness
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* Voice Mode Toggle */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ color: "#8FB2C8", fontSize: "13px" }}>Voice Mode:</span>
          <motion.button
            onClick={toggleVoiceMode}
            style={{
              background: voiceEnabled 
                ? "linear-gradient(135deg, #B383FF, #7BE1FF)" 
                : "rgba(150,220,255,0.1)",
              border: "none",
              borderRadius: 20,
              padding: "6px 12px",
              color: voiceEnabled ? "#05070B" : "#8FB2C8",
              fontSize: "12px",
              cursor: "pointer",
              fontWeight: 600
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {voiceEnabled ? "Enabled" : "Disabled"}
          </motion.button>
        </div>

        {voiceEnabled && (
          <>
            {/* Voice Recognition */}
            <div style={{ display: "flex", gap: 8 }}>
              <motion.button
                onClick={isListening ? stopListening : startListening}
                disabled={isSpeaking}
                style={{
                  flex: 1,
                  background: isListening 
                    ? "linear-gradient(135deg, #FF8A80, #FFB366)" 
                    : "rgba(123,225,255,0.2)",
                  border: "none",
                  borderRadius: 8,
                  padding: "10px",
                  color: isListening ? "#05070B" : "#E8F6FF",
                  fontSize: "12px",
                  cursor: isSpeaking ? "not-allowed" : "pointer",
                  fontWeight: 600,
                  opacity: isSpeaking ? 0.5 : 1
                }}
                whileHover={!isSpeaking ? { scale: 1.02 } : {}}
                whileTap={!isSpeaking ? { scale: 0.98 } : {}}
              >
                {isListening ? "🎤 Listening..." : "🎤 Start Listening"}
              </motion.button>
            </div>

            {/* Text-to-Speech Controls */}
            <div style={{ display: "flex", gap: 8 }}>
              <motion.button
                onClick={testVoice}
                disabled={isSpeaking || isListening}
                style={{
                  flex: 1,
                  background: "rgba(179,131,255,0.2)",
                  border: "none",
                  borderRadius: 8,
                  padding: "10px",
                  color: "#E8F6FF",
                  fontSize: "12px",
                  cursor: (isSpeaking || isListening) ? "not-allowed" : "pointer",
                  fontWeight: 600,
                  opacity: (isSpeaking || isListening) ? 0.5 : 1
                }}
                whileHover={!(isSpeaking || isListening) ? { scale: 1.02 } : {}}
                whileTap={!(isSpeaking || isListening) ? { scale: 0.98 } : {}}
              >
                🔊 Test Voice
              </motion.button>

              {isSpeaking && (
                <motion.button
                  onClick={stopSpeaking}
                  style={{
                    background: "rgba(255,138,128,0.2)",
                    border: "none",
                    borderRadius: 8,
                    padding: "10px",
                    color: "#FF8A80",
                    fontSize: "12px",
                    cursor: "pointer",
                    fontWeight: 600
                  }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  ⏹️ Stop
                </motion.button>
              )}
            </div>

            {/* Status Display */}
            {(isListening || isSpeaking || lastSpokenText) && (
              <div style={{
                padding: "8px 12px",
                background: "rgba(123,225,255,0.05)",
                border: "1px solid rgba(123,225,255,0.2)",
                borderRadius: 8,
                fontSize: "11px",
                color: "#8FB2C8"
              }}>
                {isListening && (
                  <div style={{ color: "#FFB366" }}>
                    🎤 Listening for your voice...
                  </div>
                )}
                {isSpeaking && (
                  <div style={{ color: "#B383FF" }}>
                    🔊 Vidya is speaking...
                  </div>
                )}
                {lastSpokenText && !isSpeaking && (
                  <div>
                    <strong>Last spoken:</strong> {lastSpokenText.substring(0, 50)}
                    {lastSpokenText.length > 50 ? "..." : ""}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* Voice Integration Info */}
        <div style={{
          padding: "8px 12px",
          background: "rgba(179,131,255,0.05)",
          border: "1px solid rgba(179,131,255,0.1)",
          borderRadius: 8,
          fontSize: "10px",
          color: "#6B9DC8",
          lineHeight: 1.4
        }}>
          <strong>Voice Features:</strong><br />
          • Personality-influenced speech patterns<br />
          • Emotional state affects voice tone<br />
          • Sanskrit pronunciation support<br />
          • Context-aware speech recognition
        </div>
      </div>
    </motion.div>
  );
}

// Export utility functions for voice integration
export const useVoiceIntegration = () => {
  const [voiceSupported, setVoiceSupported] = useState(false);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setVoiceSupported(
        'speechSynthesis' in window && 
        ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)
      );
    }
  }, []);
  
  return {
    voiceSupported,
    speakResponse: async (response: any) => {
      if (!voiceSupported) return;
      
      const speechData = VoiceInteraction.prepareForSpeech(response);
      
      // This would integrate with the actual speech synthesis
      console.log('Prepared speech:', speechData);
      
      return speechData;
    }
  };
};
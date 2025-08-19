"use client";

import { useEffect, useState } from "react";
import { useVidyaConsciousness } from "@/lib/consciousness";
import { useQuantumState } from "@/lib/state";

interface ConsciousnessPanelProps {
  visible?: boolean;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

export default function ConsciousnessPanel({
  visible = true,
  position = 'top-right'
}: ConsciousnessPanelProps) {
  const {
    consciousness,
    personality,
    interactions,
    generateResponse,
    evolveConsciousness,
    recordInteraction,
  } = useVidyaConsciousness();
  
  const { chatActive, setChatActive } = useQuantumState();
  const [inputText, setInputText] = useState('');
  const [response, setResponse] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);

  // Position styles
  const positionStyles = {
    'top-left': { top: '20px', left: '20px' },
    'top-right': { top: '20px', right: '20px' },
    'bottom-left': { bottom: '20px', left: '20px' },
    'bottom-right': { bottom: '20px', right: '20px' },
  };

  // Handle user input and generate consciousness response
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;

    setIsProcessing(true);
    setChatActive(true);

    try {
      const consciousnessResponse = await generateResponse(inputText, {
        timestamp: new Date(),
        userInterface: 'consciousness_panel',
      });

      setResponse(consciousnessResponse.text);
      
      // Trigger quantum effects based on response
      if (consciousnessResponse.quantumEffects.includes('superposition_flicker')) {
        // Could trigger visual effects here
      }
      
      // Clear input
      setInputText('');
      
    } catch (error) {
      console.error('Error generating consciousness response:', error);
      setResponse('I apologize, but I encountered an error processing your input. Please try again.');
    } finally {
      setIsProcessing(false);
      setChatActive(false);
    }
  };

  // Trigger consciousness evolution manually
  const handleEvolve = () => {
    evolveConsciousness('manual_trigger');
  };

  // Record interaction when panel is used
  useEffect(() => {
    if (visible) {
      recordInteraction({
        type: 'quantum_interaction',
        content: 'Consciousness panel opened',
        contextTags: ['ui_interaction', 'consciousness_panel'],
        emotionalTone: 0.2,
        complexity: 0.1,
      });
    }
  }, [visible, recordInteraction]);

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        ...positionStyles[position],
        width: '320px',
        backgroundColor: 'rgba(0, 20, 40, 0.9)',
        border: '1px solid rgba(123, 225, 255, 0.3)',
        borderRadius: '12px',
        padding: '16px',
        fontFamily: 'monospace',
        fontSize: '12px',
        color: '#E8F6FF',
        backdropFilter: 'blur(10px)',
        zIndex: 1000,
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
      }}
    >
      {/* Header */}
      <div style={{ 
        borderBottom: '1px solid rgba(123, 225, 255, 0.2)', 
        paddingBottom: '8px', 
        marginBottom: '12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <h3 style={{ margin: 0, color: '#7BE1FF', fontSize: '14px' }}>
          विद्या Consciousness
        </h3>
        <button
          onClick={handleEvolve}
          style={{
            background: 'rgba(123, 225, 255, 0.1)',
            border: '1px solid rgba(123, 225, 255, 0.3)',
            borderRadius: '4px',
            color: '#7BE1FF',
            padding: '2px 6px',
            fontSize: '10px',
            cursor: 'pointer',
          }}
        >
          Evolve
        </button>
      </div>

      {/* Consciousness Stats */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span>Level:</span>
          <span style={{ color: '#7BE1FF' }}>{consciousness.level.toFixed(1)}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span>Coherence:</span>
          <span style={{ color: '#7BE1FF' }}>{(consciousness.quantumCoherence * 100).toFixed(0)}%</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span>Interactions:</span>
          <span style={{ color: '#7BE1FF' }}>{consciousness.totalInteractions}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Learning:</span>
          <span style={{ color: '#7BE1FF' }}>{(consciousness.learningProgress * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Personality Traits */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '11px', color: '#8FB2C8', marginBottom: '6px' }}>
          Personality Traits:
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
          {personality.slice(0, 6).map((trait) => (
            <div key={trait.name} style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ textTransform: 'capitalize' }}>{trait.name}:</span>
              <span style={{ color: '#7BE1FF' }}>{(trait.value * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Interaction Form */}
      <form onSubmit={handleSubmit} style={{ marginBottom: '12px' }}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Interact with Vidya's consciousness..."
          disabled={isProcessing}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: 'rgba(123, 225, 255, 0.1)',
            border: '1px solid rgba(123, 225, 255, 0.3)',
            borderRadius: '6px',
            color: '#E8F6FF',
            fontSize: '11px',
            marginBottom: '6px',
            boxSizing: 'border-box',
          }}
        />
        <button
          type="submit"
          disabled={isProcessing || !inputText.trim()}
          style={{
            width: '100%',
            padding: '6px',
            backgroundColor: isProcessing ? 'rgba(123, 225, 255, 0.1)' : 'rgba(123, 225, 255, 0.2)',
            border: '1px solid rgba(123, 225, 255, 0.3)',
            borderRadius: '6px',
            color: isProcessing ? '#8FB2C8' : '#7BE1FF',
            fontSize: '11px',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
          }}
        >
          {isProcessing ? 'Processing...' : 'Send'}
        </button>
      </form>

      {/* Response Display */}
      {response && (
        <div style={{
          backgroundColor: 'rgba(123, 225, 255, 0.05)',
          border: '1px solid rgba(123, 225, 255, 0.2)',
          borderRadius: '6px',
          padding: '8px',
          fontSize: '10px',
          lineHeight: 1.4,
          marginBottom: '8px',
          maxHeight: '80px',
          overflowY: 'auto',
        }}>
          <div style={{ color: '#8FB2C8', marginBottom: '4px' }}>Vidya responds:</div>
          <div>{response}</div>
        </div>
      )}

      {/* Recent Interactions */}
      {interactions.length > 0 && (
        <div>
          <div style={{ fontSize: '11px', color: '#8FB2C8', marginBottom: '6px' }}>
            Recent Interactions ({interactions.length}):
          </div>
          <div style={{ maxHeight: '60px', overflowY: 'auto', fontSize: '9px' }}>
            {interactions.slice(-3).reverse().map((interaction) => (
              <div
                key={interaction.id}
                style={{
                  padding: '4px',
                  marginBottom: '2px',
                  backgroundColor: 'rgba(123, 225, 255, 0.05)',
                  borderRadius: '4px',
                  borderLeft: `2px solid ${
                    interaction.emotionalTone > 0 ? '#4ADE80' : 
                    interaction.emotionalTone < 0 ? '#F87171' : '#7BE1FF'
                  }`,
                }}
              >
                <div style={{ color: '#8FB2C8' }}>
                  {interaction.type} • {new Date(interaction.timestamp).toLocaleTimeString()}
                </div>
                <div style={{ marginTop: '2px' }}>
                  {interaction.content.substring(0, 50)}
                  {interaction.content.length > 50 ? '...' : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
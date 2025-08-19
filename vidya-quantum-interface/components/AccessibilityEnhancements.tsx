"use client";

import React, { useEffect, useState, useRef } from 'react';
import { useAccessibility } from '@/lib/accessibility';
import { useResponsive } from '@/lib/responsive';
import { useQuantumState } from '@/lib/state';

interface AccessibilityEnhancementsProps {
  children: React.ReactNode;
}

export default function AccessibilityEnhancements({ children }: AccessibilityEnhancementsProps) {
  const { settings, updateSettings, announce, describeQuantumState, provideFeedback } = useAccessibility();
  const { isMobile, isTablet } = useResponsive();
  const { quantumQuality, superposition, entanglements } = useQuantumState();
  
  const [isKeyboardNavigating, setIsKeyboardNavigating] = useState(false);
  const [currentFocus, setCurrentFocus] = useState<string | null>(null);
  const [voiceRecognition, setVoiceRecognition] = useState<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const lastAnnouncementRef = useRef<string>('');

  // Initialize voice recognition if available
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      
      recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.toLowerCase();
        handleVoiceCommand(command);
      };
      
      recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        setIsListening(false);
      };
      
      recognition.onend = () => {
        setIsListening(false);
      };
      
      setVoiceRecognition(recognition);
    }
  }, []);

  // Handle voice commands
  const handleVoiceCommand = (command: string) => {
    announce(`Voice command received: ${command}`);
    
    if (command.includes('collapse') || command.includes('measure')) {
      document.dispatchEvent(new CustomEvent('quantum-collapse-voice'));
      describeQuantumState('collapse');
    } else if (command.includes('entangle') || command.includes('connect')) {
      document.dispatchEvent(new CustomEvent('quantum-entangle-voice'));
      describeQuantumState('entanglement');
    } else if (command.includes('teleport') || command.includes('move')) {
      document.dispatchEvent(new CustomEvent('quantum-teleport-voice'));
      describeQuantumState('teleportation');
    } else if (command.includes('help') || command.includes('guide')) {
      announce('Available voice commands: collapse, entangle, teleport, high contrast, reduce motion, increase font size');
    } else if (command.includes('high contrast')) {
      updateSettings({ highContrastMode: !settings.highContrastMode });
    } else if (command.includes('reduce motion')) {
      updateSettings({ reducedMotion: !settings.reducedMotion });
    } else if (command.includes('increase font') || command.includes('larger text')) {
      const sizes = ['small', 'medium', 'large', 'extra-large'] as const;
      const currentIndex = sizes.indexOf(settings.fontSize);
      if (currentIndex < sizes.length - 1) {
        updateSettings({ fontSize: sizes[currentIndex + 1] });
      }
    } else if (command.includes('decrease font') || command.includes('smaller text')) {
      const sizes = ['small', 'medium', 'large', 'extra-large'] as const;
      const currentIndex = sizes.indexOf(settings.fontSize);
      if (currentIndex > 0) {
        updateSettings({ fontSize: sizes[currentIndex - 1] });
      }
    }
  };

  // Keyboard navigation handler
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      setIsKeyboardNavigating(true);
      
      // Voice activation shortcut
      if (event.altKey && event.key === 'v') {
        event.preventDefault();
        if (voiceRecognition && !isListening) {
          setIsListening(true);
          voiceRecognition.start();
          announce('Voice recognition activated. Speak your command.');
          provideFeedback('medium');
        }
      }
      
      // Quick accessibility toggles
      if (event.altKey && event.key === 'c') {
        event.preventDefault();
        updateSettings({ highContrastMode: !settings.highContrastMode });
        announce(`High contrast ${settings.highContrastMode ? 'disabled' : 'enabled'}`);
        provideFeedback('light');
      }
      
      if (event.altKey && event.key === 'm') {
        event.preventDefault();
        updateSettings({ reducedMotion: !settings.reducedMotion });
        announce(`Motion ${settings.reducedMotion ? 'enabled' : 'reduced'}`);
        provideFeedback('light');
      }
      
      // Quantum state descriptions
      if (event.key === 'F1') {
        event.preventDefault();
        announceCurrentQuantumState();
        provideFeedback('light');
      }
    };

    const handleMouseDown = () => {
      setIsKeyboardNavigating(false);
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, [settings, voiceRecognition, isListening, announce, updateSettings, provideFeedback]);

  // Announce current quantum state
  const announceCurrentQuantumState = () => {
    let stateDescription = `Current quantum state: Quality level ${quantumQuality}`;
    
    if (superposition) {
      stateDescription += ', Vidya is in superposition with multiple simultaneous states';
    }
    
    if (entanglements.length > 0) {
      stateDescription += `, ${entanglements.length} quantum entanglements active`;
    }
    
    announce(stateDescription);
  };

  // Monitor quantum state changes and announce them
  useEffect(() => {
    const handleQuantumStateChange = () => {
      if (settings.voiceAnnouncements) {
        announceCurrentQuantumState();
      }
    };

    // Listen for quantum state changes
    document.addEventListener('quantum-state-change', handleQuantumStateChange);
    
    return () => {
      document.removeEventListener('quantum-state-change', handleQuantumStateChange);
    };
  }, [settings.voiceAnnouncements, quantumQuality, superposition, entanglements]);

  // Focus management for quantum elements
  useEffect(() => {
    const handleFocus = (event: FocusEvent) => {
      const target = event.target as HTMLElement;
      if (target.dataset.quantumElement) {
        setCurrentFocus(target.id || target.dataset.quantumElement);
        
        if (settings.voiceAnnouncements) {
          const elementType = target.dataset.quantumElement;
          const elementName = target.getAttribute('aria-label') || target.textContent || elementType;
          announce(`Focused on ${elementName}`);
        }
        
        if (settings.hapticFeedback) {
          provideFeedback('light');
        }
      }
    };

    document.addEventListener('focusin', handleFocus);
    
    return () => {
      document.removeEventListener('focusin', handleFocus);
    };
  }, [settings, announce, provideFeedback]);

  // Screen reader optimizations
  const getAriaLiveRegionContent = () => {
    if (!settings.voiceAnnouncements) return '';
    
    let content = '';
    
    if (superposition) {
      content += 'Vidya consciousness is in quantum superposition. ';
    }
    
    if (entanglements.length > 0) {
      content += `${entanglements.length} quantum entanglements are active. `;
    }
    
    return content;
  };

  // Generate accessibility shortcuts help
  const getAccessibilityHelp = () => {
    return [
      'Alt+C: Toggle high contrast mode',
      'Alt+M: Toggle reduced motion',
      'Alt+V: Activate voice commands',
      'F1: Describe current quantum state',
      'Tab: Navigate through quantum elements',
      'Space: Collapse quantum superposition',
      'Enter: Activate focused element',
      'Escape: Exit current quantum state',
    ].join(', ');
  };

  return (
    <div
      ref={containerRef}
      className={`accessibility-enhancements ${isKeyboardNavigating ? 'keyboard-navigation' : ''}`}
      data-accessibility-enabled={settings.screenReaderEnabled}
      data-high-contrast={settings.highContrastMode}
      data-reduced-motion={settings.reducedMotion}
      data-font-size={settings.fontSize}
    >
      {/* Skip navigation link */}
      <a 
        href="#main-content" 
        className="skip-link"
        onFocus={() => announce('Skip to main content link focused')}
      >
        Skip to main content
      </a>

      {/* Live region for dynamic announcements */}
      <div
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        id="quantum-live-region"
      >
        {getAriaLiveRegionContent()}
      </div>

      {/* Assertive live region for urgent announcements */}
      <div
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
        id="quantum-urgent-region"
      />

      {/* Hidden accessibility help */}
      <div className="sr-only" id="accessibility-help">
        Vidya Quantum Sanskrit AI Interface. Use keyboard navigation with Tab key. 
        Available shortcuts: {getAccessibilityHelp()}
      </div>

      {/* Voice recognition indicator */}
      {isListening && (
        <div 
          className="voice-indicator"
          role="status"
          aria-live="polite"
        >
          <div className="voice-indicator-icon">🎤</div>
          <div className="voice-indicator-text">Listening for voice commands...</div>
        </div>
      )}

      {/* Accessibility settings panel (hidden by default) */}
      <div 
        className="accessibility-panel"
        role="dialog"
        aria-labelledby="accessibility-panel-title"
        aria-describedby="accessibility-panel-description"
        hidden={!isKeyboardNavigating}
      >
        <h2 id="accessibility-panel-title" className="sr-only">
          Accessibility Settings
        </h2>
        <p id="accessibility-panel-description" className="sr-only">
          Quick accessibility options for the quantum interface
        </p>
        
        <button
          onClick={() => updateSettings({ highContrastMode: !settings.highContrastMode })}
          aria-pressed={settings.highContrastMode}
          className="accessibility-toggle"
        >
          High Contrast: {settings.highContrastMode ? 'On' : 'Off'}
        </button>
        
        <button
          onClick={() => updateSettings({ reducedMotion: !settings.reducedMotion })}
          aria-pressed={settings.reducedMotion}
          className="accessibility-toggle"
        >
          Reduced Motion: {settings.reducedMotion ? 'On' : 'Off'}
        </button>
        
        <button
          onClick={() => updateSettings({ voiceAnnouncements: !settings.voiceAnnouncements })}
          aria-pressed={settings.voiceAnnouncements}
          className="accessibility-toggle"
        >
          Voice Announcements: {settings.voiceAnnouncements ? 'On' : 'Off'}
        </button>
      </div>

      {/* Main content with accessibility enhancements */}
      <main 
        id="main-content"
        role="main"
        aria-describedby="accessibility-help"
        tabIndex={-1}
      >
        {children}
      </main>

      <style jsx>{`
        .accessibility-enhancements {
          position: relative;
          width: 100%;
          height: 100%;
        }

        .skip-link {
          position: absolute;
          top: -40px;
          left: 6px;
          background: var(--quantum-color-background, #000);
          color: var(--quantum-color-text, #fff);
          padding: 8px;
          text-decoration: none;
          border-radius: 4px;
          z-index: 10000;
          transition: top 0.3s;
          border: 2px solid var(--quantum-color-accent, #7BE1FF);
        }

        .skip-link:focus {
          top: 6px;
        }

        .voice-indicator {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: rgba(0, 0, 0, 0.9);
          color: #7BE1FF;
          padding: 20px;
          border-radius: 12px;
          border: 2px solid #7BE1FF;
          display: flex;
          align-items: center;
          gap: 12px;
          z-index: 10000;
          box-shadow: 0 4px 20px rgba(123, 225, 255, 0.3);
        }

        .voice-indicator-icon {
          font-size: 24px;
          animation: pulse 1.5s infinite;
        }

        .voice-indicator-text {
          font-size: 16px;
          font-weight: 500;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }

        .accessibility-panel {
          position: fixed;
          top: 20px;
          right: 20px;
          background: rgba(0, 0, 0, 0.95);
          border: 2px solid var(--quantum-color-accent, #7BE1FF);
          border-radius: 8px;
          padding: 16px;
          z-index: 9999;
          display: flex;
          flex-direction: column;
          gap: 8px;
          min-width: 200px;
        }

        .accessibility-toggle {
          background: rgba(123, 225, 255, 0.1);
          border: 1px solid rgba(123, 225, 255, 0.3);
          color: #7BE1FF;
          padding: 8px 12px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          transition: all 0.2s;
        }

        .accessibility-toggle:hover,
        .accessibility-toggle:focus {
          background: rgba(123, 225, 255, 0.2);
          border-color: rgba(123, 225, 255, 0.5);
        }

        .accessibility-toggle[aria-pressed="true"] {
          background: rgba(123, 225, 255, 0.3);
          border-color: #7BE1FF;
        }

        .sr-only {
          position: absolute !important;
          width: 1px !important;
          height: 1px !important;
          padding: 0 !important;
          margin: -1px !important;
          overflow: hidden !important;
          clip: rect(0, 0, 0, 0) !important;
          white-space: nowrap !important;
          border: 0 !important;
        }

        /* Enhanced focus indicators for keyboard navigation */
        .keyboard-navigation :global(*:focus) {
          outline: 3px solid var(--quantum-color-accent, #7BE1FF) !important;
          outline-offset: 2px !important;
          box-shadow: 0 0 0 6px rgba(123, 225, 255, 0.3) !important;
        }

        .keyboard-navigation :global([data-quantum-element]:focus) {
          outline: 3px solid var(--quantum-color-accent, #B383FF) !important;
          outline-offset: 4px !important;
          box-shadow: 
            0 0 0 6px rgba(179, 131, 255, 0.3),
            0 0 20px rgba(179, 131, 255, 0.5) !important;
          transform: scale(1.05) !important;
          z-index: 1000 !important;
        }

        /* Mobile accessibility enhancements */
        @media (max-width: 768px) {
          .accessibility-panel {
            top: 10px;
            right: 10px;
            left: 10px;
            max-width: none;
          }

          .voice-indicator {
            padding: 16px;
            max-width: 90vw;
          }

          .voice-indicator-text {
            font-size: 14px;
          }
        }

        /* High contrast mode enhancements */
        :global(.high-contrast) .accessibility-enhancements {
          --quantum-color-accent: #ffff00;
        }

        :global(.high-contrast) .voice-indicator {
          background: #000000;
          color: #ffffff;
          border-color: #ffff00;
        }

        :global(.high-contrast) .accessibility-toggle {
          background: #000000;
          color: #ffffff;
          border-color: #ffff00;
        }

        /* Reduced motion mode */
        :global(.reduced-motion) .voice-indicator-icon {
          animation: none;
        }

        :global(.reduced-motion) .accessibility-enhancements * {
          transition: none !important;
          animation: none !important;
        }

        /* Large font size adjustments */
        :global(.font-large) .accessibility-panel,
        :global(.font-extra-large) .accessibility-panel {
          font-size: 18px;
        }

        :global(.font-large) .voice-indicator-text,
        :global(.font-extra-large) .voice-indicator-text {
          font-size: 20px;
        }
      `}</style>
    </div>
  );
}
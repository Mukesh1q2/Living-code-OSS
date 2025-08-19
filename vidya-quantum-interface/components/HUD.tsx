"use client";

import { useQuantumState } from "@/lib/state";
import { useResponsive, createResponsiveStyles } from "@/lib/responsive";
import DimensionalControls from "./DimensionalControls";

interface HUDProps {
  voiceEnabled?: boolean;
  onVoiceToggle?: (enabled: boolean) => void;
}

export default function HUD({ voiceEnabled = false, onVoiceToggle }: HUDProps) {
  const instances = useQuantumState((s) => s.vidyaInstances);
  const setInstances = useQuantumState((s) => s.setVidyaInstances);
  const superpos = useQuantumState((s) => s.superpositionActive);
  const setSuperpos = useQuantumState((s) => s.setSuperposition);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);
  
  const { breakpoint, isMobile, isTablet } = useResponsive();

  const hudStyles = createResponsiveStyles(
    {
      position: "fixed" as const,
      right: 16,
      top: 16,
      zIndex: 25,
      display: "flex",
      gap: 8,
      background: "rgba(8,12,20,0.55)",
      border: "1px solid rgba(150,220,255,0.12)",
      borderRadius: 12,
      padding: 8,
      backdropFilter: "blur(10px)",
    },
    {
      mobile: {
        right: 8,
        top: isMobile ? 60 : 16, // Account for mobile header
        gap: 6,
        padding: 6,
        borderRadius: 8,
        transform: isMobile ? "scale(0.9)" : "none",
      },
      tablet: {
        right: 12,
        gap: 7,
        padding: 7,
      },
    }
  );

  const buttonStyles = createResponsiveStyles(
    {
      pointerEvents: "auto" as const,
      color: "#E8F6FF",
      background: "transparent",
      border: "1px solid rgba(150,220,255,0.2)",
      borderRadius: 8,
      padding: "6px 8px",
      fontSize: "12px",
      cursor: "pointer",
      minWidth: "32px",
      height: "32px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    {
      mobile: {
        padding: "4px 6px",
        fontSize: "11px",
        minWidth: "28px",
        height: "28px",
        borderRadius: 6,
      },
      tablet: {
        padding: "5px 7px",
        fontSize: "11px",
        minWidth: "30px",
        height: "30px",
      },
    }
  );

  const textStyles = createResponsiveStyles(
    {
      color: "#E8F6FF",
      fontSize: "12px",
      alignSelf: "center",
      padding: "0 4px",
      minWidth: "16px",
      textAlign: "center" as const,
    },
    {
      mobile: {
        fontSize: "10px",
        padding: "0 2px",
        minWidth: "14px",
      },
      tablet: {
        fontSize: "11px",
        padding: "0 3px",
      },
    }
  );

  // Hide some controls on minimal quality or very small screens
  const showAdvancedControls = quantumQuality !== 'minimal' && !isMobile;

  return (
    <>
      {/* Main HUD Controls */}
      <div style={hudStyles(breakpoint)}>
        <button 
          onClick={() => setInstances(instances + 1)} 
          style={buttonStyles(breakpoint)} 
          title="Multiply Vidya"
          disabled={instances >= 8}
        >
          +
        </button>
        
        <span style={textStyles(breakpoint)}>
          {instances}
        </span>
        
        <button 
          onClick={() => setInstances(instances - 1)} 
          style={buttonStyles(breakpoint)} 
          title="Merge Vidya"
          disabled={instances <= 1}
        >
          −
        </button>
        
        {showAdvancedControls && (
          <button 
            onClick={() => setSuperpos(!superpos)} 
            style={{
              ...buttonStyles(breakpoint), 
              background: superpos ? "rgba(179,131,255,0.2)" : "transparent"
            }} 
            title="Toggle Superposition"
          >
            ⚛
          </button>
        )}
        
        {/* Voice Toggle Button */}
        {showAdvancedControls && onVoiceToggle && (
          <button 
            onClick={() => onVoiceToggle(!voiceEnabled)} 
            style={{
              ...buttonStyles(breakpoint), 
              background: voiceEnabled ? "rgba(179,131,255,0.2)" : "transparent"
            }} 
            title="Toggle Voice Interaction"
          >
            🎵
          </button>
        )}
        
        {/* Quality indicator for development */}
        {process.env.NODE_ENV === 'development' && (
          <div
            style={{
              ...textStyles(breakpoint),
              opacity: 0.5,
              fontSize: isMobile ? "8px" : "10px",
            }}
            title={`Quantum Quality: ${quantumQuality}`}
          >
            {quantumQuality[0].toUpperCase()}
          </div>
        )}
      </div>
      
      {/* Dimensional Controls */}
      <DimensionalControls
        style={{
          position: "fixed",
          right: isMobile ? 8 : 16,
          top: isMobile ? 110 : 60, // Below main HUD
          zIndex: 24,
        }}
        showLabels={!isMobile}
        orientation={isMobile ? "vertical" : "horizontal"}
      />
    </>
  );
}
"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useQuantumState } from "@/lib/state";

export default function SuperpositionOverlay() {
  const active = useQuantumState((s) => s.superpositionActive);
  const superpositionStates = useQuantumState((s) => s.superpositionStates);
  const waveformCollapsing = useQuantumState((s) => s.waveformCollapsing);
  const coherenceLevel = useQuantumState((s) => s.coherenceLevel);
  const decoherenceActive = useQuantumState((s) => s.decoherenceActive);
  const quantumQuality = useQuantumState((s) => s.quantumQuality);

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          style={{
            position: "fixed",
            top: 72,
            left: 16,
            zIndex: 10,
            pointerEvents: "none",
            color: "#DCD1FF",
            background: "rgba(30,20,50,0.45)",
            border: "1px solid rgba(200,160,255,0.18)",
            borderRadius: 10,
            padding: "8px 12px",
            fontSize: "12px",
            fontWeight: 500,
            minWidth: "200px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                backgroundColor: waveformCollapsing ? "#FF6B6B" : "#7BE1FF",
                animation: "pulse 1.5s infinite",
              }}
            />
            <span>
              {waveformCollapsing ? "Waveform Collapsing" : "Quantum Superposition"}
            </span>
          </div>
          
          {quantumQuality !== 'minimal' && (
            <>
              <div style={{ fontSize: "10px", color: "#B8C5D1", marginBottom: "2px" }}>
                States: {superpositionStates.length} | Coherence: {(coherenceLevel * 100).toFixed(0)}%
              </div>
              
              {decoherenceActive && (
                <div style={{ fontSize: "10px", color: "#FFB366" }}>
                  ⚠ Quantum decoherence detected
                </div>
              )}
              
              {waveformCollapsing && (
                <div style={{ fontSize: "10px", color: "#FF8A80" }}>
                  🌊 Probability wave collapsing...
                </div>
              )}
              
              {superpositionStates.length > 0 && !waveformCollapsing && (
                <div style={{ fontSize: "10px", color: "#63FFC9", marginTop: "4px" }}>
                  Click any Vidya state to collapse the waveform
                </div>
              )}
            </>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
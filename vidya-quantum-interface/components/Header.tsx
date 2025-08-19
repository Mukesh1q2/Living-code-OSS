"use client";

import { motion } from "framer-motion";
import { useResponsive, createResponsiveStyles } from "@/lib/responsive";

export default function Header() {
  const { breakpoint, isMobile, isTablet } = useResponsive();

  const headerStyles = createResponsiveStyles(
    {
      position: "fixed" as const,
      top: 0,
      left: 0,
      right: 0,
      zIndex: 20,
      padding: "16px 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      pointerEvents: "none" as const,
    },
    {
      mobile: {
        padding: "12px 16px",
        flexDirection: isMobile ? "column" : "row",
        gap: isMobile ? "8px" : "0",
      },
      tablet: {
        padding: "14px 20px",
      },
    }
  );

  const titleStyles = createResponsiveStyles(
    {
      color: "#E8F6FF",
      letterSpacing: "0.02em",
      fontWeight: 700,
      margin: 0,
      fontSize: "20px",
    },
    {
      mobile: {
        fontSize: "16px",
        textAlign: "center" as const,
      },
      tablet: {
        fontSize: "18px",
      },
    }
  );

  const navStyles = createResponsiveStyles(
    {
      display: "flex",
      gap: 16,
      pointerEvents: "auto" as const,
    },
    {
      mobile: {
        gap: 12,
        fontSize: "12px",
      },
      tablet: {
        gap: 14,
      },
    }
  );

  return (
    <header style={headerStyles(breakpoint)}>
      <motion.div
        style={{ pointerEvents: "auto" }}
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 style={titleStyles(breakpoint)}>
          {isMobile ? "Vidya AI" : "QuantumSanskrit AI — Vidya"}
        </h1>
      </motion.div>
      
      {!isMobile && (
        <nav style={navStyles(breakpoint)}>
          <a style={{ color: "#7BE1FF", fontSize: isTablet ? "13px" : "14px" }} href="#plans">
            Plans
          </a>
          <a style={{ color: "#7BE1FF", fontSize: isTablet ? "13px" : "14px" }} href="#docs">
            Docs
          </a>
          <a style={{ color: "#7BE1FF", fontSize: isTablet ? "13px" : "14px" }} href="#api">
            API
          </a>
        </nav>
      )}
    </header>
  );
}
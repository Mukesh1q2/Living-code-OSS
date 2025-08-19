"use client";

export default function Footer() {
  return (
    <footer
      style={{
        position: "fixed",
        bottom: 8,
        left: 12,
        right: 12,
        color: "#80caff",
        fontSize: 12,
        zIndex: 12,
        display: "flex",
        justifyContent: "space-between",
        opacity: 0.7,
      }}
    >
      <div>© {new Date().getFullYear()} QuantumSanskrit AI</div>
      <div>Roadmap: LLM → Reasoning → Code-gen → Math Engine → Learning</div>
    </footer>
  );
}
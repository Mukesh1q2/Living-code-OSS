"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useQuantumState } from "@/lib/state";

const plans = [
  {
    name: "ऋतु — Seed",
    price: "₹0",
    features: [
      "Sanskrit grammar analysis",
      "Morphology & sandhi detection",
      "Community support",
    ],
  },
  {
    name: "मेधा — Pro", 
    price: "₹1,999/mo",
    features: [
      "Priority inference lanes",
      "Custom rule plugins",
      "LLM integration hooks",
    ],
  },
  {
    name: "प्रज्ञा — Enterprise",
    price: "Contact",
    features: [
      "On-premise deployment",
      "Advanced reasoning engine",
      "Dedicated support",
    ],
  },
];

export default function PlanPanel() {
  const show = useQuantumState((s) => s.showPlanPanel);
  const setShow = useQuantumState((s) => s.setShowPlanPanel);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 40, opacity: 0 }}
          style={{
            position: "fixed",
            bottom: 28,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 30,
            background: "rgba(8,12,20,0.55)",
            border: "1px solid rgba(150,220,255,0.12)",
            borderRadius: 16,
            padding: 18,
            display: "flex",
            gap: 14,
            backdropFilter: "blur(10px)",
            maxWidth: "90vw",
            overflowX: "auto",
          }}
        >
          {plans.map((p) => (
            <motion.div
              key={p.name}
              whileHover={{ scale: 1.02 }}
              style={{
                minWidth: 260,
                maxWidth: 320,
                padding: 16,
                borderRadius: 12,
                background:
                  "linear-gradient(180deg, rgba(20,28,40,0.9), rgba(6,10,18,0.9))",
                border: "1px solid rgba(150,220,255,0.18)",
              }}
            >
              <div
                style={{
                  color: "#E8F6FF",
                  fontWeight: 700,
                  marginBottom: 8,
                  fontSize: "16px",
                }}
              >
                {p.name}
              </div>
              <div
                style={{
                  color: "#7BE1FF",
                  fontSize: 28,
                  fontWeight: 700,
                  marginBottom: 16,
                }}
              >
                {p.price}
              </div>
              <ul
                style={{
                  margin: 0,
                  padding: 0,
                  listStyle: "none",
                  color: "#8FB2C8",
                  lineHeight: 1.5,
                }}
              >
                {p.features.map((f) => (
                  <li key={f} style={{ marginBottom: 12, fontSize: "14px" }}>
                    • {f}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
          <button
            onClick={() => setShow(false)}
            style={{
              marginLeft: 12,
              color: "#8fd9ff",
              background: "transparent",
              border: "1px solid rgba(120,200,255,0.3)",
              borderRadius: 10,
              padding: "10px 14px",
            }}
          >
            Close
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
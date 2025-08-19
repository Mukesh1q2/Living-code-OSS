export const metadata = {
  title: "QuantumSanskrit AI — Vidya",
  description: "Living Code meets Sanskrit Grammar Engine with quantum UI.",
};

import "./globals.css";
import WebSocketProvider, { WebSocketStatus } from "@/components/WebSocketProvider";
import WebSocketDebugPanel from "@/components/WebSocketDebugPanel";
import { VidyaErrorBoundary } from "@/components/ErrorBoundary";
import ErrorNotificationSystem from "@/components/ErrorNotificationSystem";
import ErrorRecoverySystem from "@/components/ErrorRecoverySystem";
import { ErrorCategory } from "@/lib/error-handling";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <VidyaErrorBoundary category={ErrorCategory.SYSTEM} enableRecovery={true}>
          <WebSocketProvider autoConnect={true} debugMode={process.env.NODE_ENV === 'development'}>
            <VidyaErrorBoundary category={ErrorCategory.RENDERING} enableRecovery={true}>
              {children}
            </VidyaErrorBoundary>
            
            {/* Error handling components */}
            <ErrorRecoverySystem />
            <ErrorNotificationSystem />
            
            {/* Development components */}
            <WebSocketStatus />
            <WebSocketDebugPanel />
          </WebSocketProvider>
        </VidyaErrorBoundary>
      </body>
    </html>
  );
}
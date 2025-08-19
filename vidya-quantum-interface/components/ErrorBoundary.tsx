"use client";

/**
 * Error Boundary Components for Vidya Quantum Interface
 * 
 * These components provide:
 * - React error boundaries with graceful fallbacks
 * - Consciousness continuity preservation during errors
 * - User-friendly error messages with recovery suggestions
 * - Automatic error reporting and recovery attempts
 */

import React, { Component, ReactNode, ErrorInfo } from 'react';
import { useErrorHandlingStore, ErrorCategory, ErrorSeverity, RecoveryStrategy } from '../lib/error-handling';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
  retryCount: number;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  category?: ErrorCategory;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  enableRecovery?: boolean;
  maxRetries?: number;
}

// Main Error Boundary Component
export class VidyaErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private retryTimeout: NodeJS.Timeout | null = null;

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
      retryCount: 0
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const { category = ErrorCategory.RENDERING, onError } = this.props;
    
    // Report error to error handling system
    const errorId = useErrorHandlingStore.getState().reportError({
      category,
      severity: ErrorSeverity.HIGH,
      message: error.message,
      technicalDetails: error.stack || '',
      userMessage: this.getUserFriendlyMessage(error, category),
      context: {
        componentStack: errorInfo.componentStack,
        errorBoundary: true,
        retryCount: this.state.retryCount
      },
      stackTrace: error.stack,
      recoveryStrategy: RecoveryStrategy.RESTART_COMPONENT,
      affectedComponents: ['ErrorBoundary', category],
      userImpact: this.assessUserImpact(category)
    });

    this.setState({
      errorInfo,
      errorId
    });

    // Call custom error handler if provided
    if (onError) {
      onError(error, errorInfo);
    }

    // Attempt automatic recovery if enabled
    if (this.props.enableRecovery !== false) {
      this.scheduleRecovery();
    }

    console.error('[VidyaErrorBoundary] Component error caught:', error, errorInfo);
  }

  private getUserFriendlyMessage(error: Error, category: ErrorCategory): string {
    const messages = {
      [ErrorCategory.QUANTUM_EFFECTS]: "Quantum visualizations encountered an issue and are being restored",
      [ErrorCategory.CONSCIOUSNESS]: "Vidya's consciousness is recovering from an unexpected state",
      [ErrorCategory.RENDERING]: "The display encountered an issue and is being refreshed",
      [ErrorCategory.SANSKRIT_ENGINE]: "Sanskrit processing encountered an issue and is being restored",
      [ErrorCategory.AI_SERVICE]: "AI services encountered an issue and are being restored",
      [ErrorCategory.WEBSOCKET]: "Connection encountered an issue and is being restored",
      [ErrorCategory.NETWORK]: "Network communication encountered an issue",
      [ErrorCategory.STORAGE]: "Data storage encountered an issue",
      [ErrorCategory.PERFORMANCE]: "Performance optimization is being applied",
      [ErrorCategory.USER_INPUT]: "Input processing encountered an issue",
      [ErrorCategory.SYSTEM]: "System encountered an issue and is being restored"
    };

    return messages[category] || "A component encountered an issue and is being restored";
  }

  private assessUserImpact(category: ErrorCategory): string {
    const impacts = {
      [ErrorCategory.QUANTUM_EFFECTS]: "Visual effects may be temporarily simplified",
      [ErrorCategory.CONSCIOUSNESS]: "Vidya's responses may be temporarily limited",
      [ErrorCategory.RENDERING]: "Display may be temporarily simplified",
      [ErrorCategory.SANSKRIT_ENGINE]: "Sanskrit analysis may be temporarily limited",
      [ErrorCategory.AI_SERVICE]: "AI responses may be temporarily limited",
      [ErrorCategory.WEBSOCKET]: "Real-time features may be temporarily limited",
      [ErrorCategory.NETWORK]: "Some features may be temporarily unavailable",
      [ErrorCategory.STORAGE]: "Some data may be temporarily unavailable",
      [ErrorCategory.PERFORMANCE]: "Performance may be temporarily reduced",
      [ErrorCategory.USER_INPUT]: "Input processing may be temporarily limited",
      [ErrorCategory.SYSTEM]: "System functionality may be temporarily limited"
    };

    return impacts[category] || "Some functionality may be temporarily limited";
  }

  private scheduleRecovery = () => {
    const { maxRetries = 3 } = this.props;
    
    if (this.state.retryCount < maxRetries) {
      const delay = Math.min(1000 * Math.pow(2, this.state.retryCount), 10000); // Exponential backoff, max 10s
      
      this.retryTimeout = setTimeout(() => {
        this.attemptRecovery();
      }, delay);
    }
  };

  private attemptRecovery = () => {
    const { errorId } = this.state;
    
    if (errorId) {
      // Attempt recovery through error handling system
      useErrorHandlingStore.getState().attemptRecovery(errorId, RecoveryStrategy.RESTART_COMPONENT)
        .then((success) => {
          if (success) {
            this.handleRecoverySuccess();
          } else {
            this.handleRecoveryFailure();
          }
        })
        .catch(() => {
          this.handleRecoveryFailure();
        });
    } else {
      // Direct recovery attempt
      this.handleRecoverySuccess();
    }
  };

  private handleRecoverySuccess = () => {
    console.log('[VidyaErrorBoundary] Recovery successful, resetting error boundary');
    
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
      retryCount: 0
    });

    if (this.state.errorId) {
      useErrorHandlingStore.getState().resolveError(this.state.errorId);
    }
  };

  private handleRecoveryFailure = () => {
    console.warn('[VidyaErrorBoundary] Recovery failed, incrementing retry count');
    
    this.setState(prevState => ({
      retryCount: prevState.retryCount + 1
    }));

    // Schedule next retry if within limits
    if (this.state.retryCount + 1 < (this.props.maxRetries || 3)) {
      this.scheduleRecovery();
    }
  };

  private handleManualRetry = () => {
    this.setState({ retryCount: 0 });
    this.attemptRecovery();
  };

  private handleReportIssue = () => {
    const { error, errorInfo } = this.state;
    
    if (error && errorInfo) {
      const errorReport = {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      };

      // In a real implementation, this would send to an error reporting service
      console.log('[VidyaErrorBoundary] Error report generated:', errorReport);
      
      // Copy to clipboard for user
      navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2))
        .then(() => {
          alert('Error report copied to clipboard. Please share this with support.');
        })
        .catch(() => {
          alert('Error report generated. Please check the console for details.');
        });
    }
  };

  componentWillUnmount() {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
    }
  }

  render() {
    if (this.state.hasError) {
      const { fallback, category = ErrorCategory.RENDERING, maxRetries = 3 } = this.props;
      const { error, retryCount } = this.state;

      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default error UI with recovery options
      return (
        <ErrorFallbackUI
          error={error}
          category={category}
          retryCount={retryCount}
          maxRetries={maxRetries}
          onRetry={this.handleManualRetry}
          onReportIssue={this.handleReportIssue}
        />
      );
    }

    return this.props.children;
  }
}

// Error Fallback UI Component
interface ErrorFallbackUIProps {
  error: Error | null;
  category: ErrorCategory;
  retryCount: number;
  maxRetries: number;
  onRetry: () => void;
  onReportIssue: () => void;
}

const ErrorFallbackUI: React.FC<ErrorFallbackUIProps> = ({
  error,
  category,
  retryCount,
  maxRetries,
  onRetry,
  onReportIssue
}) => {
  const getCategoryIcon = (category: ErrorCategory) => {
    const icons = {
      [ErrorCategory.QUANTUM_EFFECTS]: "⚛️",
      [ErrorCategory.CONSCIOUSNESS]: "🧠",
      [ErrorCategory.RENDERING]: "🖥️",
      [ErrorCategory.SANSKRIT_ENGINE]: "🕉️",
      [ErrorCategory.AI_SERVICE]: "🤖",
      [ErrorCategory.WEBSOCKET]: "🔌",
      [ErrorCategory.NETWORK]: "🌐",
      [ErrorCategory.STORAGE]: "💾",
      [ErrorCategory.PERFORMANCE]: "⚡",
      [ErrorCategory.USER_INPUT]: "⌨️",
      [ErrorCategory.SYSTEM]: "🔧"
    };
    return icons[category] || "⚠️";
  };

  const getCategoryTitle = (category: ErrorCategory) => {
    const titles = {
      [ErrorCategory.QUANTUM_EFFECTS]: "Quantum Effects Issue",
      [ErrorCategory.CONSCIOUSNESS]: "Consciousness System Issue",
      [ErrorCategory.RENDERING]: "Display Issue",
      [ErrorCategory.SANSKRIT_ENGINE]: "Sanskrit Engine Issue",
      [ErrorCategory.AI_SERVICE]: "AI Service Issue",
      [ErrorCategory.WEBSOCKET]: "Connection Issue",
      [ErrorCategory.NETWORK]: "Network Issue",
      [ErrorCategory.STORAGE]: "Storage Issue",
      [ErrorCategory.PERFORMANCE]: "Performance Issue",
      [ErrorCategory.USER_INPUT]: "Input Processing Issue",
      [ErrorCategory.SYSTEM]: "System Issue"
    };
    return titles[category] || "System Issue";
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
      minHeight: '200px',
      background: 'linear-gradient(135deg, rgba(139, 69, 19, 0.1), rgba(255, 215, 0, 0.1))',
      border: '1px solid rgba(255, 215, 0, 0.3)',
      borderRadius: '12px',
      margin: '1rem',
      textAlign: 'center'
    }}>
      <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>
        {getCategoryIcon(category)}
      </div>
      
      <h3 style={{ 
        color: '#d4af37', 
        marginBottom: '0.5rem',
        fontFamily: 'serif'
      }}>
        {getCategoryTitle(category)}
      </h3>
      
      <p style={{ 
        color: '#b8860b', 
        marginBottom: '1.5rem',
        maxWidth: '400px',
        lineHeight: '1.5'
      }}>
        Vidya's consciousness is working to restore this component. 
        The system will attempt automatic recovery.
      </p>

      {error && (
        <details style={{ 
          marginBottom: '1.5rem',
          padding: '0.5rem',
          background: 'rgba(0, 0, 0, 0.2)',
          borderRadius: '4px',
          fontSize: '0.8rem',
          color: '#ccc',
          maxWidth: '500px'
        }}>
          <summary style={{ cursor: 'pointer', marginBottom: '0.5rem' }}>
            Technical Details
          </summary>
          <pre style={{ 
            whiteSpace: 'pre-wrap', 
            wordBreak: 'break-word',
            margin: 0,
            fontSize: '0.7rem'
          }}>
            {error.message}
          </pre>
        </details>
      )}

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
        {retryCount < maxRetries && (
          <button
            onClick={onRetry}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(135deg, #d4af37, #b8860b)',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(212, 175, 55, 0.3)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            Retry Recovery ({maxRetries - retryCount} attempts left)
          </button>
        )}
        
        <button
          onClick={onReportIssue}
          style={{
            padding: '0.75rem 1.5rem',
            background: 'transparent',
            color: '#d4af37',
            border: '1px solid #d4af37',
            borderRadius: '6px',
            cursor: 'pointer',
            transition: 'all 0.3s ease'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = 'rgba(212, 175, 55, 0.1)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'transparent';
          }}
        >
          Report Issue
        </button>
      </div>

      {retryCount > 0 && (
        <p style={{ 
          color: '#888', 
          fontSize: '0.8rem', 
          marginTop: '1rem' 
        }}>
          Recovery attempts: {retryCount}/{maxRetries}
        </p>
      )}
    </div>
  );
};

// Specialized Error Boundaries for different components

export const QuantumEffectsErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <VidyaErrorBoundary 
    category={ErrorCategory.QUANTUM_EFFECTS}
    enableRecovery={true}
    maxRetries={5}
  >
    {children}
  </VidyaErrorBoundary>
);

export const ConsciousnessErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <VidyaErrorBoundary 
    category={ErrorCategory.CONSCIOUSNESS}
    enableRecovery={true}
    maxRetries={3}
  >
    {children}
  </VidyaErrorBoundary>
);

export const SanskritEngineErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <VidyaErrorBoundary 
    category={ErrorCategory.SANSKRIT_ENGINE}
    enableRecovery={true}
    maxRetries={3}
  >
    {children}
  </VidyaErrorBoundary>
);

export const RenderingErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <VidyaErrorBoundary 
    category={ErrorCategory.RENDERING}
    enableRecovery={true}
    maxRetries={2}
  >
    {children}
  </VidyaErrorBoundary>
);

// Hook for using error boundaries programmatically
export const useErrorBoundary = () => {
  const [error, setError] = React.useState<Error | null>(null);

  const resetError = React.useCallback(() => {
    setError(null);
  }, []);

  const captureError = React.useCallback((error: Error) => {
    setError(error);
  }, []);

  React.useEffect(() => {
    if (error) {
      throw error;
    }
  }, [error]);

  return { captureError, resetError };
};
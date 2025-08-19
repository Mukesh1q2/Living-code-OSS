"use client";

/**
 * Error Notification System for Vidya Quantum Interface
 * 
 * This component provides:
 * - User-friendly error notifications with recovery suggestions
 * - Toast-style notifications for different error severities
 * - Interactive recovery actions
 * - System health status display
 */

import React, { useEffect, useState } from 'react';
import { useErrorHandlingStore, useUserMessages, useSystemHealth, ErrorSeverity, ErrorCategory } from '../lib/error-handling';

interface NotificationProps {
  id: string;
  title: string;
  description: string;
  severity: ErrorSeverity;
  category: ErrorCategory;
  actionSuggestions: string[];
  technicalInfo?: string;
  showTechnicalInfo: boolean;
  dismissible: boolean;
  autoHide: boolean;
  hideAfter?: number;
  onDismiss: (id: string) => void;
  onAction: (action: string) => void;
}

const ErrorNotification: React.FC<NotificationProps> = ({
  id,
  title,
  description,
  severity,
  category,
  actionSuggestions,
  technicalInfo,
  showTechnicalInfo,
  dismissible,
  autoHide,
  hideAfter,
  onDismiss,
  onAction
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [timeLeft, setTimeLeft] = useState(hideAfter ? Math.floor(hideAfter / 1000) : 0);

  useEffect(() => {
    if (autoHide && hideAfter) {
      const interval = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            onDismiss(id);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [autoHide, hideAfter, id, onDismiss]);

  const getSeverityColor = (severity: ErrorSeverity) => {
    const colors = {
      [ErrorSeverity.LOW]: '#10b981', // green
      [ErrorSeverity.MEDIUM]: '#f59e0b', // yellow
      [ErrorSeverity.HIGH]: '#ef4444', // red
      [ErrorSeverity.CRITICAL]: '#dc2626' // dark red
    };
    return colors[severity];
  };

  const getSeverityIcon = (severity: ErrorSeverity) => {
    const icons = {
      [ErrorSeverity.LOW]: 'ℹ️',
      [ErrorSeverity.MEDIUM]: '⚠️',
      [ErrorSeverity.HIGH]: '❌',
      [ErrorSeverity.CRITICAL]: '🚨'
    };
    return icons[severity];
  };

  const getCategoryIcon = (category: ErrorCategory) => {
    const icons = {
      [ErrorCategory.QUANTUM_EFFECTS]: '⚛️',
      [ErrorCategory.CONSCIOUSNESS]: '🧠',
      [ErrorCategory.RENDERING]: '🖥️',
      [ErrorCategory.SANSKRIT_ENGINE]: '🕉️',
      [ErrorCategory.AI_SERVICE]: '🤖',
      [ErrorCategory.WEBSOCKET]: '🔌',
      [ErrorCategory.NETWORK]: '🌐',
      [ErrorCategory.STORAGE]: '💾',
      [ErrorCategory.PERFORMANCE]: '⚡',
      [ErrorCategory.USER_INPUT]: '⌨️',
      [ErrorCategory.SYSTEM]: '🔧'
    };
    return icons[category] || '⚠️';
  };

  return (
    <div
      style={{
        background: 'rgba(0, 0, 0, 0.9)',
        border: `1px solid ${getSeverityColor(severity)}`,
        borderRadius: '8px',
        padding: '1rem',
        margin: '0.5rem',
        maxWidth: '400px',
        boxShadow: `0 4px 12px rgba(0, 0, 0, 0.3), 0 0 20px ${getSeverityColor(severity)}33`,
        animation: 'slideIn 0.3s ease-out',
        position: 'relative'
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
        <span style={{ fontSize: '1.2rem', marginRight: '0.5rem' }}>
          {getCategoryIcon(category)}
        </span>
        <span style={{ fontSize: '1rem', marginRight: '0.5rem' }}>
          {getSeverityIcon(severity)}
        </span>
        <h4 style={{ 
          color: getSeverityColor(severity), 
          margin: 0, 
          flex: 1,
          fontSize: '0.9rem',
          fontWeight: 'bold'
        }}>
          {title}
        </h4>
        {dismissible && (
          <button
            onClick={() => onDismiss(id)}
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              cursor: 'pointer',
              fontSize: '1.2rem',
              padding: '0',
              width: '20px',
              height: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            ×
          </button>
        )}
      </div>

      {/* Description */}
      <p style={{ 
        color: '#ccc', 
        margin: '0 0 1rem 0', 
        fontSize: '0.8rem',
        lineHeight: '1.4'
      }}>
        {description}
      </p>

      {/* Auto-hide timer */}
      {autoHide && timeLeft > 0 && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '8px 8px 0 0'
        }}>
          <div
            style={{
              height: '100%',
              background: getSeverityColor(severity),
              borderRadius: '8px 8px 0 0',
              width: `${(timeLeft / (hideAfter! / 1000)) * 100}%`,
              transition: 'width 1s linear'
            }}
          />
        </div>
      )}

      {/* Action Suggestions */}
      {actionSuggestions.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ 
            color: '#d4af37', 
            fontSize: '0.7rem', 
            margin: '0 0 0.5rem 0',
            fontWeight: 'bold'
          }}>
            Suggested Actions:
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            {actionSuggestions.slice(0, 3).map((action, index) => (
              <button
                key={index}
                onClick={() => onAction(action)}
                style={{
                  background: 'rgba(212, 175, 55, 0.1)',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  color: '#d4af37',
                  padding: '0.4rem 0.8rem',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.7rem',
                  textAlign: 'left',
                  transition: 'all 0.2s ease'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = 'rgba(212, 175, 55, 0.2)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'rgba(212, 175, 55, 0.1)';
                }}
              >
                {action}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Technical Info Toggle */}
      {technicalInfo && (
        <div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              cursor: 'pointer',
              fontSize: '0.7rem',
              padding: '0',
              textDecoration: 'underline'
            }}
          >
            {isExpanded ? 'Hide' : 'Show'} Technical Details
          </button>
          
          {isExpanded && (
            <div style={{
              marginTop: '0.5rem',
              padding: '0.5rem',
              background: 'rgba(0, 0, 0, 0.5)',
              borderRadius: '4px',
              fontSize: '0.6rem',
              color: '#aaa',
              fontFamily: 'monospace',
              maxHeight: '100px',
              overflow: 'auto'
            }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {technicalInfo}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// System Health Status Component
const SystemHealthStatus: React.FC = () => {
  const { systemHealth, fallbackState } = useSystemHealth();

  const getHealthColor = (health: string) => {
    const colors = {
      healthy: '#10b981',
      degraded: '#f59e0b',
      critical: '#ef4444',
      emergency: '#dc2626'
    };
    return colors[health as keyof typeof colors] || '#888';
  };

  const getHealthIcon = (health: string) => {
    const icons = {
      healthy: '✅',
      degraded: '⚠️',
      critical: '❌',
      emergency: '🚨'
    };
    return icons[health as keyof typeof icons] || '❓';
  };

  return (
    <div style={{
      position: 'fixed',
      top: '1rem',
      right: '1rem',
      background: 'rgba(0, 0, 0, 0.8)',
      border: `1px solid ${getHealthColor(systemHealth.overall)}`,
      borderRadius: '8px',
      padding: '0.5rem',
      fontSize: '0.7rem',
      zIndex: 1000,
      minWidth: '150px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.25rem' }}>
        <span style={{ marginRight: '0.5rem' }}>
          {getHealthIcon(systemHealth.overall)}
        </span>
        <span style={{ color: getHealthColor(systemHealth.overall), fontWeight: 'bold' }}>
          System: {systemHealth.overall.toUpperCase()}
        </span>
      </div>
      
      {Object.keys(systemHealth.components).length > 0 && (
        <div style={{ fontSize: '0.6rem', color: '#aaa' }}>
          {Object.entries(systemHealth.components).map(([component, status]) => (
            <div key={component} style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>{component}:</span>
              <span style={{ color: getHealthColor(status) }}>{status}</span>
            </div>
          ))}
        </div>
      )}

      {/* Fallback State Indicators */}
      <div style={{ marginTop: '0.5rem', fontSize: '0.6rem', color: '#888' }}>
        {!fallbackState.quantumEffectsEnabled && <div>🔴 Quantum Effects: OFF</div>}
        {!fallbackState.aiServicesAvailable && <div>🔴 AI Services: OFF</div>}
        {!fallbackState.websocketConnected && <div>🔴 WebSocket: OFF</div>}
        {!fallbackState.sanskritEngineActive && <div>🔴 Sanskrit Engine: OFF</div>}
        {fallbackState.renderingQuality !== 'high' && (
          <div>⚠️ Rendering: {fallbackState.renderingQuality.toUpperCase()}</div>
        )}
        {fallbackState.performanceMode !== 'optimal' && (
          <div>⚡ Performance: {fallbackState.performanceMode.toUpperCase()}</div>
        )}
      </div>
    </div>
  );
};

// Main Error Notification System Component
export const ErrorNotificationSystem: React.FC = () => {
  const { userMessages, dismissUserMessage } = useUserMessages();
  const { reportError, attemptRecovery } = useErrorHandlingStore();

  const handleAction = async (action: string) => {
    switch (action.toLowerCase()) {
      case 'try refreshing the page':
      case 'refresh the page to reconnect':
        window.location.reload();
        break;
        
      case 'check your internet connection':
        // Open network diagnostics or show connection status
        if (navigator.onLine) {
          alert('Internet connection appears to be working. The issue may be temporary.');
        } else {
          alert('No internet connection detected. Please check your network settings.');
        }
        break;
        
      case 'try again in a few moments':
        // Retry after a delay
        setTimeout(() => {
          window.location.reload();
        }, 3000);
        break;
        
      case 'clear your browser cache':
        alert('To clear your browser cache:\n1. Press Ctrl+Shift+Delete (or Cmd+Shift+Delete on Mac)\n2. Select "Cached images and files"\n3. Click "Clear data"');
        break;
        
      case 'update your graphics drivers':
        alert('To update graphics drivers:\n1. Visit your graphics card manufacturer\'s website\n2. Download the latest drivers for your model\n3. Install and restart your computer');
        break;
        
      case 'check if webgl is enabled in your browser':
        // Test WebGL support
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
          alert('WebGL appears to be working. The issue may be temporary.');
        } else {
          alert('WebGL is not available in your browser. Please:\n1. Update your browser\n2. Enable hardware acceleration\n3. Update your graphics drivers');
        }
        break;
        
      default:
        console.log(`Action requested: ${action}`);
        break;
    }
  };

  return (
    <>
      {/* System Health Status */}
      <SystemHealthStatus />
      
      {/* Error Notifications */}
      <div style={{
        position: 'fixed',
        bottom: '1rem',
        right: '1rem',
        zIndex: 1000,
        maxHeight: '70vh',
        overflow: 'auto'
      }}>
        {userMessages.map((message: any) => (
          <ErrorNotification
            key={message.id}
            id={message.id}
            title={message.title}
            description={message.description}
            severity={ErrorSeverity.MEDIUM} // Default severity
            category={ErrorCategory.RENDERING} // Default category
            actionSuggestions={message.actionSuggestions || []}
            technicalInfo={message.technicalInfo}
            showTechnicalInfo={message.showTechnicalInfo || false}
            dismissible={message.dismissible !== false}
            autoHide={message.autoHide || false}
            hideAfter={message.hideAfter}
            onDismiss={dismissUserMessage}
            onAction={handleAction}
          />
        ))}
      </div>

      {/* Global Styles */}
      <style jsx>{`
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </>
  );
};

export default ErrorNotificationSystem;
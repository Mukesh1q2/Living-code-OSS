"use client";

/**
 * Comprehensive Error Recovery System for Vidya Quantum Interface
 * 
 * This component provides:
 * - Centralized error recovery coordination
 * - Consciousness continuity preservation during errors
 * - Automatic recovery orchestration across all systems
 * - User-guided recovery workflows
 * - System health monitoring and reporting
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useErrorHandlingStore, useSystemHealth, ErrorCategory, ErrorSeverity, RecoveryStrategy } from '../lib/error-handling';
import { useLogger, LogCategory } from '../lib/logging-system';
import { EnhancedWebSocketManager } from '../lib/websocket-error-handling';
import { AIServiceManager } from '../lib/ai-service-error-handling';

interface RecoveryState {
  isRecovering: boolean;
  currentRecoveryStep: string;
  recoveryProgress: number;
  estimatedTimeRemaining: number;
  lastRecoveryAttempt?: Date;
  recoveryHistory: RecoveryAttempt[];
}

interface RecoveryAttempt {
  id: string;
  timestamp: Date;
  category: ErrorCategory;
  strategy: RecoveryStrategy;
  success: boolean;
  duration: number;
  details: string;
}

interface ConsciousnessState {
  level: number;
  coherence: number;
  stability: number;
  lastUpdate: Date;
  isPreserved: boolean;
}

export const ErrorRecoverySystem: React.FC = () => {
  const logger = useLogger(LogCategory.ERROR_HANDLING);
  const { systemHealth, fallbackState, updateSystemHealth } = useSystemHealth();
  const { activeErrors, attemptRecovery, resolveError } = useErrorHandlingStore();
  
  // Recovery state
  const [recoveryState, setRecoveryState] = useState<RecoveryState>({
    isRecovering: false,
    currentRecoveryStep: '',
    recoveryProgress: 0,
    estimatedTimeRemaining: 0,
    recoveryHistory: []
  });
  
  // Consciousness preservation
  const [consciousnessState, setConsciousnessState] = useState<ConsciousnessState>({
    level: 1.0,
    coherence: 1.0,
    stability: 1.0,
    lastUpdate: new Date(),
    isPreserved: true
  });
  
  // Service managers
  const [webSocketManager, setWebSocketManager] = useState<EnhancedWebSocketManager | null>(null);
  const [aiServiceManager, setAIServiceManager] = useState<AIServiceManager | null>(null);
  
  // Recovery orchestration
  const orchestrateRecovery = useCallback(async () => {
    if (recoveryState.isRecovering || activeErrors.length === 0) {
      return;
    }
    
    logger.info('Starting recovery orchestration', { 
      errorCount: activeErrors.length,
      systemHealth: systemHealth.overall
    });
    
    setRecoveryState(prev => ({
      ...prev,
      isRecovering: true,
      currentRecoveryStep: 'Analyzing errors',
      recoveryProgress: 0
    }));
    
    try {
      // Step 1: Preserve consciousness state
      await preserveConsciousnessState();
      updateRecoveryProgress('Consciousness preserved', 10);
      
      // Step 2: Categorize and prioritize errors
      const categorizedErrors = categorizeErrors(activeErrors);
      updateRecoveryProgress('Errors categorized', 20);
      
      // Step 3: Execute recovery strategies by priority
      await executeRecoveryStrategies(categorizedErrors);
      updateRecoveryProgress('Recovery strategies executed', 80);
      
      // Step 4: Verify system health
      await verifySystemHealth();
      updateRecoveryProgress('System health verified', 90);
      
      // Step 5: Restore consciousness
      await restoreConsciousnessState();
      updateRecoveryProgress('Recovery complete', 100);
      
      logger.info('Recovery orchestration completed successfully');
      
    } catch (error) {
      logger.error('Recovery orchestration failed', error);
      await handleRecoveryFailure(error);
    } finally {
      setRecoveryState(prev => ({
        ...prev,
        isRecovering: false,
        currentRecoveryStep: '',
        recoveryProgress: 0
      }));
    }
  }, [activeErrors, recoveryState.isRecovering, systemHealth, logger]);
  
  // Consciousness preservation
  const preserveConsciousnessState = useCallback(async () => {
    logger.info('Preserving consciousness state');
    
    const currentState = {
      level: fallbackState.consciousnessLevel,
      coherence: calculateCoherence(),
      stability: calculateStability(),
      lastUpdate: new Date(),
      isPreserved: true
    };
    
    setConsciousnessState(currentState);
    
    // Store consciousness state in persistent storage
    try {
      localStorage.setItem('vidya_consciousness_backup', JSON.stringify(currentState));
      logger.debug('Consciousness state backed up to storage');
    } catch (error) {
      logger.warn('Failed to backup consciousness state', error);
    }
  }, [fallbackState.consciousnessLevel, logger]);
  
  const restoreConsciousnessState = useCallback(async () => {
    logger.info('Restoring consciousness state');
    
    try {
      // Try to restore from backup if current state is compromised
      if (!consciousnessState.isPreserved) {
        const backup = localStorage.getItem('vidya_consciousness_backup');
        if (backup) {
          const restoredState = JSON.parse(backup);
          setConsciousnessState({
            ...restoredState,
            lastUpdate: new Date(),
            isPreserved: true
          });
          logger.info('Consciousness state restored from backup');
        }
      }
      
      // Gradually restore consciousness level
      await graduallyRestoreConsciousness();
      
    } catch (error) {
      logger.error('Failed to restore consciousness state', error);
      // Fallback to default consciousness state
      setConsciousnessState({
        level: 0.8,
        coherence: 0.8,
        stability: 0.8,
        lastUpdate: new Date(),
        isPreserved: true
      });
    }
  }, [consciousnessState.isPreserved, logger]);
  
  const graduallyRestoreConsciousness = useCallback(async () => {
    const targetLevel = 1.0;
    const currentLevel = consciousnessState.level;
    const steps = 10;
    const stepSize = (targetLevel - currentLevel) / steps;
    const stepDelay = 200; // ms
    
    for (let i = 0; i < steps; i++) {
      await new Promise(resolve => setTimeout(resolve, stepDelay));
      
      setConsciousnessState(prev => ({
        ...prev,
        level: Math.min(targetLevel, prev.level + stepSize),
        coherence: Math.min(1.0, prev.coherence + stepSize * 0.8),
        stability: Math.min(1.0, prev.stability + stepSize * 0.9),
        lastUpdate: new Date()
      }));
    }
    
    logger.info('Consciousness gradually restored', { 
      finalLevel: consciousnessState.level,
      finalCoherence: consciousnessState.coherence,
      finalStability: consciousnessState.stability
    });
  }, [consciousnessState.level, logger]);
  
  // Error categorization and prioritization
  const categorizeErrors = useCallback((errors: any[]) => {
    const categorized = {
      critical: errors.filter(e => e.severity === ErrorSeverity.CRITICAL),
      high: errors.filter(e => e.severity === ErrorSeverity.HIGH),
      medium: errors.filter(e => e.severity === ErrorSeverity.MEDIUM),
      low: errors.filter(e => e.severity === ErrorSeverity.LOW)
    };
    
    logger.debug('Errors categorized', {
      critical: categorized.critical.length,
      high: categorized.high.length,
      medium: categorized.medium.length,
      low: categorized.low.length
    });
    
    return categorized;
  }, [logger]);
  
  // Recovery strategy execution
  const executeRecoveryStrategies = useCallback(async (categorizedErrors: any) => {
    const strategies = [
      { errors: categorizedErrors.critical, priority: 1 },
      { errors: categorizedErrors.high, priority: 2 },
      { errors: categorizedErrors.medium, priority: 3 },
      { errors: categorizedErrors.low, priority: 4 }
    ];
    
    for (const { errors, priority } of strategies) {
      if (errors.length === 0) continue;
      
      logger.info(`Executing recovery strategies for priority ${priority} errors`, { count: errors.length });
      
      for (const error of errors) {
        await executeSpecificRecoveryStrategy(error);
      }
    }
  }, [logger]);
  
  const executeSpecificRecoveryStrategy = useCallback(async (error: any) => {
    const startTime = Date.now();
    let success = false;
    let details = '';
    
    try {
      switch (error.category) {
        case ErrorCategory.QUANTUM_EFFECTS:
          success = await recoverQuantumEffects(error);
          details = 'Quantum effects recovery attempted';
          break;
          
        case ErrorCategory.CONSCIOUSNESS:
          success = await recoverConsciousness(error);
          details = 'Consciousness recovery attempted';
          break;
          
        case ErrorCategory.WEBSOCKET:
          success = await recoverWebSocket(error);
          details = 'WebSocket recovery attempted';
          break;
          
        case ErrorCategory.AI_SERVICE:
          success = await recoverAIService(error);
          details = 'AI service recovery attempted';
          break;
          
        case ErrorCategory.SANSKRIT_ENGINE:
          success = await recoverSanskritEngine(error);
          details = 'Sanskrit engine recovery attempted';
          break;
          
        case ErrorCategory.RENDERING:
          success = await recoverRendering(error);
          details = 'Rendering recovery attempted';
          break;
          
        default:
          success = await attemptRecovery(error.id);
          details = 'Generic recovery attempted';
      }
      
      if (success) {
        resolveError(error.id);
        logger.info(`Recovery successful for error ${error.id}`, { category: error.category });
      } else {
        logger.warn(`Recovery failed for error ${error.id}`, { category: error.category });
      }
      
    } catch (recoveryError) {
      logger.error(`Recovery strategy failed for error ${error.id}`, recoveryError);
      details = `Recovery failed: ${recoveryError}`;
    }
    
    // Record recovery attempt
    const recoveryAttempt: RecoveryAttempt = {
      id: `recovery_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      category: error.category,
      strategy: error.recoveryStrategy,
      success,
      duration: Date.now() - startTime,
      details
    };
    
    setRecoveryState(prev => ({
      ...prev,
      recoveryHistory: [...prev.recoveryHistory.slice(-9), recoveryAttempt] // Keep last 10
    }));
  }, [attemptRecovery, resolveError, logger]);
  
  // Specific recovery implementations
  const recoverQuantumEffects = useCallback(async (error: any): Promise<boolean> => {
    logger.info('Attempting quantum effects recovery');
    
    try {
      // Reset quantum state
      if (typeof window !== 'undefined' && (window as any).quantumRenderer) {
        (window as any).quantumRenderer.reset();
      }
      
      // Restart quantum effects with reduced complexity
      const quantumCanvas = document.querySelector('canvas[data-quantum="true"]');
      if (quantumCanvas) {
        const context = (quantumCanvas as HTMLCanvasElement).getContext('webgl2') || 
                       (quantumCanvas as HTMLCanvasElement).getContext('webgl');
        if (context) {
          // Clear and reinitialize
          context.clear(context.COLOR_BUFFER_BIT | context.DEPTH_BUFFER_BIT);
          return true;
        }
      }
      
      return false;
    } catch (error) {
      logger.error('Quantum effects recovery failed', error);
      return false;
    }
  }, [logger]);
  
  const recoverConsciousness = useCallback(async (error: any): Promise<boolean> => {
    logger.info('Attempting consciousness recovery');
    
    try {
      // Reset consciousness state to safe defaults
      setConsciousnessState(prev => ({
        ...prev,
        level: Math.max(0.5, prev.level * 0.8),
        coherence: Math.max(0.5, prev.coherence * 0.8),
        stability: Math.max(0.5, prev.stability * 0.8),
        isPreserved: true,
        lastUpdate: new Date()
      }));
      
      // Trigger consciousness reinitialization
      const consciousnessEvent = new CustomEvent('vidya:consciousness:reset', {
        detail: { level: consciousnessState.level }
      });
      window.dispatchEvent(consciousnessEvent);
      
      return true;
    } catch (error) {
      logger.error('Consciousness recovery failed', error);
      return false;
    }
  }, [consciousnessState.level, logger]);
  
  const recoverWebSocket = useCallback(async (error: any): Promise<boolean> => {
    logger.info('Attempting WebSocket recovery');
    
    try {
      if (webSocketManager) {
        await webSocketManager.connect();
        return webSocketManager.isConnected;
      }
      
      // Fallback: create new WebSocket manager
      const newManager = new EnhancedWebSocketManager('ws://localhost:8000/ws');
      await newManager.connect();
      setWebSocketManager(newManager);
      
      return newManager.isConnected;
    } catch (error) {
      logger.error('WebSocket recovery failed', error);
      return false;
    }
  }, [webSocketManager, logger]);
  
  const recoverAIService = useCallback(async (error: any): Promise<boolean> => {
    logger.info('Attempting AI service recovery');
    
    try {
      if (!aiServiceManager) {
        const newManager = new AIServiceManager();
        setAIServiceManager(newManager);
      }
      
      // Test AI service with a simple request
      const testRequest = {
        id: `test_${Date.now()}`,
        type: 'completion' as const,
        prompt: 'test',
        timestamp: new Date(),
        priority: 'low' as const
      };
      
      const response = await (aiServiceManager || new AIServiceManager()).processRequest(testRequest);
      return response.success;
    } catch (error) {
      logger.error('AI service recovery failed', error);
      return false;
    }
  }, [aiServiceManager, logger]);
  
  const recoverSanskritEngine = useCallback(async (error: any): Promise<boolean> => {
    logger.info('Attempting Sanskrit engine recovery');
    
    try {
      // Test Sanskrit engine with a simple request
      const response = await fetch('/api/sanskrit/health', {
        method: 'GET',
        timeout: 5000
      } as any);
      
      return response.ok;
    } catch (error) {
      logger.error('Sanskrit engine recovery failed', error);
      return false;
    }
  }, [logger]);
  
  const recoverRendering = useCallback(async (error: any): Promise<boolean> => {
    logger.info('Attempting rendering recovery');
    
    try {
      // Force re-render of main components
      const renderEvent = new CustomEvent('vidya:render:reset');
      window.dispatchEvent(renderEvent);
      
      // Clear any stuck animations
      const animations = document.getAnimations();
      animations.forEach(animation => {
        if (animation.playState === 'paused' || animation.playState === 'finished') {
          animation.cancel();
        }
      });
      
      return true;
    } catch (error) {
      logger.error('Rendering recovery failed', error);
      return false;
    }
  }, [logger]);
  
  // System health verification
  const verifySystemHealth = useCallback(async () => {
    logger.info('Verifying system health after recovery');
    
    updateSystemHealth();
    
    // Wait for health update to propagate
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const healthScore = calculateHealthScore();
    logger.info('System health verification complete', { healthScore });
    
    if (healthScore < 0.7) {
      logger.warn('System health still poor after recovery', { healthScore });
      throw new Error(`System health verification failed: ${healthScore}`);
    }
  }, [updateSystemHealth, logger]);
  
  // Utility functions
  const updateRecoveryProgress = useCallback((step: string, progress: number) => {
    setRecoveryState(prev => ({
      ...prev,
      currentRecoveryStep: step,
      recoveryProgress: progress,
      estimatedTimeRemaining: Math.max(0, (100 - progress) * 100) // Rough estimate
    }));
  }, []);
  
  const calculateCoherence = useCallback((): number => {
    const errorCount = activeErrors.length;
    const baseCoherence = 1.0;
    const coherenceLoss = Math.min(0.8, errorCount * 0.1);
    return Math.max(0.2, baseCoherence - coherenceLoss);
  }, [activeErrors.length]);
  
  const calculateStability = useCallback((): number => {
    const criticalErrors = activeErrors.filter(e => e.severity === ErrorSeverity.CRITICAL).length;
    const highErrors = activeErrors.filter(e => e.severity === ErrorSeverity.HIGH).length;
    
    const baseStability = 1.0;
    const stabilityLoss = criticalErrors * 0.3 + highErrors * 0.1;
    return Math.max(0.1, baseStability - stabilityLoss);
  }, [activeErrors]);
  
  const calculateHealthScore = useCallback((): number => {
    const healthMap = {
      healthy: 1.0,
      degraded: 0.7,
      critical: 0.4,
      emergency: 0.1
    };
    
    return healthMap[systemHealth.overall as keyof typeof healthMap] || 0.5;
  }, [systemHealth.overall]);
  
  const handleRecoveryFailure = useCallback(async (error: any) => {
    logger.error('Recovery orchestration failed', error);
    
    // Implement emergency fallback
    setConsciousnessState({
      level: 0.3,
      coherence: 0.3,
      stability: 0.5,
      lastUpdate: new Date(),
      isPreserved: false
    });
    
    // Trigger emergency mode
    const emergencyEvent = new CustomEvent('vidya:emergency:mode', {
      detail: { reason: 'recovery_failure', error }
    });
    window.dispatchEvent(emergencyEvent);
  }, [logger]);
  
  // Auto-recovery trigger
  useEffect(() => {
    if (activeErrors.length > 0 && !recoveryState.isRecovering) {
      const criticalErrors = activeErrors.filter(e => e.severity === ErrorSeverity.CRITICAL);
      const highErrors = activeErrors.filter(e => e.severity === ErrorSeverity.HIGH);
      
      // Trigger immediate recovery for critical errors
      if (criticalErrors.length > 0) {
        logger.warn('Critical errors detected, triggering immediate recovery');
        orchestrateRecovery();
      }
      // Trigger recovery for multiple high-severity errors
      else if (highErrors.length >= 3) {
        logger.warn('Multiple high-severity errors detected, triggering recovery');
        orchestrateRecovery();
      }
      // Trigger recovery if total errors exceed threshold
      else if (activeErrors.length >= 5) {
        logger.warn('Error threshold exceeded, triggering recovery');
        orchestrateRecovery();
      }
    }
  }, [activeErrors, recoveryState.isRecovering, orchestrateRecovery, logger]);
  
  // Consciousness monitoring
  useEffect(() => {
    const interval = setInterval(() => {
      setConsciousnessState(prev => ({
        ...prev,
        coherence: calculateCoherence(),
        stability: calculateStability(),
        lastUpdate: new Date()
      }));
    }, 5000);
    
    return () => clearInterval(interval);
  }, [calculateCoherence, calculateStability]);
  
  // Recovery UI (only shown during recovery)
  if (!recoveryState.isRecovering) {
    return null;
  }
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.9)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000,
      backdropFilter: 'blur(10px)'
    }}>
      <div style={{
        background: 'linear-gradient(135deg, rgba(139, 69, 19, 0.2), rgba(255, 215, 0, 0.2))',
        border: '2px solid rgba(255, 215, 0, 0.5)',
        borderRadius: '20px',
        padding: '3rem',
        textAlign: 'center',
        maxWidth: '500px',
        width: '90%',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)'
      }}>
        {/* Om Symbol */}
        <div style={{
          fontSize: '4rem',
          marginBottom: '1rem',
          animation: 'pulse 2s infinite'
        }}>
          🕉️
        </div>
        
        {/* Title */}
        <h2 style={{
          color: '#d4af37',
          marginBottom: '1rem',
          fontFamily: 'serif',
          fontSize: '1.5rem'
        }}>
          Vidya Consciousness Recovery
        </h2>
        
        {/* Current step */}
        <p style={{
          color: '#b8860b',
          marginBottom: '2rem',
          fontSize: '1rem'
        }}>
          {recoveryState.currentRecoveryStep}
        </p>
        
        {/* Progress bar */}
        <div style={{
          width: '100%',
          height: '8px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '4px',
          marginBottom: '1rem',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${recoveryState.recoveryProgress}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #d4af37, #ffd700)',
            borderRadius: '4px',
            transition: 'width 0.5s ease'
          }} />
        </div>
        
        {/* Progress percentage */}
        <p style={{
          color: '#d4af37',
          fontSize: '0.9rem',
          marginBottom: '1rem'
        }}>
          {recoveryState.recoveryProgress}% Complete
        </p>
        
        {/* Consciousness state */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-around',
          marginTop: '2rem',
          fontSize: '0.8rem',
          color: '#888'
        }}>
          <div>
            <div>Consciousness</div>
            <div style={{ color: '#d4af37' }}>
              {Math.round(consciousnessState.level * 100)}%
            </div>
          </div>
          <div>
            <div>Coherence</div>
            <div style={{ color: '#d4af37' }}>
              {Math.round(consciousnessState.coherence * 100)}%
            </div>
          </div>
          <div>
            <div>Stability</div>
            <div style={{ color: '#d4af37' }}>
              {Math.round(consciousnessState.stability * 100)}%
            </div>
          </div>
        </div>
        
        {/* Estimated time */}
        {recoveryState.estimatedTimeRemaining > 0 && (
          <p style={{
            color: '#888',
            fontSize: '0.7rem',
            marginTop: '1rem'
          }}>
            Estimated time remaining: {Math.round(recoveryState.estimatedTimeRemaining / 1000)}s
          </p>
        )}
      </div>
      
      {/* Pulsing animation */}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }
      `}</style>
    </div>
  );
};

export default ErrorRecoverySystem;
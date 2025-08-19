// Shared TypeScript interfaces for frontend-backend communication

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface ComplexNumber {
  real: number;
  imaginary: number;
}

// Quantum State Interfaces
export interface QuantumState {
  stateVector: ComplexNumber[];
  entanglements: Map<string, EntanglementInfo>;
  coherenceLevel: number;
  observationHistory: QuantumMeasurement[];
}

export interface EntanglementInfo {
  partnerId: string;
  entanglementStrength: number;
  sharedProperties: string[];
  lastSynchronization: number;
}

export interface QuantumMeasurement {
  timestamp: number;
  measuredState: ComplexNumber[];
  observer: string;
}

// Vidya Consciousness Interfaces
export interface VidyaConsciousness {
  currentState: QuantumState;
  personality: PersonalityProfile;
  learningHistory: InteractionMemory[];
}

export interface PersonalityProfile {
  traits: Record<string, number>;
  preferences: Record<string, any>;
  evolutionLevel: number;
}

export interface InteractionMemory {
  timestamp: number;
  userInput: string;
  response: string;
  context: Record<string, any>;
}

// Neural Network Interfaces
export interface NetworkNode {
  id: string;
  position: Vector3;
  type: 'sanskrit-rule' | 'neural-unit' | 'quantum-gate';
  sanskritRule?: PaniniRule;
  activationLevel: number;
  quantumProperties: QuantumNodeProperties;
}

export interface NetworkConnection {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  strength: number;
  isQuantumEntangled: boolean;
}

export interface QuantumNodeProperties {
  superpositionStates: ComplexNumber[];
  entanglementPartners: string[];
  coherenceTime: number;
}

// Sanskrit Processing Interfaces
export interface PaniniRule {
  id: string;
  sutraNumber: string;
  description: string;
  conditions: string[];
  transformations: string[];
}

export interface SanskritToken {
  text: string;
  position: TextPosition;
  morphology: MorphologicalInfo;
  quantumProperties: TokenQuantumState;
  visualizationData: TokenVisualization;
}

export interface TextPosition {
  start: number;
  end: number;
  line: number;
  column: number;
}

export interface MorphologicalInfo {
  root: string;
  suffixes: string[];
  grammaticalCategory: string;
  semanticRole: string;
}

export interface TokenQuantumState {
  superposition: boolean;
  entanglements: string[];
  probability: number;
}

export interface TokenVisualization {
  color: string;
  size: number;
  animation: string;
  effects: string[];
}

// WebSocket Communication Interfaces
export interface WebSocketMessage {
  type: MessageType;
  payload: any;
  timestamp: number;
  id: string;
}

export enum MessageType {
  CONSCIOUSNESS_UPDATE = 'consciousness_update',
  QUANTUM_STATE_CHANGE = 'quantum_state_change',
  SANSKRIT_ANALYSIS = 'sanskrit_analysis',
  NEURAL_NETWORK_UPDATE = 'neural_network_update',
  USER_INPUT = 'user_input',
  SYSTEM_STATUS = 'system_status',
  ERROR = 'error'
}

// API Response Interfaces
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: number;
}

export interface SanskritAnalysisResponse {
  originalText: string;
  tokens: SanskritToken[];
  appliedRules: PaniniRuleApplication[];
  morphologicalAnalysis: MorphologicalData;
  visualizationNodes: NetworkNode[];
}

export interface PaniniRuleApplication {
  rule: PaniniRule;
  appliedTo: string;
  result: string;
  confidence: number;
}

export interface MorphologicalData {
  wordForms: WordForm[];
  etymologicalConnections: EtymologyConnection[];
  semanticGraph: SemanticNode[];
}

export interface WordForm {
  surface: string;
  lemma: string;
  pos: string;
  features: Record<string, string>;
}

export interface EtymologyConnection {
  fromWord: string;
  toWord: string;
  relationship: string;
  confidence: number;
}

export interface SemanticNode {
  concept: string;
  relations: SemanticRelation[];
  position: Vector3;
}

export interface SemanticRelation {
  target: string;
  type: string;
  strength: number;
}

// Performance and Development Interfaces
export interface PerformanceMetrics {
  fps: number;
  memoryUsage: number;
  renderTime: number;
  networkLatency: number;
}

export interface DevelopmentConfig {
  hotReload: boolean;
  debugMode: boolean;
  performanceMonitoring: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
}
import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API client configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      // Redirect to login if needed
    }
    return Promise.reject(error);
  }
);

// Type definitions
export interface Token {
  text: string;
  kind: string;
  tags: string[];
  meta: Record<string, any>;
  position?: number;
}

export interface Rule {
  id: number;
  name: string;
  priority: number;
  description: string;
  active: boolean;
  applications: number;
  max_applications?: number;
  sutra_ref?: string;
}

export interface Transformation {
  rule_name: string;
  rule_id: number;
  index: number;
  tokens_before: Token[];
  tokens_after: Token[];
  timestamp: string;
}

export interface ProcessingTrace {
  pass_number: number;
  tokens_before: Token[];
  tokens_after: Token[];
  transformations: Transformation[];
  meta_rule_applications: string[];
}

export interface SanskritProcessRequest {
  text: string;
  options?: Record<string, any>;
  enable_tracing?: boolean;
  max_passes?: number;
  active_rules?: number[];
}

export interface SanskritProcessResponse {
  input_text: string;
  input_tokens: Token[];
  output_tokens: Token[];
  converged: boolean;
  passes: number;
  traces: ProcessingTrace[];
  errors: string[];
  processing_time_ms: number;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  context?: Record<string, any>;
  stream?: boolean;
}

export interface ChatMessage {
  id: string;
  role: string;
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface ChatResponse {
  message: ChatMessage;
  conversation_id: string;
  response_time_ms: number;
  model_used: string;
  confidence: number;
}

export interface RuleTraceRequest {
  text: string;
  rule_ids?: string[];
  detail_level?: string;
}

export interface RuleTraceResponse {
  input_text: string;
  trace_data: Record<string, any>;
  rule_applications: any[];
  performance_metrics: Record<string, any>;
}

export interface FileOperation {
  operation: 'read' | 'write' | 'delete' | 'list';
  file_path: string;
  content?: string;
  options?: Record<string, any>;
}

export interface FileOperationResponse {
  success: boolean;
  operation: string;
  file_path: string;
  content?: string;
  files?: any[];
  error?: string;
}

export interface CodeExecutionRequest {
  code: string;
  language?: string;
  timeout?: number;
  context?: Record<string, any>;
}

export interface CodeExecutionResponse {
  success: boolean;
  output: string;
  error?: string;
  execution_time_ms: number;
  language: string;
}

class ApiService {
  // Health check
  async healthCheck(): Promise<any> {
    const response = await apiClient.get('/health');
    return response.data;
  }

  // Sanskrit processing
  async processSanskritText(request: SanskritProcessRequest): Promise<SanskritProcessResponse> {
    const response: AxiosResponse<SanskritProcessResponse> = await apiClient.post('/api/v1/process', request);
    return response.data;
  }

  // Chat functionality
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    const response: AxiosResponse<ChatResponse> = await apiClient.post('/api/v1/chat', request);
    return response.data;
  }

  async getConversation(conversationId: string): Promise<any> {
    const response = await apiClient.get(`/api/v1/conversations/${conversationId}`);
    return response.data;
  }

  // Rule tracing
  async traceRules(request: RuleTraceRequest): Promise<RuleTraceResponse> {
    const response: AxiosResponse<RuleTraceResponse> = await apiClient.post('/api/v1/trace', request);
    return response.data;
  }

  // File operations
  async performFileOperation(request: FileOperation): Promise<FileOperationResponse> {
    const response: AxiosResponse<FileOperationResponse> = await apiClient.post('/api/v1/files', request);
    return response.data;
  }

  async readFile(filePath: string): Promise<string> {
    const response = await this.performFileOperation({
      operation: 'read',
      file_path: filePath,
    });
    
    if (!response.success) {
      throw new Error(response.error || 'Failed to read file');
    }
    
    return response.content || '';
  }

  async writeFile(filePath: string, content: string): Promise<void> {
    const response = await this.performFileOperation({
      operation: 'write',
      file_path: filePath,
      content,
    });
    
    if (!response.success) {
      throw new Error(response.error || 'Failed to write file');
    }
  }

  async deleteFile(filePath: string): Promise<void> {
    const response = await this.performFileOperation({
      operation: 'delete',
      file_path: filePath,
    });
    
    if (!response.success) {
      throw new Error(response.error || 'Failed to delete file');
    }
  }

  async listFiles(directoryPath: string): Promise<any[]> {
    const response = await this.performFileOperation({
      operation: 'list',
      file_path: directoryPath,
    });
    
    if (!response.success) {
      throw new Error(response.error || 'Failed to list files');
    }
    
    return response.files || [];
  }

  // Code execution
  async executeCode(request: CodeExecutionRequest): Promise<CodeExecutionResponse> {
    const response: AxiosResponse<CodeExecutionResponse> = await apiClient.post('/api/v1/execute', request);
    return response.data;
  }

  // File upload
  async uploadFile(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  }

  // Rules management
  async getRules(): Promise<Rule[]> {
    const response = await apiClient.get('/api/v1/rules');
    return response.data;
  }

  async updateRule(ruleId: number, updates: Partial<Rule>): Promise<Rule> {
    const response = await apiClient.patch(`/api/v1/rules/${ruleId}`, updates);
    return response.data;
  }

  async createRule(rule: Omit<Rule, 'id'>): Promise<Rule> {
    const response = await apiClient.post('/api/v1/rules', rule);
    return response.data;
  }

  async deleteRule(ruleId: number): Promise<void> {
    await apiClient.delete(`/api/v1/rules/${ruleId}`);
  }

  async activateRule(ruleId: number): Promise<void> {
    await this.updateRule(ruleId, { active: true });
  }

  async deactivateRule(ruleId: number): Promise<void> {
    await this.updateRule(ruleId, { active: false });
  }

  // Enhanced processing with detailed tracing
  async processWithDetailedTrace(request: SanskritProcessRequest): Promise<SanskritProcessResponse> {
    const response: AxiosResponse<SanskritProcessResponse> = await apiClient.post('/api/v1/process/detailed', request);
    return response.data;
  }

  // Code generation from Sanskrit
  async generateCode(text: string, language: string = 'python'): Promise<string> {
    const response = await apiClient.post('/api/v1/generate/code', {
      text,
      language,
    });
    return response.data.code;
  }

  // Streaming chat (Server-Sent Events)
  createChatStream(conversationId: string, message: string): EventSource {
    const url = `${API_BASE_URL}/api/v1/stream/chat/${conversationId}?message=${encodeURIComponent(message)}`;
    return new EventSource(url);
  }
}

export const apiService = new ApiService();
export default apiService;
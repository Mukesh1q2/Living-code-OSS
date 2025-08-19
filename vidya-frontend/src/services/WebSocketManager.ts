import { WebSocketMessage, MessageType } from '../types/shared'

export default class WebSocketManager {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private messageQueue: WebSocketMessage[] = []
  
  public onConnect: (() => void) | null = null
  public onDisconnect: (() => void) | null = null
  public onMessage: ((message: WebSocketMessage) => void) | null = null
  public onError: ((error: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
  }

  connect(): void {
    try {
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected to Vidya backend')
        this.reconnectAttempts = 0
        
        // Send queued messages
        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift()
          if (message) {
            this.send(message)
          }
        }
        
        if (this.onConnect) {
          this.onConnect()
        }
      }
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected from Vidya backend')
        
        if (this.onDisconnect) {
          this.onDisconnect()
        }
        
        // Attempt to reconnect
        this.attemptReconnect()
      }
      
      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          if (this.onMessage) {
            this.onMessage(message)
          }
          
          // Handle specific message types
          this.handleMessage(message)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        
        if (this.onError) {
          this.onError(error)
        }
      }
      
    } catch (error) {
      console.error('Error creating WebSocket connection:', error)
      this.attemptReconnect()
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  send(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      // Queue message for when connection is restored
      this.messageQueue.push(message)
      console.warn('WebSocket not connected, message queued')
    }
  }

  sendMessage(type: MessageType, payload: any): void {
    const message: WebSocketMessage = {
      type,
      payload,
      timestamp: Date.now(),
      id: this.generateMessageId()
    }
    
    this.send(message)
  }

  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case MessageType.CONSCIOUSNESS_UPDATE:
        console.log('Vidya consciousness update:', message.payload)
        break
        
      case MessageType.QUANTUM_STATE_CHANGE:
        console.log('Quantum state change:', message.payload)
        break
        
      case MessageType.SANSKRIT_ANALYSIS:
        console.log('Sanskrit analysis result:', message.payload)
        break
        
      case MessageType.NEURAL_NETWORK_UPDATE:
        console.log('Neural network update:', message.payload)
        break
        
      case MessageType.SYSTEM_STATUS:
        console.log('System status:', message.payload)
        break
        
      case MessageType.ERROR:
        console.error('Backend error:', message.payload)
        break
        
      default:
        console.log('Unknown message type:', message.type)
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`)
      
      setTimeout(() => {
        this.connect()
      }, this.reconnectDelay * this.reconnectAttempts)
    } else {
      console.error('Max reconnection attempts reached')
    }
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
}
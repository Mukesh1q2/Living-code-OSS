"use client";

import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { usePerformanceMonitor, type PerformanceMetrics } from '@/lib/performance-monitor';
import { useMemoryManager, type MemoryUsage } from '@/lib/memory-manager';
import { usePerformanceProfiler, type PerformanceBottleneck } from '@/lib/performance-profiler';

interface PerformanceDashboardProps {
  visible: boolean;
  onClose: () => void;
}

interface ChartDataPoint {
  time: string;
  fps: number;
  memory: number;
  triangles: number;
  drawCalls: number;
}

export default function PerformanceDashboard({ visible, onClose }: PerformanceDashboardProps) {
  const performanceMonitor = usePerformanceMonitor();
  const memoryManager = useMemoryManager();
  const profiler = usePerformanceProfiler();

  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [memoryUsage, setMemoryUsage] = useState<MemoryUsage | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [bottlenecks, setBottlenecks] = useState<PerformanceBottleneck[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'memory' | 'profiler'>('overview');

  useEffect(() => {
    if (!visible) return;

    // Start performance monitoring
    performanceMonitor.start();

    // Subscribe to metrics updates
    const unsubscribe = performanceMonitor.onMetricsUpdate((newMetrics) => {
      setMetrics(newMetrics);
      
      // Update chart data
      const newDataPoint: ChartDataPoint = {
        time: new Date(newMetrics.timestamp).toLocaleTimeString(),
        fps: newMetrics.fps,
        memory: newMetrics.memoryUsage,
        triangles: Math.round(newMetrics.triangleCount / 1000), // Convert to thousands
        drawCalls: newMetrics.drawCalls,
      };

      setChartData(prev => {
        const updated = [...prev, newDataPoint];
        // Keep only last 50 data points
        return updated.slice(-50);
      });
    });

    // Update memory usage periodically
    const memoryInterval = setInterval(() => {
      setMemoryUsage(memoryManager.getMemoryUsage());
    }, 1000);

    // Update profiler data
    const updateProfilerData = () => {
      const sessions = profiler.getAllSessions();
      if (sessions.length > 0) {
        const latestSession = sessions[sessions.length - 1];
        if (latestSession.endTime > 0) {
          const sessionBottlenecks = profiler.analyzeBottlenecks(latestSession.id);
          setBottlenecks(sessionBottlenecks);
        }
      }
    };

    const profilerInterval = setInterval(updateProfilerData, 2000);

    return () => {
      unsubscribe();
      clearInterval(memoryInterval);
      clearInterval(profilerInterval);
      performanceMonitor.stop();
    };
  }, [visible, performanceMonitor, memoryManager, profiler]);

  if (!visible) return null;

  const currentQuality = performanceMonitor.getCurrentQuality();
  const qualitySettings = performanceMonitor.getQualitySettings();

  return (
    <div
      style={{
        position: 'fixed',
        top: 20,
        left: 20,
        right: 20,
        bottom: 20,
        zIndex: 10000,
        background: 'rgba(0,0,0,0.95)',
        border: '1px solid rgba(123,225,255,0.3)',
        borderRadius: 12,
        color: '#E8F6FF',
        fontSize: '12px',
        fontFamily: 'monospace',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        padding: 16,
        borderBottom: '1px solid rgba(123,225,255,0.2)',
        background: 'rgba(123,225,255,0.05)'
      }}>
        <h2 style={{ margin: 0, color: '#7BE1FF', fontSize: '16px' }}>Performance Dashboard</h2>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ 
            padding: '4px 8px', 
            background: 'rgba(99,255,201,0.2)', 
            borderRadius: 4,
            color: '#63FFC9'
          }}>
            Quality: {currentQuality}
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: '1px solid rgba(123,225,255,0.3)',
              color: '#7BE1FF',
              borderRadius: 4,
              padding: '4px 8px',
              cursor: 'pointer',
            }}
          >
            ×
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{ 
        display: 'flex', 
        borderBottom: '1px solid rgba(123,225,255,0.2)',
        background: 'rgba(0,0,0,0.3)'
      }}>
        {(['overview', 'memory', 'profiler'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              background: activeTab === tab ? 'rgba(123,225,255,0.2)' : 'transparent',
              border: 'none',
              color: activeTab === tab ? '#7BE1FF' : '#8FB2C8',
              padding: '12px 20px',
              cursor: 'pointer',
              textTransform: 'capitalize',
              borderBottom: activeTab === tab ? '2px solid #7BE1FF' : 'none',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {activeTab === 'overview' && (
          <OverviewTab 
            metrics={metrics} 
            chartData={chartData} 
            qualitySettings={qualitySettings}
            onQualityChange={(quality) => performanceMonitor.setQuality(quality)}
          />
        )}
        
        {activeTab === 'memory' && (
          <MemoryTab 
            memoryUsage={memoryUsage} 
            onCleanup={() => memoryManager.cleanup(true)}
            onSetLimits={(maxMB) => memoryManager.setMemoryLimits(maxMB)}
          />
        )}
        
        {activeTab === 'profiler' && (
          <ProfilerTab 
            bottlenecks={bottlenecks}
            profiler={profiler}
          />
        )}
      </div>
    </div>
  );
}

// Overview Tab Component
function OverviewTab({ 
  metrics, 
  chartData, 
  qualitySettings,
  onQualityChange 
}: {
  metrics: PerformanceMetrics | null;
  chartData: ChartDataPoint[];
  qualitySettings: any;
  onQualityChange: (quality: any) => void;
}) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, height: '100%' }}>
      {/* Current Metrics */}
      <div>
        <h3 style={{ margin: '0 0 16px 0', color: '#B383FF' }}>Current Metrics</h3>
        {metrics ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <MetricCard label="FPS" value={metrics.fps} unit="" color={metrics.fps > 50 ? '#63FFC9' : metrics.fps > 30 ? '#7BE1FF' : '#FF7BD5'} />
            <MetricCard label="Frame Time" value={metrics.frameTime.toFixed(1)} unit="ms" color="#7BE1FF" />
            <MetricCard label="Memory" value={metrics.memoryUsage} unit="MB" color="#B383FF" />
            <MetricCard label="Triangles" value={Math.round(metrics.triangleCount / 1000)} unit="K" color="#63FFC9" />
            <MetricCard label="Draw Calls" value={metrics.drawCalls} unit="" color="#FF7BD5" />
            <MetricCard label="Textures" value={metrics.textureMemory} unit="" color="#7BE1FF" />
          </div>
        ) : (
          <div style={{ color: '#8FB2C8' }}>No metrics available</div>
        )}

        {/* Quality Settings */}
        <div style={{ marginTop: 24 }}>
          <h3 style={{ margin: '0 0 16px 0', color: '#B383FF' }}>Quality Settings</h3>
          <div style={{ display: 'grid', gap: 8 }}>
            <div>Particles: <span style={{ color: '#63FFC9' }}>{qualitySettings.particleCount}</span></div>
            <div>Shaders: <span style={{ color: '#63FFC9' }}>{qualitySettings.shaderComplexity}</span></div>
            <div>Shadows: <span style={{ color: '#63FFC9' }}>{qualitySettings.shadowQuality}</span></div>
            <div>Antialiasing: <span style={{ color: qualitySettings.antialiasing ? '#63FFC9' : '#FF7BD5' }}>{qualitySettings.antialiasing ? 'On' : 'Off'}</span></div>
            <div>Post Processing: <span style={{ color: qualitySettings.postProcessing ? '#63FFC9' : '#FF7BD5' }}>{qualitySettings.postProcessing ? 'On' : 'Off'}</span></div>
          </div>
          
          <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {(['minimal', 'low', 'medium', 'high'] as const).map(quality => (
              <button
                key={quality}
                onClick={() => onQualityChange(quality)}
                style={{
                  background: 'rgba(123,225,255,0.1)',
                  border: '1px solid rgba(123,225,255,0.3)',
                  color: '#7BE1FF',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '10px',
                  textTransform: 'capitalize',
                }}
              >
                {quality}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Chart */}
      <div>
        <h3 style={{ margin: '0 0 16px 0', color: '#B383FF' }}>Performance History</h3>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(123,225,255,0.1)" />
              <XAxis dataKey="time" stroke="#8FB2C8" fontSize={10} />
              <YAxis stroke="#8FB2C8" fontSize={10} />
              <Tooltip 
                contentStyle={{ 
                  background: 'rgba(0,0,0,0.9)', 
                  border: '1px solid rgba(123,225,255,0.3)',
                  borderRadius: 4,
                  color: '#E8F6FF'
                }} 
              />
              <Line type="monotone" dataKey="fps" stroke="#63FFC9" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="memory" stroke="#B383FF" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ color: '#8FB2C8', textAlign: 'center', padding: 40 }}>
            Collecting performance data...
          </div>
        )}
      </div>
    </div>
  );
}

// Memory Tab Component
function MemoryTab({ 
  memoryUsage, 
  onCleanup, 
  onSetLimits 
}: {
  memoryUsage: MemoryUsage | null;
  onCleanup: () => void;
  onSetLimits: (maxMB: number) => void;
}) {
  const [memoryLimit, setMemoryLimit] = useState(256);

  return (
    <div>
      <h3 style={{ margin: '0 0 16px 0', color: '#B383FF' }}>Memory Management</h3>
      
      {memoryUsage ? (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16, marginBottom: 24 }}>
            <MetricCard label="Total Objects" value={memoryUsage.totalObjects} unit="" color="#7BE1FF" />
            <MetricCard label="Geometries" value={memoryUsage.geometries} unit="" color="#63FFC9" />
            <MetricCard label="Textures" value={memoryUsage.textures} unit="" color="#B383FF" />
            <MetricCard label="Materials" value={memoryUsage.materials} unit="" color="#FF7BD5" />
            <MetricCard label="Estimated Size" value={memoryUsage.estimatedMB.toFixed(1)} unit="MB" color="#7BE1FF" />
          </div>

          {/* Memory Usage Bar */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <span>Memory Usage</span>
              <span>{memoryUsage.estimatedMB.toFixed(1)}MB / {memoryLimit}MB</span>
            </div>
            <div style={{ 
              width: '100%', 
              height: 20, 
              background: 'rgba(123,225,255,0.1)', 
              borderRadius: 10,
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${Math.min((memoryUsage.estimatedMB / memoryLimit) * 100, 100)}%`,
                height: '100%',
                background: memoryUsage.estimatedMB > memoryLimit * 0.8 ? '#FF7BD5' : '#63FFC9',
                transition: 'width 0.3s ease',
              }} />
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={onCleanup}
              style={{
                background: 'rgba(255,123,213,0.2)',
                border: '1px solid rgba(255,123,213,0.3)',
                color: '#FF7BD5',
                borderRadius: 4,
                padding: '8px 16px',
                cursor: 'pointer',
              }}
            >
              Force Cleanup
            </button>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label>Memory Limit:</label>
              <input
                type="number"
                value={memoryLimit}
                onChange={(e) => setMemoryLimit(Number(e.target.value))}
                style={{
                  background: 'rgba(123,225,255,0.1)',
                  border: '1px solid rgba(123,225,255,0.3)',
                  color: '#E8F6FF',
                  borderRadius: 4,
                  padding: '4px 8px',
                  width: 80,
                }}
              />
              <span>MB</span>
              <button
                onClick={() => onSetLimits(memoryLimit)}
                style={{
                  background: 'rgba(123,225,255,0.1)',
                  border: '1px solid rgba(123,225,255,0.3)',
                  color: '#7BE1FF',
                  borderRadius: 4,
                  padding: '4px 8px',
                  cursor: 'pointer',
                }}
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ color: '#8FB2C8' }}>No memory data available</div>
      )}
    </div>
  );
}

// Profiler Tab Component
function ProfilerTab({ 
  bottlenecks, 
  profiler 
}: {
  bottlenecks: PerformanceBottleneck[];
  profiler: any;
}) {
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);

  useEffect(() => {
    setSessions(profiler.getAllSessions());
  }, [profiler]);

  return (
    <div>
      <h3 style={{ margin: '0 0 16px 0', color: '#B383FF' }}>Performance Profiler</h3>
      
      {/* Session Controls */}
      <div style={{ marginBottom: 24, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button
          onClick={() => {
            const sessionId = profiler.startSession('Manual Session');
            setSessions(profiler.getAllSessions());
          }}
          style={{
            background: 'rgba(99,255,201,0.2)',
            border: '1px solid rgba(99,255,201,0.3)',
            color: '#63FFC9',
            borderRadius: 4,
            padding: '8px 16px',
            cursor: 'pointer',
          }}
        >
          Start Session
        </button>
        
        <button
          onClick={() => {
            profiler.endSession();
            setSessions(profiler.getAllSessions());
          }}
          style={{
            background: 'rgba(255,123,213,0.2)',
            border: '1px solid rgba(255,123,213,0.3)',
            color: '#FF7BD5',
            borderRadius: 4,
            padding: '8px 16px',
            cursor: 'pointer',
          }}
        >
          End Session
        </button>
        
        <button
          onClick={() => {
            profiler.clearSessions();
            setSessions([]);
            setSelectedSession(null);
          }}
          style={{
            background: 'rgba(123,225,255,0.1)',
            border: '1px solid rgba(123,225,255,0.3)',
            color: '#7BE1FF',
            borderRadius: 4,
            padding: '8px 16px',
            cursor: 'pointer',
          }}
        >
          Clear All
        </button>
      </div>

      {/* Sessions List */}
      {sessions.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <h4 style={{ margin: '0 0 12px 0', color: '#B383FF' }}>Sessions</h4>
          <div style={{ display: 'grid', gap: 8 }}>
            {sessions.map(session => (
              <div
                key={session.id}
                onClick={() => setSelectedSession(session.id)}
                style={{
                  padding: 12,
                  background: selectedSession === session.id ? 'rgba(123,225,255,0.2)' : 'rgba(123,225,255,0.05)',
                  border: '1px solid rgba(123,225,255,0.2)',
                  borderRadius: 4,
                  cursor: 'pointer',
                }}
              >
                <div style={{ fontWeight: 'bold' }}>{session.name}</div>
                <div style={{ fontSize: '10px', color: '#8FB2C8' }}>
                  Duration: {session.totalDuration.toFixed(2)}ms | 
                  Entries: {session.entries.length}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bottlenecks */}
      {bottlenecks.length > 0 && (
        <div>
          <h4 style={{ margin: '0 0 12px 0', color: '#B383FF' }}>Performance Bottlenecks</h4>
          <div style={{ display: 'grid', gap: 12 }}>
            {bottlenecks.slice(0, 10).map((bottleneck, index) => (
              <div
                key={bottleneck.name}
                style={{
                  padding: 12,
                  background: 'rgba(255,123,213,0.1)',
                  border: '1px solid rgba(255,123,213,0.2)',
                  borderRadius: 4,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontWeight: 'bold' }}>{bottleneck.name}</span>
                  <span style={{ color: '#FF7BD5' }}>{bottleneck.percentage.toFixed(1)}%</span>
                </div>
                <div style={{ fontSize: '10px', color: '#8FB2C8', marginBottom: 8 }}>
                  Duration: {bottleneck.duration.toFixed(2)}ms | 
                  Occurrences: {bottleneck.occurrences} | 
                  Avg: {bottleneck.averageDuration.toFixed(2)}ms
                </div>
                {bottleneck.suggestions.length > 0 && (
                  <div style={{ fontSize: '10px' }}>
                    <div style={{ color: '#B383FF', marginBottom: 4 }}>Suggestions:</div>
                    {bottleneck.suggestions.map((suggestion, i) => (
                      <div key={i} style={{ color: '#8FB2C8' }}>• {suggestion}</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Metric Card Component
function MetricCard({ 
  label, 
  value, 
  unit, 
  color 
}: {
  label: string;
  value: number | string;
  unit: string;
  color: string;
}) {
  return (
    <div style={{
      padding: 12,
      background: 'rgba(123,225,255,0.05)',
      border: '1px solid rgba(123,225,255,0.1)',
      borderRadius: 4,
    }}>
      <div style={{ fontSize: '10px', color: '#8FB2C8', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: '16px', fontWeight: 'bold', color }}>
        {value}{unit}
      </div>
    </div>
  );
}
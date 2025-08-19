import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { apiService } from '../services/apiService';
import './IntegratedCanvas.css';

interface Token {
  text: string;
  kind: string;
  tags: string[];
  meta: Record<string, any>;
  position?: number;
}

interface Rule {
  id: number;
  name: string;
  priority: number;
  description: string;
  active: boolean;
  applications: number;
  max_applications?: number;
  sutra_ref?: string;
}

interface Transformation {
  rule_name: string;
  rule_id: number;
  index: number;
  tokens_before: Token[];
  tokens_after: Token[];
  timestamp: string;
}

interface ProcessingTrace {
  pass_number: number;
  tokens_before: Token[];
  tokens_after: Token[];
  transformations: Transformation[];
  meta_rule_applications: string[];
}

interface ProcessingResult {
  input_text: string;
  input_tokens: Token[];
  output_tokens: Token[];
  converged: boolean;
  passes: number;
  traces: ProcessingTrace[];
  errors: string[];
}

interface PipelineNode {
  id: string;
  label: string;
  type: 'input' | 'rule' | 'transformation' | 'output' | 'code' | 'result';
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  data?: any;
  stage: number;
}

interface PipelineLink {
  source: string | PipelineNode;
  target: string | PipelineNode;
  type: 'flow' | 'transformation' | 'generation';
  label?: string;
}

const IntegratedCanvas: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [sanskritInput, setSanskritInput] = useState('राम गच्छति');
  const [codeOutput, setCodeOutput] = useState('');
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [availableRules, setAvailableRules] = useState<Rule[]>([]);
  const [activeRules, setActiveRules] = useState<Set<number>>(new Set());
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTransformation, setSelectedTransformation] = useState<Transformation | null>(null);
  const [viewMode, setViewMode] = useState<'pipeline' | 'derivation' | 'rules'>('pipeline');
  const [currentPass, setCurrentPass] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);

  // D3 simulation and elements
  const simulationRef = useRef<d3.Simulation<PipelineNode, PipelineLink> | null>(null);
  const svgElementRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null);

  // Load available rules on component mount
  useEffect(() => {
    loadAvailableRules();
  }, []);

  const loadAvailableRules = async () => {
    try {
      const rules = await apiService.getRules();
      setAvailableRules(rules);
      setActiveRules(new Set(rules.filter(r => r.active).map(r => r.id)));
    } catch (err) {
      console.error('Failed to load rules:', err);
    }
  };

  const processText = useCallback(async () => {
    if (!sanskritInput.trim()) return;

    setIsProcessing(true);
    setError(null);
    setProcessingResult(null);

    try {
      const result = await apiService.processSanskritText({
        text: sanskritInput,
        enable_tracing: true,
        active_rules: Array.from(activeRules),
      });

      setProcessingResult(result);
      setCurrentPass(0);

      // Generate code output based on processing result
      const generatedCode = generateCodeFromResult(result);
      setCodeOutput(generatedCode);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process text');
    } finally {
      setIsProcessing(false);
    }
  }, [sanskritInput, activeRules]);

  const generateCodeFromResult = (result: ProcessingResult): string => {
    // Generate Python code that demonstrates the transformation
    const lines = [
      '# Generated Sanskrit Processing Code',
      `input_text = "${result.input_text}"`,
      '',
      '# Tokenization',
      `tokens = [${result.input_tokens.map(t => `"${t.text}"`).join(', ')}]`,
      '',
      '# Applied Transformations',
    ];

    result.traces.forEach((trace, passIndex) => {
      lines.push(`# Pass ${passIndex + 1}`);
      trace.transformations.forEach((transform, transformIndex) => {
        lines.push(`# Rule: ${transform.rule_name}`);
        lines.push(`before_${passIndex}_${transformIndex} = [${transform.tokens_before.map(t => `"${t.text}"`).join(', ')}]`);
        lines.push(`after_${passIndex}_${transformIndex} = [${transform.tokens_after.map(t => `"${t.text}"`).join(', ')}]`);
      });
    });

    lines.push('');
    lines.push('# Final Result');
    lines.push(`output_tokens = [${result.output_tokens.map(t => `"${t.text}"`).join(', ')}]`);
    lines.push(`output_text = "${result.output_tokens.map(t => t.text).join('')}"`);

    return lines.join('\n');
  };

  const toggleRule = (ruleId: number) => {
    const newActiveRules = new Set(activeRules);
    if (newActiveRules.has(ruleId)) {
      newActiveRules.delete(ruleId);
    } else {
      newActiveRules.add(ruleId);
    }
    setActiveRules(newActiveRules);
  };

  const createPipelineVisualization = useCallback(() => {
    if (!processingResult) return { nodes: [], links: [] };

    const nodes: PipelineNode[] = [];
    const links: PipelineLink[] = [];

    // Input node
    nodes.push({
      id: 'input',
      label: sanskritInput,
      type: 'input',
      stage: 0,
      data: processingResult.input_tokens,
    });

    // Processing stages
    processingResult.traces.forEach((trace, passIndex) => {
      const passId = `pass-${passIndex}`;
      nodes.push({
        id: passId,
        label: `Pass ${passIndex + 1}`,
        type: 'transformation',
        stage: passIndex + 1,
        data: trace,
      });

      // Link from previous stage
      const prevId = passIndex === 0 ? 'input' : `pass-${passIndex - 1}`;
      links.push({
        source: prevId,
        target: passId,
        type: 'flow',
        label: `${trace.transformations.length} rules`,
      });

      // Rule nodes for this pass
      trace.transformations.forEach((transform, transformIndex) => {
        const ruleId = `rule-${passIndex}-${transformIndex}`;
        nodes.push({
          id: ruleId,
          label: transform.rule_name,
          type: 'rule',
          stage: passIndex + 1,
          data: transform,
        });

        links.push({
          source: passId,
          target: ruleId,
          type: 'transformation',
        });
      });
    });

    // Code generation node
    nodes.push({
      id: 'code',
      label: 'Generated Code',
      type: 'code',
      stage: processingResult.traces.length + 1,
      data: codeOutput,
    });

    // Output node
    nodes.push({
      id: 'output',
      label: processingResult.output_tokens.map(t => t.text).join(''),
      type: 'output',
      stage: processingResult.traces.length + 2,
      data: processingResult.output_tokens,
    });

    // Link to code and output
    const lastPassId = `pass-${processingResult.traces.length - 1}`;
    links.push({
      source: lastPassId,
      target: 'code',
      type: 'generation',
      label: 'generate',
    });

    links.push({
      source: 'code',
      target: 'output',
      type: 'flow',
      label: 'execute',
    });

    return { nodes, links };
  }, [processingResult, sanskritInput, codeOutput]);

  // Auto-play functionality for step-by-step visualization
  useEffect(() => {
    if (!autoPlay || !processingResult) return;

    const interval = setInterval(() => {
      setCurrentPass(prev => {
        const maxPass = processingResult.traces.length - 1;
        return prev >= maxPass ? 0 : prev + 1;
      });
    }, playbackSpeed);

    return () => clearInterval(interval);
  }, [autoPlay, processingResult, playbackSpeed]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || viewMode !== 'pipeline') return;

    const { nodes, links } = createPipelineVisualization();
    if (nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svgElementRef.current = svg;

    // Clear previous content
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create container for zoomable content
    const container = svg.append('g');

    // Position nodes by stage
    nodes.forEach((node, index) => {
      node.x = (node.stage * width) / (Math.max(...nodes.map(n => n.stage)) + 1);
      node.y = height / 2 + (index % 3 - 1) * 80;
    });

    // Create links
    const link = container.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('class', 'pipeline-link')
      .attr('stroke', d => {
        switch (d.type) {
          case 'flow': return '#007bff';
          case 'transformation': return '#dc3545';
          case 'generation': return '#28a745';
          default: return '#6c757d';
        }
      })
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.7)
      .attr('marker-end', 'url(#arrowhead)');

    // Add arrowhead marker
    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#6c757d');

    // Create nodes
    const node = container.append('g')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('class', 'pipeline-node')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .on('click', (event, d) => {
        if (d.type === 'rule' && d.data) {
          setSelectedTransformation(d.data);
        }
      });

    // Add node shapes
    node.append('rect')
      .attr('width', 120)
      .attr('height', 60)
      .attr('x', -60)
      .attr('y', -30)
      .attr('rx', 8)
      .attr('fill', d => {
        switch (d.type) {
          case 'input': return '#e3f2fd';
          case 'rule': return '#fff3e0';
          case 'transformation': return '#f3e5f5';
          case 'code': return '#e8f5e8';
          case 'output': return '#fce4ec';
          default: return '#f8f9fa';
        }
      })
      .attr('stroke', d => {
        switch (d.type) {
          case 'input': return '#2196f3';
          case 'rule': return '#ff9800';
          case 'transformation': return '#9c27b0';
          case 'code': return '#4caf50';
          case 'output': return '#e91e63';
          default: return '#dee2e6';
        }
      })
      .attr('stroke-width', 2);

    // Add node labels
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '12px')
      .attr('font-weight', '500')
      .attr('fill', '#212529')
      .text(d => {
        const maxLength = 15;
        return d.label.length > maxLength ? d.label.substring(0, maxLength) + '...' : d.label;
      });

    // Add tooltips
    node.append('title')
      .text(d => `${d.type}: ${d.label}`);

    // Update link positions
    link
      .attr('x1', d => (d.source as PipelineNode).x || 0)
      .attr('y1', d => (d.source as PipelineNode).y || 0)
      .attr('x2', d => (d.target as PipelineNode).x || 0)
      .attr('y2', d => (d.target as PipelineNode).y || 0);

  }, [createPipelineVisualization, viewMode]);

  return (
    <div className="integrated-canvas">
      <div className="canvas-header">
        <h2>Sanskrit Development Canvas</h2>
        <div className="canvas-controls">
          <div className="input-group">
            <input
              type="text"
              value={sanskritInput}
              onChange={(e) => setSanskritInput(e.target.value)}
              placeholder="Enter Sanskrit text..."
              className="text-input sanskrit-text"
            />
            <button
              onClick={processText}
              disabled={isProcessing || !sanskritInput.trim()}
              className="btn btn-primary"
            >
              {isProcessing ? '⏳ Processing...' : '🔄 Process'}
            </button>
          </div>
        </div>
      </div>

      <div className="canvas-toolbar">
        <div className="view-modes">
          <button
            onClick={() => setViewMode('pipeline')}
            className={`btn ${viewMode === 'pipeline' ? 'btn-primary' : 'btn-outline'}`}
          >
            🔄 Pipeline
          </button>
          <button
            onClick={() => setViewMode('derivation')}
            className={`btn ${viewMode === 'derivation' ? 'btn-primary' : 'btn-outline'}`}
          >
            📋 Derivation
          </button>
          <button
            onClick={() => setViewMode('rules')}
            className={`btn ${viewMode === 'rules' ? 'btn-primary' : 'btn-outline'}`}
          >
            ⚙️ Rules
          </button>
        </div>

        {processingResult && (
          <div className="playback-controls">
            <button
              onClick={() => setAutoPlay(!autoPlay)}
              className={`btn ${autoPlay ? 'btn-danger' : 'btn-success'}`}
            >
              {autoPlay ? '⏸️ Pause' : '▶️ Play'}
            </button>
            <input
              type="range"
              min="0"
              max={processingResult.traces.length - 1}
              value={currentPass}
              onChange={(e) => setCurrentPass(parseInt(e.target.value))}
              className="pass-slider"
            />
            <span className="pass-indicator">
              Pass {currentPass + 1} / {processingResult.traces.length}
            </span>
            <input
              type="range"
              min="500"
              max="3000"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
              className="speed-slider"
            />
            <span className="speed-indicator">{playbackSpeed}ms</span>
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="error-close">×</button>
        </div>
      )}

      <div className="canvas-content">
        {viewMode === 'pipeline' && (
          <div className="pipeline-view">
            <svg
              ref={svgRef}
              className="pipeline-svg"
              width="100%"
              height="400"
            />
          </div>
        )}

        {viewMode === 'derivation' && processingResult && (
          <div className="derivation-view">
            <DerivationSteps
              traces={processingResult.traces}
              currentPass={currentPass}
              onTransformationSelect={setSelectedTransformation}
            />
          </div>
        )}

        {viewMode === 'rules' && (
          <div className="rules-view">
            <RuleComposer
              availableRules={availableRules}
              activeRules={activeRules}
              onToggleRule={toggleRule}
              onRuleUpdate={loadAvailableRules}
            />
          </div>
        )}
      </div>

      <div className="output-panel">
        <div className="code-output">
          <h3>Generated Code</h3>
          <pre className="code-block">
            <code>{codeOutput}</code>
          </pre>
        </div>

        {selectedTransformation && (
          <div className="transformation-details">
            <h3>Transformation Details</h3>
            <TransformationInspector transformation={selectedTransformation} />
          </div>
        )}
      </div>
    </div>
  );
};

// Sub-components for different views
interface DerivationStepsProps {
  traces: ProcessingTrace[];
  currentPass: number;
  onTransformationSelect: (transformation: Transformation) => void;
}

const DerivationSteps: React.FC<DerivationStepsProps> = ({
  traces,
  currentPass,
  onTransformationSelect,
}) => {
  const currentTrace = traces[currentPass];

  return (
    <div className="derivation-steps">
      <div className="step-header">
        <h3>Pass {currentPass + 1} - Prakriyā Steps</h3>
        <div className="tokens-before-after">
          <div className="tokens-before">
            <strong>Before:</strong> {currentTrace.tokens_before.map(t => t.text).join(' ')}
          </div>
          <div className="tokens-after">
            <strong>After:</strong> {currentTrace.tokens_after.map(t => t.text).join(' ')}
          </div>
        </div>
      </div>

      <div className="transformations-list">
        {currentTrace.transformations.map((transform, index) => (
          <div
            key={index}
            className="transformation-step"
            onClick={() => onTransformationSelect(transform)}
          >
            <div className="step-number">{index + 1}</div>
            <div className="step-content">
              <div className="rule-name">{transform.rule_name}</div>
              <div className="step-tokens">
                <span className="before-tokens">
                  {transform.tokens_before.map(t => t.text).join(' ')}
                </span>
                <span className="arrow">→</span>
                <span className="after-tokens">
                  {transform.tokens_after.map(t => t.text).join(' ')}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

interface RuleComposerProps {
  availableRules: Rule[];
  activeRules: Set<number>;
  onToggleRule: (ruleId: number) => void;
  onRuleUpdate: () => void;
}

const RuleComposer: React.FC<RuleComposerProps> = ({
  availableRules,
  activeRules,
  onToggleRule,
  onRuleUpdate,
}) => {
  return (
    <div className="rule-composer">
      <div className="rules-header">
        <h3>Rule Configuration</h3>
        <button onClick={onRuleUpdate} className="btn btn-outline">
          🔄 Refresh Rules
        </button>
      </div>

      <div className="rules-list">
        {availableRules.map((rule) => (
          <div
            key={rule.id}
            className={`rule-item ${activeRules.has(rule.id) ? 'active' : 'inactive'}`}
          >
            <div className="rule-toggle">
              <input
                type="checkbox"
                checked={activeRules.has(rule.id)}
                onChange={() => onToggleRule(rule.id)}
              />
            </div>
            <div className="rule-info">
              <div className="rule-name">{rule.name}</div>
              <div className="rule-description">{rule.description}</div>
              <div className="rule-meta">
                Priority: {rule.priority} | Applications: {rule.applications}
                {rule.max_applications && ` / ${rule.max_applications}`}
                {rule.sutra_ref && ` | Sūtra: ${rule.sutra_ref}`}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

interface TransformationInspectorProps {
  transformation: Transformation;
}

const TransformationInspector: React.FC<TransformationInspectorProps> = ({
  transformation,
}) => {
  return (
    <div className="transformation-inspector">
      <div className="inspector-header">
        <h4>{transformation.rule_name}</h4>
        <span className="rule-id">ID: {transformation.rule_id}</span>
      </div>

      <div className="token-comparison">
        <div className="tokens-section">
          <h5>Before Transformation</h5>
          <div className="token-list">
            {transformation.tokens_before.map((token, index) => (
              <div key={index} className="token-detail">
                <span className="token-text">{token.text}</span>
                <span className="token-kind">{token.kind}</span>
                {token.tags.length > 0 && (
                  <span className="token-tags">{token.tags.join(', ')}</span>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="tokens-section">
          <h5>After Transformation</h5>
          <div className="token-list">
            {transformation.tokens_after.map((token, index) => (
              <div key={index} className="token-detail">
                <span className="token-text">{token.text}</span>
                <span className="token-kind">{token.kind}</span>
                {token.tags.length > 0 && (
                  <span className="token-tags">{token.tags.join(', ')}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="transformation-metadata">
        <div className="meta-item">
          <strong>Index:</strong> {transformation.index}
        </div>
        <div className="meta-item">
          <strong>Timestamp:</strong> {new Date(transformation.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default IntegratedCanvas;
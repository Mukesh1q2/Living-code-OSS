import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { apiService } from '../services/apiService';
import './DiagramCanvas.css';

interface Node {
  id: string;
  label: string;
  type: 'token' | 'rule' | 'transformation' | 'semantic';
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface Link {
  source: string | Node;
  target: string | Node;
  type: 'transformation' | 'dependency' | 'semantic';
  label?: string;
}

interface DiagramData {
  nodes: Node[];
  links: Link[];
}

const DiagramCanvas: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [inputText, setInputText] = useState('राम गच्छति');
  const [diagramData, setDiagramData] = useState<DiagramData>({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'tokens' | 'rules' | 'semantic'>('tokens');
  const [zoomLevel, setZoomLevel] = useState(1);

  // D3 simulation and elements
  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null);
  const svgElementRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null);

  const processText = useCallback(async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await apiService.processSanskritText({
        text: inputText,
        enable_tracing: true,
      });

      // Convert processing result to diagram data
      const nodes: Node[] = [];
      const links: Link[] = [];

      if (viewMode === 'tokens') {
        // Create nodes for tokens
        result.output_tokens.forEach((token, index) => {
          nodes.push({
            id: `token-${index}`,
            label: typeof token === 'string' ? token : token.text || `Token ${index}`,
            type: 'token',
          });
        });

        // Create links between consecutive tokens
        for (let i = 0; i < nodes.length - 1; i++) {
          links.push({
            source: nodes[i].id,
            target: nodes[i + 1].id,
            type: 'dependency',
          });
        }
      } else if (viewMode === 'rules') {
        // Create nodes for transformations
        result.traces.flatMap(trace => trace.transformations).forEach((transformation, index) => {
          nodes.push({
            id: `rule-${index}`,
            label: transformation.rule_name || `Rule ${index}`,
            type: 'rule',
          });

          if (transformation.tokens_before) {
            transformation.tokens_before.forEach((token, tokenIndex) => {
              const tokenId = `before-${index}-${tokenIndex}`;
              nodes.push({
                id: tokenId,
                label: typeof token === 'string' ? token : token.text || `Token ${tokenIndex}`,
                type: 'token',
              });

              links.push({
                source: tokenId,
                target: `rule-${index}`,
                type: 'transformation',
                label: 'input',
              });
            });
          }

          if (transformation.tokens_after) {
            transformation.tokens_after.forEach((token, tokenIndex) => {
              const tokenId = `after-${index}-${tokenIndex}`;
              nodes.push({
                id: tokenId,
                label: typeof token === 'string' ? token : token.text || `Token ${tokenIndex}`,
                type: 'token',
              });

              links.push({
                source: `rule-${index}`,
                target: tokenId,
                type: 'transformation',
                label: 'output',
              });
            });
          }
        });
      } else if (viewMode === 'semantic') {
        // Create nodes for semantic graph
        // Create semantic nodes from processing metadata
        if (result.traces && result.traces.length > 0) {
          result.traces.forEach((trace, traceIndex) => {
            nodes.push({
              id: `trace-${traceIndex}`,
              label: `Pass ${trace.pass_number}`,
              type: 'semantic',
            });

            trace.transformations.forEach((transform, transformIndex) => {
              const transformId = `transform-${traceIndex}-${transformIndex}`;
              nodes.push({
                id: transformId,
                label: transform.rule_name,
                type: 'semantic',
              });

              links.push({
                source: `trace-${traceIndex}`,
                target: transformId,
                type: 'semantic',
              });
            });
          });
        }
      }

      setDiagramData({ nodes, links });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process text');
    } finally {
      setIsLoading(false);
    }
  }, [inputText, viewMode]);

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current || diagramData.nodes.length === 0) return;

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
        setZoomLevel(event.transform.k);
      });

    svg.call(zoom);

    // Create container for zoomable content
    const container = svg.append('g');

    // Create simulation
    const simulation = d3.forceSimulation<Node>(diagramData.nodes)
      .force('link', d3.forceLink<Node, Link>(diagramData.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    // Create links
    const link = container.append('g')
      .selectAll('line')
      .data(diagramData.links)
      .enter().append('line')
      .attr('class', 'link')
      .attr('stroke', d => {
        switch (d.type) {
          case 'transformation': return '#007bff';
          case 'dependency': return '#6c757d';
          case 'semantic': return '#28a745';
          default: return '#dee2e6';
        }
      })
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6);

    // Create link labels
    const linkLabel = container.append('g')
      .selectAll('text')
      .data(diagramData.links.filter(d => d.label))
      .enter().append('text')
      .attr('class', 'link-label')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#6c757d')
      .text(d => d.label || '');

    // Create nodes
    const node = container.append('g')
      .selectAll('g')
      .data(diagramData.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag<SVGGElement, Node>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    // Add circles to nodes
    node.append('circle')
      .attr('r', 20)
      .attr('fill', d => {
        switch (d.type) {
          case 'token': return '#e3f2fd';
          case 'rule': return '#fff3e0';
          case 'transformation': return '#f3e5f5';
          case 'semantic': return '#e8f5e8';
          default: return '#f8f9fa';
        }
      })
      .attr('stroke', d => {
        switch (d.type) {
          case 'token': return '#2196f3';
          case 'rule': return '#ff9800';
          case 'transformation': return '#9c27b0';
          case 'semantic': return '#4caf50';
          default: return '#dee2e6';
        }
      })
      .attr('stroke-width', 2);

    // Add labels to nodes
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '12px')
      .attr('font-weight', '500')
      .attr('fill', '#212529')
      .text(d => {
        const maxLength = 10;
        return d.label.length > maxLength ? d.label.substring(0, maxLength) + '...' : d.label;
      })
      .attr('class', d => d.label.match(/[\u0900-\u097F]/) ? 'sanskrit-text' : '');

    // Add tooltips
    node.append('title')
      .text(d => `${d.type}: ${d.label}`);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as Node).x || 0)
        .attr('y1', d => (d.source as Node).y || 0)
        .attr('x2', d => (d.target as Node).x || 0)
        .attr('y2', d => (d.target as Node).y || 0);

      linkLabel
        .attr('x', d => ((d.source as Node).x! + (d.target as Node).x!) / 2)
        .attr('y', d => ((d.source as Node).y! + (d.target as Node).y!) / 2);

      node
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [diagramData]);

  const handleReset = () => {
    if (svgElementRef.current) {
      const svg = svgElementRef.current;
      const width = svgRef.current!.clientWidth;
      const height = svgRef.current!.clientHeight;
      
      svg.transition()
        .duration(750)
        .call(
          d3.zoom<SVGSVGElement, unknown>().transform,
          d3.zoomIdentity.translate(0, 0).scale(1)
        );
      
      setZoomLevel(1);
    }
  };

  const handleExport = () => {
    if (!svgRef.current) return;

    const svgElement = svgRef.current;
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `sanskrit-diagram-${Date.now()}.svg`;
    link.click();
    
    URL.revokeObjectURL(url);
  };

  return (
    <div className="diagram-canvas">
      <div className="canvas-header">
        <h2>Sanskrit Visualization Canvas</h2>
        <div className="canvas-controls">
          <div className="input-group">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter Sanskrit text..."
              className="text-input sanskrit-text"
            />
            <button
              onClick={processText}
              disabled={isLoading || !inputText.trim()}
              className="btn btn-primary"
            >
              {isLoading ? '⏳ Processing...' : '🔍 Analyze'}
            </button>
          </div>
        </div>
      </div>

      <div className="canvas-toolbar">
        <div className="view-modes">
          <button
            onClick={() => setViewMode('tokens')}
            className={`btn ${viewMode === 'tokens' ? 'btn-primary' : 'btn-outline'}`}
          >
            📝 Tokens
          </button>
          <button
            onClick={() => setViewMode('rules')}
            className={`btn ${viewMode === 'rules' ? 'btn-primary' : 'btn-outline'}`}
          >
            ⚙️ Rules
          </button>
          <button
            onClick={() => setViewMode('semantic')}
            className={`btn ${viewMode === 'semantic' ? 'btn-primary' : 'btn-outline'}`}
          >
            🧠 Semantic
          </button>
        </div>

        <div className="canvas-actions">
          <span className="zoom-level">Zoom: {Math.round(zoomLevel * 100)}%</span>
          <button onClick={handleReset} className="btn btn-outline">
            🔄 Reset View
          </button>
          <button onClick={handleExport} className="btn btn-outline">
            💾 Export SVG
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="error-close">×</button>
        </div>
      )}

      <div className="canvas-container">
        {diagramData.nodes.length === 0 ? (
          <div className="empty-canvas">
            <div className="empty-content">
              <span className="empty-icon">🎨</span>
              <h3>Visualization Canvas</h3>
              <p>
                Enter Sanskrit text above and click "Analyze" to visualize:
              </p>
              <ul>
                <li><strong>Tokens:</strong> Word structure and boundaries</li>
                <li><strong>Rules:</strong> Grammatical transformations</li>
                <li><strong>Semantic:</strong> Meaning relationships</li>
              </ul>
            </div>
          </div>
        ) : (
          <svg
            ref={svgRef}
            className="diagram-svg"
            width="100%"
            height="100%"
          />
        )}
      </div>

      <div className="canvas-legend">
        <div className="legend-title">Legend</div>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-color token"></div>
            <span>Token</span>
          </div>
          <div className="legend-item">
            <div className="legend-color rule"></div>
            <span>Rule</span>
          </div>
          <div className="legend-item">
            <div className="legend-color semantic"></div>
            <span>Semantic</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiagramCanvas;
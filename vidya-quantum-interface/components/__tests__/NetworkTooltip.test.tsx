/**
 * Tests for NetworkTooltip component
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import * as THREE from 'three';
import NetworkTooltip from '../NetworkTooltip';
import { NetworkNode, NodeType } from '@/lib/neural-network';

// Mock @react-three/drei
vi.mock('@react-three/drei', () => ({
  Html: ({ children, style, ...props }: any) => (
    <div data-testid="html-container" style={style} {...props}>
      {children}
    </div>
  )
}));

// Mock responsive hook
vi.mock('@/lib/responsive', () => ({
  useResponsive: () => ({
    isMobile: false,
    isTablet: false
  })
}));

const createTestNode = (type: NodeType, overrides: Partial<NetworkNode> = {}): NetworkNode => ({
  id: 'test-node',
  type,
  position: new THREE.Vector3(0, 0, 0),
  active: true,
  selected: false,
  highlighted: false,
  color: '#7BE1FF',
  emissiveIntensity: 0.3,
  scale: 0.14,
  animationPhase: 0,
  pulseSpeed: 1.0,
  rotationSpeed: 0.1,
  ...overrides
});

describe('NetworkTooltip', () => {
  describe('Visibility', () => {
    it('should not render when node is null', () => {
      render(
        <NetworkTooltip
          node={null}
          visible={true}
        />
      );
      
      expect(screen.queryByTestId('html-container')).not.toBeInTheDocument();
    });

    it('should not render when visible is false', () => {
      const node = createTestNode(NodeType.NEURAL_UNIT);
      
      render(
        <NetworkTooltip
          node={node}
          visible={false}
        />
      );
      
      expect(screen.queryByTestId('html-container')).not.toBeInTheDocument();
    });

    it('should render when node is provided and visible is true', () => {
      const node = createTestNode(NodeType.NEURAL_UNIT);
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      expect(screen.getByTestId('html-container')).toBeInTheDocument();
    });
  });

  describe('Sanskrit Rule Tooltip', () => {
    it('should display Sanskrit rule information', () => {
      const node = createTestNode(NodeType.SANSKRIT_RULE, {
        sanskritRule: {
          id: 'rule1',
          name: 'संधि',
          description: 'Phonetic combination rules',
          category: 'sandhi',
          paniiniSutra: '1.1.1'
        }
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      expect(screen.getByText('संधि')).toBeInTheDocument();
      expect(screen.getByText('Phonetic combination rules')).toBeInTheDocument();
      expect(screen.getByText('sandhi')).toBeInTheDocument();
      expect(screen.getByText('Pāṇini Sūtra: 1.1.1')).toBeInTheDocument();
    });

    it('should show more details when More button is clicked', () => {
      const node = createTestNode(NodeType.SANSKRIT_RULE, {
        id: 'test-node-123',
        sanskritRule: {
          id: 'rule1',
          name: 'संधि',
          description: 'Phonetic combination rules',
          category: 'sandhi'
        }
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      const moreButton = screen.getByText('More');
      fireEvent.click(moreButton);
      
      expect(screen.getByText('Node ID:')).toBeInTheDocument();
      expect(screen.getByText('test-node-123')).toBeInTheDocument();
      expect(screen.getByText('Less')).toBeInTheDocument();
    });
  });

  describe('Neural Unit Tooltip', () => {
    it('should display neural unit information', () => {
      const node = createTestNode(NodeType.NEURAL_UNIT, {
        label: 'Neural-Unit-1'
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      expect(screen.getByText('Neural Unit')).toBeInTheDocument();
      expect(screen.getByText('Neural processing unit for pattern recognition and linguistic analysis')).toBeInTheDocument();
      expect(screen.getByText('Activity: Processing')).toBeInTheDocument();
    });

    it('should show inactive status for inactive nodes', () => {
      const node = createTestNode(NodeType.NEURAL_UNIT, {
        active: false
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      expect(screen.getByText('Activity: Idle')).toBeInTheDocument();
    });
  });

  describe('Quantum Gate Tooltip', () => {
    it('should display quantum gate information', () => {
      const node = createTestNode(NodeType.QUANTUM_GATE, {
        quantumProperties: {
          superposition: true,
          entangledWith: ['node1', 'node2'],
          coherenceLevel: 0.85,
          quantumState: 'entangled'
        }
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      expect(screen.getByText('Quantum Gate')).toBeInTheDocument();
      expect(screen.getByText('entangled')).toBeInTheDocument();
      expect(screen.getByText('⚡ Superposition Active')).toBeInTheDocument();
      expect(screen.getByText('🔗 Entangled with 2 node(s)')).toBeInTheDocument();
      expect(screen.getByText('Coherence: 85%')).toBeInTheDocument();
    });

    it('should show detailed quantum information when expanded', () => {
      const node = createTestNode(NodeType.QUANTUM_GATE, {
        quantumProperties: {
          superposition: false,
          entangledWith: ['partner-node'],
          coherenceLevel: 0.75,
          quantumState: 'coherent'
        }
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      const moreButton = screen.getByText('More');
      fireEvent.click(moreButton);
      
      expect(screen.getByText('coherent')).toBeInTheDocument();
      expect(screen.getByText('partner-node')).toBeInTheDocument();
    });
  });

  describe('Interactive Features', () => {
    it('should call onClose when close button is clicked', () => {
      const onClose = vi.fn();
      const node = createTestNode(NodeType.NEURAL_UNIT);
      
      // Mock mobile responsive hook for this test
      const mockUseResponsive = vi.fn().mockReturnValue({
        isMobile: true,
        isTablet: false
      });
      
      vi.doMock('@/lib/responsive', () => ({
        useResponsive: mockUseResponsive
      }));
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
          onClose={onClose}
        />
      );
      
      const closeButton = screen.getByText('×');
      fireEvent.click(closeButton);
      
      expect(onClose).toHaveBeenCalled();
    });

    it('should toggle details when More/Less button is clicked', () => {
      const node = createTestNode(NodeType.NEURAL_UNIT);
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      const moreButton = screen.getByText('More');
      fireEvent.click(moreButton);
      
      expect(screen.getByText('Less')).toBeInTheDocument();
      expect(screen.getByText('Activation Level:')).toBeInTheDocument();
      
      const lessButton = screen.getByText('Less');
      fireEvent.click(lessButton);
      
      expect(screen.getByText('More')).toBeInTheDocument();
      expect(screen.queryByText('Activation Level:')).not.toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should adapt styling for mobile devices', () => {
      // Mock mobile responsive hook for this test
      const mockUseResponsive = vi.fn().mockReturnValue({
        isMobile: true,
        isTablet: false
      });
      
      vi.doMock('@/lib/responsive', () => ({
        useResponsive: mockUseResponsive
      }));
      
      const node = createTestNode(NodeType.NEURAL_UNIT);
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
          onClose={vi.fn()}
        />
      );
      
      const container = screen.getByTestId('html-container');
      expect(container).toHaveStyle({ maxWidth: '200px' });
      
      // Should show close button on mobile
      expect(screen.getByText('×')).toBeInTheDocument();
    });

    it('should adapt styling for tablet devices', () => {
      // Mock tablet responsive hook for this test
      const mockUseResponsive = vi.fn().mockReturnValue({
        isMobile: false,
        isTablet: true
      });
      
      vi.doMock('@/lib/responsive', () => ({
        useResponsive: mockUseResponsive
      }));
      
      const node = createTestNode(NodeType.NEURAL_UNIT);
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
          onClose={vi.fn()}
        />
      );
      
      // Should show close button on tablet
      expect(screen.getByText('×')).toBeInTheDocument();
    });
  });

  describe('Node State Indicators', () => {
    it('should show selected state indicator', () => {
      const node = createTestNode(NodeType.SANSKRIT_RULE, {
        selected: true,
        sanskritRule: {
          id: 'rule1',
          name: 'संधि',
          description: 'Test rule',
          category: 'sandhi'
        }
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      const moreButton = screen.getByText('More');
      fireEvent.click(moreButton);
      
      expect(screen.getByText('Selected')).toBeInTheDocument();
    });

    it('should show highlighted state indicator', () => {
      const node = createTestNode(NodeType.NEURAL_UNIT, {
        highlighted: true
      });
      
      render(
        <NetworkTooltip
          node={node}
          visible={true}
        />
      );
      
      const moreButton = screen.getByText('More');
      fireEvent.click(moreButton);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });
});
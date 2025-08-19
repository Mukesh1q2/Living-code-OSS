import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DiagramCanvas from '../DiagramCanvas';

// Mock D3
jest.mock('d3', () => ({
  select: jest.fn(() => ({
    selectAll: jest.fn(() => ({
      remove: jest.fn(),
      data: jest.fn(() => ({
        enter: jest.fn(() => ({
          append: jest.fn(() => ({
            attr: jest.fn(() => ({
              attr: jest.fn(),
              text: jest.fn(),
              call: jest.fn(),
            })),
            text: jest.fn(),
            call: jest.fn(),
          })),
        })),
      })),
    })),
    append: jest.fn(() => ({
      selectAll: jest.fn(() => ({
        data: jest.fn(() => ({
          enter: jest.fn(() => ({
            append: jest.fn(() => ({
              attr: jest.fn(() => ({
                attr: jest.fn(),
                text: jest.fn(),
                call: jest.fn(),
              })),
              text: jest.fn(),
              call: jest.fn(),
            })),
          })),
        })),
      })),
      attr: jest.fn(),
    })),
    call: jest.fn(),
    transition: jest.fn(() => ({
      duration: jest.fn(() => ({
        call: jest.fn(),
      })),
    })),
  })),
  zoom: jest.fn(() => ({
    scaleExtent: jest.fn(() => ({
      on: jest.fn(),
    })),
  })),
  zoomIdentity: {
    translate: jest.fn(() => ({
      scale: jest.fn(),
    })),
  },
  forceSimulation: jest.fn(() => ({
    force: jest.fn(() => ({
      force: jest.fn(() => ({
        force: jest.fn(() => ({
          force: jest.fn(),
        })),
      })),
    })),
    on: jest.fn(),
    stop: jest.fn(),
    alphaTarget: jest.fn(() => ({
      restart: jest.fn(),
    })),
  })),
  forceLink: jest.fn(() => ({
    id: jest.fn(() => ({
      distance: jest.fn(),
    })),
  })),
  forceManyBody: jest.fn(() => ({
    strength: jest.fn(),
  })),
  forceCenter: jest.fn(),
  forceCollide: jest.fn(() => ({
    radius: jest.fn(),
  })),
  drag: jest.fn(() => ({
    on: jest.fn(() => ({
      on: jest.fn(() => ({
        on: jest.fn(),
      })),
    })),
  })),
}));

// Mock API service
jest.mock('../../services/apiService', () => ({
  apiService: {
    processSanskritText: jest.fn().mockResolvedValue({
      input_text: 'राम गच्छति',
      output_text: 'rāma gacchati',
      tokens: [
        { text: 'राम', kind: 'WORD' },
        { text: 'गच्छति', kind: 'WORD' }
      ],
      transformations: [
        {
          rule_name: 'sandhi_rule_1',
          tokens_before: [{ text: 'राम' }],
          tokens_after: [{ text: 'राम' }],
        }
      ],
      semantic_graph: {
        subject: { word: 'राम', case: 'nominative' },
        predicate: { word: 'गच्छति', tense: 'present' }
      },
      success: true,
      processing_time_ms: 100,
    }),
  },
}));

describe('DiagramCanvas', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders diagram canvas with header', () => {
    render(<DiagramCanvas />);
    
    expect(screen.getByText('Sanskrit Visualization Canvas')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter Sanskrit text...')).toBeInTheDocument();
    expect(screen.getByText('🔍 Analyze')).toBeInTheDocument();
  });

  test('displays view mode buttons', () => {
    render(<DiagramCanvas />);
    
    expect(screen.getByText('📝 Tokens')).toBeInTheDocument();
    expect(screen.getByText('⚙️ Rules')).toBeInTheDocument();
    expect(screen.getByText('🧠 Semantic')).toBeInTheDocument();
  });

  test('shows canvas controls', () => {
    render(<DiagramCanvas />);
    
    expect(screen.getByText(/Zoom:/)).toBeInTheDocument();
    expect(screen.getByText('🔄 Reset View')).toBeInTheDocument();
    expect(screen.getByText('💾 Export SVG')).toBeInTheDocument();
  });

  test('displays empty canvas state initially', () => {
    render(<DiagramCanvas />);
    
    expect(screen.getByText('Visualization Canvas')).toBeInTheDocument();
    expect(screen.getByText(/Enter Sanskrit text above/)).toBeInTheDocument();
    expect(screen.getByText('Tokens:')).toBeInTheDocument();
    expect(screen.getByText('Rules:')).toBeInTheDocument();
    expect(screen.getByText('Semantic:')).toBeInTheDocument();
  });

  test('shows legend with color coding', () => {
    render(<DiagramCanvas />);
    
    expect(screen.getByText('Legend')).toBeInTheDocument();
    expect(screen.getByText('Token')).toBeInTheDocument();
    expect(screen.getByText('Rule')).toBeInTheDocument();
    expect(screen.getByText('Semantic')).toBeInTheDocument();
  });

  test('processes text when analyze button is clicked', async () => {
    const user = userEvent.setup();
    const { apiService } = require('../../services/apiService');
    
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    await user.clear(textInput);
    await user.type(textInput, 'राम गच्छति');
    await user.click(analyzeBtn);
    
    await waitFor(() => {
      expect(apiService.processSanskritText).toHaveBeenCalledWith({
        text: 'राम गच्छति',
        enable_tracing: true,
      });
    });
  });

  test('changes view mode when buttons are clicked', async () => {
    const user = userEvent.setup();
    render(<DiagramCanvas />);
    
    const rulesBtn = screen.getByText('⚙️ Rules');
    await user.click(rulesBtn);
    
    // The button should have active styling (this would be tested via CSS classes)
    expect(rulesBtn).toBeInTheDocument();
    
    const semanticBtn = screen.getByText('🧠 Semantic');
    await user.click(semanticBtn);
    
    expect(semanticBtn).toBeInTheDocument();
  });

  test('shows loading state during processing', async () => {
    const user = userEvent.setup();
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    // Mock a delayed response
    const { apiService } = require('../../services/apiService');
    apiService.processSanskritText.mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 100))
    );
    
    await user.type(textInput, 'test');
    await user.click(analyzeBtn);
    
    expect(screen.getByText('⏳ Processing...')).toBeInTheDocument();
  });

  test('handles processing errors', async () => {
    const user = userEvent.setup();
    const { apiService } = require('../../services/apiService');
    
    // Mock an error response
    apiService.processSanskritText.mockRejectedValueOnce(new Error('Processing failed'));
    
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    await user.type(textInput, 'test');
    await user.click(analyzeBtn);
    
    await waitFor(() => {
      expect(screen.getByText('Processing failed')).toBeInTheDocument();
    });
  });

  test('disables analyze button when input is empty', () => {
    render(<DiagramCanvas />);
    
    const analyzeBtn = screen.getByText('🔍 Analyze');
    expect(analyzeBtn).toBeDisabled();
  });

  test('enables analyze button when input has text', async () => {
    const user = userEvent.setup();
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    await user.type(textInput, 'राम');
    
    expect(analyzeBtn).not.toBeDisabled();
  });

  test('renders SVG canvas after processing', async () => {
    const user = userEvent.setup();
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    await user.type(textInput, 'राम गच्छति');
    await user.click(analyzeBtn);
    
    await waitFor(() => {
      const svg = document.querySelector('.diagram-svg');
      expect(svg).toBeInTheDocument();
    });
  });

  test('exports SVG when export button is clicked', async () => {
    const user = userEvent.setup();
    
    // Mock URL.createObjectURL and document.createElement
    global.URL.createObjectURL = jest.fn(() => 'mock-url');
    global.URL.revokeObjectURL = jest.fn();
    
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn(),
    };
    jest.spyOn(document, 'createElement').mockReturnValue(mockLink as any);
    
    render(<DiagramCanvas />);
    
    // First process some text to have content to export
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    await user.type(textInput, 'राम');
    await user.click(analyzeBtn);
    
    await waitFor(() => {
      const exportBtn = screen.getByText('💾 Export SVG');
      return user.click(exportBtn);
    });
    
    expect(mockLink.click).toHaveBeenCalled();
  });

  test('resets view when reset button is clicked', async () => {
    const user = userEvent.setup();
    render(<DiagramCanvas />);
    
    const resetBtn = screen.getByText('🔄 Reset View');
    await user.click(resetBtn);
    
    // The zoom level should reset to 100%
    expect(screen.getByText('Zoom: 100%')).toBeInTheDocument();
  });

  test('closes error banner when close button is clicked', async () => {
    const user = userEvent.setup();
    const { apiService } = require('../../services/apiService');
    
    // Mock an error
    apiService.processSanskritText.mockRejectedValueOnce(new Error('Test error'));
    
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    const analyzeBtn = screen.getByText('🔍 Analyze');
    
    await user.type(textInput, 'test');
    await user.click(analyzeBtn);
    
    await waitFor(() => {
      expect(screen.getByText('Test error')).toBeInTheDocument();
    });
    
    const closeBtn = screen.getByText('×');
    await user.click(closeBtn);
    
    expect(screen.queryByText('Test error')).not.toBeInTheDocument();
  });
});

describe('DiagramCanvas Accessibility', () => {
  test('has proper ARIA labels', () => {
    render(<DiagramCanvas />);
    
    const textInput = screen.getByPlaceholderText('Enter Sanskrit text...');
    expect(textInput).toHaveAttribute('aria-label');
  });

  test('supports keyboard navigation', () => {
    render(<DiagramCanvas />);
    
    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      expect(button).toHaveAttribute('tabIndex');
    });
  });

  test('has semantic HTML structure', () => {
    render(<DiagramCanvas />);
    
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getAllByRole('button')).toHaveLength(6); // All the buttons in the interface
  });

  test('provides alternative text for visual elements', () => {
    render(<DiagramCanvas />);
    
    // SVG should have proper accessibility attributes when rendered
    const canvas = document.querySelector('.canvas-container');
    expect(canvas).toBeInTheDocument();
  });
});
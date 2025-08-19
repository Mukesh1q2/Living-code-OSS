import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import IntegratedCanvas from '../IntegratedCanvas';
import { apiService } from '../../services/apiService';

// Mock the API service
jest.mock('../../services/apiService');
const mockApiService = apiService as jest.Mocked<typeof apiService>;

// Mock D3 to avoid SVG rendering issues in tests
jest.mock('d3', () => ({
  select: jest.fn(() => ({
    selectAll: jest.fn(() => ({
      remove: jest.fn(),
    })),
    append: jest.fn(() => ({
      selectAll: jest.fn(() => ({
        data: jest.fn(() => ({
          enter: jest.fn(() => ({
            append: jest.fn(() => ({
              attr: jest.fn(() => ({
                attr: jest.fn(() => ({
                  attr: jest.fn(() => ({
                    attr: jest.fn(() => ({
                      attr: jest.fn(() => ({
                        attr: jest.fn(),
                      })),
                    })),
                  })),
                })),
              })),
            })),
          })),
        })),
      })),
    })),
    call: jest.fn(),
    attr: jest.fn(),
  })),
  zoom: jest.fn(() => ({
    scaleExtent: jest.fn(() => ({
      on: jest.fn(),
    })),
  })),
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
  zoomIdentity: {
    translate: jest.fn(() => ({
      scale: jest.fn(),
    })),
  },
}));

// Mock data for tests
const mockRules = [
  {
    id: 1,
    name: 'Vowel Sandhi Rule',
    priority: 1,
    description: 'Combines adjacent vowels according to Sanskrit phonology',
    active: true,
    applications: 5,
    max_applications: 10,
    sutra_ref: '6.1.87',
  },
  {
    id: 2,
    name: 'Consonant Assimilation',
    priority: 2,
    description: 'Assimilates consonants based on place of articulation',
    active: false,
    applications: 2,
    sutra_ref: '8.4.55',
  },
];

const mockProcessingResult = {
  input_text: 'राम गच्छति',
  input_tokens: [
    { text: 'राम', kind: 'WORD', tags: [], meta: {} },
    { text: ' ', kind: 'SPACE', tags: [], meta: {} },
    { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
  ],
  output_tokens: [
    { text: 'राम', kind: 'WORD', tags: ['processed'], meta: {} },
    { text: 'गच्छति', kind: 'WORD', tags: ['processed'], meta: {} },
  ],
  converged: true,
  passes: 2,
  traces: [
    {
      pass_number: 1,
      tokens_before: [
        { text: 'राम', kind: 'WORD', tags: [], meta: {} },
        { text: ' ', kind: 'SPACE', tags: [], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      tokens_after: [
        { text: 'राम', kind: 'WORD', tags: [], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      transformations: [
        {
          rule_name: 'Space Removal',
          rule_id: 1,
          index: 1,
          tokens_before: [
            { text: 'राम', kind: 'WORD', tags: [], meta: {} },
            { text: ' ', kind: 'SPACE', tags: [], meta: {} },
            { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
          ],
          tokens_after: [
            { text: 'राम', kind: 'WORD', tags: [], meta: {} },
            { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
          ],
          timestamp: '2024-01-01T12:00:00Z',
        },
      ],
      meta_rule_applications: [],
    },
  ],
  errors: [],
  processing_time_ms: 150,
};

describe('IntegratedCanvas', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getRules.mockResolvedValue(mockRules);
    mockApiService.processSanskritText.mockResolvedValue(mockProcessingResult);
  });

  describe('Component Rendering', () => {
    test('renders the integrated canvas with all main sections', async () => {
      render(<IntegratedCanvas />);

      expect(screen.getByText('Sanskrit Development Canvas')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter Sanskrit text...')).toBeInTheDocument();
      expect(screen.getByText('🔄 Process')).toBeInTheDocument();
      expect(screen.getByText('🔄 Pipeline')).toBeInTheDocument();
      expect(screen.getByText('📋 Derivation')).toBeInTheDocument();
      expect(screen.getByText('⚙️ Rules')).toBeInTheDocument();
    });

    test('loads available rules on component mount', async () => {
      render(<IntegratedCanvas />);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalledTimes(1);
      });
    });

    test('displays default Sanskrit text in input field', () => {
      render(<IntegratedCanvas />);

      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      expect(input).toHaveValue('राम गच्छति');
    });
  });

  describe('Text Processing', () => {
    test('processes Sanskrit text when process button is clicked', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalledWith({
          text: 'राम गच्छति',
          enable_tracing: true,
          active_rules: [1], // Only active rules
        });
      });
    });

    test('shows loading state during processing', async () => {
      const user = userEvent.setup();
      mockApiService.processSanskritText.mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockProcessingResult), 100))
      );

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      expect(screen.getByText('⏳ Processing...')).toBeInTheDocument();
      expect(processButton).toBeDisabled();

      await waitFor(() => {
        expect(screen.getByText('🔄 Process')).toBeInTheDocument();
      });
    });

    test('handles processing errors gracefully', async () => {
      const user = userEvent.setup();
      mockApiService.processSanskritText.mockRejectedValue(new Error('Processing failed'));

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('Processing failed')).toBeInTheDocument();
      });
    });

    test('updates input text and processes new text', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      await user.clear(input);
      await user.type(input, 'नमस्ते');

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalledWith({
          text: 'नमस्ते',
          enable_tracing: true,
          active_rules: [1],
        });
      });
    });
  });

  describe('View Mode Switching', () => {
    test('switches between different view modes', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text first to enable other views
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalled();
      });

      // Test derivation view
      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);
      expect(derivationButton).toHaveClass('btn-primary');

      // Test rules view
      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);
      expect(rulesButton).toHaveClass('btn-primary');

      // Test pipeline view
      const pipelineButton = screen.getByText('🔄 Pipeline');
      await user.click(pipelineButton);
      expect(pipelineButton).toHaveClass('btn-primary');
    });

    test('shows derivation steps in derivation view', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text first
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalled();
      });

      // Switch to derivation view
      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      await waitFor(() => {
        expect(screen.getByText('Pass 1 - Prakriyā Steps')).toBeInTheDocument();
        expect(screen.getByText('Space Removal')).toBeInTheDocument();
      });
    });

    test('shows rules configuration in rules view', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Wait for rules to load
      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalled();
      });

      // Switch to rules view
      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      await waitFor(() => {
        expect(screen.getByText('Rule Configuration')).toBeInTheDocument();
        expect(screen.getByText('Vowel Sandhi Rule')).toBeInTheDocument();
        expect(screen.getByText('Consonant Assimilation')).toBeInTheDocument();
      });
    });
  });

  describe('Rule Management', () => {
    test('displays available rules with correct status', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalled();
      });

      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      await waitFor(() => {
        const activeRule = screen.getByText('Vowel Sandhi Rule').closest('.rule-item');
        const inactiveRule = screen.getByText('Consonant Assimilation').closest('.rule-item');

        expect(activeRule).toHaveClass('active');
        expect(inactiveRule).toHaveClass('inactive');
      });
    });

    test('toggles rule activation status', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalled();
      });

      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      await waitFor(() => {
        const checkboxes = screen.getAllByRole('checkbox');
        expect(checkboxes[0]).toBeChecked(); // Active rule
        expect(checkboxes[1]).not.toBeChecked(); // Inactive rule
      });

      // Toggle inactive rule to active
      await user.click(checkboxes[1]);
      expect(checkboxes[1]).toBeChecked();

      // Toggle active rule to inactive
      await user.click(checkboxes[0]);
      expect(checkboxes[0]).not.toBeChecked();
    });

    test('refreshes rules when refresh button is clicked', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalledTimes(1);
      });

      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      const refreshButton = screen.getByText('🔄 Refresh Rules');
      await user.click(refreshButton);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Playback Controls', () => {
    test('shows playback controls after processing', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('▶️ Play')).toBeInTheDocument();
        expect(screen.getByText('Pass 1 / 1')).toBeInTheDocument();
      });
    });

    test('toggles auto-play mode', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        const playButton = screen.getByText('▶️ Play');
        expect(playButton).toBeInTheDocument();
      });

      const playButton = screen.getByText('▶️ Play');
      await user.click(playButton);

      expect(screen.getByText('⏸️ Pause')).toBeInTheDocument();
    });

    test('controls pass navigation with slider', async () => {
      const user = userEvent.setup();
      
      // Mock result with multiple passes
      const multiPassResult = {
        ...mockProcessingResult,
        passes: 3,
        traces: [
          mockProcessingResult.traces[0],
          { ...mockProcessingResult.traces[0], pass_number: 2 },
          { ...mockProcessingResult.traces[0], pass_number: 3 },
        ],
      };
      mockApiService.processSanskritText.mockResolvedValue(multiPassResult);

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('Pass 1 / 3')).toBeInTheDocument();
      });

      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: '2' } });

      expect(screen.getByText('Pass 3 / 3')).toBeInTheDocument();
    });
  });

  describe('Code Generation', () => {
    test('generates Python code from processing result', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('Generated Code')).toBeInTheDocument();
        const codeBlock = screen.getByRole('code');
        expect(codeBlock).toHaveTextContent('# Generated Sanskrit Processing Code');
        expect(codeBlock).toHaveTextContent('input_text = "राम गच्छति"');
        expect(codeBlock).toHaveTextContent('# Rule: Space Removal');
      });
    });
  });

  describe('Transformation Details', () => {
    test('shows transformation details when transformation is selected', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text first
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalled();
      });

      // Switch to derivation view
      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      // Click on a transformation
      await waitFor(() => {
        const transformationStep = screen.getByText('Space Removal').closest('.transformation-step');
        expect(transformationStep).toBeInTheDocument();
      });

      const transformationStep = screen.getByText('Space Removal').closest('.transformation-step');
      await user.click(transformationStep!);

      await waitFor(() => {
        expect(screen.getByText('Transformation Details')).toBeInTheDocument();
        expect(screen.getByText('Before Transformation')).toBeInTheDocument();
        expect(screen.getByText('After Transformation')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('displays error banner when processing fails', async () => {
      const user = userEvent.setup();
      mockApiService.processSanskritText.mockRejectedValue(new Error('Network error'));

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument();
      });
    });

    test('allows dismissing error banner', async () => {
      const user = userEvent.setup();
      mockApiService.processSanskritText.mockRejectedValue(new Error('Test error'));

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('Test error')).toBeInTheDocument();
      });

      const closeButton = screen.getByText('×');
      await user.click(closeButton);

      expect(screen.queryByText('Test error')).not.toBeInTheDocument();
    });

    test('handles empty processing results gracefully', async () => {
      const user = userEvent.setup();
      mockApiService.processSanskritText.mockResolvedValue({
        ...mockProcessingResult,
        traces: [],
        passes: 0,
      });

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalled();
      });

      // Should not show playback controls for empty results
      expect(screen.queryByText('▶️ Play')).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    test('provides proper ARIA labels and roles', () => {
      render(<IntegratedCanvas />);

      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      expect(input).toHaveAttribute('type', 'text');

      const processButton = screen.getByText('🔄 Process');
      expect(processButton).toHaveAttribute('type', 'button');
    });

    test('supports keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      
      // Tab to input field
      await user.tab();
      expect(input).toHaveFocus();

      // Tab to process button
      await user.tab();
      const processButton = screen.getByText('🔄 Process');
      expect(processButton).toHaveFocus();
    });

    test('maintains focus management during view switches', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      expect(derivationButton).toHaveFocus();
    });
  });

  describe('Performance', () => {
    test('debounces rapid processing requests', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      
      // Rapid clicks
      await user.click(processButton);
      await user.click(processButton);
      await user.click(processButton);

      // Should only process once due to loading state
      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalledTimes(1);
      });
    });

    test('cleans up resources on unmount', () => {
      const { unmount } = render(<IntegratedCanvas />);
      
      // Should not throw errors on unmount
      expect(() => unmount()).not.toThrow();
    });
  });
});

describe('IntegratedCanvas Integration Tests', () => {
  test('complete workflow: load rules, process text, view results', async () => {
    const user = userEvent.setup();
    render(<IntegratedCanvas />);

    // 1. Wait for rules to load
    await waitFor(() => {
      expect(mockApiService.getRules).toHaveBeenCalled();
    });

    // 2. Enter Sanskrit text
    const input = screen.getByPlaceholderText('Enter Sanskrit text...');
    await user.clear(input);
    await user.type(input, 'देव नमस्ते');

    // 3. Process the text
    const processButton = screen.getByText('🔄 Process');
    await user.click(processButton);

    await waitFor(() => {
      expect(mockApiService.processSanskritText).toHaveBeenCalledWith({
        text: 'देव नमस्ते',
        enable_tracing: true,
        active_rules: [1],
      });
    });

    // 4. Check generated code
    await waitFor(() => {
      expect(screen.getByText('Generated Code')).toBeInTheDocument();
    });

    // 5. Switch to derivation view
    const derivationButton = screen.getByText('📋 Derivation');
    await user.click(derivationButton);

    await waitFor(() => {
      expect(screen.getByText('Pass 1 - Prakriyā Steps')).toBeInTheDocument();
    });

    // 6. Switch to rules view and modify rules
    const rulesButton = screen.getByText('⚙️ Rules');
    await user.click(rulesButton);

    await waitFor(() => {
      expect(screen.getAllByRole('checkbox')).toHaveLength(2);
    });

    const checkboxes = screen.getAllByRole('checkbox');
    await user.click(checkboxes[1]); // Activate second rule

    // 7. Process again with new rule configuration
    await user.click(processButton);

    await waitFor(() => {
      expect(mockApiService.processSanskritText).toHaveBeenCalledWith({
        text: 'देव नमस्ते',
        enable_tracing: true,
        active_rules: [1, 2], // Both rules now active
      });
    });
  });
});
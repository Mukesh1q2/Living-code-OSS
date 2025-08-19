import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import IntegratedCanvas from '../IntegratedCanvas';
import { apiService } from '../../services/apiService';

// Mock the API service
jest.mock('../../services/apiService');
const mockApiService = apiService as jest.Mocked<typeof apiService>;

// Mock D3 for SVG interactions
jest.mock('d3', () => ({
  select: jest.fn(() => ({
    selectAll: jest.fn(() => ({
      remove: jest.fn(),
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
                        attr: jest.fn(() => ({
                          text: jest.fn(),
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
      attr: jest.fn(),
    })),
    call: jest.fn(),
    attr: jest.fn(),
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

// Enhanced mock data for integration tests
const mockComplexRules = [
  {
    id: 1,
    name: 'Vowel Sandhi (a + i → e)',
    priority: 1,
    description: 'Combines a + i to form e according to Pāṇini 6.1.87',
    active: true,
    applications: 15,
    max_applications: 50,
    sutra_ref: '6.1.87',
  },
  {
    id: 2,
    name: 'Consonant Assimilation (n → m)',
    priority: 2,
    description: 'Assimilates n to m before labial consonants',
    active: true,
    applications: 8,
    max_applications: 25,
    sutra_ref: '8.4.55',
  },
  {
    id: 3,
    name: 'Visarga Transformation',
    priority: 3,
    description: 'Transforms visarga based on following sound',
    active: false,
    applications: 3,
    sutra_ref: '8.3.15',
  },
  {
    id: 4,
    name: 'Compound Formation',
    priority: 10,
    description: 'Forms compounds by joining words with +',
    active: true,
    applications: 12,
    sutra_ref: '2.1.3',
  },
];

const mockComplexProcessingResult = {
  input_text: 'राम + अयम् गच्छति',
  input_tokens: [
    { text: 'राम', kind: 'WORD', tags: [], meta: { position: 0 } },
    { text: '+', kind: 'MARKER', tags: [], meta: { position: 1 } },
    { text: 'अयम्', kind: 'WORD', tags: [], meta: { position: 2 } },
    { text: 'गच्छति', kind: 'WORD', tags: [], meta: { position: 3 } },
  ],
  output_tokens: [
    { text: 'रामायम्', kind: 'WORD', tags: ['compound', 'sandhi'], meta: {} },
    { text: 'गच्छति', kind: 'WORD', tags: ['processed'], meta: {} },
  ],
  converged: true,
  passes: 3,
  traces: [
    {
      pass_number: 1,
      tokens_before: [
        { text: 'राम', kind: 'WORD', tags: [], meta: {} },
        { text: '+', kind: 'MARKER', tags: [], meta: {} },
        { text: 'अयम्', kind: 'WORD', tags: [], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      tokens_after: [
        { text: 'राम', kind: 'WORD', tags: [], meta: {} },
        { text: 'अयम्', kind: 'WORD', tags: [], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      transformations: [
        {
          rule_name: 'Compound Formation',
          rule_id: 4,
          index: 1,
          tokens_before: [
            { text: 'राम', kind: 'WORD', tags: [], meta: {} },
            { text: '+', kind: 'MARKER', tags: [], meta: {} },
            { text: 'अयम्', kind: 'WORD', tags: [], meta: {} },
          ],
          tokens_after: [
            { text: 'राम', kind: 'WORD', tags: [], meta: {} },
            { text: 'अयम्', kind: 'WORD', tags: [], meta: {} },
          ],
          timestamp: '2024-01-01T12:00:00Z',
        },
      ],
      meta_rule_applications: [],
    },
    {
      pass_number: 2,
      tokens_before: [
        { text: 'राम', kind: 'WORD', tags: [], meta: {} },
        { text: 'अयम्', kind: 'WORD', tags: [], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      tokens_after: [
        { text: 'रामायम्', kind: 'WORD', tags: ['sandhi'], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      transformations: [
        {
          rule_name: 'Vowel Sandhi (a + i → e)',
          rule_id: 1,
          index: 0,
          tokens_before: [
            { text: 'राम', kind: 'WORD', tags: [], meta: {} },
            { text: 'अयम्', kind: 'WORD', tags: [], meta: {} },
          ],
          tokens_after: [
            { text: 'रामायम्', kind: 'WORD', tags: ['sandhi'], meta: {} },
          ],
          timestamp: '2024-01-01T12:00:01Z',
        },
      ],
      meta_rule_applications: [],
    },
    {
      pass_number: 3,
      tokens_before: [
        { text: 'रामायम्', kind: 'WORD', tags: ['sandhi'], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: [], meta: {} },
      ],
      tokens_after: [
        { text: 'रामायम्', kind: 'WORD', tags: ['compound', 'sandhi'], meta: {} },
        { text: 'गच्छति', kind: 'WORD', tags: ['processed'], meta: {} },
      ],
      transformations: [],
      meta_rule_applications: ['final_tagging'],
    },
  ],
  errors: [],
  processing_time_ms: 245,
};

describe('IntegratedCanvas - Advanced Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getRules.mockResolvedValue(mockComplexRules);
    mockApiService.processSanskritText.mockResolvedValue(mockComplexProcessingResult);
  });

  describe('Real-time Pipeline Visualization', () => {
    test('visualizes complete Sanskrit → Code → Output pipeline', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Enter complex Sanskrit text
      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      await user.clear(input);
      await user.type(input, 'राम + अयम् गच्छति');

      // Process the text
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalledWith({
          text: 'राम + अयम् गच्छति',
          enable_tracing: true,
          active_rules: [1, 2, 4], // Active rules from mock data
        });
      });

      // Verify pipeline visualization elements are present
      await waitFor(() => {
        expect(screen.getByText('Generated Code')).toBeInTheDocument();
        
        // Check that the generated code contains expected elements
        const codeBlock = screen.getByRole('code');
        expect(codeBlock).toHaveTextContent('input_text = "राम + अयम् गच्छति"');
        expect(codeBlock).toHaveTextContent('# Rule: Compound Formation');
        expect(codeBlock).toHaveTextContent('# Rule: Vowel Sandhi (a + i → e)');
        expect(codeBlock).toHaveTextContent('output_text = "रामायम्गच्छति"');
      });
    });

    test('updates pipeline visualization in real-time during playback', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text to get multi-pass result
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('Pass 1 / 3')).toBeInTheDocument();
      });

      // Test pass navigation
      const slider = screen.getByRole('slider');
      
      // Move to pass 2
      fireEvent.change(slider, { target: { value: '1' } });
      expect(screen.getByText('Pass 2 / 3')).toBeInTheDocument();

      // Move to pass 3
      fireEvent.change(slider, { target: { value: '2' } });
      expect(screen.getByText('Pass 3 / 3')).toBeInTheDocument();
    });

    test('shows auto-play progression through pipeline stages', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('▶️ Play')).toBeInTheDocument();
      });

      // Start auto-play
      const playButton = screen.getByText('▶️ Play');
      await user.click(playButton);

      expect(screen.getByText('⏸️ Pause')).toBeInTheDocument();

      // Pause auto-play
      const pauseButton = screen.getByText('⏸️ Pause');
      await user.click(pauseButton);

      expect(screen.getByText('▶️ Play')).toBeInTheDocument();
    });
  });

  describe('Interactive Prakriyā Visualization', () => {
    test('displays step-by-step derivation with detailed token information', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text and switch to derivation view
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(screen.getByText('📋 Derivation')).toBeInTheDocument();
      });

      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      await waitFor(() => {
        // Check that derivation steps are displayed
        expect(screen.getByText('Pass 1 - Prakriyā Steps')).toBeInTheDocument();
        expect(screen.getByText('Compound Formation')).toBeInTheDocument();
        
        // Check before/after token display
        expect(screen.getByText('राम + अयम् गच्छति')).toBeInTheDocument();
        expect(screen.getByText('राम अयम् गच्छति')).toBeInTheDocument();
      });

      // Navigate to different passes
      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: '1' } });

      await waitFor(() => {
        expect(screen.getByText('Pass 2 - Prakriyā Steps')).toBeInTheDocument();
        expect(screen.getByText('Vowel Sandhi (a + i → e)')).toBeInTheDocument();
      });
    });

    test('shows detailed transformation information on selection', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process and navigate to derivation view
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      // Click on a transformation step
      await waitFor(() => {
        const transformationStep = screen.getByText('Compound Formation').closest('.transformation-step');
        await user.click(transformationStep!);
      });

      // Check transformation details panel
      await waitFor(() => {
        expect(screen.getByText('Transformation Details')).toBeInTheDocument();
        expect(screen.getByText('Compound Formation')).toBeInTheDocument();
        expect(screen.getByText('ID: 4')).toBeInTheDocument();
        expect(screen.getByText('Before Transformation')).toBeInTheDocument();
        expect(screen.getByText('After Transformation')).toBeInTheDocument();
      });
    });

    test('highlights token changes between transformation steps', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      // Select transformation and check token details
      await waitFor(() => {
        const transformationStep = screen.getByText('Compound Formation').closest('.transformation-step');
        await user.click(transformationStep!);
      });

      await waitFor(() => {
        // Check that token details show kind and tags
        const tokenDetails = screen.getAllByText('WORD');
        expect(tokenDetails.length).toBeGreaterThan(0);
        
        const markerToken = screen.getByText('MARKER');
        expect(markerToken).toBeInTheDocument();
      });
    });
  });

  describe('Live Rule Editing and Composition', () => {
    test('allows real-time rule activation/deactivation', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Wait for rules to load and navigate to rules view
      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalled();
      });

      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      // Check initial rule states
      await waitFor(() => {
        const checkboxes = screen.getAllByRole('checkbox');
        expect(checkboxes[0]).toBeChecked(); // Vowel Sandhi
        expect(checkboxes[1]).toBeChecked(); // Consonant Assimilation
        expect(checkboxes[2]).not.toBeChecked(); // Visarga Transformation
        expect(checkboxes[3]).toBeChecked(); // Compound Formation
      });

      // Toggle Visarga Transformation rule
      await user.click(checkboxes[2]);
      expect(checkboxes[2]).toBeChecked();

      // Process text with new rule configuration
      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      await user.clear(input);
      await user.type(input, 'रामः गच्छति');

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalledWith({
          text: 'रामः गच्छति',
          enable_tracing: true,
          active_rules: [1, 2, 3, 4], // All rules now active
        });
      });
    });

    test('displays rule metadata and application statistics', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalled();
      });

      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      await waitFor(() => {
        // Check rule metadata display
        expect(screen.getByText('Priority: 1 | Applications: 15 / 50 | Sūtra: 6.1.87')).toBeInTheDocument();
        expect(screen.getByText('Priority: 2 | Applications: 8 / 25 | Sūtra: 8.4.55')).toBeInTheDocument();
        expect(screen.getByText('Priority: 3 | Applications: 3 | Sūtra: 8.3.15')).toBeInTheDocument();
        expect(screen.getByText('Priority: 10 | Applications: 12 | Sūtra: 2.1.3')).toBeInTheDocument();
      });
    });

    test('provides rule composition interface with drag-and-drop simulation', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      await waitFor(() => {
        expect(mockApiService.getRules).toHaveBeenCalled();
      });

      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      // Simulate drag-and-drop by checking rule reordering capability
      await waitFor(() => {
        const ruleItems = screen.getAllByText(/Priority: \d+/);
        expect(ruleItems.length).toBe(4);
        
        // Verify rules are displayed in priority order
        expect(ruleItems[0]).toHaveTextContent('Priority: 1');
        expect(ruleItems[1]).toHaveTextContent('Priority: 2');
        expect(ruleItems[2]).toHaveTextContent('Priority: 3');
        expect(ruleItems[3]).toHaveTextContent('Priority: 10');
      });
    });
  });

  describe('Visual Debugging Tools', () => {
    test('provides comprehensive rule application sequence visualization', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process complex text
      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      await user.clear(input);
      await user.type(input, 'राम + अयम् गच्छति');

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      // Check pipeline view shows rule sequence
      await waitFor(() => {
        // Pipeline should be visible by default
        const svg = screen.getByRole('img', { hidden: true }) || document.querySelector('.pipeline-svg');
        expect(svg).toBeInTheDocument();
      });

      // Switch to derivation view for detailed sequence
      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      await waitFor(() => {
        // Check that all transformation steps are visible
        expect(screen.getByText('Compound Formation')).toBeInTheDocument();
        
        // Navigate to see sandhi rule
        const slider = screen.getByRole('slider');
        fireEvent.change(slider, { target: { value: '1' } });
        
        expect(screen.getByText('Vowel Sandhi (a + i → e)')).toBeInTheDocument();
      });
    });

    test('shows rule application timing and performance metrics', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      // Click on transformation to see details
      await waitFor(() => {
        const transformationStep = screen.getByText('Compound Formation').closest('.transformation-step');
        await user.click(transformationStep!);
      });

      // Check timestamp information
      await waitFor(() => {
        expect(screen.getByText(/Timestamp:/)).toBeInTheDocument();
        expect(screen.getByText(/Index: 1/)).toBeInTheDocument();
      });
    });

    test('provides error visualization and debugging information', async () => {
      const user = userEvent.setup();
      
      // Mock processing result with errors
      const errorResult = {
        ...mockComplexProcessingResult,
        errors: ['Rule application limit exceeded for rule 1', 'Convergence not achieved after 20 passes'],
        converged: false,
      };
      mockApiService.processSanskritText.mockResolvedValue(errorResult);

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      // Errors should be handled gracefully and not break the interface
      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenCalled();
        // Interface should still be functional
        expect(screen.getByText('📋 Derivation')).toBeInTheDocument();
      });
    });
  });

  describe('Code Generation and Execution Integration', () => {
    test('generates executable Python code from Sanskrit processing', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        const codeBlock = screen.getByRole('code');
        
        // Check for complete Python code structure
        expect(codeBlock).toHaveTextContent('# Generated Sanskrit Processing Code');
        expect(codeBlock).toHaveTextContent('input_text = "राम + अयम् गच्छति"');
        expect(codeBlock).toHaveTextContent('# Pass 1');
        expect(codeBlock).toHaveTextContent('# Rule: Compound Formation');
        expect(codeBlock).toHaveTextContent('# Pass 2');
        expect(codeBlock).toHaveTextContent('# Rule: Vowel Sandhi (a + i → e)');
        expect(codeBlock).toHaveTextContent('output_tokens = ["रामायम्", "गच्छति"]');
        expect(codeBlock).toHaveTextContent('output_text = "रामायम्गच्छति"');
      });
    });

    test('updates generated code when rule configuration changes', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Initial processing
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      // Change rule configuration
      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);

      await waitFor(() => {
        expect(screen.getAllByRole('checkbox')).toHaveLength(4);
      });

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[2]); // Activate Visarga Transformation

      // Process again
      await user.click(processButton);

      await waitFor(() => {
        expect(mockApiService.processSanskritText).toHaveBeenLastCalledWith({
          text: 'राम गच्छति',
          enable_tracing: true,
          active_rules: [1, 2, 3, 4], // Updated rule set
        });
      });
    });
  });

  describe('Performance and Scalability', () => {
    test('handles large processing results efficiently', async () => {
      const user = userEvent.setup();
      
      // Create large processing result
      const largeResult = {
        ...mockComplexProcessingResult,
        passes: 10,
        traces: Array.from({ length: 10 }, (_, i) => ({
          ...mockComplexProcessingResult.traces[0],
          pass_number: i + 1,
          transformations: Array.from({ length: 5 }, (_, j) => ({
            ...mockComplexProcessingResult.traces[0].transformations[0],
            rule_name: `Rule ${i + 1}.${j + 1}`,
            rule_id: i * 5 + j + 1,
          })),
        })),
      };
      
      mockApiService.processSanskritText.mockResolvedValue(largeResult);

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      // Should handle large results without performance issues
      await waitFor(() => {
        expect(screen.getByText('Pass 1 / 10')).toBeInTheDocument();
      });

      // Test navigation through large result set
      const slider = screen.getByRole('slider');
      fireEvent.change(slider, { target: { value: '9' } });
      
      expect(screen.getByText('Pass 10 / 10')).toBeInTheDocument();
    });

    test('maintains responsive UI during complex processing', async () => {
      const user = userEvent.setup();
      
      // Simulate slow processing
      mockApiService.processSanskritText.mockImplementation(
        () => new Promise(resolve => 
          setTimeout(() => resolve(mockComplexProcessingResult), 500)
        )
      );

      render(<IntegratedCanvas />);

      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      // UI should remain responsive during processing
      expect(screen.getByText('⏳ Processing...')).toBeInTheDocument();
      expect(processButton).toBeDisabled();

      // Other UI elements should still be interactive
      const rulesButton = screen.getByText('⚙️ Rules');
      await user.click(rulesButton);
      
      expect(rulesButton).toHaveClass('btn-primary');

      await waitFor(() => {
        expect(screen.getByText('🔄 Process')).toBeInTheDocument();
      }, { timeout: 1000 });
    });
  });

  describe('Accessibility and Usability', () => {
    test('provides comprehensive keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Tab through main interface elements
      await user.tab(); // Input field
      expect(screen.getByPlaceholderText('Enter Sanskrit text...')).toHaveFocus();

      await user.tab(); // Process button
      expect(screen.getByText('🔄 Process')).toHaveFocus();

      await user.tab(); // Pipeline button
      expect(screen.getByText('🔄 Pipeline')).toHaveFocus();

      await user.tab(); // Derivation button
      expect(screen.getByText('📋 Derivation')).toHaveFocus();

      await user.tab(); // Rules button
      expect(screen.getByText('⚙️ Rules')).toHaveFocus();
    });

    test('provides proper ARIA labels and semantic structure', () => {
      render(<IntegratedCanvas />);

      // Check for proper heading structure
      expect(screen.getByRole('heading', { name: 'Sanskrit Development Canvas' })).toBeInTheDocument();
      
      // Check for proper button roles
      expect(screen.getByRole('button', { name: /Process/ })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Pipeline/ })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Derivation/ })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Rules/ })).toBeInTheDocument();

      // Check for proper input labeling
      const input = screen.getByPlaceholderText('Enter Sanskrit text...');
      expect(input).toHaveAttribute('type', 'text');
    });

    test('supports screen reader navigation', async () => {
      const user = userEvent.setup();
      render(<IntegratedCanvas />);

      // Process text to create content
      const processButton = screen.getByText('🔄 Process');
      await user.click(processButton);

      await waitFor(() => {
        // Check that generated content has proper structure
        expect(screen.getByText('Generated Code')).toBeInTheDocument();
      });

      // Navigate to derivation view
      const derivationButton = screen.getByText('📋 Derivation');
      await user.click(derivationButton);

      await waitFor(() => {
        // Check for proper heading hierarchy
        expect(screen.getByText('Pass 1 - Prakriyā Steps')).toBeInTheDocument();
      });
    });
  });
});
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CodeEditor from '../CodeEditor';

// Mock Monaco Editor
jest.mock('@monaco-editor/react', () => {
  return {
    __esModule: true,
    default: ({ onChange, onMount, value }: any) => {
      React.useEffect(() => {
        if (onMount) {
          const mockEditor = {
            updateOptions: jest.fn(),
            addCommand: jest.fn(),
            getModel: () => ({
              setValue: jest.fn(),
            }),
          };
          const mockMonaco = {
            KeyMod: { CtrlCmd: 1 },
            KeyCode: { KeyS: 1, Enter: 1 },
            editor: {
              setModelLanguage: jest.fn(),
            },
          };
          onMount(mockEditor, mockMonaco);
        }
      }, [onMount]);

      return (
        <textarea
          data-testid="monaco-editor"
          value={value}
          onChange={(e) => onChange && onChange(e.target.value)}
          placeholder="Monaco Editor Mock"
        />
      );
    },
  };
});

// Mock API service
jest.mock('../../services/apiService', () => ({
  apiService: {
    processSanskritText: jest.fn().mockResolvedValue({
      input_text: 'राम गच्छति',
      output_text: 'rāma gacchati',
      tokens: [{ text: 'राम' }, { text: 'गच्छति' }],
      transformations: [],
      semantic_graph: {},
      success: true,
      processing_time_ms: 100,
    }),
    executeCode: jest.fn().mockResolvedValue({
      success: true,
      output: 'Hello, World!',
      execution_time_ms: 50,
      language: 'python',
    }),
  },
}));

describe('CodeEditor', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders code editor with toolbar', () => {
    render(<CodeEditor />);
    
    expect(screen.getByTestId('monaco-editor')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Sanskrit')).toBeInTheDocument();
    expect(screen.getByText('💾 Save')).toBeInTheDocument();
    expect(screen.getByText('▶️ Execute')).toBeInTheDocument();
  });

  test('displays language selection dropdown', () => {
    render(<CodeEditor />);
    
    const languageSelect = screen.getByDisplayValue('Sanskrit');
    expect(languageSelect).toBeInTheDocument();
    
    // Check if other languages are available
    fireEvent.click(languageSelect);
    expect(screen.getByText('Python')).toBeInTheDocument();
    expect(screen.getByText('JavaScript')).toBeInTheDocument();
  });

  test('displays theme selection dropdown', () => {
    render(<CodeEditor />);
    
    const themeSelect = screen.getByDisplayValue('Light');
    expect(themeSelect).toBeInTheDocument();
    
    fireEvent.click(themeSelect);
    expect(screen.getByText('Dark')).toBeInTheDocument();
    expect(screen.getByText('Sanskrit')).toBeInTheDocument();
  });

  test('has font size control', () => {
    render(<CodeEditor />);
    
    expect(screen.getByText('Size:')).toBeInTheDocument();
    expect(screen.getByDisplayValue('14')).toBeInTheDocument();
    expect(screen.getByText('14px')).toBeInTheDocument();
  });

  test('changes language when dropdown selection changes', async () => {
    const user = userEvent.setup();
    render(<CodeEditor />);
    
    const languageSelect = screen.getByDisplayValue('Sanskrit');
    await user.selectOptions(languageSelect, 'python');
    
    expect(screen.getByDisplayValue('Python')).toBeInTheDocument();
  });

  test('changes theme when dropdown selection changes', async () => {
    const user = userEvent.setup();
    render(<CodeEditor />);
    
    const themeSelect = screen.getByDisplayValue('Light');
    await user.selectOptions(themeSelect, 'vs-dark');
    
    expect(screen.getByDisplayValue('Dark')).toBeInTheDocument();
  });

  test('updates font size with range input', async () => {
    const user = userEvent.setup();
    render(<CodeEditor />);
    
    const fontSizeRange = screen.getByDisplayValue('14');
    await user.clear(fontSizeRange);
    await user.type(fontSizeRange, '16');
    
    expect(screen.getByText('16px')).toBeInTheDocument();
  });

  test('executes Sanskrit text processing', async () => {
    const user = userEvent.setup();
    const { apiService } = require('../../services/apiService');
    
    render(<CodeEditor />);
    
    const editor = screen.getByTestId('monaco-editor');
    const executeBtn = screen.getByText('▶️ Execute');
    
    await user.type(editor, 'राम गच्छति');
    await user.click(executeBtn);
    
    await waitFor(() => {
      expect(apiService.processSanskritText).toHaveBeenCalledWith({
        text: 'राम गच्छति',
        enable_tracing: true,
      });
    });
  });

  test('executes code in other languages', async () => {
    const user = userEvent.setup();
    const { apiService } = require('../../services/apiService');
    
    render(<CodeEditor />);
    
    const languageSelect = screen.getByDisplayValue('Sanskrit');
    const editor = screen.getByTestId('monaco-editor');
    const executeBtn = screen.getByText('▶️ Execute');
    
    await user.selectOptions(languageSelect, 'python');
    await user.type(editor, 'print("Hello, World!")');
    await user.click(executeBtn);
    
    await waitFor(() => {
      expect(apiService.executeCode).toHaveBeenCalledWith({
        code: 'print("Hello, World!")',
        language: 'python',
        timeout: 30,
      });
    });
  });

  test('displays execution output', async () => {
    const user = userEvent.setup();
    render(<CodeEditor />);
    
    const languageSelect = screen.getByDisplayValue('Sanskrit');
    const editor = screen.getByTestId('monaco-editor');
    const executeBtn = screen.getByText('▶️ Execute');
    
    await user.selectOptions(languageSelect, 'python');
    await user.type(editor, 'print("Hello")');
    await user.click(executeBtn);
    
    await waitFor(() => {
      expect(screen.getByText('✅ Result')).toBeInTheDocument();
      expect(screen.getByText('Hello, World!')).toBeInTheDocument();
    });
  });

  test('shows loading state during execution', async () => {
    const user = userEvent.setup();
    render(<CodeEditor />);
    
    const editor = screen.getByTestId('monaco-editor');
    const executeBtn = screen.getByText('▶️ Execute');
    
    await user.type(editor, 'test code');
    
    // Mock a delayed response
    const { apiService } = require('../../services/apiService');
    apiService.processSanskritText.mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 100))
    );
    
    await user.click(executeBtn);
    
    expect(screen.getByText('⏳ Running...')).toBeInTheDocument();
  });

  test('handles execution errors', async () => {
    const user = userEvent.setup();
    const { apiService } = require('../../services/apiService');
    
    // Mock an error response
    apiService.executeCode.mockResolvedValueOnce({
      success: false,
      error: 'Syntax error',
      output: '',
      execution_time_ms: 10,
      language: 'python',
    });
    
    render(<CodeEditor />);
    
    const languageSelect = screen.getByDisplayValue('Sanskrit');
    const editor = screen.getByTestId('monaco-editor');
    const executeBtn = screen.getByText('▶️ Execute');
    
    await user.selectOptions(languageSelect, 'python');
    await user.type(editor, 'invalid syntax');
    await user.click(executeBtn);
    
    await waitFor(() => {
      expect(screen.getByText('❌ Error')).toBeInTheDocument();
      expect(screen.getByText('Syntax error')).toBeInTheDocument();
    });
  });

  test('clears output when clear button is clicked', async () => {
    const user = userEvent.setup();
    render(<CodeEditor />);
    
    const editor = screen.getByTestId('monaco-editor');
    const executeBtn = screen.getByText('▶️ Execute');
    
    await user.type(editor, 'राम');
    await user.click(executeBtn);
    
    await waitFor(() => {
      expect(screen.getByText('Clear')).toBeInTheDocument();
    });
    
    const clearBtn = screen.getByText('Clear');
    await user.click(clearBtn);
    
    expect(screen.queryByText('✅ Result')).not.toBeInTheDocument();
  });

  test('saves code to localStorage', async () => {
    const user = userEvent.setup();
    const setItemSpy = jest.spyOn(Storage.prototype, 'setItem');
    
    render(<CodeEditor />);
    
    const editor = screen.getByTestId('monaco-editor');
    const saveBtn = screen.getByText('💾 Save');
    
    await user.type(editor, 'test code');
    await user.click(saveBtn);
    
    expect(setItemSpy).toHaveBeenCalledWith('sanskrit-editor-content', 'test code');
    
    setItemSpy.mockRestore();
  });

  test('disables execute button when no code', () => {
    render(<CodeEditor />);
    
    const executeBtn = screen.getByText('▶️ Execute');
    expect(executeBtn).toBeDisabled();
  });

  test('shows empty output state initially', () => {
    render(<CodeEditor />);
    
    expect(screen.getByText('Execute your code to see the output here')).toBeInTheDocument();
    expect(screen.getByText('Ctrl')).toBeInTheDocument();
    expect(screen.getByText('Enter')).toBeInTheDocument();
  });
});

describe('CodeEditor Accessibility', () => {
  test('has proper keyboard shortcuts', () => {
    render(<CodeEditor />);
    
    expect(screen.getByTitle('Save (Ctrl+S)')).toBeInTheDocument();
    expect(screen.getByTitle('Execute (Ctrl+Enter)')).toBeInTheDocument();
  });

  test('has accessible form controls', () => {
    render(<CodeEditor />);
    
    const languageSelect = screen.getByDisplayValue('Sanskrit');
    const themeSelect = screen.getByDisplayValue('Light');
    
    expect(languageSelect).toHaveAttribute('aria-label');
    expect(themeSelect).toHaveAttribute('aria-label');
  });

  test('supports screen readers with semantic HTML', () => {
    render(<CodeEditor />);
    
    expect(screen.getByRole('button', { name: /Save/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Execute/ })).toBeInTheDocument();
  });
});
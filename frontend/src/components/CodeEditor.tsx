import React, { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import { apiService } from '../services/apiService';
import './CodeEditor.css';

interface CodeEditorProps {
  initialValue?: string;
  language?: string;
  theme?: string;
  readOnly?: boolean;
  onSave?: (content: string) => void;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  initialValue = '',
  language = 'sanskrit',
  theme = 'vs-light',
  readOnly = false,
  onSave,
}) => {
  const [code, setCode] = useState(initialValue);
  const [output, setOutput] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentLanguage, setCurrentLanguage] = useState(language);
  const [currentTheme, setCurrentTheme] = useState(theme);
  const [fontSize, setFontSize] = useState(14);
  const editorRef = useRef<any>(null);

  // Sanskrit syntax highlighting configuration
  useEffect(() => {
    const configureSanskritLanguage = (monaco: any) => {
      // Register Sanskrit language
      monaco.languages.register({ id: 'sanskrit' });

      // Define Sanskrit tokens
      monaco.languages.setMonarchTokensProvider('sanskrit', {
        tokenizer: {
          root: [
            // Devanagari characters
            [/[\u0900-\u097F]+/, 'sanskrit.devanagari'],
            
            // IAST transliteration
            [/[aāiīuūṛṝḷḹeēoōṃḥ]+/, 'sanskrit.vowel'],
            [/[kgṅcjñṭḍṇtdnpbmyrlvśṣsh]+/, 'sanskrit.consonant'],
            
            // Morphological markers
            [/[+_:]/, 'sanskrit.marker'],
            
            // Sandhi boundaries
            [/[-]/, 'sanskrit.sandhi'],
            
            // Comments (for analysis)
            [/#.*$/, 'comment'],
            
            // Numbers
            [/\d+/, 'number'],
            
            // Whitespace
            [/\s+/, 'white'],
          ],
        },
      });

      // Define Sanskrit theme
      monaco.editor.defineTheme('sanskrit-theme', {
        base: 'vs',
        inherit: true,
        rules: [
          { token: 'sanskrit.devanagari', foreground: '1a237e', fontStyle: 'bold' },
          { token: 'sanskrit.vowel', foreground: '2e7d32' },
          { token: 'sanskrit.consonant', foreground: 'c62828' },
          { token: 'sanskrit.marker', foreground: 'ff6f00', fontStyle: 'bold' },
          { token: 'sanskrit.sandhi', foreground: '6a1b9a' },
          { token: 'comment', foreground: '757575', fontStyle: 'italic' },
        ],
        colors: {
          'editor.background': '#fafafa',
          'editor.lineHighlightBackground': '#e8f5e8',
        },
      });
    };

    // Configure Monaco when it loads
    const handleEditorWillMount = (monaco: any) => {
      configureSanskritLanguage(monaco);
    };

    // Store the configuration function
    (window as any).monacoWillMount = handleEditorWillMount;
  }, []);

  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor;
    
    // Configure editor options
    editor.updateOptions({
      fontSize: fontSize,
      fontFamily: 'Noto Sans Devanagari, Consolas, Monaco, Courier New, monospace',
      lineNumbers: 'on',
      minimap: { enabled: true },
      wordWrap: 'on',
      automaticLayout: true,
    });

    // Add keyboard shortcuts
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      handleSave();
    });

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      handleExecute();
    });
  };

  const handleEditorChange = (value: string | undefined) => {
    setCode(value || '');
    setError(null);
  };

  const handleSave = () => {
    if (onSave) {
      onSave(code);
    }
    // You could also save to localStorage or API here
    localStorage.setItem('sanskrit-editor-content', code);
  };

  const handleExecute = async () => {
    if (!code.trim()) return;

    setIsExecuting(true);
    setError(null);
    setOutput('');

    try {
      if (currentLanguage === 'sanskrit') {
        // Process Sanskrit text
        const result = await apiService.processSanskritText({
          text: code,
          enable_tracing: true,
        });

        setOutput(JSON.stringify(result, null, 2));
      } else {
        // Execute code in other languages
        const result = await apiService.executeCode({
          code,
          language: currentLanguage,
          timeout: 30,
        });

        if (result.success) {
          setOutput(result.output);
        } else {
          setError(result.error || 'Execution failed');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Execution failed');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleLanguageChange = (newLanguage: string) => {
    setCurrentLanguage(newLanguage);
    
    // Update editor language
    if (editorRef.current) {
      const model = editorRef.current.getModel();
      if (model) {
        const monaco = (window as any).monaco;
        monaco.editor.setModelLanguage(model, newLanguage);
      }
    }
  };

  const handleThemeChange = (newTheme: string) => {
    setCurrentTheme(newTheme);
  };

  const handleFontSizeChange = (newSize: number) => {
    setFontSize(newSize);
    if (editorRef.current) {
      editorRef.current.updateOptions({ fontSize: newSize });
    }
  };

  const languages = [
    { id: 'sanskrit', name: 'Sanskrit' },
    { id: 'python', name: 'Python' },
    { id: 'javascript', name: 'JavaScript' },
    { id: 'typescript', name: 'TypeScript' },
    { id: 'json', name: 'JSON' },
    { id: 'markdown', name: 'Markdown' },
  ];

  const themes = [
    { id: 'vs-light', name: 'Light' },
    { id: 'vs-dark', name: 'Dark' },
    { id: 'sanskrit-theme', name: 'Sanskrit' },
  ];

  return (
    <div className="code-editor-container">
      <div className="editor-layout">
        <div className="editor-panel">
          <div className="editor-toolbar">
            <div className="toolbar-left">
              <select
                value={currentLanguage}
                onChange={(e) => handleLanguageChange(e.target.value)}
                className="language-select"
              >
                {languages.map((lang) => (
                  <option key={lang.id} value={lang.id}>
                    {lang.name}
                  </option>
                ))}
              </select>

              <select
                value={currentTheme}
                onChange={(e) => handleThemeChange(e.target.value)}
                className="theme-select"
              >
                {themes.map((theme) => (
                  <option key={theme.id} value={theme.id}>
                    {theme.name}
                  </option>
                ))}
              </select>

              <div className="font-size-control">
                <label>Size:</label>
                <input
                  type="range"
                  min="10"
                  max="24"
                  value={fontSize}
                  onChange={(e) => handleFontSizeChange(Number(e.target.value))}
                />
                <span>{fontSize}px</span>
              </div>
            </div>

            <div className="toolbar-right">
              <button
                onClick={handleSave}
                className="btn btn-outline"
                title="Save (Ctrl+S)"
              >
                💾 Save
              </button>
              <button
                onClick={handleExecute}
                disabled={isExecuting || !code.trim()}
                className="btn btn-primary"
                title="Execute (Ctrl+Enter)"
              >
                {isExecuting ? '⏳ Running...' : '▶️ Execute'}
              </button>
            </div>
          </div>

          <div className="editor-wrapper">
            <Editor
              height="100%"
              language={currentLanguage}
              theme={currentTheme}
              value={code}
              onChange={handleEditorChange}
              onMount={handleEditorDidMount}
              beforeMount={(window as any).monacoWillMount}
              options={{
                readOnly,
                fontSize,
                fontFamily: 'Noto Sans Devanagari, Consolas, Monaco, Courier New, monospace',
                lineNumbers: 'on',
                minimap: { enabled: true },
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                renderWhitespace: 'selection',
                bracketPairColorization: { enabled: true },
              }}
            />
          </div>
        </div>

        <div className="output-panel">
          <div className="output-header">
            <h3>Output</h3>
            {(output || error) && (
              <button
                onClick={() => {
                  setOutput('');
                  setError(null);
                }}
                className="btn btn-outline clear-btn"
              >
                Clear
              </button>
            )}
          </div>

          <div className="output-content">
            {error ? (
              <div className="error-output">
                <div className="error-header">❌ Error</div>
                <pre>{error}</pre>
              </div>
            ) : output ? (
              <div className="success-output">
                <div className="output-header">✅ Result</div>
                <pre>{output}</pre>
              </div>
            ) : (
              <div className="empty-output">
                <div className="empty-content">
                  <span className="empty-icon">📝</span>
                  <p>Execute your code to see the output here</p>
                  <div className="shortcuts">
                    <div className="shortcut">
                      <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to execute
                    </div>
                    <div className="shortcut">
                      <kbd>Ctrl</kbd> + <kbd>S</kbd> to save
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodeEditor;
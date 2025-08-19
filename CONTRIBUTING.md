# Contributing to Vidya Quantum Interface

Thank you for your interest in contributing to Vidya! This project combines ancient Sanskrit wisdom with modern quantum computing concepts, and we welcome contributions from developers, linguists, Sanskrit scholars, and quantum computing enthusiasts.

## 🌟 Ways to Contribute

### 🐛 Bug Reports
- Use the [GitHub Issues](https://github.com/Mukesh1q2/Living-code-OSS/issues) page
- Include detailed reproduction steps
- Provide system information (OS, Python version, Node.js version)
- Include relevant logs and error messages

### ✨ Feature Requests
- Check existing issues to avoid duplicates
- Describe the feature's purpose and benefits
- Consider how it fits with the project's Sanskrit/quantum theme
- Provide mockups or examples if applicable

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch
- Follow our coding standards
- Add tests for new functionality
- Update documentation as needed

### 📚 Documentation
- Improve existing documentation
- Add examples and tutorials
- Translate documentation (especially Sanskrit-related content)
- Create video tutorials or demos

### 🕉️ Sanskrit Expertise
- Improve Sanskrit grammar rules
- Add new morphological analysis patterns
- Enhance etymology databases
- Validate linguistic accuracy

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/Mukesh1q2/Living-code-OSS.git
   cd Living-code-OSS
   ```

2. **Set Up Development Environment**
   ```bash
   # Quick setup
   .\start-simple.ps1 setup
   
   # Or manual setup
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Install Frontend Dependencies**
   ```bash
   cd vidya-quantum-interface
   npm install
   cd ..
   ```

4. **Run Tests**
   ```bash
   # Backend tests
   pytest
   
   # Frontend tests
   cd vidya-quantum-interface
   npm test
   ```

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest
   cd vidya-quantum-interface && npm test
   
   # Run linting
   flake8 src/ tests/
   black src/ tests/
   mypy src/
   
   # Frontend linting
   cd vidya-quantum-interface
   npm run lint
   npm run type-check
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use the GitHub interface
   - Fill out the PR template
   - Link related issues

## 📋 Coding Standards

### Python Code Style

- **Formatting**: Use [Black](https://black.readthedocs.io/) with default settings
- **Linting**: Use [flake8](https://flake8.pycqa.org/) with project configuration
- **Type Hints**: Use [mypy](https://mypy.readthedocs.io/) for type checking
- **Docstrings**: Use Google-style docstrings

```python
def analyze_sanskrit_word(word: str, apply_sandhi: bool = True) -> Dict[str, Any]:
    """Analyze a Sanskrit word using morphological rules.
    
    Args:
        word: The Sanskrit word to analyze
        apply_sandhi: Whether to apply sandhi rules
        
    Returns:
        Dictionary containing morphological analysis results
        
    Raises:
        ValueError: If word contains invalid characters
    """
    pass
```

### TypeScript/React Code Style

- **Formatting**: Use [Prettier](https://prettier.io/) with project configuration
- **Linting**: Use [ESLint](https://eslint.org/) with TypeScript rules
- **Components**: Use functional components with hooks
- **Props**: Define interfaces for all component props

```typescript
interface SanskritAnalyzerProps {
  text: string;
  onAnalysisComplete: (result: AnalysisResult) => void;
  enableQuantumEffects?: boolean;
}

const SanskritAnalyzer: React.FC<SanskritAnalyzerProps> = ({
  text,
  onAnalysisComplete,
  enableQuantumEffects = true
}) => {
  // Component implementation
};
```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(sanskrit): add new morphological analysis rules
fix(quantum): resolve WebGL context loss issue
docs(api): update Sanskrit analysis endpoint documentation
test(frontend): add integration tests for quantum visualizations
```

## 🧪 Testing Guidelines

### Quick start (from bundle recommendations)

- Create a virtual environment:
  - PowerShell: .\.venv\Scripts\Activate.ps1 after python -m venv .venv
  - Bash: source .venv/bin/activate
- Install dev tools: pip install -U pip && pip install -e .[dev] (or use requirements.txt)
- Install pre-commit hooks: pre-commit install
- Run full validation locally:
  - make validate (or scripts/validate.sh)
  - Frontend: npm ci && npm run build in the chosen frontend directory

### Backend Testing

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test API endpoints and database interactions
- **Performance Tests**: Test Sanskrit processing performance

```python
def test_sanskrit_morphological_analysis():
    """Test basic morphological analysis functionality."""
    analyzer = SanskritAnalyzer()
    result = analyzer.analyze("गच्छति")
    
    assert result["root"] == "गम्"
    assert result["form"] == "present_tense"
    assert result["person"] == "third"
```

### Frontend Testing

- **Unit Tests**: Test individual components and utilities
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user workflows

```typescript
describe('SanskritAnalyzer', () => {
  it('should display analysis results', async () => {
    render(<SanskritAnalyzer text="गच्छति" onAnalysisComplete={jest.fn()} />);
    
    const analyzeButton = screen.getByText('Analyze');
    fireEvent.click(analyzeButton);
    
    await waitFor(() => {
      expect(screen.getByText('Root: गम्')).toBeInTheDocument();
    });
  });
});
```

## 📚 Documentation Standards

### Code Documentation

- **Python**: Use Google-style docstrings
- **TypeScript**: Use JSDoc comments
- **API**: Document all endpoints with OpenAPI/Swagger
- **Components**: Document props and usage examples

### User Documentation

- **Clear Examples**: Provide working code examples
- **Screenshots**: Include visual examples for UI features
- **Step-by-Step**: Break down complex processes
- **Troubleshooting**: Include common issues and solutions

## 🕉️ Sanskrit Contributions

### Linguistic Accuracy

- **Grammar Rules**: Follow Paninian grammar principles
- **Transliteration**: Use IAST (International Alphabet of Sanskrit Transliteration)
- **Etymology**: Provide accurate word origins and meanings
- **Context**: Consider historical and cultural context

### Sanskrit Resources

- **Primary Sources**: Reference classical texts when possible
- **Modern Scholarship**: Include contemporary Sanskrit research
- **Validation**: Have Sanskrit experts review linguistic contributions
- **Documentation**: Explain Sanskrit concepts for non-experts

## 🔮 Quantum Computing Contributions

### Quantum Concepts

- **Accuracy**: Ensure quantum mechanics concepts are scientifically accurate
- **Visualization**: Create intuitive quantum state representations
- **Performance**: Optimize quantum simulations for real-time interaction
- **Education**: Make quantum concepts accessible to general users

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers bumped
- [ ] Security review completed
- [ ] Performance benchmarks run

## 🤝 Community Guidelines

### Code of Conduct

- **Respectful**: Treat all contributors with respect
- **Inclusive**: Welcome contributors from all backgrounds
- **Constructive**: Provide helpful feedback and suggestions
- **Patient**: Help newcomers learn and contribute
- **Cultural Sensitivity**: Respect Sanskrit's cultural significance

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and ideas
- **Pull Request Reviews**: For code-specific discussions
- **Email**: For private or sensitive matters

## 🏆 Recognition

Contributors will be recognized in:
- **README.md**: Major contributors listed
- **CHANGELOG.md**: Contributors noted for each release
- **GitHub**: Contributor graphs and statistics
- **Documentation**: Expert contributors acknowledged

## 📞 Getting Help

- **Documentation**: Check existing docs first
- **GitHub Issues**: Search existing issues
- **GitHub Discussions**: Ask questions in discussions
- **Code Review**: Request review from maintainers

## 🎯 Contribution Areas

### High Priority
- Sanskrit grammar rule improvements
- Quantum visualization enhancements
- Performance optimizations
- Mobile responsiveness
- Accessibility improvements

### Medium Priority
- Additional AI model integrations
- Voice interaction features
- Advanced analytics
- Internationalization
- Plugin system

### Future Considerations
- VR/AR interface support
- Collaborative features
- Educational modules
- API marketplace
- Enterprise features

Thank you for contributing to Vidya Quantum Interface! Together, we're bridging ancient wisdom and modern technology. 🕉️✨
# In-System Help and Contextual Guidance

## Overview

This document defines the in-system help system for the Sanskrit Rewrite Engine, providing contextual guidance, tooltips, and interactive help features.

## Help System Architecture

### Context-Aware Help
```python
class HelpSystem:
    def __init__(self):
        self.help_content = {}
        self.context_handlers = {}
        self.user_progress = {}
    
    def get_contextual_help(self, context, user_level="beginner"):
        """Get help content based on current context"""
        if context in self.help_content:
            return self.help_content[context].get(user_level, 
                   self.help_content[context]["beginner"])
        return self.get_default_help()
    
    def register_context_handler(self, context, handler):
        """Register handler for specific context"""
        self.context_handlers[context] = handler
```

### Interactive Tooltips
```python
class TooltipManager:
    def __init__(self):
        self.tooltips = {
            "tokenization": {
                "title": "Tokenization",
                "content": "Breaking Sanskrit text into meaningful units",
                "example": "rāma → [r][ā][m][a]",
                "learn_more": "/docs/tokenization"
            },
            "sandhi": {
                "title": "Sandhi Rules",
                "content": "Phonological changes at word boundaries",
                "example": "rāma + iti → rāmeti (a + i → e)",
                "learn_more": "/docs/sandhi"
            },
            "morphology": {
                "title": "Morphological Analysis",
                "content": "Analysis of word structure and inflection",
                "example": "rāmasya → rāma (stem) + asya (genitive)",
                "learn_more": "/docs/morphology"
            }
        }
```

## Help Content Definitions

### Beginner Level Help

#### Getting Started
```yaml
getting_started:
  title: "Getting Started with Sanskrit Processing"
  sections:
    - title: "What is the Sanskrit Rewrite Engine?"
      content: |
        The Sanskrit Rewrite Engine automatically applies grammatical rules
        to transform Sanskrit text. It handles sandhi (phonological changes),
        morphology (word structure), and compound formation.
      
    - title: "Basic Usage"
      content: |
        1. Enter your Sanskrit text (Devanagari or IAST)
        2. Click 'Process' to apply transformations
        3. Review the results and transformation traces
      example: |
        Input: rāma + iti
        Output: rāmeti
        Rule Applied: Vowel Sandhi (a + i → e)

    - title: "Input Formats"
      content: |
        • Devanagari: राम इति
        • IAST: rāma iti  
        • Mixed: rāma + इति
        • Marked: rāma : GEN (for morphology)
```

#### Common Tasks
```yaml
common_tasks:
  sandhi_processing:
    title: "Processing Sandhi"
    description: "Combine words using Sanskrit phonological rules"
    steps:
      - "Enter words separated by '+': rāma + iti"
      - "Click 'Process' to apply sandhi rules"
      - "View the combined result: rāmeti"
    tips:
      - "Use '+' to mark word boundaries"
      - "The engine applies rules automatically"
      - "Check traces to see which rules fired"
  
  morphology:
    title: "Morphological Analysis"
    description: "Analyze or generate inflected word forms"
    steps:
      - "Enter stem with case marker: rāma : GEN"
      - "Enable morphological analysis"
      - "Process to get inflected form: rāmasya"
    tips:
      - "Use ':' followed by case abbreviation"
      - "Available cases: NOM, GEN, DAT, ACC, etc."
      - "Works with verbs and nouns"
```

### Intermediate Level Help

#### Advanced Features
```yaml
advanced_features:
  custom_rules:
    title: "Creating Custom Rules"
    description: "Add your own transformation rules"
    example: |
      ```python
      def match_pattern(tokens, index):
          return tokens[index].text == "target"
      
      def apply_transformation(tokens, index):
          new_token = Token("replacement", TokenKind.OTHER)
          return tokens[:index] + [new_token] + tokens[index+1:], index+1
      
      rule = Rule(
          priority=1,
          name="custom_rule",
          match_fn=match_pattern,
          apply_fn=apply_transformation
      )
      engine.add_rule(rule)
      ```
  
  performance_tuning:
    title: "Performance Optimization"
    description: "Optimize processing for large texts"
    techniques:
      - "Enable performance mode for speed"
      - "Disable tracing for large batches"
      - "Use chunking for very large texts"
      - "Cache frequently processed patterns"
```

### Expert Level Help

#### Research Applications
```yaml
research_applications:
  corpus_analysis:
    title: "Corpus-Scale Analysis"
    description: "Process and analyze large Sanskrit corpora"
    workflow: |
      1. Prepare corpus files in UTF-8 encoding
      2. Set up batch processing pipeline
      3. Configure analysis parameters
      4. Process corpus systematically
      5. Extract statistical patterns
      6. Generate research reports
  
  rule_development:
    title: "Developing New Rules"
    description: "Create and test new grammatical rules"
    methodology: |
      1. Study traditional grammar sources
      2. Define rule conditions precisely
      3. Implement match and apply functions
      4. Create comprehensive test cases
      5. Validate against known examples
      6. Integrate with existing rule set
```

## Interactive Guidance System

### Onboarding Flow
```python
class OnboardingGuide:
    def __init__(self):
        self.steps = [
            {
                "id": "welcome",
                "title": "Welcome to Sanskrit Processing",
                "content": "Let's start with a simple example",
                "action": "show_example",
                "example": "rāma + iti"
            },
            {
                "id": "process_text",
                "title": "Process Your First Text",
                "content": "Click the Process button to see the transformation",
                "highlight": "#process-button",
                "wait_for": "process_complete"
            },
            {
                "id": "view_results",
                "title": "Understanding Results",
                "content": "The output shows the transformed text and applied rules",
                "highlight": "#results-panel"
            },
            {
                "id": "explore_traces",
                "title": "Transformation Traces",
                "content": "Click on traces to see step-by-step transformations",
                "highlight": "#traces-section"
            }
        ]
    
    def start_onboarding(self, user_id):
        """Start interactive onboarding for new user"""
        return self.steps[0]
    
    def next_step(self, current_step, user_action):
        """Progress to next onboarding step"""
        current_index = next(i for i, step in enumerate(self.steps) 
                           if step["id"] == current_step)
        
        if current_index < len(self.steps) - 1:
            return self.steps[current_index + 1]
        return None  # Onboarding complete
```

### Smart Suggestions
```python
class SmartSuggestions:
    def __init__(self):
        self.suggestion_rules = {
            "empty_input": {
                "condition": lambda context: not context.get("input_text"),
                "suggestion": "Try entering 'rāma + iti' to see sandhi in action",
                "type": "example"
            },
            "no_transformations": {
                "condition": lambda context: context.get("passes") == 0,
                "suggestion": "No rules applied. Try adding '+' between words for sandhi",
                "type": "tip"
            },
            "non_convergence": {
                "condition": lambda context: not context.get("converged"),
                "suggestion": "Processing didn't converge. Try reducing max passes or checking for rule conflicts",
                "type": "warning"
            },
            "performance_issue": {
                "condition": lambda context: context.get("processing_time", 0) > 5.0,
                "suggestion": "Slow processing detected. Consider enabling performance mode",
                "type": "optimization"
            }
        }
    
    def get_suggestions(self, context):
        """Get smart suggestions based on current context"""
        suggestions = []
        for rule_name, rule in self.suggestion_rules.items():
            if rule["condition"](context):
                suggestions.append({
                    "text": rule["suggestion"],
                    "type": rule["type"],
                    "action": rule.get("action")
                })
        return suggestions
```

## Error Messages and Recovery

### User-Friendly Error Messages
```python
class ErrorMessageSystem:
    def __init__(self):
        self.error_messages = {
            "TokenizationError": {
                "user_message": "Unable to process the input text",
                "explanation": "The text contains characters that cannot be tokenized",
                "suggestions": [
                    "Check that text is in valid Sanskrit script",
                    "Ensure proper UTF-8 encoding",
                    "Remove any special characters"
                ],
                "help_link": "/docs/troubleshooting#tokenization"
            },
            "RuleApplicationError": {
                "user_message": "A transformation rule encountered an error",
                "explanation": "One of the grammatical rules failed to apply correctly",
                "suggestions": [
                    "Try processing a simpler text first",
                    "Check if custom rules are properly defined",
                    "Report this issue if it persists"
                ],
                "help_link": "/docs/troubleshooting#rules"
            },
            "ConvergenceError": {
                "user_message": "Processing did not complete within the maximum passes",
                "explanation": "The transformation process didn't reach a stable state",
                "suggestions": [
                    "Increase the maximum number of passes",
                    "Check for conflicting rules",
                    "Simplify the input text"
                ],
                "help_link": "/docs/troubleshooting#convergence"
            }
        }
    
    def format_error_message(self, error, context=None):
        """Format user-friendly error message"""
        error_type = type(error).__name__
        
        if error_type in self.error_messages:
            template = self.error_messages[error_type]
            return {
                "title": template["user_message"],
                "description": template["explanation"],
                "suggestions": template["suggestions"],
                "help_link": template["help_link"],
                "technical_details": str(error) if context and context.get("show_technical") else None
            }
        
        # Generic error message
        return {
            "title": "An unexpected error occurred",
            "description": "Please try again or contact support if the problem persists",
            "suggestions": ["Check your input text", "Try a simpler example"],
            "technical_details": str(error)
        }
```

## Progressive Disclosure

### Layered Information Architecture
```python
class ProgressiveHelp:
    def __init__(self):
        self.help_layers = {
            "basic": {
                "sandhi": "Combines words: rāma + iti → rāmeti",
                "morphology": "Changes word forms: rāma → rāmasya (genitive)",
                "compounds": "Forms compound words: deva + rāja → devarāja"
            },
            "detailed": {
                "sandhi": {
                    "description": "Sandhi rules handle phonological changes at word boundaries",
                    "types": ["vowel sandhi", "consonant sandhi", "visarga sandhi"],
                    "examples": {
                        "vowel": "a + i → e (rāma + iti → rāmeti)",
                        "consonant": "t + c → c (tat + ca → tac ca)",
                        "visarga": "ḥ + vowel → r (namaḥ + iti → namar iti)"
                    }
                }
            },
            "technical": {
                "sandhi": {
                    "implementation": "Token-based pattern matching with priority-ordered rules",
                    "algorithm": "Left-to-right scan with guard system for loop prevention",
                    "customization": "Add custom rules via Rule class with match_fn and apply_fn"
                }
            }
        }
    
    def get_help_content(self, topic, level="basic"):
        """Get help content at specified detail level"""
        if topic in self.help_layers[level]:
            return self.help_layers[level][topic]
        return self.help_layers["basic"].get(topic, "No help available")
```

## Contextual Examples

### Dynamic Example Generation
```python
class ExampleGenerator:
    def __init__(self):
        self.example_templates = {
            "sandhi": [
                {"pattern": "WORD + iti", "examples": ["rāma + iti", "deva + iti", "guru + iti"]},
                {"pattern": "WORD + indra", "examples": ["deva + indra", "mahā + indra"]},
                {"pattern": "WORD + ātman", "examples": ["mahā + ātman", "param + ātman"]}
            ],
            "morphology": [
                {"pattern": "WORD : GEN", "examples": ["rāma : GEN", "deva : GEN", "guru : GEN"]},
                {"pattern": "WORD : DAT", "examples": ["rāma : DAT", "deva : DAT", "guru : DAT"]},
                {"pattern": "WORD : ACC", "examples": ["rāma : ACC", "deva : ACC", "guru : ACC"]}
            ],
            "compounds": [
                {"pattern": "ADJECTIVE + NOUN", "examples": ["mahā + rāja", "su + putra", "param + guru"]},
                {"pattern": "NOUN + NOUN", "examples": ["deva + rāja", "guru + kula", "dharma + śāstra"]}
            ]
        }
    
    def get_contextual_examples(self, feature, user_input=None):
        """Generate examples relevant to current context"""
        if feature in self.example_templates:
            templates = self.example_templates[feature]
            
            # If user provided input, try to generate similar examples
            if user_input:
                return self.generate_similar_examples(user_input, templates)
            
            # Otherwise return default examples
            return [template["examples"][0] for template in templates]
        
        return []
    
    def generate_similar_examples(self, user_input, templates):
        """Generate examples similar to user input"""
        # Simple pattern matching - could be more sophisticated
        examples = []
        for template in templates:
            if any(word in user_input for word in template["examples"][0].split()):
                examples.extend(template["examples"][:2])
        return examples[:3]  # Limit to 3 examples
```

## Help Integration Points

### Web Interface Integration
```javascript
// Help system integration for web interface
class WebHelpSystem {
    constructor() {
        this.helpAPI = new HelpAPI();
        this.currentContext = null;
        this.userLevel = 'beginner';
    }
    
    // Show contextual help tooltip
    showTooltip(element, topic) {
        const helpContent = this.helpAPI.getTooltip(topic, this.userLevel);
        this.displayTooltip(element, helpContent);
    }
    
    // Show help panel
    showHelpPanel(topic) {
        const helpContent = this.helpAPI.getHelp(topic, this.userLevel);
        this.displayHelpPanel(helpContent);
    }
    
    // Smart suggestions based on user actions
    updateSuggestions(context) {
        const suggestions = this.helpAPI.getSuggestions(context);
        this.displaySuggestions(suggestions);
    }
    
    // Progressive help disclosure
    expandHelp(topic, currentLevel) {
        const nextLevel = this.getNextLevel(currentLevel);
        const expandedContent = this.helpAPI.getHelp(topic, nextLevel);
        this.displayExpandedHelp(expandedContent);
    }
}
```

### CLI Integration
```python
class CLIHelpSystem:
    def __init__(self):
        self.help_content = load_help_content()
    
    def show_command_help(self, command):
        """Show help for specific CLI command"""
        if command in self.help_content:
            print(f"Help for '{command}':")
            print(self.help_content[command]["description"])
            print("\nUsage:")
            print(self.help_content[command]["usage"])
            print("\nExamples:")
            for example in self.help_content[command]["examples"]:
                print(f"  {example}")
    
    def show_interactive_help(self):
        """Start interactive help session"""
        print("Sanskrit Rewrite Engine - Interactive Help")
        print("Type 'help <topic>' for specific help, 'quit' to exit")
        
        while True:
            user_input = input("Help> ").strip()
            
            if user_input == "quit":
                break
            elif user_input.startswith("help "):
                topic = user_input[5:]
                self.show_topic_help(topic)
            elif user_input == "topics":
                self.list_help_topics()
            else:
                print("Available commands: help <topic>, topics, quit")
```

This comprehensive in-system help system provides contextual guidance, progressive disclosure, and user-friendly error handling to make the Sanskrit Rewrite Engine accessible to users of all levels.
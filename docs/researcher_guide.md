# Sanskrit Rewrite Engine - Researcher's Guide

## Introduction

This guide is designed for computational linguists, digital humanities researchers, and computer scientists working with Sanskrit computational systems. It covers advanced research applications, methodologies, and integration possibilities.

## Research Applications

### Computational Linguistics Research

#### Morphological Analysis Studies
```python
from sanskrit_rewrite_engine import MorphologicalAnalyzer

analyzer = MorphologicalAnalyzer()
results = analyzer.analyze_corpus("mahabharata_corpus.txt")

# Extract morphological patterns
patterns = results.get_morphological_patterns()
frequency_dist = results.get_morpheme_frequencies()
```

#### Phonological Process Investigation
```python
from sanskrit_rewrite_engine import PhonologicalAnalyzer

# Study sandhi application patterns
analyzer = PhonologicalAnalyzer()
sandhi_data = analyzer.extract_sandhi_contexts(corpus)

# Analyze rule productivity
productivity_metrics = analyzer.calculate_rule_productivity(sandhi_data)
```

#### Syntactic Pattern Mining
```python
from sanskrit_rewrite_engine import SyntacticAnalyzer

# Extract compound formation patterns
analyzer = SyntacticAnalyzer()
compounds = analyzer.extract_compounds(corpus)
compound_types = analyzer.classify_compound_types(compounds)
```

### Digital Humanities Research

#### Manuscript Variation Studies
```python
from sanskrit_rewrite_engine import VariationAnalyzer

# Compare manuscript variants
analyzer = VariationAnalyzer()
variants = analyzer.compare_manuscripts([
    "manuscript_a.txt",
    "manuscript_b.txt", 
    "manuscript_c.txt"
])

# Generate variation maps
variation_map = analyzer.create_variation_map(variants)
```

#### Historical Linguistics Analysis
```python
from sanskrit_rewrite_engine import HistoricalAnalyzer

# Track linguistic changes over time
analyzer = HistoricalAnalyzer()
changes = analyzer.track_changes_over_time([
    ("vedic_corpus", -1500),
    ("classical_corpus", 0),
    ("medieval_corpus", 1000)
])
```

#### Authorship Attribution
```python
from sanskrit_rewrite_engine import StyleAnalyzer

# Extract stylistic features
analyzer = StyleAnalyzer()
features = analyzer.extract_stylistic_features(text)

# Compare with known authors
similarity = analyzer.compare_with_authors(features, author_database)
```

### Machine Learning Integration

#### Feature Extraction for NLP Models
```python
from sanskrit_rewrite_engine import FeatureExtractor

extractor = FeatureExtractor()

# Extract linguistic features
features = extractor.extract_features(text, feature_types=[
    'morphological',
    'phonological', 
    'syntactic',
    'semantic'
])

# Use for ML training
X = features.to_matrix()
y = labels
model.fit(X, y)
```

#### Training Data Generation
```python
from sanskrit_rewrite_engine import DataGenerator

# Generate synthetic Sanskrit data
generator = DataGenerator()
synthetic_data = generator.generate_training_data(
    patterns=['sandhi', 'compounds', 'inflections'],
    size=10000
)
```

## Advanced Analysis Techniques

### Corpus-Scale Processing

#### Batch Processing Framework
```python
from sanskrit_rewrite_engine import BatchProcessor
import multiprocessing as mp

def process_file(filename):
    engine = SanskritRewriteEngine()
    with open(filename, 'r') as f:
        text = f.read()
    return engine.process(text)

# Parallel processing
with mp.Pool() as pool:
    results = pool.map(process_file, file_list)
```

#### Streaming Analysis
```python
from sanskrit_rewrite_engine import StreamingProcessor

# Process large texts incrementally
processor = StreamingProcessor()
for chunk in processor.stream_process("large_corpus.txt", chunk_size=1000):
    # Analyze chunk
    analysis = processor.analyze_chunk(chunk)
    # Store results
    database.store_analysis(analysis)
```

### Statistical Analysis

#### Rule Application Statistics
```python
from sanskrit_rewrite_engine import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
stats = analyzer.analyze_rule_applications(corpus)

# Rule frequency analysis
rule_frequencies = stats.get_rule_frequencies()
rule_contexts = stats.get_rule_contexts()

# Statistical significance testing
significance = analyzer.test_rule_significance(rule_frequencies)
```

#### Linguistic Complexity Metrics
```python
# Calculate complexity measures
complexity = analyzer.calculate_complexity_metrics(text, metrics=[
    'morphological_complexity',
    'syntactic_depth',
    'lexical_diversity',
    'phonological_complexity'
])
```

### Network Analysis

#### Morphological Networks
```python
from sanskrit_rewrite_engine import NetworkAnalyzer
import networkx as nx

analyzer = NetworkAnalyzer()

# Build morphological derivation network
network = analyzer.build_morphological_network(corpus)

# Analyze network properties
centrality = nx.betweenness_centrality(network)
clusters = nx.community.greedy_modularity_communities(network)
```

#### Semantic Relationship Networks
```python
# Build semantic networks
semantic_net = analyzer.build_semantic_network(corpus)

# Find semantic clusters
semantic_clusters = analyzer.find_semantic_clusters(semantic_net)
```

## Experimental Methodologies

### Controlled Experiments

#### Rule Ablation Studies
```python
from sanskrit_rewrite_engine import ExperimentFramework

# Test impact of individual rules
framework = ExperimentFramework()

baseline_engine = SanskritRewriteEngine()
baseline_results = framework.evaluate(baseline_engine, test_corpus)

# Remove specific rule and test
ablated_engine = baseline_engine.copy()
ablated_engine.remove_rule("vowel_sandhi_a_i")
ablated_results = framework.evaluate(ablated_engine, test_corpus)

# Compare performance
performance_diff = framework.compare_results(baseline_results, ablated_results)
```

#### Parameter Sensitivity Analysis
```python
# Test different parameter settings
parameters = {
    'max_passes': [10, 20, 50],
    'rule_priorities': ['traditional', 'frequency_based', 'optimized'],
    'tokenization_mode': ['strict', 'lenient', 'adaptive']
}

results = framework.parameter_sweep(parameters, test_corpus)
optimal_params = framework.find_optimal_parameters(results)
```

### Evaluation Methodologies

#### Gold Standard Evaluation
```python
from sanskrit_rewrite_engine import EvaluationFramework

evaluator = EvaluationFramework()

# Load gold standard annotations
gold_standard = evaluator.load_gold_standard("annotated_corpus.json")

# Evaluate system performance
metrics = evaluator.evaluate_against_gold_standard(
    system_output, 
    gold_standard,
    metrics=['accuracy', 'precision', 'recall', 'f1']
)
```

#### Cross-Validation Studies
```python
# K-fold cross-validation
cv_results = evaluator.cross_validate(
    engine, 
    corpus, 
    k=10,
    stratify_by='text_type'
)

# Statistical significance testing
significance = evaluator.test_significance(cv_results)
```

### Reproducibility Framework

#### Experiment Configuration
```python
from sanskrit_rewrite_engine import ExperimentConfig

config = ExperimentConfig({
    'engine_version': '2.1.0',
    'rule_set': 'classical_v1.2',
    'parameters': {
        'max_passes': 20,
        'enable_tracing': True
    },
    'corpus': {
        'name': 'test_corpus_v1.0',
        'size': 10000,
        'preprocessing': 'standard'
    },
    'evaluation': {
        'metrics': ['accuracy', 'coverage'],
        'gold_standard': 'manual_annotations_v1.0'
    }
})

# Run reproducible experiment
results = framework.run_experiment(config)
```

#### Result Archival
```python
# Archive experiment results
archive = ExperimentArchive()
archive.store_experiment(
    config=config,
    results=results,
    code_version=git_commit_hash,
    timestamp=datetime.now()
)

# Retrieve previous experiments
previous_results = archive.query_experiments(
    filters={'corpus.name': 'test_corpus_v1.0'}
)
```

## Integration with Research Tools

### Jupyter Notebook Integration

```python
# Install Jupyter extension
%load_ext sanskrit_rewrite_engine.jupyter

# Interactive analysis
%%sanskrit_analysis
rāma + iti
```

### R Integration

```r
# R package for Sanskrit analysis
library(sanskritR)

# Process Sanskrit text from R
result <- process_sanskrit("rāma + iti")
plot_rule_trace(result)
```

### NLTK Integration

```python
import nltk
from sanskrit_rewrite_engine.nltk_bridge import SanskritTokenizer

# Use as NLTK tokenizer
tokenizer = SanskritTokenizer()
nltk.data.register('tokenizers/sanskrit', tokenizer)

# NLTK corpus integration
from nltk.corpus import PlaintextCorpusReader
sanskrit_corpus = PlaintextCorpusReader('sanskrit_texts/', '.*\.txt')
```

### Pandas Integration

```python
import pandas as pd
from sanskrit_rewrite_engine import DataFrameProcessor

# Process DataFrame of Sanskrit texts
df = pd.DataFrame({'text': sanskrit_texts, 'author': authors})
processor = DataFrameProcessor()

# Add analysis columns
df = processor.add_analysis_columns(df, [
    'processed_text',
    'morphological_analysis',
    'rule_trace'
])
```

## Performance Optimization for Research

### Profiling and Benchmarking

```python
from sanskrit_rewrite_engine import Profiler

profiler = Profiler()

# Profile rule application
with profiler.profile('rule_application'):
    result = engine.process(text)

# Analyze performance bottlenecks
bottlenecks = profiler.analyze_bottlenecks()
optimization_suggestions = profiler.suggest_optimizations()
```

### Parallel Processing Strategies

```python
from sanskrit_rewrite_engine import ParallelProcessor
from dask import delayed, compute

# Dask integration for large-scale processing
@delayed
def process_text_delayed(text):
    return engine.process(text)

# Process corpus in parallel
tasks = [process_text_delayed(text) for text in corpus]
results = compute(*tasks)
```

### Memory Optimization

```python
from sanskrit_rewrite_engine import MemoryOptimizer

optimizer = MemoryOptimizer()

# Optimize for large corpus processing
optimized_engine = optimizer.optimize_for_corpus_processing(
    engine,
    expected_corpus_size=1000000,
    available_memory='8GB'
)
```

## Data Management and Sharing

### Corpus Management

```python
from sanskrit_rewrite_engine import CorpusManager

manager = CorpusManager()

# Create standardized corpus
corpus = manager.create_corpus(
    name="research_corpus_v1",
    texts=text_files,
    metadata=metadata_dict,
    preprocessing_config=preprocess_config
)

# Version control for corpora
manager.version_corpus(corpus, version="1.0", description="Initial release")
```

### Data Export and Sharing

```python
# Export results in standard formats
exporter = DataExporter()

# TEI XML export
exporter.export_to_tei(results, "results.xml")

# JSON-LD export for linked data
exporter.export_to_jsonld(results, "results.jsonld")

# CSV export for statistical analysis
exporter.export_to_csv(results, "results.csv")
```

### Collaboration Tools

```python
from sanskrit_rewrite_engine import CollaborationTools

# Share analysis pipelines
pipeline = AnalysisPipeline([
    TokenizationStep(),
    MorphologicalAnalysisStep(),
    StatisticalAnalysisStep()
])

tools = CollaborationTools()
tools.share_pipeline(pipeline, "morphological_analysis_v1")

# Collaborative annotation
annotator = CollaborativeAnnotator()
annotator.create_annotation_project(
    corpus=corpus,
    annotation_schema=schema,
    collaborators=["researcher1", "researcher2"]
)
```

## Quality Assurance and Validation

### Automated Testing

```python
from sanskrit_rewrite_engine import TestSuite

# Comprehensive test suite
test_suite = TestSuite()

# Regression testing
test_suite.add_regression_tests(known_good_examples)

# Property-based testing
test_suite.add_property_tests([
    'sandhi_reversibility',
    'morphological_consistency',
    'rule_ordering_invariants'
])

# Run tests
test_results = test_suite.run_all_tests()
```

### Validation Against Linguistic Theory

```python
from sanskrit_rewrite_engine import LinguisticValidator

validator = LinguisticValidator()

# Validate against Pāṇinian principles
paninian_validation = validator.validate_paninian_compliance(results)

# Check morphological plausibility
morphological_validation = validator.validate_morphology(results)

# Verify phonological naturalness
phonological_validation = validator.validate_phonology(results)
```

## Publication and Dissemination

### Academic Paper Integration

```python
# Generate figures for papers
from sanskrit_rewrite_engine import FigureGenerator

generator = FigureGenerator()

# Rule application frequency chart
fig1 = generator.create_rule_frequency_chart(results)
fig1.save("rule_frequency.pdf")

# Morphological complexity distribution
fig2 = generator.create_complexity_distribution(complexity_data)
fig2.save("complexity_distribution.pdf")
```

### Reproducible Research Packages

```python
# Create research package
package = ResearchPackage()
package.add_code(analysis_scripts)
package.add_data(corpus_data)
package.add_results(experiment_results)
package.add_documentation(paper_draft)

# Generate DOI and archive
doi = package.publish_to_zenodo(
    title="Sanskrit Morphological Analysis Study",
    authors=["Author1", "Author2"],
    description="Computational analysis of Sanskrit morphology"
)
```

## Future Research Directions

### Machine Learning Applications

- Neural morphological analyzers
- Transformer models for Sanskrit
- Reinforcement learning for rule optimization
- Unsupervised discovery of linguistic patterns

### Cross-Linguistic Studies

- Comparative Indo-European morphology
- Universal grammar validation
- Typological studies using computational methods
- Historical reconstruction algorithms

### Digital Humanities Integration

- Manuscript digitization pipelines
- Automated paleographic analysis
- Cultural analytics applications
- Network analysis of textual traditions

## Community and Collaboration

### Research Networks

- International Sanskrit computational linguistics community
- Digital humanities Sanskrit working groups
- Computational linguistics conferences and workshops
- Collaborative research projects and grants

### Open Science Practices

- Open source development
- Open data sharing
- Reproducible research standards
- Community-driven validation efforts

### Training and Education

- Graduate student training programs
- Workshop and tutorial development
- Online course materials
- Mentorship opportunities

This guide provides a comprehensive framework for researchers to leverage the Sanskrit Rewrite Engine in their computational linguistics and digital humanities research.
#
# Advanced Research Methodologies

### Large-Scale Corpus Analysis

#### Processing Sanskrit Digital Libraries
```python
import asyncio
from pathlib import Path
from sanskrit_rewrite_engine import SanskritRewriteEngine
from sanskrit_rewrite_engine.corpus import CorpusProcessor

async def process_digital_library(library_path, output_path):
    """Process entire digital Sanskrit libraries"""
    
    processor = CorpusProcessor(
        engine_config=EngineConfig(
            performance_mode=True,
            enable_parallel_processing=True,
            worker_count=8
        )
    )
    
    # Process all texts in parallel
    results = await processor.process_directory(
        input_dir=library_path,
        output_dir=output_path,
        file_pattern="*.txt",
        batch_size=100
    )
    
    # Generate corpus statistics
    stats = processor.generate_statistics(results)
    
    return results, stats

# Process large digital library
library_results, corpus_stats = asyncio.run(
    process_digital_library("gretil_corpus/", "processed_output/")
)

print(f"Processed {corpus_stats['total_texts']} texts")
print(f"Total transformations: {corpus_stats['total_transformations']}")
print(f"Most common rules: {corpus_stats['top_rules'][:10]}")
```

#### Statistical Analysis of Grammatical Patterns
```python
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def statistical_grammar_analysis(corpus_results):
    """Perform statistical analysis of grammatical patterns"""
    
    # Extract feature vectors for each text
    feature_vectors = []
    text_metadata = []
    
    for text_result in corpus_results:
        # Create feature vector from rule applications
        rule_counts = text_result.get_transformation_summary()
        
        # Normalize by text length
        text_length = len(text_result.input_text.split())
        normalized_counts = {rule: count/text_length for rule, count in rule_counts.items()}
        
        # Convert to vector (using consistent rule ordering)
        all_rules = sorted(set().union(*[r.get_transformation_summary().keys() for r in corpus_results]))
        vector = [normalized_counts.get(rule, 0) for rule in all_rules]
        
        feature_vectors.append(vector)
        text_metadata.append({
            'title': text_result.metadata.get('title', 'Unknown'),
            'period': text_result.metadata.get('period', 'Unknown'),
            'genre': text_result.metadata.get('genre', 'Unknown')
        })
    
    feature_matrix = np.array(feature_vectors)
    
    # Perform clustering analysis
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(feature_matrix)
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(feature_matrix)
    
    # Statistical tests
    results = {
        'feature_matrix': feature_matrix,
        'clusters': clusters,
        'pca_features': reduced_features,
        'cluster_centers': kmeans.cluster_centers_,
        'explained_variance': pca.explained_variance_ratio_
    }
    
    return results, text_metadata

# Perform statistical analysis
stats_results, metadata = statistical_grammar_analysis(library_results)
```

### Machine Learning Integration

#### Rule Learning from Corpus Data
```python
from sanskrit_rewrite_engine.ml import RuleLearner
from sklearn.ensemble import RandomForestClassifier

def learn_rules_from_corpus(training_data):
    """Learn new grammatical rules from corpus data"""
    
    learner = RuleLearner(
        base_engine=SanskritRewriteEngine(),
        ml_model=RandomForestClassifier(n_estimators=100)
    )
    
    # Prepare training data
    X, y = learner.prepare_training_data(training_data)
    
    # Train rule learning model
    learner.fit(X, y)
    
    # Extract learned rules
    learned_rules = learner.extract_rules(
        confidence_threshold=0.8,
        support_threshold=10
    )
    
    # Validate learned rules
    validation_results = learner.validate_rules(
        learned_rules, 
        validation_corpus
    )
    
    return learned_rules, validation_results

# Learn new rules from corpus
new_rules, validation = learn_rules_from_corpus(training_corpus)

# Add validated rules to engine
for rule in new_rules:
    if validation[rule.id]['accuracy'] > 0.9:
        engine.add_rule(rule)
```

#### Predictive Modeling for Sanskrit Processing
```python
from sanskrit_rewrite_engine.ml import SanskritPredictor
import torch
import torch.nn as nn

class SanskritTransformationPredictor(nn.Module):
    """Neural network for predicting Sanskrit transformations"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        predictions = self.classifier(lstm_out)
        return predictions

def train_transformation_predictor(training_data):
    """Train neural network to predict transformations"""
    
    # Prepare data
    tokenizer = SanskritTokenizer()
    vocab = tokenizer.build_vocab(training_data)
    
    model = SanskritTransformationPredictor(len(vocab))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(100):
        total_loss = 0
        for batch in training_data:
            optimizer.zero_grad()
            
            inputs, targets = prepare_batch(batch, tokenizer)
            predictions = model(inputs)
            
            loss = criterion(predictions.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return model, tokenizer

# Train predictive model
predictor_model, tokenizer = train_transformation_predictor(ml_training_data)
```

### Comparative Studies

#### Cross-Linguistic Analysis
```python
def compare_with_other_languages(sanskrit_patterns, other_language_data):
    """Compare Sanskrit patterns with other Indo-European languages"""
    
    comparison_results = {}
    
    for language, patterns in other_language_data.items():
        # Find similar phonological processes
        similar_processes = find_similar_processes(sanskrit_patterns, patterns)
        
        # Calculate similarity metrics
        similarity_score = calculate_pattern_similarity(sanskrit_patterns, patterns)
        
        # Identify unique Sanskrit features
        unique_features = identify_unique_features(sanskrit_patterns, patterns)
        
        comparison_results[language] = {
            'similarity_score': similarity_score,
            'similar_processes': similar_processes,
            'unique_sanskrit_features': unique_features
        }
    
    return comparison_results

# Compare with related languages
indo_european_data = {
    'greek': greek_phonological_patterns,
    'latin': latin_phonological_patterns,
    'avestan': avestan_phonological_patterns
}

cross_linguistic_analysis = compare_with_other_languages(
    sanskrit_phonological_patterns, 
    indo_european_data
)
```

#### Historical Development Studies
```python
def trace_historical_development(diachronic_corpus):
    """Trace historical development of Sanskrit features"""
    
    periods = ['vedic', 'classical', 'epic', 'medieval', 'modern']
    development_timeline = {}
    
    for period in periods:
        period_engine = SanskritRewriteEngine(
            config=EngineConfig(historical_period=period)
        )
        
        period_texts = diachronic_corpus[period]
        period_analysis = analyze_period_features(period_engine, period_texts)
        
        development_timeline[period] = {
            'rule_productivity': period_analysis['rule_productivity'],
            'new_patterns': period_analysis['new_patterns'],
            'obsolete_patterns': period_analysis['obsolete_patterns'],
            'frequency_changes': period_analysis['frequency_changes']
        }
    
    # Identify development trends
    trends = identify_development_trends(development_timeline)
    
    return development_timeline, trends

# Trace historical development
timeline, trends = trace_historical_development(historical_corpus)
```

## Research Data Management

### Corpus Annotation and Metadata
```python
from sanskrit_rewrite_engine.annotation import CorpusAnnotator

def create_annotated_corpus(raw_texts, annotation_schema):
    """Create richly annotated Sanskrit corpus"""
    
    annotator = CorpusAnnotator(schema=annotation_schema)
    
    annotated_corpus = []
    
    for text_id, text in enumerate(raw_texts):
        # Process with full tracing
        result = engine.process(text, enable_tracing=True)
        
        # Create comprehensive annotations
        annotations = {
            'text_id': text_id,
            'original_text': text,
            'processed_text': result.get_output_text(),
            'transformations': [],
            'linguistic_features': {},
            'metadata': {}
        }
        
        # Annotate transformations
        for pass_trace in result.traces:
            for transform in pass_trace.transformations:
                annotations['transformations'].append({
                    'rule_name': transform.rule_name,
                    'rule_type': transform.rule_type,
                    'position': transform.index,
                    'before': transform.before_pattern,
                    'after': transform.after_pattern,
                    'sutra_reference': transform.sutra_reference,
                    'confidence': transform.confidence_score
                })
        
        # Extract linguistic features
        annotations['linguistic_features'] = extract_linguistic_features(result)
        
        # Add metadata
        annotations['metadata'] = extract_metadata(text, result)
        
        annotated_corpus.append(annotations)
    
    return annotated_corpus

# Create annotated research corpus
annotation_schema = load_annotation_schema("sanskrit_research_schema.json")
annotated_corpus = create_annotated_corpus(research_texts, annotation_schema)
```

### Data Export and Interoperability
```python
def export_research_data(corpus_data, format='tei'):
    """Export research data in standard formats"""
    
    if format == 'tei':
        return export_to_tei(corpus_data)
    elif format == 'conllu':
        return export_to_conllu(corpus_data)
    elif format == 'json_ld':
        return export_to_json_ld(corpus_data)
    elif format == 'rdf':
        return export_to_rdf(corpus_data)
    else:
        raise ValueError(f"Unsupported format: {format}")

def export_to_tei(corpus_data):
    """Export to TEI XML format for digital humanities"""
    
    tei_documents = []
    
    for text_data in corpus_data:
        tei_doc = create_tei_document(
            text=text_data['processed_text'],
            annotations=text_data['transformations'],
            metadata=text_data['metadata']
        )
        tei_documents.append(tei_doc)
    
    return tei_documents

# Export in multiple formats
tei_export = export_research_data(annotated_corpus, 'tei')
conllu_export = export_research_data(annotated_corpus, 'conllu')
json_ld_export = export_research_data(annotated_corpus, 'json_ld')
```

## Publication and Reproducibility

### Reproducible Research Workflows
```python
import yaml
from datetime import datetime

def create_research_workflow(experiment_config):
    """Create reproducible research workflow"""
    
    workflow = {
        'experiment_id': generate_experiment_id(),
        'timestamp': datetime.now().isoformat(),
        'config': experiment_config,
        'environment': {
            'sanskrit_engine_version': sanskrit_rewrite_engine.__version__,
            'python_version': sys.version,
            'dependencies': get_dependency_versions()
        },
        'steps': []
    }
    
    # Define workflow steps
    steps = [
        {'name': 'data_preparation', 'function': prepare_data},
        {'name': 'preprocessing', 'function': preprocess_corpus},
        {'name': 'analysis', 'function': run_analysis},
        {'name': 'statistical_tests', 'function': run_statistical_tests},
        {'name': 'visualization', 'function': create_visualizations},
        {'name': 'export_results', 'function': export_results}
    ]
    
    # Execute workflow with logging
    for step in steps:
        print(f"Executing step: {step['name']}")
        
        step_start = datetime.now()
        step_result = step['function'](experiment_config)
        step_end = datetime.now()
        
        workflow['steps'].append({
            'name': step['name'],
            'start_time': step_start.isoformat(),
            'end_time': step_end.isoformat(),
            'duration': (step_end - step_start).total_seconds(),
            'result_summary': summarize_step_result(step_result)
        })
    
    # Save workflow log
    with open(f"workflow_{workflow['experiment_id']}.yaml", 'w') as f:
        yaml.dump(workflow, f)
    
    return workflow

# Create and execute reproducible workflow
experiment_config = {
    'corpus_path': 'research_corpus/',
    'analysis_type': 'morphological_complexity',
    'statistical_tests': ['anova', 'chi_square'],
    'visualization_types': ['histogram', 'scatter', 'heatmap']
}

workflow_log = create_research_workflow(experiment_config)
```

### Academic Publication Support
```python
def generate_publication_materials(research_results):
    """Generate materials for academic publication"""
    
    materials = {}
    
    # Generate LaTeX tables
    materials['latex_tables'] = generate_latex_tables(research_results)
    
    # Create publication-quality figures
    materials['figures'] = create_publication_figures(research_results)
    
    # Generate statistical summaries
    materials['statistics'] = generate_statistical_summaries(research_results)
    
    # Create supplementary data files
    materials['supplementary_data'] = create_supplementary_data(research_results)
    
    # Generate citation information
    materials['citations'] = generate_citations(research_results)
    
    return materials

def generate_latex_tables(results):
    """Generate LaTeX tables for publication"""
    
    tables = {}
    
    # Rule frequency table
    rule_freq_table = create_latex_table(
        data=results['rule_frequencies'],
        caption="Frequency of Sanskrit transformation rules in corpus",
        label="tab:rule_frequencies",
        columns=['Rule Name', 'Frequency', 'Percentage', 'Sūtra Reference']
    )
    
    tables['rule_frequencies'] = rule_freq_table
    
    # Statistical test results
    stats_table = create_latex_table(
        data=results['statistical_tests'],
        caption="Statistical test results for grammatical pattern analysis",
        label="tab:statistical_tests",
        columns=['Test', 'Statistic', 'p-value', 'Effect Size']
    )
    
    tables['statistical_tests'] = stats_table
    
    return tables

# Generate publication materials
pub_materials = generate_publication_materials(research_results)

# Save LaTeX tables
for table_name, table_latex in pub_materials['latex_tables'].items():
    with open(f"tables/{table_name}.tex", 'w') as f:
        f.write(table_latex)
```

## Integration with Research Infrastructure

### Database Integration
```python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class SanskritText(Base):
    __tablename__ = 'sanskrit_texts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    author = Column(String(255))
    period = Column(String(50))
    genre = Column(String(100))
    original_text = Column(Text)
    processed_text = Column(Text)
    processing_metadata = Column(Text)  # JSON
    created_at = Column(DateTime)

class TransformationRecord(Base):
    __tablename__ = 'transformations'
    
    id = Column(Integer, primary_key=True)
    text_id = Column(Integer)
    rule_name = Column(String(100))
    position = Column(Integer)
    before_pattern = Column(String(100))
    after_pattern = Column(String(100))
    sutra_reference = Column(String(50))

def setup_research_database(database_url):
    """Set up research database for Sanskrit corpus"""
    
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    return Session()

def store_corpus_in_database(corpus_data, session):
    """Store processed corpus in research database"""
    
    for text_data in corpus_data:
        # Store text record
        text_record = SanskritText(
            title=text_data['metadata'].get('title'),
            author=text_data['metadata'].get('author'),
            period=text_data['metadata'].get('period'),
            genre=text_data['metadata'].get('genre'),
            original_text=text_data['original_text'],
            processed_text=text_data['processed_text'],
            processing_metadata=json.dumps(text_data['linguistic_features']),
            created_at=datetime.now()
        )
        
        session.add(text_record)
        session.flush()  # Get the ID
        
        # Store transformation records
        for transform in text_data['transformations']:
            transform_record = TransformationRecord(
                text_id=text_record.id,
                rule_name=transform['rule_name'],
                position=transform['position'],
                before_pattern=transform['before'],
                after_pattern=transform['after'],
                sutra_reference=transform['sutra_reference']
            )
            session.add(transform_record)
    
    session.commit()

# Set up and populate research database
session = setup_research_database('postgresql://user:pass@localhost/sanskrit_research')
store_corpus_in_database(annotated_corpus, session)
```

### Cloud Computing Integration
```python
import boto3
from google.cloud import storage
import azure.storage.blob as azure_blob

def setup_cloud_processing(cloud_provider='aws'):
    """Set up cloud infrastructure for large-scale processing"""
    
    if cloud_provider == 'aws':
        return setup_aws_processing()
    elif cloud_provider == 'gcp':
        return setup_gcp_processing()
    elif cloud_provider == 'azure':
        return setup_azure_processing()

def setup_aws_processing():
    """Set up AWS infrastructure for Sanskrit processing"""
    
    # Set up S3 for data storage
    s3_client = boto3.client('s3')
    bucket_name = 'sanskrit-research-corpus'
    
    # Set up Lambda for serverless processing
    lambda_client = boto3.client('lambda')
    
    # Set up SQS for job queuing
    sqs_client = boto3.client('sqs')
    queue_url = sqs_client.create_queue(
        QueueName='sanskrit-processing-queue'
    )['QueueUrl']
    
    return {
        's3_client': s3_client,
        'lambda_client': lambda_client,
        'sqs_client': sqs_client,
        'bucket_name': bucket_name,
        'queue_url': queue_url
    }

def process_corpus_on_cloud(corpus_files, cloud_config):
    """Process large corpus using cloud infrastructure"""
    
    # Upload corpus files to cloud storage
    for file_path in corpus_files:
        upload_to_cloud_storage(file_path, cloud_config)
    
    # Submit processing jobs
    job_ids = []
    for file_path in corpus_files:
        job_id = submit_processing_job(file_path, cloud_config)
        job_ids.append(job_id)
    
    # Monitor job completion
    results = monitor_jobs(job_ids, cloud_config)
    
    # Download and aggregate results
    aggregated_results = aggregate_cloud_results(results, cloud_config)
    
    return aggregated_results

# Set up cloud processing
cloud_config = setup_cloud_processing('aws')
cloud_results = process_corpus_on_cloud(large_corpus_files, cloud_config)
```

## Best Practices for Researchers

### 1. Experimental Design
```python
def design_sanskrit_experiment(research_question, hypotheses):
    """Design rigorous Sanskrit computational linguistics experiment"""
    
    experiment_design = {
        'research_question': research_question,
        'hypotheses': hypotheses,
        'variables': {
            'independent': [],
            'dependent': [],
            'control': []
        },
        'methodology': {},
        'statistical_plan': {},
        'validation_strategy': {}
    }
    
    # Define variables based on research question
    if 'morphological_complexity' in research_question.lower():
        experiment_design['variables']['dependent'] = [
            'transformation_count',
            'rule_diversity',
            'convergence_passes'
        ]
        experiment_design['variables']['independent'] = [
            'text_period',
            'text_genre',
            'author'
        ]
    
    # Plan statistical analysis
    experiment_design['statistical_plan'] = {
        'descriptive_stats': ['mean', 'std', 'median', 'iqr'],
        'inferential_tests': ['anova', 'chi_square', 'correlation'],
        'effect_size_measures': ['cohens_d', 'eta_squared'],
        'multiple_comparison_correction': 'bonferroni'
    }
    
    return experiment_design

# Design experiment
research_q = "How does morphological complexity vary across Sanskrit literary periods?"
hypotheses = [
    "Classical Sanskrit shows higher morphological complexity than Vedic",
    "Epic Sanskrit shows intermediate complexity between Vedic and Classical"
]

experiment = design_sanskrit_experiment(research_q, hypotheses)
```

### 2. Data Quality Assurance
```python
def validate_corpus_quality(corpus_data):
    """Validate quality of Sanskrit corpus data"""
    
    quality_report = {
        'total_texts': len(corpus_data),
        'encoding_issues': [],
        'processing_errors': [],
        'linguistic_anomalies': [],
        'metadata_completeness': {},
        'recommendations': []
    }
    
    for i, text_data in enumerate(corpus_data):
        # Check encoding
        encoding_issues = check_text_encoding(text_data['original_text'])
        if encoding_issues:
            quality_report['encoding_issues'].append({
                'text_id': i,
                'issues': encoding_issues
            })
        
        # Check processing results
        if not text_data.get('processed_text'):
            quality_report['processing_errors'].append({
                'text_id': i,
                'error': 'No processed text found'
            })
        
        # Check for linguistic anomalies
        anomalies = detect_linguistic_anomalies(text_data)
        if anomalies:
            quality_report['linguistic_anomalies'].extend(anomalies)
        
        # Check metadata completeness
        required_metadata = ['title', 'author', 'period', 'genre']
        missing_metadata = [field for field in required_metadata 
                           if not text_data['metadata'].get(field)]
        if missing_metadata:
            quality_report['metadata_completeness'][i] = missing_metadata
    
    # Generate recommendations
    if quality_report['encoding_issues']:
        quality_report['recommendations'].append(
            "Fix encoding issues before analysis"
        )
    
    if len(quality_report['processing_errors']) > len(corpus_data) * 0.1:
        quality_report['recommendations'].append(
            "High processing error rate - review engine configuration"
        )
    
    return quality_report

# Validate corpus quality
quality_report = validate_corpus_quality(research_corpus)
print(f"Quality Report: {quality_report['recommendations']}")
```

### 3. Ethical Considerations
```python
def ensure_research_ethics(corpus_metadata):
    """Ensure ethical compliance in Sanskrit research"""
    
    ethics_checklist = {
        'copyright_compliance': False,
        'attribution_complete': False,
        'cultural_sensitivity': False,
        'data_sharing_permissions': False,
        'traditional_knowledge_respect': False
    }
    
    # Check copyright status
    for text_meta in corpus_metadata:
        if text_meta.get('copyright_status') != 'public_domain':
            if not text_meta.get('permission_granted'):
                print(f"Warning: Copyright permission needed for {text_meta['title']}")
                return ethics_checklist
    
    ethics_checklist['copyright_compliance'] = True
    
    # Check attribution completeness
    required_attribution = ['title', 'author', 'source', 'editor']
    for text_meta in corpus_metadata:
        missing_attribution = [field for field in required_attribution 
                             if not text_meta.get(field)]
        if missing_attribution:
            print(f"Warning: Missing attribution for {text_meta['title']}: {missing_attribution}")
            return ethics_checklist
    
    ethics_checklist['attribution_complete'] = True
    
    # Additional ethical checks...
    ethics_checklist['cultural_sensitivity'] = True
    ethics_checklist['data_sharing_permissions'] = True
    ethics_checklist['traditional_knowledge_respect'] = True
    
    return ethics_checklist

# Ensure ethical compliance
ethics_status = ensure_research_ethics(corpus_metadata)
if all(ethics_status.values()):
    print("✅ All ethical requirements met")
else:
    print("⚠️ Ethical review required")
```

---

This comprehensive researcher guide provides advanced methodologies, tools, and best practices for conducting rigorous computational linguistics research with Sanskrit texts. The engine's sophisticated architecture supports complex research workflows while maintaining scholarly standards and ethical practices.

*For implementation details, see the [Developer Guide](developer_guide.md). For linguistic applications, see the [Linguist Guide](linguist_guide.md).*

*Last updated: January 15, 2024*
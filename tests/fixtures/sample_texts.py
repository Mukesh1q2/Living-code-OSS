"""
Sample Sanskrit texts for testing.
"""

# Basic Sanskrit words for unit testing
BASIC_WORDS = [
    "rama",
    "sita", 
    "dharma",
    "karma",
    "yoga",
    "guru",
    "mantra",
    "sutra"
]

# Sanskrit text with morphological markers
MORPHOLOGICAL_EXAMPLES = [
    "rāma + iti",
    "dharma + a",
    "karma + ṇi",
    "yoga + īn",
    "guru + bhyām"
]

# Compound formation examples
COMPOUND_EXAMPLES = [
    ("rāja + putra", "rājaputra"),
    ("dharma + śāstra", "dharmaśāstra"),
    ("yoga + śāstra", "yogaśāstra"),
    ("guru + kula", "gurukula")
]

# Sandhi transformation examples
SANDHI_EXAMPLES = [
    ("rāma + iti", "rāmeti"),
    ("dharma + artha", "dharmārtha"),
    ("yoga + īśvara", "yogeśvara"),
    ("guru + upadeśa", "gurupadeśa")
]

# Devanagari to IAST examples
TRANSLITERATION_EXAMPLES = [
    ("राम", "rāma"),
    ("सीता", "sītā"),
    ("धर्म", "dharma"),
    ("कर्म", "karma"),
    ("योग", "yoga")
]

# Complex Sanskrit sentences for integration testing
COMPLEX_SENTENCES = [
    "rāmo rājā daśarathasya putraḥ",
    "sītā janakasya duhitā rāmasya patnī",
    "dharmo rakṣati rakṣitaḥ",
    "satyam eva jayate nānṛtam"
]

# Error cases for robustness testing
ERROR_CASES = [
    "",  # Empty string
    "   ",  # Whitespace only
    "123",  # Numbers only
    "!@#$%",  # Special characters only
    "a" * 1000,  # Very long string
    "mixed english and sanskrit राम",  # Mixed scripts
]

# Performance test data
PERFORMANCE_TEST_TEXTS = [
    "rāma",  # Tiny
    "rāma + iti",  # Small
    "rāma + iti dharma + artha",  # Medium
    " ".join(MORPHOLOGICAL_EXAMPLES),  # Large
    " ".join(COMPLEX_SENTENCES),  # Very large
]

# Unicode test cases
UNICODE_TEST_CASES = [
    ("Basic Latin", "hello world"),
    ("Latin with diacritics", "café naïve résumé"),
    ("Devanagari", "नमस्ते दुनिया"),
    ("Mixed scripts", "hello नमस्ते world"),
    ("Emoji", "🙏 नमस्ते 🌍"),
    ("Greek", "αβγδε"),
    ("Arabic", "السلام عليكم"),
    ("Cyrillic", "Привет мир"),
]

# Edge cases for testing
EDGE_CASES = [
    "\n",  # Newline only
    "\t",  # Tab only
    "\r\n",  # Windows line ending
    "a",  # Single character
    "ab",  # Two characters
    "a b",  # Space separated
    "a\nb",  # Newline separated
    "a\tb",  # Tab separated
    "a  b",  # Multiple spaces
    "  a  ",  # Leading/trailing spaces
]

# Stress test data
STRESS_TEST_TEXTS = [
    "rāma + iti " * 100,  # Repeated pattern
    "a" * 10000,  # Very long single word
    " ".join(["word"] * 1000),  # Many short words
    "\n".join(COMPLEX_SENTENCES * 10),  # Many lines
]
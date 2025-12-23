# Chroma Generative Features

## Overview

Chroma now includes built-in support for **Retrieval-Augmented Generation (RAG)**, combining the power of vector search with large language models (LLMs) to generate accurate, context-aware responses based on your document collections.

## Features

### 🤖 **Generative Functions**

Support for multiple LLM providers:
- **OpenAI** (GPT-3.5, GPT-4, GPT-4o)
- **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus, Claude 2)
- **Extensible protocol** for custom providers

### 🧠 **RAG Algorithms**

Multiple strategies for combining retrieval with generation:

1. **Simple RAG** - Straightforward top-k document retrieval
2. **Reranking RAG** - Advanced relevance scoring with cross-encoders
3. **Hybrid RAG** - Combines multiple retrieval strategies (dense + sparse)
4. **Adaptive RAG** - Dynamically adjusts based on query complexity

### ⚙️ **Advanced Features**

- Context compression and optimization
- Configurable relevance thresholds
- Batch generation support
- Custom system prompts
- Metadata filtering in retrieval
- Token limit management

---

## Quick Start

### Installation

```bash
pip install chromadb openai anthropic
```

### Basic Usage

```python
import chromadb
from chromadb.utils.generative_functions import OpenAIGenerativeFunction

# Initialize Chroma
client = chromadb.Client()
collection = client.create_collection(name="my_docs")

# Add documents
collection.add(
    documents=[
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "NLP helps computers understand text.",
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Create generative function
gen_fn = OpenAIGenerativeFunction(
    model_name="gpt-4",
    temperature=0.7
)

# Generate with RAG
result = collection.generate(
    query_text="What is machine learning?",
    generative_function=gen_fn,
    n_results=2
)

print(result['response'])
```

---

## API Reference

### GenerativeFunction Protocol

All generative functions implement this protocol:

```python
class GenerativeFunction(Protocol):
    def __call__(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response given prompt and context."""
        ...

    def generate_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        ...
```

### OpenAIGenerativeFunction

```python
from chromadb.utils.generative_functions import OpenAIGenerativeFunction

gen_fn = OpenAIGenerativeFunction(
    api_key=None,                    # Optional, uses OPENAI_API_KEY env var
    model_name="gpt-4",              # Model to use
    temperature=0.7,                  # 0-2, higher = more creative
    max_tokens=500,                   # Maximum response length
    system_prompt=None,               # Custom system prompt
    api_key_env_var="OPENAI_API_KEY", # Environment variable for API key
)
```

**Supported Models:**
- `gpt-4`, `gpt-4-turbo`, `gpt-4o`
- `gpt-3.5-turbo`

### AnthropicGenerativeFunction

```python
from chromadb.utils.generative_functions import AnthropicGenerativeFunction

gen_fn = AnthropicGenerativeFunction(
    api_key=None,                        # Optional, uses ANTHROPIC_API_KEY
    model_name="claude-3-5-sonnet-20241022",  # Claude model
    temperature=0.7,                      # 0-1, higher = more creative
    max_tokens=1024,                      # Maximum response length
    system_prompt=None,                   # Custom system prompt
    api_key_env_var="ANTHROPIC_API_KEY",  # Environment variable
)
```

**Supported Models:**
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-2.1`, `claude-2.0`

### Collection.generate()

```python
result = collection.generate(
    query_text: str,                      # Required: The query/question
    generative_function: GenerativeFunction, # Required: LLM to use
    n_results: int = 5,                   # Number of documents to retrieve
    where: Optional[Where] = None,        # Metadata filter
    where_document: Optional[WhereDocument] = None,  # Document filter
    rag_algorithm: str = "simple",        # RAG strategy
    **generation_kwargs                    # Additional parameters
)
```

**Returns:**
```python
{
    'response': str,                      # Generated text
    'source_documents': List[str],        # Documents used as context
    'source_metadatas': List[dict],       # Metadata of source docs
    'source_distances': List[float],      # Relevance scores
}
```

**RAG Algorithms:**
- `"simple"` - Basic top-k retrieval
- `"reranking"` - Advanced relevance scoring
- `"hybrid"` - Multi-strategy fusion
- `"adaptive"` - Query-complexity-aware

**Generation Kwargs:**
- `temperature` - Override default temperature
- `max_tokens` - Override max response length
- `relevance_threshold` - Minimum relevance score (0-1)
- `max_context_length` - Maximum context characters

---

## Examples

### Example 1: Basic Q&A

```python
collection = client.create_collection(name="knowledge_base")

collection.add(
    documents=["Python is a programming language.", "It is used for AI."],
    ids=["1", "2"]
)

gen_fn = OpenAIGenerativeFunction(model_name="gpt-4")

result = collection.generate(
    query_text="What is Python used for?",
    generative_function=gen_fn
)

print(result['response'])
# Output: "Based on the provided context, Python is used for AI..."
```

### Example 2: With Metadata Filtering

```python
collection.add(
    documents=[
        "Machine learning algorithms learn from data.",
        "Deep learning is a type of machine learning.",
        "Computer vision processes images.",
    ],
    ids=["1", "2", "3"],
    metadatas=[
        {"category": "ML"},
        {"category": "DL"},
        {"category": "CV"}
    ]
)

result = collection.generate(
    query_text="Tell me about deep learning",
    generative_function=gen_fn,
    where={"category": {"$in": ["ML", "DL"]}}  # Filter to ML/DL docs only
)
```

### Example 3: Adaptive RAG

```python
# Simple query - uses fewer documents
result1 = collection.generate(
    query_text="What is AI?",
    generative_function=gen_fn,
    rag_algorithm="adaptive"  # Adapts to query complexity
)

# Complex query - uses more documents
result2 = collection.generate(
    query_text="Compare and contrast supervised and unsupervised learning approaches",
    generative_function=gen_fn,
    rag_algorithm="adaptive"
)
```

### Example 4: Custom System Prompt

```python
gen_fn = OpenAIGenerativeFunction(
    model_name="gpt-4",
    system_prompt="You are a helpful coding assistant. Provide code examples."
)

result = collection.generate(
    query_text="How do I use list comprehension in Python?",
    generative_function=gen_fn
)
```

### Example 5: Using Anthropic Claude

```python
from chromadb.utils.generative_functions import AnthropicGenerativeFunction

claude = AnthropicGenerativeFunction(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=2000
)

result = collection.generate(
    query_text="Explain quantum computing in simple terms",
    generative_function=claude,
    n_results=3
)
```

### Example 6: Direct RAG Algorithm Usage

```python
from chromadb.utils.generative_functions.rag_algorithms import (
    create_rag_algorithm,
    RAGConfig
)

# Create custom RAG configuration
config = RAGConfig(
    top_k=5,
    relevance_threshold=0.3,
    max_context_length=3000
)

rag = create_rag_algorithm("adaptive", config)

# Use with your own documents
context = rag.prepare_context(
    documents=["doc1", "doc2", "doc3"],
    distances=[0.1, 0.2, 0.5],
    query="my question"
)
```

---

## RAG Algorithms in Detail

### Simple RAG

The most straightforward approach - retrieves top-k most relevant documents.

```python
result = collection.generate(
    query_text="What is Python?",
    generative_function=gen_fn,
    rag_algorithm="simple",
    n_results=3
)
```

**When to use:**
- Quick prototyping
- Simple Q&A scenarios
- When you have high-quality documents

### Adaptive RAG

Automatically adjusts retrieval based on query complexity.

```python
result = collection.generate(
    query_text="Complex multi-part question...",
    generative_function=gen_fn,
    rag_algorithm="adaptive"
)
```

**When to use:**
- Varying query complexity
- Production systems with diverse user queries
- When you want automatic optimization

### Hybrid RAG

Combines multiple retrieval strategies (coming soon).

**When to use:**
- When you need both semantic and keyword matching
- Complex domain-specific applications
- Maximum accuracy requirements

### Reranking RAG

Uses advanced models to rerank retrieved documents.

**When to use:**
- High-precision requirements
- When initial retrieval needs refinement
- Long-form document collections

---

## Best Practices

### 1. Choose the Right Model

- **GPT-3.5-turbo**: Fast, cost-effective for simple queries
- **GPT-4**: Best quality for complex reasoning
- **Claude 3.5 Sonnet**: Excellent for long-context tasks

### 2. Optimize Context Length

```python
result = collection.generate(
    query_text=query,
    generative_function=gen_fn,
    n_results=5,
    max_context_length=2000  # Prevent token limit issues
)
```

### 3. Use Metadata Filtering

```python
result = collection.generate(
    query_text="Recent developments in AI",
    generative_function=gen_fn,
    where={"year": {"$gte": 2023}}  # Only recent documents
)
```

### 4. Set Appropriate Temperature

```python
# Factual answers - low temperature
result = collection.generate(
    query_text="What is the speed of light?",
    generative_function=gen_fn,
    temperature=0.1
)

# Creative answers - higher temperature
result = collection.generate(
    query_text="Write a story about AI",
    generative_function=gen_fn,
    temperature=0.9
)
```

### 5. Handle API Keys Securely

```bash
# Use environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
# Don't hardcode API keys
gen_fn = OpenAIGenerativeFunction()  # Uses env var
```

---

## Advanced Usage

### Custom Generative Function

Implement your own provider:

```python
from chromadb.api.types import GenerativeFunction
from typing import List, Optional, Any, Dict

class CustomGenerativeFunction(GenerativeFunction):
    def __init__(self, **kwargs):
        # Initialize your model
        pass

    def __call__(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        # Your generation logic
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt

        # Call your model
        response = your_model.generate(full_prompt, **kwargs)
        return response

    @staticmethod
    def name() -> str:
        return "custom"

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "custom"}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "CustomGenerativeFunction":
        return CustomGenerativeFunction(**config)
```

### Context Compression

```python
from chromadb.utils.generative_functions.rag_algorithms import ContextCompressor

compressor = ContextCompressor(max_length=2000)

compressed_docs = compressor.compress(
    documents=long_documents,
    strategy="extract"  # or "truncate"
)
```

---

## Troubleshooting

### Issue: API Key Errors

```
ValueError: The OPENAI_API_KEY environment variable is not set.
```

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Issue: Token Limit Exceeded

```
Error: This model's maximum context length is...
```

**Solution:**
```python
result = collection.generate(
    query_text=query,
    generative_function=gen_fn,
    n_results=3,  # Reduce number of documents
    max_context_length=2000,  # Limit context size
    max_tokens=500  # Limit response length
)
```

### Issue: Poor Response Quality

**Solutions:**
1. Increase `n_results` to provide more context
2. Try `rag_algorithm="adaptive"`
3. Use a more powerful model (GPT-4 vs GPT-3.5)
4. Add a custom `system_prompt`
5. Adjust `temperature` (lower for factual, higher for creative)

---

## Performance Optimization

### Batch Processing

```python
queries = ["Query 1", "Query 2", "Query 3"]

results = []
for query in queries:
    result = collection.generate(
        query_text=query,
        generative_function=gen_fn
    )
    results.append(result)
```

### Caching

Consider caching generated responses for common queries:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(query: str) -> str:
    result = collection.generate(
        query_text=query,
        generative_function=gen_fn
    )
    return result['response']
```

---

## License

This feature is part of Chroma and follows the same Apache 2.0 license.

## Contributing

Contributions welcome! To add new generative providers:

1. Implement the `GenerativeFunction` protocol
2. Add to `chromadb/utils/generative_functions/`
3. Register in `__init__.py`
4. Add tests and examples

## Support

- **Documentation**: https://docs.trychroma.com
- **GitHub Issues**: https://github.com/chroma-core/chroma/issues
- **Discord**: https://discord.gg/chroma

---

**Happy generating! 🚀**

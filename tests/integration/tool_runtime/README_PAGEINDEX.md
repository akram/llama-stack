# PageIndex RAG Tests

This directory contains tests for the PageIndex RAG (vectorless, reasoning-based retrieval) implementation.

## Test Files

### 1. Unit Tests (`tests/unit/rag/test_pageindex_rag.py`)

Unit tests that verify core PageIndex functionality with mocked dependencies:

- **Query Tests**: Verify PageIndex can handle queries, empty results, and multiple stores
- **Metadata Tests**: Ensure tree structure metadata is preserved
- **Configuration Tests**: Verify RAGQueryConfig parameters are respected
- **Insert Tests**: Test document insertion and tree building

**Run unit tests:**
```bash
pytest tests/unit/rag/test_pageindex_rag.py -v
```

### 2. Integration Tests (`tests/integration/tool_runtime/test_pageindex_rag.py`)

Full end-to-end tests using the Responses API with real documents:

- **Basic Query**: Simple PageIndex queries
- **Hierarchical Retrieval**: Testing tree structure navigation
- **Multi-Document Query**: Reasoning across multiple documents
- **Streaming**: PageIndex with streaming responses
- **Various Queries**: Parameterized tests with different query types

**Run integration tests:**
```bash
# Run all integration tests
pytest tests/integration/tool_runtime/test_pageindex_rag.py -v

# Run specific test
pytest tests/integration/tool_runtime/test_pageindex_rag.py::test_pageindex_basic_query -v

# Run with a specific provider
pytest tests/integration/tool_runtime/test_pageindex_rag.py --provider=inline -v
```

### 3. Standalone Test Script (`tests/test_pageindex_standalone.py`)

A self-contained test script that can be run directly without pytest:

**Run standalone test:**
```bash
# Use default models
python tests/test_pageindex_standalone.py

# Use custom models
PAGEINDEX_TREE_BUILDER_MODEL="llama3.2:3b" \
PAGEINDEX_REASONING_MODEL="llama3.2:3b" \
python tests/test_pageindex_standalone.py
```

This script:
- Creates a PageIndex RAG implementation
- Inserts test documents and builds trees
- Performs multiple test queries
- Reports detailed results and term matching

## What PageIndex Tests Verify

### Core Capabilities

1. **Vectorless Retrieval**: No embeddings needed, uses reasoning instead
2. **Tree Structure**: Hierarchical document organization
3. **Semantic Understanding**: Finds relevant content through reasoning
4. **Multi-Document**: Queries across multiple documents
5. **Section Navigation**: Retrieves specific document sections

### PageIndex vs Vector RAG

PageIndex differs from traditional vector RAG:

| Feature | Vector RAG | PageIndex RAG |
|---------|-----------|---------------|
| Embeddings | Required | Not needed |
| Retrieval | Similarity search | Reasoning-based |
| Structure | Flat chunks | Hierarchical tree |
| Setup | Embedding model needed | Uses LLM directly |
| Best for | Semantic similarity | Conceptual queries |

## Environment Variables

Configure PageIndex models via environment variables:

```bash
# Tree building model (used during document insertion)
export PAGEINDEX_TREE_BUILDER_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Query reasoning model (used during queries)
export PAGEINDEX_REASONING_MODEL="meta-llama/Llama-3.1-8B-Instruct"
```

## Test Data

The tests use machine learning documentation as test data:
- Machine Learning introduction (supervised, unsupervised, reinforcement learning)
- Neural Networks architecture and training
- Data Preprocessing techniques

This allows testing:
- Specific fact retrieval ("What are the types of machine learning?")
- Conceptual queries ("How does training work?")
- Multi-document reasoning ("How do CNNs relate to ML?")
- Section-specific queries ("What normalization techniques exist?")

## Expected Behavior

### Successful Test Run

```
✓ Query successful
✓ Retrieved 3 chunks
✓ Average relevance score: 0.85
✓ Found terms: supervised, unsupervised, reinforcement
```

### Common Issues

1. **No Results**: If queries return no results, check:
   - PageIndexEngine implementation is complete
   - Models are accessible and working
   - Documents were successfully inserted

2. **Low Relevance Scores**: May indicate:
   - Query doesn't match document content well
   - Tree structure needs adjustment
   - Reasoning model needs tuning

3. **Timeout During Tree Building**: Large documents may need:
   - Increase `max_wait` timeout
   - Reduce `max_tree_depth` in config
   - Use smaller/faster model

## Debugging

Enable detailed logging:

```bash
export LLAMA_STACK_LOG_LEVEL=DEBUG
pytest tests/integration/tool_runtime/test_pageindex_rag.py -v -s
```

Check PageIndex-specific logs:
```bash
pytest tests/integration/tool_runtime/test_pageindex_rag.py -v -s 2>&1 | grep "pageindex"
```

## Contributing

When adding new PageIndex tests:

1. **Unit tests** for isolated functionality
2. **Integration tests** for end-to-end scenarios
3. **Update standalone script** if adding core features
4. **Document expected behavior** in docstrings

## Related Documentation

- [PageIndex RAG Provider Documentation](../../../docs/providers/pageindex_rag.md)
- [RAG API Documentation](../../../docs/api/rag.md)
- [Tool Runtime Documentation](../../../docs/api/tool_runtime.md)

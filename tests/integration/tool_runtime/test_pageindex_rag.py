# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for PageIndex RAG (vectorless reasoning-based retrieval).

These tests verify that PageIndex can:
1. Build hierarchical tree structures from documents
2. Perform reasoning-based retrieval without embeddings
3. Handle complex queries across multiple documents
4. Work with the Responses API for file search

To run these tests with a specific provider:
    pytest tests/integration/tool_runtime/test_pageindex_rag.py --provider=inline
"""

import time

import pytest


@pytest.fixture
def pageindex_store(responses_client, tmp_path_factory):
    """Create a PageIndex store with test documents."""
    # Note: PageIndex stores are created similar to vector stores
    # but use a different provider (pageindex_rag instead of rag)
    
    store_name = "test_pageindex_store"
    
    # Clean up any existing store with the same name
    try:
        stores = responses_client.vector_stores.list()
        for store in stores:
            if store.name == store_name:
                responses_client.vector_stores.delete(vector_store_id=store.id)
    except Exception:
        pass  # Store might not exist yet
    
    # Create PageIndex store (uses same API as vector stores)
    store = responses_client.vector_stores.create(
        name=store_name,
        # PageIndex doesn't need embedding model/dimension
        extra_body={"provider": "pageindex_rag"},
    )
    
    # Create test documents
    tmp_path = tmp_path_factory.mktemp("pageindex_test_docs")
    
    documents = [
        {
            "name": "machine_learning_intro.txt",
            "content": """Machine Learning Introduction
            
Machine learning is a branch of artificial intelligence that focuses on building systems 
that can learn from data. There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data and makes predictions.
   Examples include classification and regression tasks.

2. Unsupervised Learning: The algorithm finds patterns in data without labeled outputs.
   Examples include clustering and dimensionality reduction.

3. Reinforcement Learning: The algorithm learns by interacting with an environment and 
   receiving rewards or penalties. It's commonly used in robotics and game playing.

Deep learning is a subset of machine learning that uses neural networks with multiple layers.
It has achieved remarkable success in computer vision, natural language processing, and 
speech recognition.
""",
        },
        {
            "name": "neural_networks.txt",
            "content": """Neural Networks Deep Dive

Neural networks are computing systems inspired by biological neural networks. 
They consist of layers of interconnected nodes (neurons).

Architecture:
- Input Layer: Receives the initial data
- Hidden Layers: Process the data through weighted connections
- Output Layer: Produces the final predictions

Training Process:
1. Forward Propagation: Data flows through the network to generate predictions
2. Loss Calculation: Compare predictions with actual values
3. Backpropagation: Calculate gradients and update weights
4. Iteration: Repeat until convergence

Popular architectures include:
- Convolutional Neural Networks (CNNs) for image processing
- Recurrent Neural Networks (RNNs) for sequential data
- Transformers for natural language understanding
""",
        },
        {
            "name": "data_preprocessing.txt",
            "content": """Data Preprocessing Techniques

Data preprocessing is crucial for machine learning success. Key techniques include:

Data Cleaning:
- Handling missing values (imputation, deletion)
- Removing duplicates
- Fixing inconsistent data

Feature Engineering:
- Creating new features from existing ones
- Feature selection to reduce dimensionality
- Encoding categorical variables (one-hot, label encoding)

Normalization and Scaling:
- Min-Max Scaling: Scale features to a fixed range [0, 1]
- Standardization: Transform features to have mean=0 and std=1
- Robust Scaling: Use median and IQR for outlier resistance

Data Augmentation:
- Increase training data size artificially
- Common in computer vision (rotation, flipping, cropping)
- Also useful for text data (synonym replacement, back-translation)
""",
        },
    ]
    
    file_ids = []
    for doc in documents:
        # Write file
        file_path = tmp_path / doc["name"]
        file_path.write_text(doc["content"])
        
        # Upload file
        with open(file_path, "rb") as f:
            file_response = responses_client.files.create(file=f, purpose="assistants")
        file_ids.append(file_response.id)
        
        # Attach to PageIndex store
        file_attach_response = responses_client.vector_stores.files.create(
            vector_store_id=store.id,
            file_id=file_response.id,
        )
        
        # Wait for PageIndex tree building to complete
        max_wait = 30  # seconds
        start_time = time.time()
        while file_attach_response.status == "in_progress":
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"PageIndex tree building timed out for {doc['name']}")
            time.sleep(0.5)
            file_attach_response = responses_client.vector_stores.files.retrieve(
                vector_store_id=store.id,
                file_id=file_response.id,
            )
        
        assert file_attach_response.status == "completed", (
            f"PageIndex tree building failed for {doc['name']}: {file_attach_response.last_error}"
        )
    
    yield store
    
    # Cleanup
    try:
        responses_client.vector_stores.delete(vector_store_id=store.id)
        for file_id in file_ids:
            try:
                responses_client.files.delete(file_id=file_id)
            except Exception:
                pass
    except Exception:
        pass


def test_pageindex_basic_query(responses_client, text_model_id, pageindex_store):
    """Test basic PageIndex query without embeddings."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    response = responses_client.responses.create(
        model=text_model_id,
        input="What are the main types of machine learning?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    # Verify file search was performed
    assert len(response.output) > 1
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    
    # Verify relevant content was retrieved
    results_text = " ".join([r.text for r in response.output[0].results]).lower()
    assert "supervised" in results_text or "unsupervised" in results_text or "reinforcement" in results_text
    
    # Verify final response mentions the types
    output_text = response.output_text.lower()
    assert "supervised" in output_text or "unsupervised" in output_text or "learning" in output_text


def test_pageindex_hierarchical_retrieval(responses_client, text_model_id, pageindex_store):
    """Test that PageIndex can retrieve from document hierarchies."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    response = responses_client.responses.create(
        model=text_model_id,
        input="Explain the neural network training process",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    
    # PageIndex should retrieve the section about training process
    results_text = " ".join([r.text for r in response.output[0].results]).lower()
    assert "backpropagation" in results_text or "forward propagation" in results_text or "training" in results_text


def test_pageindex_multi_document_query(responses_client, text_model_id, pageindex_store):
    """Test PageIndex reasoning across multiple documents."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    # Query that requires information from multiple documents
    response = responses_client.responses.create(
        model=text_model_id,
        input="How do CNNs relate to deep learning and machine learning?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    
    # Should retrieve information from both ML intro and neural networks docs
    results_text = " ".join([r.text for r in response.output[0].results]).lower()
    assert "neural" in results_text or "cnn" in results_text or "convolutional" in results_text


def test_pageindex_specific_section_query(responses_client, text_model_id, pageindex_store):
    """Test PageIndex retrieval of specific document sections."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    response = responses_client.responses.create(
        model=text_model_id,
        input="What normalization techniques are available for preprocessing?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    
    # Should retrieve the normalization section from preprocessing doc
    results_text = " ".join([r.text for r in response.output[0].results]).lower()
    assert any(term in results_text for term in ["min-max", "standardization", "scaling", "normalization"])


def test_pageindex_streaming(responses_client, text_model_id, pageindex_store):
    """Test PageIndex with streaming responses."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    stream = responses_client.responses.create(
        model=text_model_id,
        input="What is data augmentation?",
        tools=tools,
        stream=True,
    )
    
    chunks = list(stream)
    event_types = [chunk.type for chunk in chunks]
    
    # Verify streaming events
    assert "response.file_search_call.in_progress" in event_types
    assert "response.file_search_call.completed" in event_types
    
    # Verify final response
    final_chunk = chunks[-1]
    if hasattr(final_chunk, "response"):
        file_search_calls = [output for output in final_chunk.response.output if output.type == "file_search_call"]
        assert len(file_search_calls) > 0
        assert file_search_calls[0].status == "completed"


def test_pageindex_vs_vector_comparison(responses_client, text_model_id, pageindex_store):
    """
    Test that PageIndex can handle queries that would be difficult for pure vector search.
    
    PageIndex uses reasoning to understand document structure and relationships,
    which can be more effective for conceptual queries than vector similarity alone.
    """
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    # Complex reasoning query that benefits from tree structure
    response = responses_client.responses.create(
        model=text_model_id,
        input="What steps should I follow to prepare data before training a neural network?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    
    # Should retrieve relevant preprocessing and training information
    results_text = " ".join([r.text for r in response.output[0].results]).lower()
    assert any(term in results_text for term in ["preprocessing", "cleaning", "scaling", "normalization", "feature"])


def test_pageindex_empty_query_handling(responses_client, text_model_id, pageindex_store):
    """Test PageIndex behavior with queries that have no relevant results."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    # Query about something completely unrelated to the documents
    response = responses_client.responses.create(
        model=text_model_id,
        input="What is the recipe for chocolate cake?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    # Should still complete successfully, even if results are less relevant
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    # May or may not return results depending on PageIndex's reasoning


@pytest.mark.parametrize(
    "query,expected_terms",
    [
        ("What is supervised learning?", ["supervised", "labeled", "classification", "regression"]),
        ("Explain transformers", ["transformer", "natural language", "architecture"]),
        ("How to handle missing values?", ["missing", "imputation", "deletion", "cleaning"]),
        ("What are hidden layers?", ["hidden", "layer", "neural", "network"]),
    ],
)
def test_pageindex_various_queries(responses_client, text_model_id, pageindex_store, query, expected_terms):
    """Test PageIndex with various types of queries."""
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [pageindex_store.id],
        }
    ]
    
    response = responses_client.responses.create(
        model=text_model_id,
        input=query,
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )
    
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    
    if response.output[0].results:
        results_text = " ".join([r.text for r in response.output[0].results]).lower()
        # At least one expected term should be present
        assert any(term.lower() in results_text for term in expected_terms), (
            f"None of the expected terms {expected_terms} found in results for query: {query}"
        )

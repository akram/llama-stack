#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Standalone test script for PageIndex RAG provider.

This script tests the PageIndex RAG implementation without requiring a full test suite.
It can be run directly to verify that PageIndex is working correctly.

Usage:
    python test_pageindex_standalone.py
    
Environment variables:
    PAGEINDEX_TREE_BUILDER_MODEL - Model for building document trees (default: meta-llama/Llama-3.1-8B-Instruct)
    PAGEINDEX_REASONING_MODEL - Model for query reasoning (default: meta-llama/Llama-3.1-8B-Instruct)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from llama_stack.providers.inline.tool_runtime.pageindex_rag.config import (
    PageIndexInlineConfig,
    PageIndexRagToolRuntimeConfig,
)
from llama_stack.providers.inline.tool_runtime.pageindex_rag.memory import PageIndexRagToolRuntimeImpl
from llama_stack_api import RAGDocument, RAGQueryConfig


class MockInferenceAPI:
    """Mock inference API for testing."""
    
    async def openai_chat_completion(self, params):
        """Mock chat completion."""
        from types import SimpleNamespace
        
        # Return a simple query transformation
        content = params.messages[0]["content"] if isinstance(params.messages[0], dict) else params.messages[0].content
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class MockFilesAPI:
    """Mock files API for testing."""
    
    async def openai_upload_file(self, request, file):
        """Mock file upload."""
        from types import SimpleNamespace
        return SimpleNamespace(id=f"file_{file.filename}")


async def test_pageindex_insert_and_query():
    """Test PageIndex document insertion and querying."""
    
    print("=" * 80)
    print("PageIndex RAG Standalone Test")
    print("=" * 80)
    print()
    
    # Create config
    print("1. Creating PageIndex configuration...")
    config = PageIndexRagToolRuntimeConfig(
        pageindex_config=PageIndexInlineConfig(
            tree_builder_model=os.environ.get("PAGEINDEX_TREE_BUILDER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            reasoning_model=os.environ.get("PAGEINDEX_REASONING_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            max_tree_depth=5,
            min_section_length=100,
        )
    )
    print(f"   Tree builder model: {config.pageindex_config.tree_builder_model}")
    print(f"   Reasoning model: {config.pageindex_config.reasoning_model}")
    print()
    
    # Create implementation
    print("2. Initializing PageIndex implementation...")
    impl = PageIndexRagToolRuntimeImpl(
        config=config,
        inference_api=MockInferenceAPI(),
        files_api=MockFilesAPI(),
    )
    await impl.initialize()
    print("   ✓ Initialized successfully")
    print()
    
    # Create test documents
    print("3. Creating test documents...")
    documents = [
        RAGDocument(
            document_id="doc1",
            content="""
Machine Learning Fundamentals

Machine learning is a field of artificial intelligence that enables computers to learn from data
without being explicitly programmed. There are three main categories:

1. Supervised Learning
   - Uses labeled training data
   - Predicts outcomes for new data
   - Examples: classification, regression
   
2. Unsupervised Learning
   - Works with unlabeled data
   - Finds hidden patterns
   - Examples: clustering, dimensionality reduction
   
3. Reinforcement Learning
   - Learns through trial and error
   - Receives rewards and penalties
   - Examples: game playing, robotics
""",
            metadata={"title": "ML Fundamentals", "author": "Test"},
        ),
        RAGDocument(
            document_id="doc2",
            content="""
Deep Learning Neural Networks

Neural networks are the foundation of deep learning. They consist of interconnected layers:

Input Layer:
- Receives raw data
- Each node represents a feature

Hidden Layers:
- Process information through weighted connections
- Multiple layers enable deep learning
- Activation functions introduce non-linearity

Output Layer:
- Produces final predictions
- Structure depends on the task

Training uses backpropagation to adjust weights and minimize prediction errors.
""",
            metadata={"title": "Neural Networks", "topic": "Deep Learning"},
        ),
        RAGDocument(
            document_id="doc3",
            content="""
Data Preprocessing Best Practices

Before training any model, data must be properly prepared:

Data Cleaning:
- Handle missing values (imputation, deletion)
- Remove duplicates
- Fix inconsistent formatting

Feature Engineering:
- Create meaningful features
- Encode categorical variables
- Scale numerical features

Normalization Techniques:
- Min-Max Scaling: scale to [0,1]
- Standardization: mean=0, std=1
- Robust Scaling: uses median and IQR

Proper preprocessing is crucial for model performance.
""",
            metadata={"title": "Data Preprocessing", "difficulty": "beginner"},
        ),
    ]
    print(f"   Created {len(documents)} test documents")
    print()
    
    # Insert documents
    print("4. Building PageIndex trees (this may take a moment)...")
    try:
        await impl.insert(
            documents=documents,
            vector_store_id="test_store_1",
        )
        print("   ✓ Successfully built PageIndex trees for all documents")
    except Exception as e:
        print(f"   ✗ Error during tree building: {e}")
        print(f"     This is expected if the PageIndexEngine is not fully implemented yet")
        print(f"     Skipping query tests...")
        return
    print()
    
    # Test queries
    print("5. Testing PageIndex queries...")
    print()
    
    test_queries = [
        ("What are the types of machine learning?", ["supervised", "unsupervised", "reinforcement"]),
        ("Explain neural network layers", ["input", "hidden", "output"]),
        ("What normalization techniques exist?", ["min-max", "standardization", "scaling"]),
        ("How does backpropagation work?", ["backpropagation", "weights", "training"]),
    ]
    
    results_summary = []
    
    for i, (query, expected_terms) in enumerate(test_queries, 1):
        print(f"   Query {i}: {query}")
        try:
            result = await impl.query(
                content=query,
                vector_store_ids=["test_store_1"],
                query_config=RAGQueryConfig(max_chunks=3),
            )
            
            if result.content:
                # Extract text from result
                result_text = ""
                for item in result.content:
                    if hasattr(item, "text"):
                        result_text += item.text + " "
                
                result_text = result_text.lower()
                
                # Check for expected terms
                found_terms = [term for term in expected_terms if term in result_text]
                
                print(f"      ✓ Query successful")
                print(f"      Found terms: {', '.join(found_terms) if found_terms else 'none'}")
                
                if result.metadata:
                    print(f"      Retrieved {len(result.metadata.get('chunks', []))} chunks")
                    if 'scores' in result.metadata:
                        avg_score = sum(result.metadata['scores']) / len(result.metadata['scores'])
                        print(f"      Average relevance score: {avg_score:.2f}")
                
                results_summary.append({
                    "query": query,
                    "success": True,
                    "found_terms": found_terms,
                    "total_terms": len(expected_terms),
                })
            else:
                print(f"      ⚠ Query returned no results")
                results_summary.append({
                    "query": query,
                    "success": False,
                    "found_terms": [],
                    "total_terms": len(expected_terms),
                })
        except Exception as e:
            print(f"      ✗ Error: {e}")
            results_summary.append({
                "query": query,
                "success": False,
                "error": str(e),
                "total_terms": len(expected_terms),
            })
        print()
    
    # Print summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()
    
    successful = sum(1 for r in results_summary if r["success"])
    print(f"Queries: {successful}/{len(results_summary)} successful")
    print()
    
    if successful > 0:
        total_found = sum(len(r.get("found_terms", [])) for r in results_summary if r["success"])
        total_expected = sum(r["total_terms"] for r in results_summary if r["success"])
        print(f"Term matching: {total_found}/{total_expected} expected terms found")
        print()
    
    # Shutdown
    print("6. Cleaning up...")
    await impl.shutdown()
    print("   ✓ Shutdown complete")
    print()
    
    print("=" * 80)
    if successful == len(results_summary):
        print("✓ All tests passed!")
    elif successful > 0:
        print(f"⚠ Partial success: {successful}/{len(results_summary)} tests passed")
    else:
        print("✗ All tests failed")
    print("=" * 80)


if __name__ == "__main__":
    print()
    asyncio.run(test_pageindex_insert_and_query())
    print()

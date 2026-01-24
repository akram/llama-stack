# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.providers.inline.tool_runtime.pageindex_rag.config import (
    PageIndexInlineConfig,
    PageIndexRagToolRuntimeConfig,
)
from llama_stack.providers.inline.tool_runtime.pageindex_rag.memory import PageIndexRagToolRuntimeImpl
from llama_stack_api import ChunkMetadata, EmbeddedChunk, QueryChunksResponse, RAGQueryConfig


class TestPageIndexRagQuery:
    """Unit tests for PageIndex RAG query functionality."""

    async def test_query_raises_on_empty_vector_store_ids(self):
        """Test that query raises ValueError when no store IDs are provided."""
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )
        
        with pytest.raises(ValueError, match="No PageIndex stores were provided"):
            await rag_tool.query(content="test query", vector_store_ids=[])

    async def test_query_chunk_metadata_handling(self):
        """Test that PageIndex query properly handles chunk metadata."""
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )
        
        content = "test query content"
        store_ids = ["pageindex_store_1"]

        # Mock PageIndex query results
        pageindex_results = [
            {
                "node_id": "node_1",
                "content": "This is test content from PageIndex tree node",
                "document_id": "doc1",
                "relevance_score": 0.95,
                "title": "Test Document",
                "start_index": 0,
                "end_index": 100,
                "source": "test_source",
            }
        ]

        # Mock the PageIndexEngine query method
        with patch.object(rag_tool.pageindex_engine, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = pageindex_results
            
            # Mock generate_rag_query
            with patch("llama_stack.providers.inline.tool_runtime.pageindex_rag.memory.generate_rag_query") as mock_gen:
                mock_gen.return_value = "processed test query"
                
                result = await rag_tool.query(content=content, vector_store_ids=store_ids)

        assert result is not None
        assert result.content is not None
        assert len(result.content) > 0
        
        # Check metadata
        assert result.metadata is not None
        assert "document_ids" in result.metadata
        assert "chunks" in result.metadata
        assert "scores" in result.metadata
        assert "store_ids" in result.metadata
        
        # Verify PageIndex-specific metadata is preserved
        assert result.metadata["document_ids"] == ["doc1"]
        assert result.metadata["scores"][0] == 0.95

    async def test_query_multiple_stores_parallel(self):
        """Test that PageIndex queries multiple stores in parallel."""
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )
        
        store_ids = ["store_1", "store_2", "store_3"]
        
        # Mock results from different stores
        pageindex_results = [
            {
                "node_id": f"node_{i}",
                "content": f"Content from store {i}",
                "document_id": f"doc{i}",
                "relevance_score": 0.9 - (i * 0.1),
                "title": f"Document {i}",
            }
            for i in range(len(store_ids))
        ]

        with patch.object(rag_tool.pageindex_engine, "query", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = [[result] for result in pageindex_results]
            
            with patch("llama_stack.providers.inline.tool_runtime.pageindex_rag.memory.generate_rag_query") as mock_gen:
                mock_gen.return_value = "test query"
                
                result = await rag_tool.query(
                    content="test query",
                    vector_store_ids=store_ids,
                )

        # Verify all stores were queried
        assert mock_query.call_count == len(store_ids)
        
        # Verify results from all stores are merged and sorted by score
        assert len(result.metadata["store_ids"]) == len(store_ids)
        assert result.metadata["scores"][0] >= result.metadata["scores"][-1]

    async def test_query_config_parameters(self):
        """Test that RAGQueryConfig parameters are properly used."""
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )
        
        query_config = RAGQueryConfig(
            max_chunks=3,
            max_tokens_in_context=2048,
        )
        
        # Create more chunks than max_chunks to test limiting
        pageindex_results = [
            {
                "node_id": f"node_{i}",
                "content": f"Content {i}",
                "document_id": f"doc{i}",
                "relevance_score": 1.0 - (i * 0.1),
                "title": f"Doc {i}",
            }
            for i in range(10)
        ]

        with patch.object(rag_tool.pageindex_engine, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = pageindex_results
            
            with patch("llama_stack.providers.inline.tool_runtime.pageindex_rag.memory.generate_rag_query") as mock_gen:
                mock_gen.return_value = "test query"
                
                result = await rag_tool.query(
                    content="test query",
                    vector_store_ids=["store_1"],
                    query_config=query_config,
                )

        # Verify only max_chunks are returned
        assert len(result.metadata["chunks"]) <= query_config.max_chunks

    async def test_query_empty_results(self):
        """Test handling of empty query results."""
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )

        with patch.object(rag_tool.pageindex_engine, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []
            
            with patch("llama_stack.providers.inline.tool_runtime.pageindex_rag.memory.generate_rag_query") as mock_gen:
                mock_gen.return_value = "test query"
                
                result = await rag_tool.query(
                    content="test query",
                    vector_store_ids=["store_1"],
                )

        # Should return empty result but not crash
        assert result is not None
        assert result.content is None

    async def test_query_with_tree_structure_metadata(self):
        """Test that PageIndex tree structure metadata is preserved."""
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )
        
        # Simulate hierarchical tree node results
        pageindex_results = [
            {
                "node_id": "root_section_1",
                "content": "This is a root section of the document",
                "document_id": "doc1",
                "relevance_score": 0.9,
                "title": "Introduction",
                "start_index": 0,
                "end_index": 500,
                "depth": 1,
                "parent_id": None,
            },
            {
                "node_id": "subsection_1_1",
                "content": "This is a subsection with more details",
                "document_id": "doc1",
                "relevance_score": 0.85,
                "title": "Background",
                "start_index": 100,
                "end_index": 300,
                "depth": 2,
                "parent_id": "root_section_1",
            },
        ]

        with patch.object(rag_tool.pageindex_engine, "query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = pageindex_results
            
            with patch("llama_stack.providers.inline.tool_runtime.pageindex_rag.memory.generate_rag_query") as mock_gen:
                mock_gen.return_value = "test query"
                
                result = await rag_tool.query(
                    content="test query about background",
                    vector_store_ids=["store_1"],
                )

        # Verify hierarchical metadata is preserved
        assert result is not None
        assert len(result.metadata["chunks"]) == 2
        
        # Check that content from both tree levels is included
        content_text = " ".join([item.text for item in result.content if hasattr(item, "text")])
        assert "root section" in content_text.lower()
        assert "subsection" in content_text.lower()


class TestPageIndexRagInsert:
    """Unit tests for PageIndex RAG document insertion."""

    async def test_insert_builds_tree_structure(self):
        """Test that document insertion builds PageIndex tree structure."""
        from llama_stack_api import RAGDocument
        
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        files_api = MagicMock()
        files_api.openai_upload_file = AsyncMock(return_value=MagicMock(id="file_123"))
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=files_api,
        )
        
        document = RAGDocument(
            document_id="doc1",
            content="This is a test document with multiple sections.",
            metadata={"title": "Test Document"},
        )

        with patch.object(rag_tool.pageindex_engine, "build_index", new_callable=AsyncMock) as mock_build:
            await rag_tool.insert(
                documents=[document],
                vector_store_id="store_1",
            )

        # Verify tree building was called
        assert mock_build.called
        assert files_api.openai_upload_file.called

    async def test_insert_multiple_documents(self):
        """Test inserting multiple documents in batch."""
        from llama_stack_api import RAGDocument
        
        config = PageIndexRagToolRuntimeConfig(
            pageindex_config=PageIndexInlineConfig()
        )
        
        files_api = MagicMock()
        files_api.openai_upload_file = AsyncMock(side_effect=[
            MagicMock(id=f"file_{i}") for i in range(3)
        ])
        
        rag_tool = PageIndexRagToolRuntimeImpl(
            config=config,
            inference_api=MagicMock(),
            files_api=files_api,
        )
        
        documents = [
            RAGDocument(
                document_id=f"doc{i}",
                content=f"Document {i} content",
                metadata={"title": f"Doc {i}"},
            )
            for i in range(3)
        ]

        with patch.object(rag_tool.pageindex_engine, "build_index", new_callable=AsyncMock):
            await rag_tool.insert(
                documents=documents,
                vector_store_id="store_1",
            )

        # Verify all documents were uploaded
        assert files_api.openai_upload_file.call_count == len(documents)

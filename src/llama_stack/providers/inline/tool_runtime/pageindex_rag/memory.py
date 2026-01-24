# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
import mimetypes
from typing import Any

import httpx
from fastapi import UploadFile
from pydantic import TypeAdapter

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.memory.vector_store import parse_data_url, content_from_data_and_mime_type
from llama_stack_api import (
    URL,
    ChunkMetadata,
    EmbeddedChunk,
    Files,
    Inference,
    InterleavedContent,
    InterleavedContentItem,
    ListToolDefsResponse,
    OpenAIFilePurpose,
    QueryChunksResponse,
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
    TextContentItem,
    ToolDef,
    ToolGroup,
    ToolGroupsProtocolPrivate,
    ToolInvocationResult,
    ToolRuntime,
    UploadFileRequest,
)

from .config import PageIndexRagToolRuntimeConfig
from .pageindex_engine import PageIndexEngine
from ..rag.context_retriever import generate_rag_query

log = get_logger(name=__name__, category="tool_runtime::pageindex_rag")


async def raw_data_from_doc(doc: RAGDocument) -> tuple[bytes, str]:
    """Get raw binary data and mime type from a RAGDocument for file upload."""
    if isinstance(doc.content, URL):
        uri = doc.content.uri
        if uri.startswith("file://"):
            raise ValueError("file:// URIs are not supported. Please use the Files API (/v1/files) to upload files.")
        if uri.startswith("data:"):
            parts = parse_data_url(uri)
            mime_type = parts["mimetype"]
            data = parts["data"]

            if parts["is_base64"]:
                file_data = base64.b64decode(data)
            else:
                file_data = data.encode("utf-8")

            return file_data, mime_type
        else:
            async with httpx.AsyncClient() as client:
                r = await client.get(uri)
                r.raise_for_status()
                mime_type = r.headers.get("content-type", "application/octet-stream")
                return r.content, mime_type
    else:
        if isinstance(doc.content, str):
            content_str = doc.content
        else:
            content_str = interleaved_content_as_str(doc.content)

        if content_str.startswith("file://"):
            raise ValueError("file:// URIs are not supported. Please use the Files API (/v1/files) to upload files.")
        if content_str.startswith("data:"):
            parts = parse_data_url(content_str)
            mime_type = parts["mimetype"]
            data = parts["data"]

            if parts["is_base64"]:
                file_data = base64.b64decode(data)
            else:
                file_data = data.encode("utf-8")

            return file_data, mime_type
        else:
            return content_str.encode("utf-8"), "text/plain"


class PageIndexRagToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime):
    """Inline vectorless RAG implementation using PageIndexEngine for reasoning-based retrieval."""

    def __init__(
        self,
        config: PageIndexRagToolRuntimeConfig,
        inference_api: Inference,
        files_api: Files,
    ):
        self.config = config
        self.inference_api = inference_api
        self.files_api = files_api
        self.pageindex_engine = PageIndexEngine(config.pageindex_config, inference_api)

    async def initialize(self):
        """Initialize the PageIndex engine (sets up KV store)."""
        await self.pageindex_engine.initialize()

    async def shutdown(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def insert(
        self,
        documents: list[RAGDocument],
        vector_store_id: str,  # Actually a PageIndex store_id, but keeping same interface
        chunk_size_in_tokens: int | None = None,  # Ignored for PageIndex
    ) -> None:
        """Insert documents into PageIndex (no chunking, builds tree structure)."""
        if not documents:
            return

        for doc in documents:
            try:
                try:
                    file_data, mime_type = await raw_data_from_doc(doc)
                except Exception as e:
                    log.error(f"Failed to extract content from document {doc.document_id}: {e}")
                    continue

                file_extension = mimetypes.guess_extension(mime_type) or ".txt"
                filename = doc.metadata.get("filename", f"{doc.document_id}{file_extension}")

                file_obj = io.BytesIO(file_data)
                file_obj.name = filename
                upload_file = UploadFile(file=file_obj, filename=filename)

                try:
                    created_file = await self.files_api.openai_upload_file(
                        request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
                        file=upload_file,
                    )
                except Exception as e:
                    log.error(f"Failed to upload file for document {doc.document_id}: {e}")
                    continue

                # Extract text content from file
                try:
                    document_text = content_from_data_and_mime_type(file_data, mime_type)
                except Exception as e:
                    log.error(f"Failed to extract text from file {created_file.id}: {e}")
                    continue

                # Build PageIndex tree using LLM (no chunking, no embeddings)
                try:
                    await self.pageindex_engine.build_index(
                        store_id=vector_store_id,
                        file_id=created_file.id,
                        document_text=document_text,
                        metadata={**doc.metadata, "filename": filename, "document_id": doc.document_id},
                    )

                    # Update manifest to track file_ids for this store
                    await self._update_manifest(vector_store_id, created_file.id)

                except Exception as e:
                    log.error(
                        f"Failed to build PageIndex tree for file {created_file.id} "
                        f"in store {vector_store_id} for document {doc.document_id}: {e}"
                    )
                    continue

            except Exception as e:
                log.error(f"Unexpected error processing document {doc.document_id}: {e}")
                continue

    async def _update_manifest(self, store_id: str, file_id: str) -> None:
        """Update the manifest to track file_ids for a store."""
        manifest_key = f"pageindex_manifest:{store_id}"
        manifest_data = await self.pageindex_engine.kvstore.get(manifest_key)
        
        if manifest_data:
            manifest = json.loads(manifest_data)
        else:
            manifest = {"file_ids": []}
        
        if file_id not in manifest["file_ids"]:
            manifest["file_ids"].append(file_id)
            await self.pageindex_engine.kvstore.set(manifest_key, json.dumps(manifest))

    async def query(
        self,
        content: InterleavedContent,
        vector_store_ids: list[str],  # Actually PageIndex store_ids
        query_config: RAGQueryConfig | None = None,
    ) -> RAGQueryResult:
        """Query PageIndex using reasoning-based tree search."""
        if not vector_store_ids:
            raise ValueError(
                "No PageIndex stores were provided to the knowledge search tool. "
                "Please provide at least one store ID."
            )

        query_config = query_config or RAGQueryConfig(
            max_tokens_in_context=self.config.vector_stores_config.chunk_retrieval_params.max_tokens_in_context
        )

        # Generate query (reuse same query generation logic)
        query = await generate_rag_query(
            query_config.query_generator_config,
            content,
            inference_api=self.inference_api,
        )

        # Query all stores in parallel
        tasks = [
            self._query_pageindex_store(
                store_id=store_id,
                query=query,
                max_results=query_config.max_chunks,
            )
            for store_id in vector_store_ids
        ]
        results: list[QueryChunksResponse] = await asyncio.gather(*tasks)

        # Merge and format results (same as vector RAG)
        chunks = []
        scores = []

        for store_id, result in zip(vector_store_ids, results, strict=False):
            for embedded_chunk, score in zip(result.chunks, result.scores, strict=False):
                chunk = embedded_chunk
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["store_id"] = store_id
                chunks.append(chunk)
                scores.append(score)

        if not chunks:
            return RAGQueryResult(content=None)

        # Sort by score
        chunks, scores = zip(*sorted(zip(chunks, scores, strict=False), key=lambda x: x[1], reverse=True), strict=False)  # type: ignore
        chunks = chunks[: query_config.max_chunks]

        # Format using same templates as vector RAG
        tokens = 0
        vector_stores_config = self.config.vector_stores_config
        header_template = vector_stores_config.file_search_params.header_template
        footer_template = vector_stores_config.file_search_params.footer_template
        chunk_template = vector_stores_config.context_prompt_params.chunk_annotation_template
        context_template = vector_stores_config.context_prompt_params.context_template

        picked: list[InterleavedContentItem] = [TextContentItem(text=header_template.format(num_chunks=len(chunks)))]
        for i, embedded_chunk in enumerate(chunks):
            metadata = embedded_chunk.metadata
            tokens += metadata.get("token_count", 0)
            tokens += metadata.get("metadata_token_count", 0)

            if tokens > query_config.max_tokens_in_context:
                log.error(f"Using {len(picked)} chunks; reached max tokens in context: {tokens}")
                break

            chunk_metadata_keys_to_include_from_context = [
                "chunk_id",
                "document_id",
                "source",
                "pageindex_node_id",
                "start_index",
                "end_index",
                "title",
            ]
            metadata_keys_to_exclude_from_context = [
                "token_count",
                "metadata_token_count",
                "store_id",
            ]
            metadata_for_context = {}
            for k in chunk_metadata_keys_to_include_from_context:
                if hasattr(embedded_chunk.chunk_metadata, k):
                    metadata_for_context[k] = getattr(embedded_chunk.chunk_metadata, k)
            for k in metadata:
                if k not in metadata_keys_to_exclude_from_context:
                    metadata_for_context[k] = metadata[k]

            text_content = chunk_template.format(index=i + 1, chunk=embedded_chunk, metadata=metadata_for_context)
            picked.append(TextContentItem(text=text_content))

        picked.append(TextContentItem(text=footer_template))
        picked.append(
            TextContentItem(
                text=context_template.format(query=interleaved_content_as_str(content), annotation_instruction="")
            )
        )

        return RAGQueryResult(
            content=picked,
            metadata={
                "document_ids": [c.document_id for c in chunks[: len(picked)]],
                "chunks": [c.content for c in chunks[: len(picked)]],
                "scores": scores[: len(picked)],
                "store_ids": [c.metadata["store_id"] for c in chunks[: len(picked)]],
            },
        )

    async def _query_pageindex_store(
        self,
        store_id: str,
        query: str,
        max_results: int,
    ) -> QueryChunksResponse:
        """Query a single PageIndex store and convert to QueryChunksResponse."""
        try:
            pageindex_results = await self.pageindex_engine.query(
                store_id=store_id,
                query=query,
                max_results=max_results,
            )
        except Exception as e:
            log.error(f"Failed to query PageIndex store {store_id}: {e}")
            return QueryChunksResponse(chunks=[], scores=[])

        # Convert PageIndex tree nodes to EmbeddedChunk format
        chunks = []
        scores = []

        for node in pageindex_results:
            # Extract content from node (could be summary or full content)
            content = node.get("content") or node.get("summary", "")
            node_id = node.get("node_id", "")
            title = node.get("title", "")

            chunk = EmbeddedChunk(
                content=content,
                chunk_id=node_id,
                metadata={
                    "document_id": node.get("document_id", store_id),
                    "pageindex_node_id": node_id,
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "title": title,
                },
                chunk_metadata=ChunkMetadata(
                    chunk_id=node_id,
                    document_id=node.get("document_id", store_id),
                    source=node.get("source"),
                ),
                # Dummy embedding (not used for PageIndex)
                embedding=[0.0] * 384,
                embedding_model="pageindex_inline",
                embedding_dimension=384,
            )
            chunks.append(chunk)
            # Use PageIndex relevance score
            scores.append(node.get("relevance_score", 1.0))

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def list_runtime_tools(
        self,
        tool_group_id: str | None = None,
        mcp_endpoint: URL | None = None,
        authorization: str | None = None,
    ) -> ListToolDefsResponse:
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="insert_into_memory",
                    description="Insert documents into memory using PageIndex (inline, vectorless, tree-based)",
                ),
                ToolDef(
                    name="knowledge_search",
                    description="Search for information using reasoning-based retrieval (PageIndex inline).",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for. Can be a natural language sentence.",
                            }
                        },
                        "required": ["query"],
                    },
                ),
            ]
        )

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None
    ) -> ToolInvocationResult:
        store_ids = kwargs.get("vector_store_ids", [])  # Keep same param name for compatibility
        query_config = kwargs.get("query_config")
        if query_config:
            query_config = TypeAdapter(RAGQueryConfig).validate_python(query_config)
        else:
            query_config = RAGQueryConfig(
                max_tokens_in_context=self.config.vector_stores_config.chunk_retrieval_params.max_tokens_in_context
            )

        query = kwargs["query"]
        result = await self.query(
            content=query,
            vector_store_ids=store_ids,
            query_config=query_config,
        )

        return ToolInvocationResult(
            content=result.content or [],
            metadata={
                **(result.metadata or {}),
                "citation_files": getattr(result, "citation_files", None),
            },
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

from llama_stack.core.storage.kvstore.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.vector_store import content_from_data_and_mime_type
from llama_stack_api import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)

from .config import PageIndexInlineConfig

log = get_logger(name=__name__, category="tool_runtime::pageindex_rag")


class PageIndexEngine:
    """Inline PageIndex implementation using LLM for tree building and reasoning."""

    def __init__(
        self,
        config: PageIndexInlineConfig,
        inference_api: Any,  # Inference API
    ):
        self.config = config
        self.inference_api = inference_api
        self.kvstore = None

    async def initialize(self):
        """Initialize KV store for tree storage."""
        self.kvstore = await kvstore_impl(self.config.kvstore)

    async def build_index(
        self,
        store_id: str,
        file_id: str,
        document_text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build PageIndex tree from document text using LLM."""

        # Step 1: Use LLM to analyze document structure and create tree
        tree_prompt = self._build_tree_prompt(document_text)

        tree_response = await self.inference_api.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model=self.config.tree_builder_model,
                messages=[
                    OpenAIUserMessageParam(content=tree_prompt)
                ],
                response_format={"type": "json_object"},  # Force JSON output
            )
        )

        tree_structure = json.loads(tree_response.choices[0].message.content)

        # Step 2: Store tree in KV store
        tree_key = f"pageindex_tree:{store_id}:{file_id}"
        await self.kvstore.set(
            key=tree_key,
            value=json.dumps({
                "tree": tree_structure,
                "metadata": metadata,
                "document_text": document_text,  # Store full text for retrieval
            })
        )

        return {"tree": tree_structure, "file_id": file_id}

    async def query(
        self,
        store_id: str,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Query PageIndex using reasoning-based tree search."""

        # Step 1: Load all trees for this store
        trees = await self._load_store_trees(store_id)

        if not trees:
            log.warning(f"No trees found for store {store_id}")
            return []

        # Step 2: Use LLM to reason over trees and find relevant nodes
        reasoning_prompt = self._build_reasoning_prompt(query, trees)

        reasoning_response = await self.inference_api.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model=self.config.reasoning_model,
                messages=[
                    OpenAIUserMessageParam(content=reasoning_prompt)
                ],
                response_format={"type": "json_object"},
            )
        )

        relevant_nodes = json.loads(reasoning_response.choices[0].message.content)

        # Step 3: Extract content for relevant nodes
        results = []
        for node_ref in relevant_nodes.get("nodes", [])[:max_results]:
            node = self._get_node_by_id(trees, node_ref.get("node_id"))
            if node:
                content = self._extract_node_content(node, trees)
                results.append({
                    "node_id": node.get("node_id", ""),
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "content": content,
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "relevance_score": node_ref.get("score", 1.0),
                    "document_id": node.get("document_id", store_id),
                })

        return results

    def _build_tree_prompt(self, document_text: str) -> str:
        """Build prompt for LLM to generate tree structure."""
        # Truncate document if too long (keep first 50000 chars for prompt)
        truncated_text = document_text[:50000]
        if len(document_text) > 50000:
            truncated_text += "\n\n[... document truncated ...]"

        return f"""Analyze the following document and create a hierarchical tree structure similar to a table of contents.

The tree should have:
- title: Section title
- summary: Brief summary of the section
- start_index: Character index where section starts
- end_index: Character index where section ends
- node_id: Unique identifier for the node (e.g., "0001", "0002")
- nodes: Nested child sections (if any)

Document:
{truncated_text}

Return JSON in this format:
{{
  "title": "Document Title",
  "summary": "Overall summary",
  "node_id": "0000",
  "start_index": 0,
  "end_index": {len(document_text)},
  "nodes": [
    {{
      "title": "Section 1",
      "summary": "Section summary",
      "node_id": "0001",
      "start_index": 100,
      "end_index": 500,
      "nodes": []
    }}
  ]
}}

Create a meaningful hierarchy that captures the document's structure. Each node should represent a logical section or subsection."""

    def _build_reasoning_prompt(self, query: str, trees: list[dict]) -> str:
        """Build prompt for LLM to reason over trees and find relevant nodes."""
        trees_summary = self._summarize_trees(trees)

        return f"""Given the following query and document tree structures, identify the most relevant sections.

Query: {query}

Document Trees:
{trees_summary}

Reason about which tree nodes are most relevant to answer the query. Consider:
1. Semantic relevance of node titles and summaries
2. Hierarchical relationships (parent-child context)
3. Document structure and organization

Return JSON with relevant nodes:
{{
  "nodes": [
    {{
      "node_id": "node_id_here",
      "score": 0.95,
      "reasoning": "Why this node is relevant"
    }}
  ]
}}

Order nodes by relevance score (highest first)."""

    async def _load_store_trees(self, store_id: str) -> list[dict]:
        """Load all trees for a given store."""
        # List all keys for this store
        prefix = f"pageindex_tree:{store_id}:"
        trees = []

        # Note: KV store doesn't have a list operation in the protocol
        # We'll need to store a manifest or use a different approach
        # For now, we'll try to load trees by iterating through known file_ids
        # This is a limitation - in production, you'd want a manifest/index

        # Try to get a manifest if it exists
        manifest_key = f"pageindex_manifest:{store_id}"
        manifest_data = await self.kvstore.get(manifest_key)
        if manifest_data:
            manifest = json.loads(manifest_data)
            for file_id in manifest.get("file_ids", []):
                tree_key = f"{prefix}{file_id}"
                tree_data = await self.kvstore.get(tree_key)
                if tree_data:
                    trees.append(json.loads(tree_data))

        return trees

    def _summarize_trees(self, trees: list[dict]) -> str:
        """Create a summary of tree structures for the reasoning prompt."""
        summaries = []
        for i, tree_data in enumerate(trees):
            tree = tree_data.get("tree", {})
            summary = self._tree_to_summary(tree, 0)
            summaries.append(f"Tree {i+1}:\n{summary}")
        return "\n\n".join(summaries)

    def _tree_to_summary(self, node: dict, depth: int) -> str:
        """Convert a tree node to a summary string."""
        indent = "  " * depth
        title = node.get("title", "")
        summary = node.get("summary", "")
        node_id = node.get("node_id", "")
        start = node.get("start_index", 0)
        end = node.get("end_index", 0)

        result = f"{indent}- [{node_id}] {title} (chars {start}-{end})\n"
        if summary:
            result += f"{indent}  Summary: {summary[:100]}\n"

        for child in node.get("nodes", []):
            result += self._tree_to_summary(child, depth + 1)

        return result

    def _get_node_by_id(self, trees: list[dict], node_id: str) -> dict | None:
        """Find a node by ID across all trees."""
        for tree_data in trees:
            tree = tree_data.get("tree", {})
            node = self._find_node_in_tree(tree, node_id)
            if node:
                # Add document_id from tree metadata
                node["document_id"] = tree_data.get("metadata", {}).get("document_id", "")
                return node
        return None

    def _find_node_in_tree(self, node: dict, node_id: str) -> dict | None:
        """Recursively find a node by ID in a tree."""
        if node.get("node_id") == node_id:
            return node
        for child in node.get("nodes", []):
            found = self._find_node_in_tree(child, node_id)
            if found:
                return found
        return None

    def _extract_node_content(self, node: dict, trees: list[dict]) -> str:
        """Extract the actual text content for a node."""
        # Find the tree that contains this node
        for tree_data in trees:
            tree = tree_data.get("tree", {})
            if self._node_in_tree(tree, node.get("node_id")):
                document_text = tree_data.get("document_text", "")
                start = node.get("start_index", 0)
                end = node.get("end_index", len(document_text))
                return document_text[start:end]
        return node.get("summary", "")

    def _node_in_tree(self, node: dict, node_id: str) -> bool:
        """Check if a node with given ID exists in tree."""
        if node.get("node_id") == node_id:
            return True
        for child in node.get("nodes", []):
            if self._node_in_tree(child, node_id):
                return True
        return False

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import KVStoreReference, VectorStoresConfig


class PageIndexInlineConfig(BaseModel):
    """Configuration for inline PageIndex implementation."""

    # LLM models for tree building and reasoning
    tree_builder_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Model to use for building document tree structures",
    )
    reasoning_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Model to use for reasoning-based retrieval",
    )

    # KV store for tree storage
    kvstore: KVStoreReference = Field(
        default_factory=lambda: KVStoreReference(backend="kv_default", namespace="pageindex"),
        description="KV store reference for storing PageIndex trees",
    )

    # Tree building parameters
    max_tree_depth: int = Field(
        default=5,
        description="Maximum depth of tree hierarchy",
    )
    min_section_length: int = Field(
        default=100,
        description="Minimum character length for a tree node",
    )


class PageIndexRagToolRuntimeConfig(BaseModel):
    """Configuration for inline vectorless RAG tool runtime using PageIndex."""

    pageindex_config: PageIndexInlineConfig = Field(
        description="PageIndex inline configuration"
    )

    # Reuse vector_stores_config for prompt templates and formatting
    vector_stores_config: VectorStoresConfig = Field(
        default_factory=VectorStoresConfig,
        description="Configuration for prompt templates and behavior (reused from vector stores config)",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "pageindex_config": {
                "tree_builder_model": "${env.PAGEINDEX_TREE_BUILDER_MODEL:=meta-llama/Llama-3.1-8B-Instruct}",
                "reasoning_model": "${env.PAGEINDEX_REASONING_MODEL:=meta-llama/Llama-3.1-8B-Instruct}",
                "kvstore": {
                    "backend": "kv_default",
                    "namespace": "pageindex",
                },
            }
        }

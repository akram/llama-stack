# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import httpx

from llama_stack.log import get_logger

from .config import PageIndexConfig

log = get_logger(name=__name__, category="tool_runtime::pageindex_rag")


class PageIndexClient:
    """Client for PageIndex API - vectorless, reasoning-based document retrieval."""

    def __init__(self, config: PageIndexConfig):
        self.config = config
        self.base_url = config.api_url.rstrip("/")
        self.headers = {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"
        self.timeout = httpx.Timeout(config.timeout_seconds)

    async def build_index(
        self,
        store_id: str,
        file_id: str,
        file_data: bytes,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build PageIndex tree from a document.

        Args:
            store_id: The store/index ID
            file_id: File identifier
            file_data: Raw file bytes
            metadata: Document metadata

        Returns:
            Response from PageIndex API with tree structure
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Upload file and build tree
            files = {"file": (metadata.get("filename", file_id), file_data)}
            data = {
                "index_id": store_id,
                "file_id": file_id,
                "metadata": metadata,
            }

            try:
                response = await client.post(
                    f"{self.base_url}/v1/indexes/{store_id}/documents",
                    headers=self.headers,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                log.error(f"Failed to build PageIndex tree for store {store_id}, file {file_id}: {e}")
                raise

    async def query(
        self,
        store_id: str,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Query PageIndex using reasoning-based tree search.

        Args:
            store_id: The store/index ID
            query: Natural language query
            max_results: Maximum number of results to return

        Returns:
            List of tree nodes with content, scores, and metadata
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/indexes/{store_id}/query",
                    headers=self.headers,
                    json={
                        "query": query,
                        "max_results": max_results,
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("nodes", [])
            except httpx.HTTPError as e:
                log.error(f"Failed to query PageIndex store {store_id}: {e}")
                raise

    async def get_index_info(self, store_id: str) -> dict[str, Any]:
        """Get information about a PageIndex."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/v1/indexes/{store_id}",
                    headers=self.headers,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                log.error(f"Failed to get PageIndex info for store {store_id}: {e}")
                raise

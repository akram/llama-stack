#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
End-to-end test for PageIndex RAG against a running llama-stack server.

This script tests PageIndex RAG by making real API calls to a running server.
It creates stores, uploads documents, performs queries, and validates results.

Usage:
    # Test against local server
    python tests/test_pageindex_e2e.py
    
    # Test against custom server
    python tests/test_pageindex_e2e.py --url http://localhost:8321
    
    # With specific model
    python tests/test_pageindex_e2e.py --model llama3.2:3b

Requirements:
    - llama-stack server running (default: http://localhost:8321)
    - PageIndex RAG provider configured
    - Inference provider available
"""

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx package not found. Install with: pip install httpx")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


class PageIndexE2ETest:
    """End-to-end test for PageIndex RAG."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.Client(base_url=self.base_url, timeout=60.0)
        self.created_resources = {
            "vector_stores": [],
            "files": [],
        }
    
    def cleanup(self):
        """Clean up all created resources."""
        print_info("Cleaning up resources...")
        
        # Delete vector stores
        for store_id in self.created_resources["vector_stores"]:
            try:
                response = self.client.delete(f"/vector-stores/{store_id}")
                if response.status_code in [200, 204]:
                    print_info(f"  Deleted vector store: {store_id}")
                else:
                    print_warning(f"  Could not delete vector store {store_id}: {response.status_code}")
            except Exception as e:
                print_warning(f"  Could not delete vector store {store_id}: {e}")
        
        # Delete files
        for file_id in self.created_resources["files"]:
            try:
                response = self.client.delete(f"/v1/files/{file_id}")
                if response.status_code in [200, 204]:
                    print_info(f"  Deleted file: {file_id}")
                else:
                    print_warning(f"  Could not delete file {file_id}: {response.status_code}")
            except Exception as e:
                print_warning(f"  Could not delete file {file_id}: {e}")
    
    def create_test_documents(self):
        """Create test documents for PageIndex."""
        print_info("Creating test documents...")
        
        documents = {
            "machine_learning.txt": """Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.

Types of Machine Learning:

1. Supervised Learning
   - Training with labeled data
   - Predicts outcomes based on input features
   - Examples: classification, regression
   - Common algorithms: linear regression, decision trees, neural networks

2. Unsupervised Learning
   - Finding patterns in unlabeled data
   - No predefined outputs
   - Examples: clustering, dimensionality reduction
   - Common algorithms: K-means, PCA, autoencoders

3. Reinforcement Learning
   - Learning through trial and error
   - Agent receives rewards or penalties
   - Examples: game playing, robotics
   - Common algorithms: Q-learning, policy gradients

Applications:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Fraud detection
""",
            "neural_networks.txt": """Neural Networks Architecture

Neural networks are the foundation of deep learning, inspired by biological neural networks.

Network Structure:

Input Layer:
- Receives raw input data
- One node per feature
- No computation, just passes data forward

Hidden Layers:
- Process information through weighted connections
- Apply activation functions for non-linearity
- Multiple layers enable deep learning
- Each layer learns different levels of abstraction

Output Layer:
- Produces final predictions
- Structure depends on task (classification vs regression)
- Activation function chosen based on output type

Training Process:

Forward Propagation:
1. Input data flows through network
2. Each layer applies weights and activation
3. Output is generated

Backpropagation:
1. Calculate loss (prediction error)
2. Compute gradients using chain rule
3. Update weights to minimize loss
4. Iterate until convergence

Activation Functions:
- ReLU: most common, prevents vanishing gradients
- Sigmoid: for binary classification
- Tanh: similar to sigmoid, centered at zero
- Softmax: for multi-class classification
""",
            "data_preprocessing.txt": """Data Preprocessing Best Practices

Proper data preparation is crucial for machine learning success.

Data Cleaning:

Missing Values:
- Deletion: Remove rows/columns with missing data
- Imputation: Fill with mean, median, or mode
- Advanced: Use ML algorithms to predict missing values

Duplicates:
- Identify exact duplicates
- Handle near-duplicates carefully
- Consider domain context

Outliers:
- Detect using statistical methods (IQR, Z-score)
- Decide: remove, cap, or transform
- Domain knowledge is critical

Feature Engineering:

Creating Features:
- Polynomial features for non-linear relationships
- Interaction terms between features
- Domain-specific transformations

Encoding Categorical Variables:
- One-hot encoding for nominal categories
- Label encoding for ordinal categories
- Target encoding for high-cardinality features

Normalization Techniques:

Min-Max Scaling:
- Scale features to range [0, 1]
- Formula: (x - min) / (max - min)
- Sensitive to outliers

Standardization:
- Scale to mean=0, std=1
- Formula: (x - mean) / std
- Less sensitive to outliers

Robust Scaling:
- Use median and IQR
- Best for data with outliers
- Formula: (x - median) / IQR
""",
        }
        
        # Create temp directory and write files
        temp_dir = tempfile.mkdtemp(prefix="pageindex_e2e_")
        file_paths = {}
        
        for filename, content in documents.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            file_paths[filename] = str(file_path)
            print_info(f"  Created: {filename}")
        
        return file_paths
    
    def test_server_connectivity(self):
        """Test connection to llama-stack server."""
        print_info(f"Testing connection to {self.base_url}...")
        
        try:
            # Try the OpenAI-compatible models endpoint
            response = self.client.get("/v1/models")
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("data", [])
                print_success(f"Connected to server at {self.base_url}")
                print_info(f"  Available models: {len(models)}")
                return True
            else:
                print_error(f"Server returned status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Failed to connect to server: {e}")
            return False
    
    def create_pageindex_store(self, name: str):
        """Create a PageIndex vector store."""
        print_info(f"Creating PageIndex store: {name}...")
        
        try:
            # Create vector store via llama-stack API
            response = self.client.post(
                "/v1/vector_stores",
                json={"name": name}
            )
            
            if response.status_code == 200:
                store = response.json()
                store_id = store.get("id")
                self.created_resources["vector_stores"].append(store_id)
                print_success(f"Created store: {store_id}")
                return store
            else:
                print_error(f"Failed to create store: HTTP {response.status_code}")
                print_info(f"  Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Failed to create store: {e}")
            return None
    
    def upload_and_attach_files(self, store_id: str, file_paths: dict):
        """Upload files and attach them to the store."""
        print_info(f"Uploading {len(file_paths)} files...")
        
        attached_files = []
        
        for filename, file_path in file_paths.items():
            try:
                # Upload file
                with open(file_path, "rb") as f:
                    files = {"file": (filename, f, "text/plain")}
                    data = {"purpose": "assistants"}
                    response = self.client.post("/v1/files", files=files, data=data)
                
                if response.status_code != 200:
                    print_error(f"  Failed to upload {filename}: HTTP {response.status_code}")
                    continue
                
                file_data = response.json()
                file_id = file_data.get("id")
                self.created_resources["files"].append(file_id)
                print_info(f"  Uploaded: {filename} ({file_id})")
                
                # Attach to vector store
                response = self.client.post(
                    f"/v1/vector_stores/{store_id}/files",
                    json={"file_id": file_id}
                )
                
                if response.status_code != 200:
                    print_error(f"  Failed to attach {filename}: HTTP {response.status_code}")
                    continue
                
                # Wait for processing
                max_wait = 60  # seconds
                start_time = time.time()
                status = "in_progress"
                
                while status == "in_progress":
                    if time.time() - start_time > max_wait:
                        print_warning(f"  Timeout waiting for {filename}")
                        break
                    time.sleep(1)
                    
                    response = self.client.get(f"/v1/vector_stores/{store_id}/files/{file_id}")
                    if response.status_code == 200:
                        file_status = response.json()
                        status = file_status.get("status", "unknown")
                    else:
                        break
                
                if status == "completed":
                    print_success(f"  Indexed: {filename}")
                    attached_files.append((filename, file_id))
                else:
                    print_error(f"  Failed to index {filename}: {status}")
                
            except Exception as e:
                print_error(f"  Failed to process {filename}: {e}")
        
        return attached_files
    
    def test_query(self, store_id: str, query: str, expected_terms: list):
        """Test a query against the PageIndex store."""
        print_info(f'Query: "{query}"')
        
        try:
            # Use responses API directly
            response = self.client.post(
                "/responses/create",
                json={
                    "model": self.model,
                    "input": query,
                    "tools": [
                        {
                            "type": "file_search",
                            "vector_store_ids": [store_id],
                        }
                    ],
                    "stream": False,
                },
            )
            
            if response.status_code != 200:
                print_error(f"  Query failed: HTTP {response.status_code}")
                print_info(f"  Response: {response.text[:200]}")
                return False
            
            result = response.json()
            
            # Extract response text
            output_text = result.get("output_text", "")
            response_text = output_text.lower()
            
            # Check for expected terms
            found_terms = [term for term in expected_terms if term.lower() in response_text]
            
            print_success(f"  Query successful")
            print_info(f"  Found terms: {', '.join(found_terms) if found_terms else 'none'}")
            print_info(f"  Response preview: {output_text[:150]}...")
            
            return len(found_terms) > 0
                
        except Exception as e:
            print_error(f"  Query error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Run the complete end-to-end test."""
        print_header("PageIndex RAG End-to-End Test")
        
        print_info(f"Server: {self.base_url}")
        print_info(f"Model: {self.model}")
        print()
        
        try:
            # Test 1: Server connectivity
            print_header("1. Testing Server Connectivity")
            if not self.test_server_connectivity():
                print_error("Cannot proceed without server connection")
                return False
            
            # Test 2: Create test documents
            print_header("2. Creating Test Documents")
            file_paths = self.create_test_documents()
            
            # Test 3: Create PageIndex store
            print_header("3. Creating PageIndex Store")
            store = self.create_pageindex_store("e2e_test_pageindex_store")
            if not store:
                return False
            
            # Test 4: Upload and index documents
            print_header("4. Uploading and Indexing Documents")
            attached_files = self.upload_and_attach_files(store["id"], file_paths)
            if not attached_files:
                print_error("No files were successfully indexed")
                return False
            
            # Test 5: Query tests
            print_header("5. Testing Queries")
            
            test_queries = [
                ("What are the main types of machine learning?", ["supervised", "unsupervised", "reinforcement"]),
                ("Explain the neural network training process", ["forward propagation", "backpropagation", "gradient"]),
                ("What normalization techniques are available?", ["min-max", "standardization", "scaling"]),
                ("How do you handle missing values?", ["imputation", "deletion", "missing"]),
            ]
            
            results = []
            for query, expected_terms in test_queries:
                success = self.test_query(store["id"], query, expected_terms)
                results.append(success)
                print()
            
            # Summary
            print_header("Test Summary")
            successful = sum(results)
            total = len(results)
            
            print_info(f"Documents indexed: {len(attached_files)}")
            print_info(f"Queries: {successful}/{total} successful")
            print()
            
            if successful == total:
                print_success("All tests passed! ✨")
                return True
            elif successful > 0:
                print_warning(f"Partial success: {successful}/{total} tests passed")
                return False
            else:
                print_error("All tests failed")
                return False
                
        except KeyboardInterrupt:
            print_warning("\nTest interrupted by user")
            return False
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            print_header("Cleanup")
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="PageIndex RAG End-to-End Test")
    parser.add_argument(
        "--url",
        default="http://localhost:8321",
        help="Llama-stack server URL (default: http://localhost:8321)"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for queries"
    )
    
    args = parser.parse_args()
    
    test = PageIndexE2ETest(base_url=args.url, model=args.model)
    success = test.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

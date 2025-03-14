# Marqo Essentials - Distilled Documentation

## Core Concept
Marqo is an open-source vector search engine that supports both semantic (tensor) and lexical search across text, images, and other media. It enables powerful hybrid search capabilities that combine both methods.

## Installation

```bash
# Run Marqo using Docker
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest

# Install Python client
pip install marqo
```

## Basic Usage Pattern

```python
import marqo

# Initialize client
mq = marqo.Client(url="http://localhost:8882")

# Create an index
mq.create_index("my-index")

# Add documents
mq.index("my-index").add_documents([
    {"_id": "doc1", "text": "Content for document 1", "image": "https://example.com/image1.jpg"},
    {"_id": "doc2", "text": "Content for document 2", "image": "https://example.com/image2.jpg"}
], tensor_fields=["text", "image"])

# Search (semantic/tensor search by default)
results = mq.index("my-index").search("query text")

# Lexical search
lexical_results = mq.index("my-index").search("query text", search_method="LEXICAL")

# Hybrid search
hybrid_results = mq.index("my-index").search(
    q="query text",
    search_method="HYBRID",
    hybrid_parameters={
        "retrievalMethod": "disjunction",
        "rankingMethod": "rrf",
        "alpha": 0.5  # Balance between tensor (1.0) and lexical (0.0)
    }
)

# Filter search results
filtered_results = mq.index("my-index").search(
    q="query text", 
    filter_string="field1:value OR field2:(value1 AND value2)"
)
```

## Index Types

### Unstructured Indexes
- Flexible schema - fields can be added at any time
- Must specify tensor_fields during document addition
- Default index type

### Structured Indexes 
- Fixed schema defined at creation time
- Better performance, especially for large indexes
- Explicit control over which fields are searchable, filterable, etc.

```python
# Create structured index
settings = {
    "type": "structured",
    "model": "hf/e5-base-v2",
    "allFields": [
        {"name": "title", "type": "text", "features": ["lexical_search"]},
        {"name": "description", "type": "text", "features": ["lexical_search", "filter"]},
        {"name": "image", "type": "image_pointer"},
        {"name": "multimodal_field", "type": "multimodal_combination", 
         "dependentFields": {"title": 0.2, "image": 0.8}}
    ],
    "tensorFields": ["multimodal_field"]
}
mq.create_index("structured-index", settings_dict=settings)
```

## Key Operations

### Creating Indexes
- Specify model (default: "hf/e5-base-v2" for text)
- Configure preprocessing options
- Set HNSW parameters for vector search

### Adding Documents
- Documents are JSON objects
- Special "_id" field for document ID (auto-generated if not provided)
- Specify tensor_fields for unstructured indexes
- Support for multimodal fields, combining text and images

### Search Types
- Tensor search (default): Semantic understanding
- Lexical search: Keyword matching
- Hybrid search: Combines both methods

### Filtering
- DSL syntax: `field:(value1 OR value2) AND another_field:value3`
- Range filters: `numeric_field:[0 TO 100]`
- Boolean operators: AND, OR, NOT

## Model Options

### Text Models
- Default: "hf/e5-base-v2"
- Performance options: "hf/e5-small-v2" (fastest), "hf/e5-large-v2" (most accurate)
- Multilingual: "hf/multilingual-e5-base"

### Image Models
- Default: "open_clip/ViT-B-32/laion2b_s34b_b79k"
- Fashion-specific: "Marqo/marqo-fashionCLIP"
- Best image understanding: "open_clip/ViT-H-14/laion2b_s32b_b79k"

## Advanced Features

### Multimodal Search
```python
# Create multimodal index
settings = {
    "treatUrlsAndPointersAsImages": True,  # Required for image URLs
    "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
}
mq.create_index("multimodal-index", settings_dict=settings)

# Add documents with multimodal combination field
docs = [{"text": "red shirt", "image": "https://example.com/shirt.jpg"}]
mq.index("multimodal-index").add_documents(
    docs,
    mappings={
        "combined_field": {
            "type": "multimodal_combination",
            "weights": {"text": 0.3, "image": 0.7}
        }
    },
    tensor_fields=["combined_field"]
)

# Search with weighted queries
results = mq.index("multimodal-index").search({
    "red shirt": 1.0,
    "https://example.com/query-image.jpg": 0.8,
    "low quality": -0.3  # Negative weights push away certain results
})
```

### Custom Vectors
```python
# Create index for custom vectors
settings = {
    "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",  # Dimension must match your vectors
    "annParameters": {
        "spaceType": "angular",  # NOT "prenormalized-angular" for custom vectors
    }
}
mq.create_index("custom-vector-index", settings_dict=settings)

# Add documents with custom vectors
mq.index("custom-vector-index").add_documents(
    [{
        "_id": "doc1",
        "my_vector_field": {
            "vector": [0.1, 0.2, ..., 0.5],  # Must match model dimension
            "content": "Optional text for filtering/lexical search"
        }
    }],
    mappings={"my_vector_field": {"type": "custom_vector"}},
    tensor_fields=["my_vector_field"]
)

# Search with custom vectors
mq.index("custom-vector-index").search(
    q=None,  # No query text needed
    context={"tensor": [{"vector": [0.1, 0.2, ..., 0.5], "weight": 1.0}]}
)
```

### Score Modifiers
```python
# Boost results based on metadata fields
results = mq.index("my-index").search(
    q="shirt",
    score_modifiers={
        "multiply_score_by": [{"field_name": "popularity", "weight": 1.2}],
        "add_to_score": [{"field_name": "recency", "weight": 0.1}]
    }
)
```

### Recommendations
```python
# Get similar items based on existing documents
recommendations = mq.index("my-index").recommend(
    documents=["doc1", "doc5"],  # Document IDs
    limit=10,
    exclude_input_documents=True
)

# With document weights
recommendations = mq.index("my-index").recommend(
    documents={"doc1": 1.0, "doc2": 0.5},
    limit=10
)
```

## Configuration & Optimization

### HNSW Parameters
```python
# Configure vector search algorithm for better recall or performance
settings = {
    "annParameters": {
        "spaceType": "prenormalized-angular",
        "parameters": {
            "efConstruction": 512,  # Higher = better index quality but slower indexing
            "m": 16  # Higher = better recall but more memory usage
        }
    }
}
mq.create_index("optimized-index", settings_dict=settings)
```

### Text Preprocessing
```python
# Configure text chunking for better search results
settings = {
    "textPreprocessing": {
        "splitLength": 2,       # Chunks of 2 sentences
        "splitOverlap": 1,      # 1 sentence overlap
        "splitMethod": "sentence"  # Other options: "word", "character", "passage"
    }
}
```

### Image Preprocessing
```python
# Configure image patching for better results
settings = {
    "imagePreprocessing": {
        "patchMethod": "dino-v2"  # Options: "simple", "overlap", "frcnn", "marqo-yolo", "dino-v1", "dino-v2"
    }
}
```

### GPU Acceleration
```bash
# Run Marqo with GPU support
docker run --name marqo --gpus all -p 8882:8882 marqoai/marqo:latest
```

```python
# Specify device during operations
mq.index("my-index").add_documents(documents, device="cuda")
mq.index("my-index").search("query", device="cuda")
```

## Best Practices

### Performance
- Limit tensor fields to necessary fields only
- Use structured indexes for production systems
- Set appropriate batch sizes for document addition (client_batch_size=64)
- Increase efSearch parameter for better recall at cost of latency

### Search Quality
- Use hybrid search to balance precision and recall
- Adjust multimodal weights based on your data (text-heavy: 0.6/0.4, image-heavy: 0.2/0.8)
- Use weighted query terms to control influence of different aspects
- For better recall, increase limit parameter

### Resource Management
- Set environment variables to control resource usage:
  ```bash
  docker run --name marqo -p 8882:8882 \
      -e "MARQO_MAX_DOC_BYTES=200000" \
      -e "MARQO_MAX_CUDA_MODEL_MEMORY=5" \
      marqoai/marqo:latest
  ```

### Data Persistence
```bash
# Create a named volume for data persistence
docker volume create --name opt_vespa_var
docker run --name marqo -p 8882:8882 -v opt_vespa_var:/opt/vespa/var marqoai/marqo:latest
```

## Filtering & Advanced Queries

### Filter Syntax
```python
# Basic filtering
results = mq.index("my-index").search("query", filter_string="field1:value")

# Complex filtering with boolean operators
results = mq.index("my-index").search(
    q="query",
    filter_string="(category:(shirts OR pants) AND NOT color:red) OR (brand:nike AND price:[20 TO 50])"
)

# Filtering on array fields
results = mq.index("my-index").search(
    q="query",
    filter_string="tags:summer AND tags:sale"  # Both tags must exist
)

# Filtering with IN operator (structured indexes only)
results = mq.index("my-index").search(
    q="query",
    filter_string="category IN (shirts, pants, (summer clothes))"
)
```

### Exact Match Lexical Search
```python
# Double quotes for required terms
results = mq.index("my-index").search(
    q='query with "exact phrase"',
    search_method="LEXICAL"
)

# Escaping special characters (note double backslashes in Python)
results = mq.index("my-index").search(
    q="Dwayne \\"The Rock\\" Johnson",
    search_method="LEXICAL"
)
```

## Embedding & Vector Operations

### Generate Embeddings
```python
# Get embeddings for content without storing
embeddings = mq.index("my-index").embed(
    content=["Text to embed", "Another text to embed"]
)

# With weighted content
embeddings = mq.index("my-index").embed(
    content=[{
        "Text part 1": 0.7,
        "https://example.com/image.jpg": 0.3
    }]
)

# Control prefixing behavior
embeddings = mq.index("my-index").embed(
    content=["Query text"],
    content_type="query"  # Options: "query", "document", None
)
```

### Using Embeddings in Searches
```python
# Search with pre-computed vectors
results = mq.index("my-index").search(
    q={"Text query": 1.0},  # Can be combined with vector context
    context={"tensor": [
        {"vector": embeddings["embeddings"][0], "weight": 0.8}
    ]}
)
```

## Document Management

### Get Documents
```python
# Get single document
doc = mq.index("my-index").get_document(document_id="doc1")

# Get multiple documents
docs = mq.index("my-index").get_documents(
    document_ids=["doc1", "doc2", "doc3"]
)

# Get document with vector data
doc_with_vectors = mq.index("my-index").get_document(
    document_id="doc1",
    expose_facets=True  # Includes _tensor_facets field with vectors
)
```

### Update Documents
```python
# Same as add_documents, existing documents will be updated
mq.index("my-index").add_documents(
    [{"_id": "doc1", "updated_field": "new value"}],
    tensor_fields=["updated_field"]
)

# Use existing tensors when updating non-tensor fields
mq.index("my-index").add_documents(
    [{"_id": "doc1", "non_tensor_field": "new value"}],
    use_existing_tensors=True  # Avoids recomputing tensors
)
```

### Delete Documents
```python
# Delete single document
mq.index("my-index").delete_documents(ids="doc1")

# Delete multiple documents
mq.index("my-index").delete_documents(ids=["doc1", "doc2", "doc3"])
```

## Common Recipes & Patterns

### Semantic Filtering with Prompts
```python
# Filter for specific styles without metadata
style_templates = {
    "photorealistic": "A high-resolution, lifelike stock photo of a <QUERY>",
    "cartoon": "An image in the style of Saturday morning cartoons featuring a <QUERY>",
    "minimalistic": "A photo emphasizing simplicity and minimalism, showing a <QUERY>"
}

# Apply to search
query = "dog"
styled_query = style_templates["cartoon"].replace("<QUERY>", query)
results = mq.index("my-index").search(styled_query)
```

### Personalized Search
```python
# Inject user context into search
def personalize_query(query, user_context, context_weight=0.2):
    composed_query = {query: 1.0}
    weight_per_item = context_weight / len(user_context)
    
    for item in user_context:
        composed_query[item] = weight_per_item
        
    return composed_query

# User's previous interactions
user_context = [
    "I love bold patterns",
    "https://example.com/previous_interaction.jpg"
]

results = mq.index("my-index").search(
    personalize_query("t-shirt", user_context)
)
```

### Reciprocal Rank Fusion
```python
# For structured indexes, use hybrid search:
results = mq.index("my-index").search(
    q="query",
    search_method="HYBRID",
    hybrid_parameters={
        "retrievalMethod": "disjunction",
        "rankingMethod": "rrf",
        "alpha": 0.5,
        "rrfK": 60
    }
)

# For unstructured indexes, implement RRF manually:
def reciprocal_rank_fusion(list1, list2, k=60):
    scores = {}
    
    for rank, item in enumerate(list1, start=1):
        scores[item] = scores.get(item, 0) + 1/(k + rank)
        
    for rank, item in enumerate(list2, start=1):
        scores[item] = scores.get(item, 0) + 1/(k + rank)
        
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# Get results from both methods
lexical_results = mq.index("my-index").search(q="query", search_method="LEXICAL")
tensor_results = mq.index("my-index").search(q="query")

# Combine using RRF
lexical_ids = [r["_id"] for r in lexical_results["hits"]]
tensor_ids = [r["_id"] for r in tensor_results["hits"]]
rrf_ids = reciprocal_rank_fusion(lexical_ids, tensor_ids)
```

### Duplicate Detection
```python
# Find duplicates of a document
doc = mq.index("my-index").get_document(
    document_id="doc1",
    expose_facets=True
)

# Extract vector from the document
vector = doc["_tensor_facets"][0]["_embedding"]

# Search for similar documents
duplicates = mq.index("my-index").search(
    q=None,
    context={"tensor": [{"vector": vector, "weight": 1.0}]},
    filter_string=f"_id:!doc1"  # Exclude the original
)

# Check scores - typically >0.99 for strong duplicates
potential_duplicates = [hit for hit in duplicates["hits"] if hit["_score"] > 0.99]
```

## Media & Multimodal Support

### Video and Audio Processing
```python
# Create index with media support
settings = {
    "treatUrlsAndPointersAsMedia": True,  # Required for video/audio
    "model": "LanguageBind/Video_V1.5_FT_Audio_FT_Image",  # Supports video, audio, image, text
    "videoPreprocessing": {
        "splitLength": 20,  # Video chunks in seconds
        "splitOverlap": 3   # Overlap between chunks
    },
    "audioPreprocessing": {
        "splitLength": 10,  # Audio chunks in seconds 
        "splitOverlap": 3   # Overlap between chunks
    }
}
mq.create_index("media-index", settings_dict=settings)

# Add video, audio, and images
mq.index("media-index").add_documents([
    {
        "_id": "video1",
        "video": "https://example.com/video.mp4"
    },
    {
        "_id": "audio1",
        "audio": "https://example.com/audio.wav"
    },
    {
        "_id": "multimodal1",
        "text": "Description of content",
        "image": "https://example.com/image.jpg",
        "video": "https://example.com/video.mp4"
    }
], tensor_fields=["video", "audio", "image", "text"])
```

### Media Download Configuration
```python
# Control threading for media downloads
mq.index("media-index").add_documents(
    documents,
    media_download_headers={"Authorization": "Bearer token"},  # Authentication
    media_download_thread_count=5  # Default is 5 for media, 20 for images only
)

# Configure media file size limits
docker_cmd = """
docker run --name marqo -p 8882:8882 \
    -e MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE=387973120 \
    -e MARQO_MAX_ADD_DOCS_VIDEO_AUDIO_FILE_SIZE=387973120 \
    marqoai/marqo:latest
"""
```

## Custom & Bring-Your-Own Models

### Using Custom Text Models
```python
# Load Hugging Face model
settings = {
    "model": "custom-sbert-model",
    "modelProperties": {
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # HF model
        "dimensions": 384,
        "type": "hf",
        "tokens": 128,
        "poolingMethod": "mean"  # Options: "mean", "cls"
    }
}
mq.create_index("custom-model-index", settings_dict=settings)

# Load private HF model
settings = {
    "model": "private-model",
    "modelProperties": {
        "modelLocation": {
            "hf": {
                "repoId": "username/private-model-name"
            },
            "authRequired": True
        },
        "dimensions": 768,
        "type": "hf"
    }
}
mq.create_index("private-model-index", settings_dict=settings)

# When using the model
results = mq.index("private-model-index").search(
    "query",
    model_auth={"hf": {"token": "YOUR_HF_TOKEN"}}
)
```

### Using Custom CLIP Models
```python
# Load custom CLIP model
settings = {
    "treatUrlsAndPointersAsImages": True,
    "model": "custom-clip-model",
    "modelProperties": {
        "name": "ViT-B-32",  # Model architecture
        "dimensions": 512,
        "url": "https://path-to-model-checkpoint.pt",
        "type": "open_clip"  # Or "clip" for OpenAI CLIP
    }
}
mq.create_index("custom-clip-index", settings_dict=settings)

# Advanced configuration options
settings["modelProperties"].update({
    "jit": False,               # JIT compilation
    "precision": "fp32",        # "fp32" or "fp16"
    "imagePreprocessor": "OpenCLIP",  # "SigLIP", "OpenAI", "OpenCLIP", "MobileCLIP", "CLIPA"
})
```

### Using Custom LanguageBind Models
```python
# Load custom LanguageBind model
settings = {
    "treatUrlsAndPointersAsMedia": True,
    "model": "custom-languagebind",
    "modelProperties": {
        "dimensions": 768,
        "type": "languagebind",
        "supportedModalities": ["text", "video", "audio", "image"],
        "modelLocation": {
            "video": {
                "hf": {"repoId": "path/to/video-model"}
            },
            "audio": {
                "url": "https://path-to-audio-model.zip"
            },
            "image": {
                "s3": {
                    "Bucket": "my-bucket",
                    "Key": "image-model.zip"
                }
            }
        }
    }
}
```

### No-Model Option (Use External Vectors)
```python
# Create index without model for custom vectors only
settings = {
    "model": "no_model",
    "modelProperties": {
        "dimensions": 384,
        "type": "no_model"
    }
}
mq.create_index("custom-vectors-only", settings_dict=settings)
```

## Index Management & Analytics

### Index Operations
```python
# List all indexes
indexes = mq.get_indexes()

# Get index settings
settings = mq.index("my-index").get_settings()

# Get index statistics
stats = mq.index("my-index").get_stats()
# Returns: {"numberOfDocuments": 123, "numberOfVectors": 456, ...}

# Check index health
health = mq.index("my-index").get_health()
# Returns status for index, backend, and inference components

# Delete index
mq.delete_index("my-index")
```

### Performance Analysis
```python
# Enable telemetry in client
mq = marqo.Client(return_telemetry=True)

# Get timing info in responses
search_response = mq.index("my-index").search("query")
timings = search_response["telemetry"]["timesMs"]
# Contains breakdown of times for various operations

# For REST API directly:
curl_cmd = 'curl -XPOST "http://localhost:8882/indexes/my-index/search?telemetry=true" -d \'{"q":"query"}\''
```

### Model Management
```python
# Get loaded models information
models_info = mq.get_models()
# Returns list of models and their devices

# Get CUDA information
cuda_info = mq.get_cuda_info()
# Returns CUDA device stats if available

# Preload models at startup
docker_cmd = """
docker run --name marqo -p 8882:8882 \
    -e 'MARQO_MODELS_TO_PRELOAD=["hf/e5-base-v2", "open_clip/ViT-B-32/laion2b_s34b_b79k"]' \
    marqoai/marqo:latest
"""
```

### Calculate Recall
```python
def calculate_recall(mq, index_name, query, limit=10):
    # Run approximate search (default)
    approx_results = mq.index(index_name).search(q=query, limit=limit)
    approx_ids = [hit["_id"] for hit in approx_results["hits"]]
    
    # Run exact search (slower but more accurate)
    exact_results = mq.index(index_name).search(
        q=query, limit=limit, approximate=False
    )
    exact_ids = [hit["_id"] for hit in exact_results["hits"]]
    
    # Calculate recall
    intersection = set(approx_ids).intersection(exact_ids)
    recall = len(intersection) / len(exact_ids)
    
    return recall
```

## System Configuration & Deployment

### Resource Configuration
```bash
# Configure Marqo resource limits
docker run --name marqo -p 8882:8882 \
    -e MARQO_MAX_DOC_BYTES=100000 \
    -e MARQO_MAX_RETRIEVABLE_DOCS=10000 \
    -e MARQO_MAX_CUDA_MODEL_MEMORY=4 \
    -e MARQO_MAX_CPU_MODEL_MEMORY=4 \
    -e MARQO_MAX_VECTORISE_BATCH_SIZE=16 \
    -e MARQO_MAX_DOCUMENTS_BATCH_SIZE=128 \
    -e MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST=5 \
    marqoai/marqo:latest
```

### GPU Configuration
```bash
# Run with GPU support
docker run --name marqo --gpus all -p 8882:8882 marqoai/marqo:latest

# Enable video GPU acceleration
docker run --name marqo --gpus all -p 8882:8882 \
    -e "MARQO_ENABLE_VIDEO_GPU_ACCELERATION=TRUE" \
    marqoai/marqo:latest
```

### Throttling & Concurrency
```bash
# Configure concurrent operations
docker run --name marqo -p 8882:8882 \
    -e MARQO_ENABLE_THROTTLING=TRUE \
    -e MARQO_MAX_CONCURRENT_INDEX=8 \
    -e MARQO_MAX_CONCURRENT_SEARCH=8 \
    -e MARQO_MAX_CONCURRENT_PARTIAL_UPDATE=100 \
    marqoai/marqo:latest
```

### Data Persistence
```bash
# Create and use a named volume
docker volume create --name opt_vespa_var
docker run --name marqo -p 8882:8882 \
    -v opt_vespa_var:/opt/vespa/var \
    marqoai/marqo:latest

# Optional: Mount logs directory
docker run --name marqo -p 8882:8882 \
    -v opt_vespa_var:/opt/vespa/var \
    -v $(pwd)/logs:/opt/vespa/logs \
    marqoai/marqo:latest
```

### External Vespa Configuration
```bash
# Run with external Vespa backend
docker run --name marqo -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e VESPA_CONFIG_URL="http://host.docker.internal:19071" \
    -e VESPA_DOCUMENT_URL="http://host.docker.internal:8080" \
    -e VESPA_QUERY_URL="http://host.docker.internal:8080" \
    -e ZOOKEEPER_HOSTS="host.docker.internal:2181" \
    marqoai/marqo:latest
```

### Caching Configuration
```bash
# Configure inference cache
docker run --name marqo -p 8882:8882 \
    -e MARQO_INFERENCE_CACHE_SIZE=20 \
    -e MARQO_INFERENCE_CACHE_TYPE=LRU \
    marqoai/marqo:latest
```

## Reranking & Advanced Search Features

### Image Reranking with OWL-ViT
```python
# Enable image localization and reranking
results = mq.index("image-index").search(
    "dog in the garden",
    searchable_attributes=["image_field"],  # Must include image field first
    reranker="owl/ViT-B/32"  # Options: "owl/ViT-B/32", "owl/ViT-B/16", "owl/ViT-L/14"
)

# Results include bounding box coordinates in _highlights field
# Format: [x1, y1, x2, y2] - coordinates of detected object
```

### Advanced HNSW Parameters
```python
# Detailed HNSW configuration
settings = {
    "annParameters": {
        "spaceType": "prenormalized-angular",  # For normalized vectors
        # Other options: "angular", "euclidean", "dotproduct"
        "parameters": {
            "efConstruction": 512,  # Higher = better quality index, slower indexing
                                    # Range: 2-4096, default: 512
            "m": 16,               # Higher = better recall, more memory
                                    # Range: 2-100, default: 16
        }
    }
}

# Control search quality/speed tradeoff
results = mq.index("my-index").search(
    "query",
    efSearch=2000,     # Higher = better recall, slower search (default: 2000)
    approximate=True   # Set False for exact (exhaustive) search
)

# Calculate performance metrics
recall = calculate_average_recall(mq, "my-index", ["query1", "query2"], limit=100)
```

## Local File & Image Handling

### Working with Local Images
```python
# Option 1: Simple HTTP Server
"""
1. Navigate to image directory: cd /path/to/images/
2. Start HTTP server: python3 -m http.server 8222
3. Reference images as: http://host.docker.internal:8222/image.jpg
"""

# Option 2: Mount local directory to Docker
"""
docker run --name marqo --mount type=bind,source=/path/to/images/,target=/path/to/images/ -p 8882:8882 marqoai/marqo:latest
"""

# Add documents with local images
docs = [{
    "_id": "doc1",
    "image": "http://host.docker.internal:8222/image.jpg",  # Option 1
    # "image": "/path/to/images/image.jpg",                 # Option 2
    "text": "Image description"
}]
```

## Troubleshooting

### Common Issues & Solutions
```python
# Problem: Insufficient memory
# Solution: Use smaller models or increase memory limits
docker_cmd = """
docker run --name marqo -p 8882:8882 \\
    -e MARQO_MAX_CUDA_MODEL_MEMORY=8 \\
    -e MARQO_MAX_CPU_MODEL_MEMORY=8 \\
    marqoai/marqo:latest
"""

# Problem: Slow image/media downloads
# Solution: Adjust thread count
mq.index("my-index").add_documents(
    documents,
    media_download_thread_count=3  # Lower for stability, higher for speed
)

# Problem: Search not finding expected results
# Solution: Check prefixes for E5 models
mq.index("e5-index").search(
    "query text",
    text_query_prefix="query: "  # Default prefix for E5 models
)

# Problem: Batch indexing failures
# Solution: Reduce batch size
mq.index("my-index").add_documents(
    documents,
    client_batch_size=16  # Default is larger, reduce for complex documents
)

# Problem: Out of disk space 
# Solution: Check Docker volume usage
"""
docker system df
docker volume prune  # Remove unused volumes
"""
```

### Version Compatibility & Upgrades
```bash
# Check Marqo version
curl -XGET 'http://localhost:8882/'
# Returns: {"message":"Welcome to Marqo", "version":"X.Y.Z"}

# Upgrade while preserving data
# For Marqo 2.9+
docker rm -f marqo
docker run --name marqo -it -p 8882:8882 \
    -v opt_vespa_var:/opt/vespa/var \
    marqoai/marqo:latest

# For Marqo before 2.9 -> 2.9+
docker volume create --name opt_vespa_var
docker run --rm -it --entrypoint='' \
    -v old_volume_name:/opt/vespa_old \
    -v opt_vespa_var:/opt/vespa/var \
    marqoai/marqo:latest sh -c "cd /opt/vespa_old/var ; cp -a . /opt/vespa/var"
```

## Additional API Operations

### Health & Monitoring
```python
# Get Marqo server info
info = mq.index("any-index").get_marqo()

# Get health of specific index
health = mq.index("my-index").get_health()
# Contains status of index, backend, and inference components

# Monitor CUDA usage
cuda_info = mq.get_cuda_info()
```

### Working with the Embed API
```python
# Generate embeddings with document prefixes
embeddings = mq.index("my-index").embed(
    content=["Document text to embed"],
    content_type="document"  # Adds document prefix, e.g. "passage: "
)

# Generate embeddings with query prefixes
embeddings = mq.index("my-index").embed(
    content=["Query text to embed"],
    content_type="query"  # Adds query prefix, e.g. "query: "
)

# Generate embeddings without prefixes
embeddings = mq.index("my-index").embed(
    content=["Text to embed as-is"],
    content_type=None  # No prefix added
)
```

## Best Practices by Use Case

### E-commerce Search
```python
# Create structured index
settings = {
    "type": "structured",
    "model": "Marqo/marqo-fashionCLIP",  # For fashion items
    "treatUrlsAndPointersAsImages": True,
    "allFields": [
        {"name": "title", "type": "text", "features": ["lexical_search", "filter"]},
        {"name": "description", "type": "text", "features": ["lexical_search"]},
        {"name": "category", "type": "text", "features": ["filter"]},
        {"name": "price", "type": "float", "features": ["filter", "score_modifier"]},
        {"name": "image", "type": "image_pointer"},
        {"name": "popularity", "type": "float", "features": ["score_modifier"]},
        {"name": "product_field", "type": "multimodal_combination", 
         "dependentFields": {"title": 0.3, "description": 0.1, "image": 0.6}}
    ],
    "tensorFields": ["product_field"]
}

# Example search
results = mq.index("ecommerce").search(
    "blue summer dress",
    filter_string="category:clothing AND price:[20 TO 100]",
    score_modifiers={
        "multiply_score_by": [{"field_name": "popularity", "weight": 1.2}]
    },
    searchable_attributes=["product_field"],
    limit=20
)
```

### Knowledge Base / Document Search
```python
# Create index with effective text chunking
settings = {
    "model": "hf/e5-large-v2",  # Highest quality for text
    "textPreprocessing": {
        "splitLength": 4,        # 4 sentences per chunk
        "splitOverlap": 1,       # 1 sentence overlap
        "splitMethod": "sentence"
    }
}

# Add documents with metadata
docs = [{
    "_id": "doc1",
    "title": "Document Title",
    "content": "Long document content with multiple paragraphs...",
    "metadata": {
        "author": "Author Name",
        "date": "2023-01-01",
        "category": "Technical"
    }
}]

mq.index("knowledge-base").add_documents(
    docs,
    tensor_fields=["title", "content"]
)

# Search with hybrid approach
results = mq.index("knowledge-base").search(
    "technical concept explanation",
    search_method="HYBRID",
    hybrid_parameters={
        "retrievalMethod": "disjunction",
        "rankingMethod": "rrf",
        "alpha": 0.7  # Favor semantic understanding
    },
    filter_string="metadata.category:Technical"
)
```

### Content Moderation
```python
# Create index for content moderation
settings = {
    "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",  # Strong image understanding
    "treatUrlsAndPointersAsImages": True
}
mq.create_index("moderation", settings_dict=settings)

# Add content to moderate
mq.index("moderation").add_documents(
    [{"_id": "content1", "image": "https://example.com/image.jpg"}],
    tensor_fields=["image"]
)

# Check for problematic content
moderation_terms = {
    "explicit adult content": 1.0,
    "violent imagery": 1.0,
    "hate symbols": 1.0
}

# For each item to check
for content_id in content_ids:
    results = mq.index("moderation").search(
        moderation_terms,
        filter_string=f"_id:{content_id}",
        limit=1
    )
    
    # Check score - higher means more similar to problematic content
    score = results["hits"][0]["_score"] if results["hits"] else 0
    if score > 0.25:  # Threshold depends on your use case
        print(f"Content {content_id} may be problematic (score: {score})")
```

## Advanced Optimizations

### Memory Optimization
```python
# Reduce memory usage
settings = {
    "model": "hf/e5-small-v2",  # Smallest model (134MB)
    "textPreprocessing": {
        "splitLength": 5,  # Fewer chunks per document
        "splitOverlap": 0  # No overlap
    }
}

# Selective tensor fields
mq.index("my-index").add_documents(
    documents,
    tensor_fields=["main_content_only"]  # Only vectorize essential fields
)

# Limit preloaded models
docker_cmd = """
docker run --name marqo -p 8882:8882 \\
    -e 'MARQO_MODELS_TO_PRELOAD=[]' \\
    marqoai/marqo:latest
"""
```

### Storage Requirements Estimation
```
# Estimated storage per document:
- Text only (default model): ~5-10 KB
- Image (CLIP model): ~15-20 KB
- Document with multiple fields: ~10-20 KB

# Storage reduction strategies:
1. Include specific tensor_fields only
2. Increase splitLength in textPreprocessing
3. Use smaller dimension models
```

### Performance Tuning
```python
# Configure Vespa parameters for search performance
docker_cmd = """
docker run --name marqo -p 8882:8882 \\
    -e VESPA_POOL_SIZE=20 \\
    -e VESPA_SEARCH_TIMEOUT_MS=2000 \\
    marqoai/marqo:latest
"""

# Tune operational parameters
settings = {
    "annParameters": {
        "parameters": {
            "efConstruction": 256,  # Lower = faster indexing, less quality
            "m": 16
        }
    }
}

# Default parameters by importance
"""
For best accuracy (slower): 
- efSearch: 4000, efConstruction: 1024, m: 32

For best performance (less accurate):
- efSearch: 100, efConstruction: 128, m: 8

For balanced approach (default):
- efSearch: 2000, efConstruction: 512, m: 16
"""
```

## Edge Cases & Advanced Scenarios

### Working with Map Fields
```python
# Using map fields as score modifiers (structured index)
settings = {
    "type": "structured",
    "allFields": [
        {"name": "title", "type": "text", "features": ["lexical_search"]},
        {"name": "score_map", "type": "map<text, float>", "features": ["score_modifier"]}
    ],
    "tensorFields": ["title"]
}

# Add documents with map fields
mq.index("map-index").add_documents([{
    "_id": "doc1",
    "title": "Product title",
    "score_map": {
        "quality": 0.9,
        "popularity": 0.7,
        "freshness": 0.5
    }
}])

# Search with map field score modifier
results = mq.index("map-index").search(
    "product",
    score_modifiers={
        "add_to_score": [{"field_name": "score_map.quality", "weight": 0.5}]
    }
)
```

### Handling Large Text Documents
```python
# For very large documents (books, articles, etc.)
settings = {
    "textPreprocessing": {
        "splitLength": 5,       # More sentences per chunk
        "splitOverlap": 2,      # Good overlap for context
        "splitMethod": "sentence"
    }
}

# Process large documents in chunks
def add_large_document(mq, index_name, document, chunk_size=5000):
    """Add a very large document by breaking it into manageable pieces"""
    text = document["content"]
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        doc_chunk = document.copy()
        doc_chunk["content"] = chunk
        doc_chunk["_id"] = f"{document['_id']}_{i}"
        
        mq.index(index_name).add_documents(
            [doc_chunk],
            tensor_fields=["content"]
        )
```

## Summary of Key Concepts

### Vector Search vs. Lexical Search
- **Vector/Tensor Search**: Semantic understanding, finds conceptually similar items
- **Lexical Search**: Keyword matching, finds exact text matches
- **Hybrid Search**: Combines both methods using Reciprocal Rank Fusion

### Index Types
- **Unstructured**: Flexible schema, define tensor fields during indexing
- **Structured**: Fixed schema, better performance, explicit field features

### Models
- **Text**: E5 models (small/base/large), multilingual, Stella models
- **Image**: CLIP models, OpenCLIP, FashionCLIP, multilingual CLIP
- **Video/Audio**: LanguageBind models for multimodal processing
- **Custom**: Support for your own models from Hugging Face or other sources

### Document Processing
- **Text**: Chunking by sentence, word, character, or passage
- **Image**: Various patching methods (simple, DINO, YOLO, etc.)
- **Video/Audio**: Chunking by time segments with overlap

### Search Features
- **Filtering**: Complex DSL with boolean operators, ranges, array filters
- **Score Modifiers**: Boost results based on metadata fields
- **Recommendations**: Find similar items to existing documents
- **Personalization**: Inject context into queries using weighted terms

### Deployment Considerations
- **Resources**: Configure memory limits, threading, batch sizes
- **Storage**: Persistent volumes for data, estimate storage requirements
- **Performance**: GPU acceleration, HNSW parameter tuning, caching
- **Monitoring**: Telemetry, health checks, stats

# Part 6: Additional Important Features & Details

## Field Types & Special Objects

### Multimodal Combination Objects
```python
# More details on multimodal combination fields
doc = {
    "_id": "doc1",
    "text_field": "Product description",
    "image_field": "https://example.com/image.jpg",
    # The multimodal_field will be created via mappings
}

# When adding document
mq.index("my-index").add_documents(
    [doc],
    mappings={
        "multimodal_field": {
            "type": "multimodal_combination",
            "weights": {"text_field": 0.3, "image_field": 0.7}
        }
    },
    tensor_fields=["multimodal_field"]
)

# Important notes:
# - Only generates a single vector (no chunking)
# - Child fields can be used for lexical search/filtering
# - All child fields and content must be strings
```

### Custom Vector Objects in Detail
```python
# Required format for custom vector objects
doc = {
    "_id": "doc1",
    "vector_field": {
        "vector": [0.1, 0.2, ..., 0.5],  # Required, must match model dimensions
        "content": "Optional text"        # Optional, for lexical search/filtering
    }
}

# Important constraints:
# - Cannot use prenormalized-angular space type with custom vectors
# - Cannot be dependent fields in multimodal combinations
# - No normalization is performed on custom vectors
# - Zero magnitude vectors will cause errors
```

### Map Fields for Score Modifiers
```python
# Create structured index with map fields
settings = {
    "type": "structured",
    "allFields": [
        {"name": "map_score_mods", "type": "map<text, float>", "features": ["score_modifier"]},
        {"name": "map_score_mods_int", "type": "map<text, int>", "features": ["score_modifier"]}
        # Other supported types: map<text, long>, map<text, double>
    ]
}

# Add documents with map fields
mq.index("map-fields").add_documents([{
    "_id": "doc1",
    "map_score_mods": {"a": 0.5, "b": 0.7, "c": 0.9},
    "map_score_mods_int": {"x": 1, "y": 2, "z": 3}
}])

# Use dot notation to access subfields in score modifiers
results = mq.index("map-fields").search(
    "query",
    score_modifiers={
        "add_to_score": [
            {"field_name": "map_score_mods.a", "weight": 2.0},
            {"field_name": "map_score_mods_int.z", "weight": 0.5}
        ]
    }
)
```

### Array Fields
```python
# Add document with array fields
mq.index("my-index").add_documents([{
    "_id": "doc1",
    "tags": ["summer", "sale", "new"],  # Array of strings
    "title": "Summer collection"
}], tensor_fields=["title"])  # Array fields cannot be tensor fields

# Filter on array fields
results = mq.index("my-index").search(
    "summer clothes",
    filter_string="tags:summer AND tags:new"  # Must contain both values
)

# For structured indexes
settings = {
    "type": "structured",
    "allFields": [
        {"name": "title", "type": "text", "features": ["lexical_search"]},
        {"name": "tags", "type": "array<text>", "features": ["filter"]},
        {"name": "numbers", "type": "array<int>", "features": ["filter"]}
        # Other supported: array<float>, array<long>, array<double>
    ]
}
```

## Advanced Search Techniques

### Pagination with Deep Results
```python
# Retrieve large result sets with pagination
def paginated_search(mq, index, query, page_size=10, max_pages=10):
    all_results = []
    
    for page in range(max_pages):
        offset = page * page_size
        
        # Important: efSearch must be > limit+offset
        results = mq.index(index).search(
            q=query,
            limit=page_size,
            offset=offset,
            efSearch=page_size + offset + 100  # Add buffer
        )
        
        hits = results["hits"]
        all_results.extend(hits)
        
        # Stop if fewer results than page_size (reached end)
        if len(hits) < page_size:
            break
    
    return all_results

# Pagination limitations:
# - limit + offset must be ≤ 10,000
# - efSearch must be > limit + offset
# - Some results may be duplicated or skipped between pages
```

### Working with Tensor Facets
```python
# Retrieve documents with vectors
docs = mq.index("my-index").get_documents(
    document_ids=["doc1", "doc2"],
    expose_facets=True
)

# Extract vectors for reuse
vectors = []
for doc in docs["results"]:
    if "_tensor_facets" in doc and doc["_tensor_facets"]:
        for facet in doc["_tensor_facets"]:
            vectors.append({
                "id": doc["_id"],
                "field": facet["_field_name"],
                "vector": facet["_embedding"]
            })

# Use vector in context search
if vectors:
    results = mq.index("my-index").search(
        q=None,  # No text query
        context={"tensor": [{"vector": vectors[0]["vector"], "weight": 1.0}]}
    )
```

### Advanced Global Score Modifiers
```python
# Global score modifiers with hybrid search and reranking
results = mq.index("my-index").search(
    q="query text",
    search_method="HYBRID",
    hybrid_parameters={
        "retrievalMethod": "disjunction",
        "rankingMethod": "rrf",
        # Field-specific score modifiers for tensor component
        "scoreModifiersTensor": {
            "add_to_score": [{"field_name": "freshness", "weight": 0.05}]
        },
        # Field-specific score modifiers for lexical component
        "scoreModifiersLexical": {
            "multiply_score_by": [{"field_name": "relevance", "weight": 1.2}]
        }
    },
    # Global score modifiers applied after fusion
    score_modifiers={
        "add_to_score": [{"field_name": "priority", "weight": 0.1}]
    },
    # Number of results to rerank with global score modifiers
    rerankDepth=50,
    # Number of results to return
    limit=20
)
```

## Configuration Details

### Normalization & Vector Spaces
```python
# Vector normalization options
settings = {
    "normalizeEmbeddings": True,  # Vectors normalized to unit length
    "vectorNumericType": "float",  # Options: "float" (default)
    "annParameters": {
        # Space types depend on normalizeEmbeddings
        "spaceType": "prenormalized-angular",  # For normalizeEmbeddings=True
        # Other options when normalizeEmbeddings=False:
        # "angular", "euclidean", "dotproduct", "geodegrees", "hamming"
    }
}

# Impact on operations:
# - For normalizeEmbeddings=True: Use "slerp" interpolation in recommend
# - For normalizeEmbeddings=False: Use "lerp" interpolation in recommend
```

### Text Chunk & Query Prefixes
```python
# Index settings for prefixes
settings = {
    "textChunkPrefix": "passage: ",  # Default for E5 models
    "textQueryPrefix": "query: "      # Default for E5 models
}

# Override during operations
mq.index("my-index").add_documents(
    documents,
    text_chunk_prefix="override passage: "
)

mq.index("my-index").search(
    "query text",
    text_query_prefix="override query: "
)

# When generating embeddings
embeddings = mq.index("my-index").embed(
    content=["text to embed"],
    content_type="query"  # Options: "query", "document", None
)
```

### Advanced CUDA Configuration
```bash
# Check CUDA device availability and utilization
curl -XGET 'http://localhost:8882/device/cuda'

# Response format:
# {
#   "cuda_devices": [
#     {
#       "device_id": 0,
#       "device_name": "Tesla T4",
#       "memory_used": "1.7 GiB",
#       "total_memory": "14.6 GiB",
#       "utilization": "11.0 %",
#       "memory_used_percent": "25.0 %"
#     }
#   ]
# }
```

### Optimizing Media Processing
```python
# Video preprocessing options
settings = {
    "videoPreprocessing": {
        "splitLength": 20,  # Chunk size in seconds
        "splitOverlap": 3   # Overlap in seconds
    }
}

# Audio preprocessing options
settings = {
    "audioPreprocessing": {
        "splitLength": 10,  # Chunk size in seconds
        "splitOverlap": 3   # Overlap in seconds
    }
}

# Enable video GPU acceleration (requires NVIDIA drivers 550.54.14+)
docker_cmd = """
docker run --name marqo --gpus all -p 8882:8882 \\
    -e "MARQO_ENABLE_VIDEO_GPU_ACCELERATION=TRUE" \\
    marqoai/marqo:latest
"""
```

## Edge Cases & Gotchas

### ZeroDivisionError in SLERP
```python
# Avoid weights that sum to zero when using SLERP
# This will cause an error:
mq.index("my-index").recommend(
    documents={"doc1": 0.5, "doc2": -0.5}  # Weights sum to zero
)

# Fix: Ensure weights don't sum to zero
# Either add more documents with non-zero weights
# Or use a different interpolation method:
mq.index("my-index").recommend(
    documents={"doc1": 0.5, "doc2": -0.5},
    interpolation_method="lerp"  # Use linear interpolation instead
)
```

### Model Loading Authorization
```python
# Working with private models
settings = {
    "model": "private-model",
    "modelProperties": {
        "model_location": {
            "s3": {
                "Bucket": "my-bucket",
                "Key": "model.pt",
            },
            "auth_required": True
        },
        "dimensions": 512,
        "type": "open_clip"
    }
}
mq.create_index("private-model-index", settings_dict=settings)

# Provide auth during operations
auth = {
    "s3": {
        "aws_access_key_id": "ACCESS_KEY",
        "aws_secret_access_key": "SECRET_KEY"
    }
}

# Must provide auth for all operations using the model
mq.index("private-model-index").add_documents(documents, model_auth=auth)
mq.index("private-model-index").search("query", model_auth=auth)
mq.index("private-model-index").embed("content", model_auth=auth)
```

### Exact Match vs Lexical Search
```python
# Exact match requires double quotes
results = mq.index("my-index").search(
    q='query with "exact phrase"',
    search_method="LEXICAL"
)

# Important quote syntax rules:
# - Quotes must have spaces before/after
# - Unescaped/unbalanced quotes are treated as whitespace
# - Escape quotes with backslash (\\")
# - Every 2 quotes are paired to create a required term
```

### Inference Cache Configuration
```bash
# Configure inference cache
docker run --name marqo -p 8882:8882 \
    -e "MARQO_INFERENCE_CACHE_SIZE=20" \
    -e "MARQO_INFERENCE_CACHE_TYPE=LRU" \
    marqoai/marqo:latest

# Cache notes:
# - Caches query-embedding pairs to improve search latency
# - LRU: Least Recently Used eviction policy
# - LFU: Least Frequently Used eviction policy
# - Only applies to search queries, not document indexing
# - Size is measured by number of query-embedding pairs
```

## Additional Model Information

### Model Selection Guide

#### Text Models (by size)
- **Small/Fast**: 
  - hf/e5-small-v2 (384d, ~134MB)
  - hf/bge-small-en-v1.5 (384d)
  - sentence-transformers/all-MiniLM-L6-v2 (384d)
- **Balanced**: 
  - hf/e5-base-v2 (768d, default)
  - hf/bge-base-en-v1.5 (768d)
  - hf/multilingual-e5-base (768d)
- **Large/Accurate**: 
  - hf/e5-large-v2 (1024d)
  - hf/GIST-large-Embedding-v0 (1024d)
  - hf/multilingual-e5-large (1024d)

#### Image Models (by quality/speed)
- **Fast/Basic**: 
  - open_clip/ViT-B-32/laion2b_s34b_b79k (512d)
  - fp16/ViT-B/32 (512d, requires GPU)
- **Balanced**: 
  - open_clip/ViT-L-14/laion2b_s32b_b82k (768d)
  - Marqo/marqo-fashionCLIP (512d, fashion-specific)
- **Highest Quality**: 
  - open_clip/ViT-H-14/laion2b_s32b_b79k (1024d)
  - open_clip/ViT-g-14/laion2b_s34b_b88k (1536d)

#### Language Support
- **Multilingual Text**: 
  - hf/multilingual-e5-base (supports 100+ languages)
  - sentence-transformers/stsb-xlm-r-multilingual
- **Multilingual Image-Text**: 
  - visheratin/nllb-clip-base-siglip (supports 200+ languages)
  - open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k

#### Video/Audio Models
- **Full Multimodal**: LanguageBind/Video_V1.5_FT_Audio_FT_Image (video+audio+image+text)
- **Video+Text**: LanguageBind/Video_V1.5_FT (video+text)
- **Audio+Text**: LanguageBind/Audio_FT (audio+text)

### Model Prefixing Details
```python
# Default prefixes by model
prefixes = {
    "e5": {"query": "query: ", "document": "passage: "},
    "e5-instruct": {"query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "},
    "bge": {"query": "Represent this sentence for searching relevant passages: "}
}

# Set custom prefix at index creation
settings = {
    "textChunkPrefix": "custom document prefix: ",
    "textQueryPrefix": "custom query prefix: "
}

# Override at runtime
mq.index("my-index").add_documents(
    documents, 
    text_chunk_prefix="override prefix: "
)

mq.index("my-index").search(
    "query",
    text_query_prefix="override prefix: "
)
```

## Docker Advanced Configuration

### Custom Storage Location
```bash
# Change default Docker storage location
sudo mkdir /new/storage/path
sudo systemctl stop docker
sudo mount --rbind /new/storage/path /var/lib/docker
sudo systemctl start docker
```

### Running Multiple Marqo Instances
```bash
# Run multiple instances on different ports
docker run --name marqo1 -p 8882:8882 marqoai/marqo:latest
docker run --name marqo2 -p 8883:8882 marqoai/marqo:latest

# With different configurations
docker run --name marqo-text -p 8882:8882 \
    -e 'MARQO_MODELS_TO_PRELOAD=["hf/e5-base-v2"]' \
    marqoai/marqo:latest

docker run --name marqo-image -p 8883:8882 \
    -e 'MARQO_MODELS_TO_PRELOAD=["open_clip/ViT-B-32/laion2b_s34b_b79k"]' \
    marqoai/marqo:latest
```

### Controlling Log Level
```bash
# Set log level
docker run --name marqo -p 8882:8882 \
    -e MARQO_LOG_LEVEL=WARNING \
    marqoai/marqo:latest
    
# Available levels: DEBUG, INFO, WARNING, ERROR
# Default: INFO
```

## Additional Best Practices

### Working with Large Documents
```python
# Strategy 1: Custom chunking
def chunk_large_document(doc, chunk_size=4000, overlap=1000):
    text = doc["content"]
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        end = min(i + chunk_size, len(text))
        chunks.append(text[i:end])
        
        if end == len(text):
            break
    
    return chunks

# Strategy 2: Utilize Marqo's text preprocessing
settings = {
    "textPreprocessing": {
        "splitLength": 3,        # Sentences per chunk
        "splitOverlap": 1,       # Sentences overlapping
        "splitMethod": "passage" # For document with natural breaks
    }
}

# Strategy 3: Identify important fields only
mq.index("my-index").add_documents(
    documents,
    tensor_fields=["title", "abstract"]  # Skip full content for vectors
)
```

### Filtering for Security/Multi-tenancy
```python
# Implement tenant isolation using filters
def search_for_tenant(mq, index, query, tenant_id):
    return mq.index(index).search(
        q=query,
        filter_string=f"tenant_id:{tenant_id}"
    )

# Add tenant information to all documents
for doc in documents:
    doc["tenant_id"] = current_tenant_id

mq.index("shared-index").add_documents(documents)
```

### Handling Sparse Data
```python
# When many documents have missing fields
def add_with_defaults(mq, index, documents, defaults):
    # Add default values for missing fields
    for doc in documents:
        for field, default_value in defaults.items():
            if field not in doc:
                doc[field] = default_value
    
    mq.index(index).add_documents(documents)

# Example usage
defaults = {
    "description": "",
    "category": "uncategorized",
    "rating": 0.0
}
add_with_defaults(mq, "my-index", documents, defaults)
```

## Resource Planning & Hardware Requirements

### Recommended Hardware
```
# Minimal setup (development):
- CPU: 2+ cores
- RAM: 8+ GB
- Storage: 20+ GB
- Docker memory limit: 6+ GB

# Production setup:
- CPU: 4+ cores
- RAM: 16+ GB 
- Storage: 100+ GB (depends on dataset)
- GPU: NVIDIA GPU with 8+ GB VRAM (optional but recommended)

# For video/audio processing:
- RAM: 16+ GB
- GPU: NVIDIA GPU with 8+ GB VRAM
- NVIDIA drivers: 550.54.14+
```

### Storage Estimation
```
# Approximate storage per document:
- Text only: 5-10 KB per document
- Image only: 15-20 KB per document
- Text with multiple fields: 10-20 KB per document
- Video/audio: Depends on chunk settings

# For 1M documents:
- Text-only index: ~5-10 GB
- Image-only index: ~15-20 GB
- Mixed content index: ~10-30 GB
```
# AutoCAD Documentation MCP Server

An MCP (Model Context Protocol) server that exposes Autodesk AutoCAD SDK documentation taken from CHM files distributed together with the official Autodesk Autocad SDK documenation, enabling AI agents to query and retrieve structured documentation content.

## Architecture

```
CHM Files → 7-zip → HTML → Parser → Chunker → Indexer → MCP Server → AI Agent
    ↓         ↓       ↓        ↓        ↓         ↓          ↓
  arxmgd.chm  Extract  Topics  Chunks  FAISS+BM25  Tools   AI Agent
```

### Components

- **TOC Parser**: Parses HHC/HHK files for table of contents and index
- **Topic Parser**: Extracts HTML content using BeautifulSoup
- **Link Graph Builder**: Builds parent/children/see_also relationships
- **Heading-Aware Chunker**: Splits content while preserving document structure
- **Hybrid Indexer**: Builds FAISS vector index and BM25 lexical index
- **MCP Server**: Exposes search tools via Model Context Protocol

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mcp-autocad-api
   ```

2. **Install dependencies**:
   ```bash
   # For ingestion (heavy dependencies)   
   pip install -r requirements-ingestion.txt # (sorry it is dirty)
   
   # OR for runtime only (lightweight)
   pip install -r requirements-runtime.txt
   ```

3. **Extract CHM files** using 7-zip:
   ```bash
   # Extract main .NET API documentation
   7z x data/chm/arxmgd.chm -odata/chm/arxmgd/
   
   # Extract other CHM files as needed
   7z x data/chm/arxdev.chm -odata/chm/arxdev/
   ```

## Usage

### 1. Ingest CHM Documentation

**First, install ingestion dependencies:**
```bash
pip install -r requirements-ingestion.txt
```

**Then build search indices:**
```bash
# Ingest default source (arxmgd)
python -m ingester.ingest

# Ingest specific source
python -m ingester.ingest --source arxdev

# List available sources
python -m ingester.ingest --list-sources
```

### 2. Run the MCP Server

**Install lightweight runtime dependencies:**
```bash
pip install -r requirements-runtime.txt
```

**Start the server:**
```bash
# Run with default source (arxmgd)
python -m server.mcp_server

# Run with specific source
python -m server.mcp_server --source arxdev
```

## Example Queries

Here are example queries that AI agents can use:

- **"What class represents a revision cloud?"**
- **"What are the ways to construct AcDbArc?"**
- **"What methods are available on AcDbBlockReference?"**
- **"How does AcDbDimension store information about text size?"**

## MCP Tools

The server exposes the following tools:

### `docs.search`
Search documentation using hybrid semantic and lexical search.

**Parameters**:
- `query` (required): Search query
- `k` (optional): Number of results (default: 10)
- `source` (optional): CHM source to search (arxmgd, arxdev, etc.)

**Example**:
```json
{
  "query": "revision cloud",
  "k": 5,
  "source": "arxmgd"
}
```

### `docs.get`
Get full content of a documentation topic by ID.

**Parameters**:
- `id` (required): Document chunk ID
- `format` (optional): Content format ("text" or "html", default: "text")
- `source` (optional): CHM source filter

### `docs.toc`
Get table of contents for a CHM source.

**Parameters**:
- `source` (optional): CHM source (default: "arxmgd")

### `docs.neighbors`
Get related documentation (parent, children, see also).

**Parameters**:
- `id` (required): Document chunk ID
- `source` (optional): CHM source filter

### `docs.list_sources`
List available CHM documentation sources.

### `docs.health`
Get server health and version information.

## Project Structure

```
mcp-autocad-api/
├── data/
│   ├── chm/           # Input CHM files
│   └── index/         # FAISS + BM25 artifacts
├── ingester/
│   ├── models.py      # Data models
│   ├── chm_parser.py  # CHM file parser
│   ├── chunker.py     # Heading-aware chunking
│   ├── indexer.py     # Hybrid search indexer
│   └── ingest.py      # Main ingestion pipeline
├── server/
│   └── mcp_server.py  # MCP server implementation
├── tests/
│   └── test_queries.py # Test queries
├── requirements.txt
└── README.md
```

## Configuration

### Embedding Model

The default embedding model is `all-MiniLM-L6-v2`. You can change it in the ingestion pipeline:

```bash
python -m ingester.ingest --embedding-model "sentence-transformers/all-mpnet-base-v2"
```

### Chunking Parameters

Default chunking parameters:
- Target tokens: 1000
- Overlap tokens: 200
- Minimum chunk tokens: 200

These can be modified in `ingester/chunker.py`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Need MCP Server for your business ?

Contact me: https://dmytro-prototypes.net/

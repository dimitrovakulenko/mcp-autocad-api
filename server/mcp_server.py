#!/usr/bin/env python3
"""
MCP Server for AutoCAD SDK Documentation
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # Log to stderr to avoid interfering with MCP protocol
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info(f"Starting MCP Server")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Current working directory: {Path.cwd()}")
logger.info(f"Project root: {project_root}")
logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolRequest, CallToolResult, ListResourcesRequest, ListResourcesResult,
    ListToolsRequest, ListToolsResult, ReadResourceRequest, ReadResourceResult
)

from ingester.models import SourceType, SearchResult, NeighborInfo
from ingester.indexer import HybridIndexer
from ingester.link_graph import LinkGraphBuilder


class AutoCADMCPServer:
    """MCP Server for AutoCAD SDK documentation search"""
    
    def __init__(self, source: SourceType = SourceType.ARXMGD):
        self.source = source
        self.indexer = None
        self.link_graph = None
        self.server = Server("autocad-sdk-mcp")
        
        logger.info(f"Initializing AutoCADMCPServer with source: {source.value}")
        logger.info(f"Current working directory: {Path.cwd()}")
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="docs.search",
                    description="Search AutoCAD SDK documentation using hybrid semantic and lexical search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                                "default": 10
                            },
                            "source": {
                                "type": "string",
                                "description": "SDK documentation source to search (default: arxmgd)",
                                "enum": [s.value for s in SourceType],
                                "default": "arxmgd"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="docs.get",
                    description="Get full content of an SDK documentation topic by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Document chunk ID"
                            },
                            "format": {
                                "type": "string",
                                "description": "Content format (text or html)",
                                "enum": ["text", "html"],
                                "default": "text"
                            },
                            "source": {
                                "type": "string",
                                "description": "CHM source filter (default: arxmgd)",
                                "enum": [s.value for s in SourceType],
                                "default": "arxmgd"
                            }
                        },
                        "required": ["id"]
                    }
                ),
                Tool(
                    name="docs.toc",
                    description="Get table of contents for an SDK documentation source",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "CHM source to get TOC for",
                                "enum": [s.value for s in SourceType],
                                "default": "arxmgd"
                            }
                        }
                    }
                ),
                Tool(
                    name="docs.neighbors",
                    description="Get related SDK documentation (parent, children, see also)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Document chunk ID"
                            },
                            "source": {
                                "type": "string",
                                "description": "CHM source filter (default: arxmgd)",
                                "enum": [s.value for s in SourceType],
                                "default": "arxmgd"
                            }
                        },
                        "required": ["id"]
                    }
                ),
                Tool(
                    name="docs.list_sources",
                    description="List available SDK documentation sources",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="docs.health",
                    description="Get server health and version information",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "docs.search":
                    return await self._handle_search(arguments)
                elif name == "docs.get":
                    return await self._handle_get(arguments)
                elif name == "docs.toc":
                    return await self._handle_toc(arguments)
                elif name == "docs.neighbors":
                    return await self._handle_neighbors(arguments)
                elif name == "docs.list_sources":
                    return await self._handle_list_sources(arguments)
                elif name == "docs.health":
                    return await self._handle_health(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_search(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle search requests"""
        query = args.get("query", "")
        k = args.get("k", 10)
        source_str = args.get("source", "arxmgd")
        
        logger.info(f"Search request: query='{query}', k={k}, source='{source_str}'")
        
        if not query:
            return [TextContent(type="text", text="Error: Query is required")]
        
        source = SourceType(source_str)
        
        # Ensure indexer is loaded
        if not self.indexer or self.indexer.source != source:
            logger.info(f"Loading indexer for source: {source.value}")
            logger.info(f"Current working directory: {Path.cwd()}")
            
            # Check if index files exist - use absolute path from project root
            index_dir = project_root / "data" / "index" / source.value
            faiss_path = index_dir / "faiss.index"
            bm25_path = index_dir / "bm25.pkl"
            
            logger.info(f"Looking for index files in: {index_dir.absolute()}")
            logger.info(f"FAISS index path: {faiss_path.absolute()}")
            logger.info(f"BM25 index path: {bm25_path.absolute()}")
            logger.info(f"FAISS index exists: {faiss_path.exists()}")
            logger.info(f"BM25 index exists: {bm25_path.exists()}")
            
            # List directory contents for debugging
            if index_dir.exists():
                logger.info(f"Index directory contents: {list(index_dir.iterdir())}")
            else:
                logger.error(f"Index directory does not exist: {index_dir.absolute()}")
                error_msg = (
                    f"Error: Index directory does not exist at {index_dir.absolute()}\n"
                    f"Project root: {project_root.absolute()}\n"
                    f"Expected path: {index_dir.absolute()}\n"
                    f"Please run the ingestion process to create the index files."
                )
                return [TextContent(type="text", text=error_msg)]
            
            if not faiss_path.exists():
                logger.error(f"FAISS index not found at: {faiss_path.absolute()}")
                logger.error(f"Project root: {project_root.absolute()}")
                logger.error(f"Index dir: {index_dir.absolute()}")
                error_msg = (
                    f"Error: FAISS index not found at {faiss_path.absolute()}\n"
                    f"Project root: {project_root.absolute()}\n"
                    f"Index directory: {index_dir.absolute()}\n"
                    f"Please run the ingestion process to create the index files."
                )
                return [TextContent(type="text", text=error_msg)]
            
            if not bm25_path.exists():
                logger.error(f"BM25 index not found at: {bm25_path.absolute()}")
                logger.error(f"Project root: {project_root.absolute()}")
                logger.error(f"Index dir: {index_dir.absolute()}")
                error_msg = (
                    f"Error: BM25 index not found at {bm25_path.absolute()}\n"
                    f"Project root: {project_root.absolute()}\n"
                    f"Index directory: {index_dir.absolute()}\n"
                    f"Please run the ingestion process to create the index files."
                )
                return [TextContent(type="text", text=error_msg)]
            
            try:
                self.indexer = HybridIndexer(source=source)
                logger.info("Created HybridIndexer instance")
                self.indexer.load_indices()
                logger.info("Successfully loaded indices")
            except Exception as e:
                logger.error(f"Failed to load indices: {e}")
                import traceback
                logger.error(traceback.format_exc())
                error_msg = (
                    f"Error loading indices: {str(e)}\n"
                    f"Project root: {project_root.absolute()}\n"
                    f"Index directory: {index_dir.absolute()}\n"
                    f"FAISS path: {faiss_path.absolute()}\n"
                    f"BM25 path: {bm25_path.absolute()}\n"
                    f"Please check that the index files exist and are accessible."
                )
                return [TextContent(type="text", text=error_msg)]
        
        # Perform search
        results = self.indexer.search(query, k=k)
        
        # Format results
        if not results:
            return [TextContent(type="text", text="No results found")]
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            anchor_info = ""
            if result.chunk_index is not None:
                anchor_info = f" (Chunk {result.chunk_index + 1})"
            
            formatted_results.append(
                f"{i}. **{result.title}**{anchor_info} (Score: {result.score:.3f})\n"
                f"   Path: {result.path}\n"
                f"   Snippet: {result.snippet}\n"
                f"   ID: {result.id}\n"
            )
        
        return [TextContent(type="text", text="\n".join(formatted_results))]
    
    async def _handle_get(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle get content requests"""
        chunk_id = args.get("id", "")
        format_type = args.get("format", "text")
        source_str = args.get("source", "arxmgd")
        
        if not chunk_id:
            return [TextContent(type="text", text="Error: ID is required")]
        
        source = SourceType(source_str)
        
        # Ensure indexer is loaded
        if not self.indexer or self.indexer.source != source:
            logger.info(f"Loading indexer for source: {source.value} (get request)")
            self.indexer = HybridIndexer(source=source)
            self.indexer.load_indices()
        
        # Get chunk
        chunk = self.indexer.get_chunk_by_id(chunk_id)
        if not chunk:
            return [TextContent(type="text", text="Document not found")]
        
        # Return content in requested format
        if format_type == "html":
            content = chunk.html_content
        else:
            content = chunk.content
        
        # Add metadata
        metadata = (
            f"Title: {chunk.title}\n"
            f"Path: {chunk.path}\n"
            f"Source: {chunk.source.value}\n"
            f"Chunk: {chunk.chunk_index + 1}/{chunk.total_chunks}\n"
            f"ID: {chunk.id}\n\n"
        )
        
        return [TextContent(type="text", text=metadata + content)]
    
    async def _handle_toc(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle table of contents requests"""
        source_str = args.get("source", "arxmgd")
        source = SourceType(source_str)
        
        # Load TOC from graph
        if not self.link_graph:
            graph_path = project_root / "data" / "index" / source.value / "graph.json"
            if graph_path.exists():
                self.link_graph = LinkGraphBuilder(source)
                self.link_graph.load_graph(graph_path)
        
        if not self.link_graph:
            return [TextContent(
                type="text", 
                text=f"Table of Contents for {source.value} not found. "
                     f"Use docs.search to find specific topics."
            )]
        
        # Build TOC tree from graph
        toc_text = self._build_toc_text(source.value)
        return [TextContent(type="text", text=toc_text)]
    
    async def _handle_neighbors(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle neighbor requests"""
        chunk_id = args.get("id", "")
        source_str = args.get("source", "arxmgd")
        
        if not chunk_id:
            return [TextContent(type="text", text="Error: ID is required")]
        
        source = SourceType(source_str)
        
        # Ensure indexer is loaded
        if not self.indexer or self.indexer.source != source:
            logger.info(f"Loading indexer for source: {source.value} (neighbors request)")
            self.indexer = HybridIndexer(source=source)
            self.indexer.load_indices()
        
        # Load link graph
        if not self.link_graph:
            graph_path = project_root / "data" / "index" / source.value / "graph.json"
            if graph_path.exists():
                self.link_graph = LinkGraphBuilder(source)
                self.link_graph.load_graph(graph_path)
        
        # Get chunk
        chunk = self.indexer.get_chunk_by_id(chunk_id)
        if not chunk:
            return [TextContent(type="text", text="Document not found")]
        
        # Get neighbors from graph
        if self.link_graph:
            neighbors = self.link_graph.get_neighbors(chunk_id)
            
            neighbor_info = f"Neighbors for: {chunk.title}\n"
            neighbor_info += f"ID: {chunk.id}\n\n"
            
            if neighbors['parent']:
                neighbor_info += f"Parent: {neighbors['parent']}\n"
            
            if neighbors['children']:
                neighbor_info += f"Children: {', '.join(neighbors['children'])}\n"
            
            if neighbors['see_also']:
                neighbor_info += f"See also: {', '.join(neighbors['see_also'])}\n"
        else:
            neighbor_info = (
                f"Neighbors for: {chunk.title}\n"
                f"ID: {chunk.id}\n"
                f"Source: {chunk.source.value}\n\n"
                f"Link graph not available."
            )
        
        return [TextContent(type="text", text=neighbor_info)]
    
    async def _handle_list_sources(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle list sources requests"""
        sources = [source.value for source in SourceType]
        sources_text = "Available CHM documentation sources:\n\n"
        for source in sources:
            sources_text += f"- {source}\n"
        
        return [TextContent(type="text", text=sources_text)]
    
    async def _handle_health(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle health check requests"""
        health_info = (
            "AutoCAD CHM MCP Server\n"
            "Version: 1.0.0\n"
            "Status: Running\n"
            f"Source: {self.source.value}\n"
            f"Indexer Loaded: {self.indexer is not None}\n"
            f"Link Graph Loaded: {self.link_graph is not None}\n"
        )
        
        return [TextContent(type="text", text=health_info)]
    
    def _build_toc_text(self, source: str) -> str:
        """Build TOC text from link graph"""
        if not self.link_graph:
            return "TOC not available"
        
        # Simple TOC representation
        toc_text = f"Table of Contents for {source}:\n\n"
        
        # Find root nodes (nodes without parents)
        root_nodes = []
        for page_id, neighbors in self.link_graph.graph.items():
            if not neighbors.get('parent'):
                root_nodes.append(page_id)
        
        for root_id in root_nodes[:10]:  # Limit to first 10 root nodes
            chunk = self.indexer.get_chunk_by_id(root_id) if self.indexer else None
            if chunk:
                toc_text += f"- {chunk.title}\n"
                
                # Add children
                children = neighbors.get('children', [])
                for child_id in children[:5]:  # Limit children
                    child_chunk = self.indexer.get_chunk_by_id(child_id) if self.indexer else None
                    if child_chunk:
                        toc_text += f"  - {child_chunk.title}\n"
        
        return toc_text
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="autocad-sdk-mcp",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(
                            prompts_changed=False,
                            resources_changed=False,
                            tools_changed=False
                        ),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AutoCAD SDK MCP Server")
    parser.add_argument("--source", default="arxmgd", 
                       choices=[s.value for s in SourceType],
                       help="SDK documentation source to serve")
    
    args = parser.parse_args()
    
    logger.info(f"Starting MCP server with arguments: {args}")
    logger.info(f"Current working directory: {Path.cwd()}")
    
    source = SourceType(args.source)
    logger.info(f"Using source: {source.value}")
    
    server = AutoCADMCPServer(source=source)
    logger.info("Server initialized, starting MCP protocol...")
    
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())

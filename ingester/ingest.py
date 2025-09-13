#!/usr/bin/env python3
"""
CHM ingestion pipeline for AutoCAD documentation
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from ingester.models import SourceType, DocumentPage, DocumentChunk
from ingester.toc_parser import TOCParser
from ingester.topic_parser import TopicParser
from ingester.link_graph import LinkGraphBuilder
from ingester.chunker import HeadingAwareChunker
from ingester.indexer import HybridIndexer


class CHMIngestionPipeline:
    """Complete pipeline for ingesting CHM files into searchable indices"""
    
    def __init__(self, 
                 source: SourceType = SourceType.ARXMGD,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.source = source
        self.embedding_model = embedding_model
        
        # Set up paths
        self.extraction_root = Path(f"data/chm/{source.value}")
        self.index_dir = Path(f"data/index/{source.value}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.toc_parser = TOCParser(str(self.extraction_root))
        self.topic_parser = TopicParser(str(self.extraction_root), source)
        self.link_builder = LinkGraphBuilder(source)
        self.chunker = HeadingAwareChunker()
        self.indexer = HybridIndexer(embedding_model, source)
    
    def ingest_source(self) -> int:
        """Ingest the configured CHM source"""
        print(f"\n=== Ingesting {self.source.value} ===")
        
        # Check if extraction directory exists
        if not self.extraction_root.exists():
            print(f"Error: Extraction directory {self.extraction_root} not found.")
            print("Please extract the CHM file using 7-zip first:")
            print(f"7z x data/chm/{self.source.value}.chm -o{self.extraction_root}")
            return 0
        
        # Parse TOC
        print("Parsing TOC...")
        toc_nodes = self.toc_parser.parse_hhc("*.hhc")  # Look for HHC files
        
        # Parse topics
        print("Parsing topics...")
        pages = self.topic_parser.parse_all_topics()
        print(f"Extracted {len(pages)} pages")
        
        # Build link graph
        print("Building link graph...")
        link_graph = self.link_builder.build_graph(toc_nodes, pages)
        self.link_builder.save_graph(self.index_dir / "graph.json")
        
        # Chunk pages
        print("Chunking pages...")
        all_chunks = []
        for page in tqdm(pages, desc="Chunking"):
            chunks = self.chunker.chunk_page(page)
            
            # Update total_chunks for each chunk
            for chunk in chunks:
                chunk.total_chunks = len(chunks)
            
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Build index
        print("Building search index...")
        anchor_map = self.chunker.get_anchor_map()
        self.indexer.build_index(all_chunks, anchor_map)
        
        return len(all_chunks)
    
    def get_available_sources(self) -> List[SourceType]:
        """Get list of available CHM sources"""
        available = []
        for source in SourceType:
            extraction_dir = Path(f"data/chm/{source.value}")
            if extraction_dir.exists():
                available.append(source)
        return available


def main():
    """Main entry point for the ingestion pipeline"""
    parser = argparse.ArgumentParser(description="Ingest AutoCAD CHM documentation")
    parser.add_argument("--source", default="arxmgd", 
                       choices=[s.value for s in SourceType],
                       help="CHM source to ingest")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--list-sources", action="store_true", 
                       help="List available CHM sources and exit")
    
    args = parser.parse_args()
    
    if args.list_sources:
        # Create a temporary pipeline to check sources
        pipeline = CHMIngestionPipeline()
        available = pipeline.get_available_sources()
        print("Available CHM sources:")
        for source in available:
            print(f"  - {source.value}")
        return
    
    # Create pipeline for specified source
    source = SourceType(args.source)
    pipeline = CHMIngestionPipeline(
        source=source,
        embedding_model=args.embedding_model
    )
    
    # Ingest the source
    chunk_count = pipeline.ingest_source()
    
    print(f"\n=== Ingestion Complete ===")
    print(f"{source.value}: {chunk_count} chunks indexed")


if __name__ == "__main__":
    main()

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from .models import DocumentChunk, SearchResult, SourceType


class HybridIndexer:
    """Hybrid search indexer using FAISS (semantic) and BM25 (lexical)"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 source: SourceType = SourceType.ARXMGD):
        self.embedding_model_name = embedding_model
        self.source = source
        
        # Use absolute path from project root to avoid working directory issues
        project_root = Path(__file__).parent.parent
        self.index_dir = project_root / "data" / "index" / source.value
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Index storage
        self.faiss_index = None
        self.bm25_index = None
        self.chunk_metadata = {}
        self.anchor_map = {}
        
    def build_index(self, chunks: List[DocumentChunk], anchor_map: Dict = None):
        """Build both FAISS and BM25 indices from chunks"""
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        print(f"Building index for {len(chunks)} chunks...")
        
        # Store anchor map
        if anchor_map:
            self.anchor_map = anchor_map
        
        # Prepare data
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]
        
        # Build FAISS index (semantic search)
        self._build_faiss_index(texts, chunk_ids)
        
        # Build BM25 index (lexical search)
        self._build_bm25_index(texts, chunk_ids)
        
        # Store metadata
        self._store_metadata(chunks)
        
        # Save indices
        self._save_indices()
        
        print("Index building completed!")
    
    def _build_faiss_index(self, texts: List[str], chunk_ids: List[str]):
        """Build FAISS vector index"""
        print("Building FAISS index...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index - use IndexFlatL2 for simplicity
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def _build_bm25_index(self, texts: List[str], chunk_ids: List[str]):
        """Build BM25 lexical index"""
        print("Building BM25 index...")
        
        # Tokenize texts for BM25
        tokenized_texts = [text.lower().split() for text in texts]
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        print(f"BM25 index built with {len(tokenized_texts)} documents")
    
    def _store_metadata(self, chunks: List[DocumentChunk]):
        """Store chunk metadata for retrieval"""
        self.chunk_metadata = {}
        
        for i, chunk in enumerate(chunks):
            self.chunk_metadata[chunk.id] = {
                'chunk': chunk,
                'index': i
            }
    
    def _save_indices(self):
        """Save indices to disk"""
        print("Saving indices...")
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(self.index_dir / "faiss.index"))
        
        # Save BM25 index
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25_index, f)
        
        # Save metadata
        with open(self.index_dir / "metadata.json", "w") as f:
            # Convert chunks to dict for JSON serialization
            metadata_dict = {}
            for chunk_id, data in self.chunk_metadata.items():
                chunk_dict = data['chunk'].model_dump()
                metadata_dict[chunk_id] = {
                    'chunk': chunk_dict,
                    'index': data['index']
                }
            json.dump(metadata_dict, f, indent=2)
        
        # Save anchor map
        with open(self.index_dir / "anchor_map.json", "w") as f:
            json.dump(self.anchor_map, f, indent=2)
        
        print("Indices saved successfully!")
    
    def load_indices(self):
        """Load indices from disk"""
        print("Loading indices...")
        
        # Load FAISS index
        faiss_path = self.index_dir / "faiss.index"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        else:
            raise FileNotFoundError("FAISS index not found")
        
        # Load BM25 index
        bm25_path = self.index_dir / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)
        else:
            raise FileNotFoundError("BM25 index not found")
        
        # Load metadata
        metadata_path = self.index_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                self.chunk_metadata = {}
                for chunk_id, data in metadata_dict.items():
                    chunk = DocumentChunk(**data['chunk'])
                    self.chunk_metadata[chunk_id] = {
                        'chunk': chunk,
                        'index': data['index']
                    }
        else:
            raise FileNotFoundError("Metadata not found")
        
        # Load anchor map
        anchor_map_path = self.index_dir / "anchor_map.json"
        if anchor_map_path.exists():
            with open(anchor_map_path, "r") as f:
                self.anchor_map = json.load(f)
        
        print("Indices loaded successfully!")
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Hybrid search using both FAISS and BM25"""
        if not self.faiss_index or not self.bm25_index:
            raise RuntimeError("Indices not loaded")
        
        # Semantic search with FAISS
        faiss_scores = self._faiss_search(query, k * 2)
        
        # Lexical search with BM25
        bm25_scores = self._bm25_search(query, k * 2)
        
        # Combine scores using Reciprocal Rank Fusion
        combined_scores = self._reciprocal_rank_fusion(faiss_scores, bm25_scores)
        
        # Get top k results
        top_results = combined_scores[:k]
        
        # Convert to SearchResult objects
        results = []
        for chunk_id, score in top_results:
            if chunk_id in self.chunk_metadata:
                chunk = self.chunk_metadata[chunk_id]['chunk']
                snippet = self._create_snippet(chunk.content, query)
                
                result = SearchResult(
                    id=chunk.id,
                    title=chunk.title,
                    path=chunk.path,
                    snippet=snippet,
                    score=score,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index
                )
                results.append(result)
        
        return results
    
    def _faiss_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Perform semantic search with FAISS"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        # Convert to chunk IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                # Find chunk ID for this index
                for chunk_id, data in self.chunk_metadata.items():
                    if data['index'] == idx:
                        results.append((chunk_id, float(score)))
                        break
        
        return results
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Perform lexical search with BM25"""
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        # Convert to chunk IDs
        results = []
        for idx in top_indices:
            for chunk_id, data in self.chunk_metadata.items():
                if data['index'] == idx:
                    results.append((chunk_id, float(scores[idx])))
                    break
        
        return results
    
    def _reciprocal_rank_fusion(self, faiss_scores: List[Tuple[str, float]], 
                               bm25_scores: List[Tuple[str, float]], 
                               k: int = 60) -> List[Tuple[str, float]]:
        """Combine scores using Reciprocal Rank Fusion"""
        # Normalize scores to ranks
        faiss_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(faiss_scores)}
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_scores)}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            faiss_rank = faiss_ranks.get(chunk_id, k)
            bm25_rank = bm25_ranks.get(chunk_id, k)
            
            rrf_score = 1.0 / (k + faiss_rank + 1) + 1.0 / (k + bm25_rank + 1)
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID"""
        if chunk_id in self.chunk_metadata:
            return self.chunk_metadata[chunk_id]['chunk']
        return None
    
    def get_chunk_by_anchor(self, page_id: str, anchor: str) -> Optional[DocumentChunk]:
        """Get a chunk by page ID and anchor"""
        anchor_key = f"{page_id}#{anchor}"
        if anchor_key in self.anchor_map:
            chunk_id = self.anchor_map[anchor_key]['chunk_id']
            return self.get_chunk_by_id(chunk_id)
        return None
    
    def _create_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Create a snippet highlighting the query"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find query position
        query_pos = content_lower.find(query_lower)
        if query_pos == -1:
            # Query not found, return beginning
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Create snippet around query
        start = max(0, query_pos - max_length // 2)
        end = min(len(content), query_pos + len(query) + max_length // 2)
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID"""
        if chunk_id in self.chunk_metadata:
            return self.chunk_metadata[chunk_id]['chunk']
        return None

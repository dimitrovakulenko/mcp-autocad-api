#!/usr/bin/env python3
"""
Test queries for AutoCAD SDK MCP Server
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from ingester.indexer import HybridIndexer
from ingester.models import SourceType


class TestQueries:
    """Test queries to verify the MCP server functionality"""
    
    def __init__(self, source: SourceType = SourceType.ARXMGD):
        self.source = source
        self.indexer = None
    
    def load_indexer(self):
        """Load the search indexer"""
        if not self.indexer:
            self.indexer = HybridIndexer(source=self.source)
            self.indexer.load_indices()
    
    def test_search_queries(self):
        """Test various search queries"""
        self.load_indexer()
        
        # Golden queries from requirements
        test_queries = [
            "class represents a revision cloud",
            "ways to construct AcDbArc",
            "methods on AcDbBlockReference",
            "how AcDbDimension stores text size"
        ]
        
        print("=== Testing Search Queries ===\n")
        
        for query in test_queries:
            print(f"Query: '{query}'")
            try:
                results = self.indexer.search(query, k=3)
                if results:
                    print(f"Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result.title} (Score: {result.score:.3f})")
                        print(f"     {result.snippet[:100]}...")
                        print(f"     ID: {result.id}")
                else:
                    print("  No results found")
            except Exception as e:
                print(f"  Error: {e}")
            print()
    
    def test_anchor_retrieval(self):
        """Test anchor-based retrieval"""
        self.load_indexer()
        
        print("=== Testing Anchor Retrieval ===\n")
        
        # First, get some search results to find chunk IDs
        results = self.indexer.search("revision cloud", k=3)
        
        if results:
            for result in results:
                print(f"Testing anchor retrieval for: {result.id}")
                try:
                    chunk = self.indexer.get_chunk_by_id(result.id)
                    if chunk:
                        print(f"  Title: {chunk.title}")
                        print(f"  Source: {chunk.source.value}")
                        print(f"  Content length: {len(chunk.content)} chars")
                        print(f"  Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
                        
                        # Test anchor retrieval if anchors exist
                        if chunk.metadata.get('anchors'):
                            anchor = chunk.metadata['anchors'][0]
                            anchor_chunk = self.indexer.get_chunk_by_anchor(chunk.page_id, anchor)
                            if anchor_chunk:
                                print(f"  Anchor '{anchor}' found in chunk: {anchor_chunk.id}")
                    else:
                        print("  Chunk not found")
                except Exception as e:
                    print(f"  Error: {e}")
                print()
        else:
            print("No search results available for anchor retrieval test")
    
    def test_golden_queries(self):
        """Test the golden queries and verify expected results"""
        self.load_indexer()
        
        print("=== Testing Golden Queries ===\n")
        
        golden_tests = [
            {
                "query": "class represents a revision cloud",
                "expected_keywords": ["revision", "cloud", "class"]
            },
            {
                "query": "ways to construct AcDbArc",
                "expected_keywords": ["AcDbArc", "construct", "constructor"]
            },
            {
                "query": "methods on AcDbBlockReference",
                "expected_keywords": ["AcDbBlockReference", "method"]
            },
            {
                "query": "how AcDbDimension stores text size",
                "expected_keywords": ["AcDbDimension", "text", "size"]
            }
        ]
        
        for test in golden_tests:
            query = test["query"]
            expected = test["expected_keywords"]
            
            print(f"Query: '{query}'")
            try:
                results = self.indexer.search(query, k=3)
                if results:
                    print(f"Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result.title} (Score: {result.score:.3f})")
                        
                        # Check if result contains expected keywords
                        content_lower = result.snippet.lower()
                        found_keywords = [kw for kw in expected if kw.lower() in content_lower]
                        if found_keywords:
                            print(f"     ✓ Contains keywords: {found_keywords}")
                        else:
                            print(f"     ⚠ Missing expected keywords: {expected}")
                else:
                    print("  No results found")
            except Exception as e:
                print(f"  Error: {e}")
            print()
    
    def run_all_tests(self):
        """Run all test queries"""
        print("AutoCAD SDK MCP Server - Test Queries")
        print("=" * 50)
        
        try:
            self.test_golden_queries()
            self.test_search_queries()
            self.test_anchor_retrieval()
            print("All tests completed!")
        except Exception as e:
            print(f"Test failed: {e}")


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AutoCAD SDK MCP Server")
    parser.add_argument("--source", default="arxmgd", 
                       choices=[s.value for s in SourceType],
                       help="SDK documentation source to test")
    
    args = parser.parse_args()
    
    source = SourceType(args.source)
    tester = TestQueries(source=source)
    tester.run_all_tests()


if __name__ == "__main__":
    main()

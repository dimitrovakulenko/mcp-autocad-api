"""
Link graph builder for CHM documentation
"""

import json
from typing import Dict, List, Set
from pathlib import Path
from .toc_parser import TOCTreeNode
from .models import DocumentPage, SourceType


class LinkGraphBuilder:
    """Builds and manages link graph from TOC and topics"""
    
    def __init__(self, source: SourceType):
        self.source = source
        self.graph = {}
        self.path_to_id = {}  # Map path to page_id
    
    def build_graph(self, toc_nodes: List[TOCTreeNode], pages: List[DocumentPage]) -> Dict:
        """Build link graph from TOC and pages"""
        # Initialize graph
        self.graph = {}
        self.path_to_id = {}
        
        # Build path to ID mapping
        for page in pages:
            self.path_to_id[page.path] = page.id
            self.graph[page.id] = {
                'parent': None,
                'children': [],
                'see_also': []
            }
        
        # Add TOC relationships (parent/children)
        self._add_toc_relationships(toc_nodes)
        
        # Add see_also relationships from pages
        self._add_see_also_relationships(pages)
        
        return self.graph
    
    def _add_toc_relationships(self, toc_nodes: List[TOCTreeNode]):
        """Add parent/children relationships from TOC"""
        for node in toc_nodes:
            self._process_toc_node(node, None)
    
    def _process_toc_node(self, node: TOCTreeNode, parent_id: str = None):
        """Process a TOC node and its children"""
        if not node.path:
            # Node without path, process children with same parent
            for child in node.children:
                self._process_toc_node(child, parent_id)
            return
        
        # Find page ID for this path
        page_id = self.path_to_id.get(node.path)
        if not page_id:
            # Path not found, process children with same parent
            for child in node.children:
                self._process_toc_node(child, parent_id)
            return
        
        # Set parent relationship
        if parent_id and page_id in self.graph:
            self.graph[page_id]['parent'] = parent_id
            if page_id not in self.graph[parent_id]['children']:
                self.graph[parent_id]['children'].append(page_id)
        
        # Process children
        for child in node.children:
            self._process_toc_node(child, page_id)
    
    def _add_see_also_relationships(self, pages: List[DocumentPage]):
        """Add see_also relationships from page links"""
        for page in pages:
            page_id = page.id
            if page_id not in self.graph:
                continue
            
            for link_path in page.see_also:
                # Try to find the target page
                target_id = self._resolve_link_path(link_path, page.path)
                if target_id and target_id != page_id:
                    if target_id not in self.graph[page_id]['see_also']:
                        self.graph[page_id]['see_also'].append(target_id)
    
    def _resolve_link_path(self, link_path: str, current_path: str) -> str:
        """Resolve a link path to a page ID"""
        # Try direct path match
        if link_path in self.path_to_id:
            return self.path_to_id[link_path]
        
        # Try relative path resolution
        current_dir = Path(current_path).parent
        resolved_path = current_dir / link_path
        resolved_str = str(resolved_path)
        
        if resolved_str in self.path_to_id:
            return self.path_to_id[resolved_str]
        
        # Try with different extensions
        for ext in ['.html', '.htm']:
            if not link_path.endswith(ext):
                test_path = link_path + ext
                if test_path in self.path_to_id:
                    return self.path_to_id[test_path]
                
                # Try relative resolution with extension
                resolved_with_ext = current_dir / test_path
                resolved_str_ext = str(resolved_with_ext)
                if resolved_str_ext in self.path_to_id:
                    return self.path_to_id[resolved_str_ext]
        
        return None
    
    def save_graph(self, output_path: Path):
        """Save link graph to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, indent=2, ensure_ascii=False)
    
    def load_graph(self, input_path: Path) -> Dict:
        """Load link graph from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            self.graph = json.load(f)
        return self.graph
    
    def get_neighbors(self, page_id: str) -> Dict:
        """Get neighbors for a specific page"""
        if page_id not in self.graph:
            return {'parent': None, 'children': [], 'see_also': []}
        
        return self.graph[page_id].copy()

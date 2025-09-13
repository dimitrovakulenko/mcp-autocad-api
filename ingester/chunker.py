import re
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from .models import DocumentPage, DocumentChunk, SourceType


class HeadingAwareChunker:
    """Chunks documents while preserving heading structure"""
    
    def __init__(self, 
                 target_tokens: int = 1000,
                 overlap_tokens: int = 150,  # 10-15% of 1000
                 min_chunk_tokens: int = 200):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.anchor_map = {}  # page_id#anchor -> byte/char offsets
        
    def chunk_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """Split a page into heading-aware chunks"""
        # Parse HTML to identify heading structure
        soup = BeautifulSoup(page.html_content, 'html.parser')
        heading_sections = self._extract_heading_sections(soup)
        
        if not heading_sections:
            # No headings found, chunk by paragraphs
            return self._chunk_by_paragraphs(page)
        
        chunks = []
        current_chunk_text = ""
        current_chunk_html = ""
        current_offset = 0
        chunk_index = 0
        
        for section in heading_sections:
            section_text = section['text']
            section_html = section['html']
            section_tokens = self._estimate_tokens(section_text)
            
            # If adding this section would exceed target, finalize current chunk
            if (current_chunk_text and 
                self._estimate_tokens(current_chunk_text + " " + section_text) > self.target_tokens):
                
                if self._estimate_tokens(current_chunk_text) >= self.min_chunk_tokens:
                    chunk = self._create_chunk(
                        page, chunk_index, current_chunk_text, current_chunk_html,
                        current_offset, current_offset + len(current_chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text + " " + section_text if overlap_text else section_text
                current_chunk_html = section_html
                current_offset = current_offset + len(current_chunk_text) - len(overlap_text) - len(section_text)
            else:
                # Add section to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + section_text
                    current_chunk_html += section_html
                else:
                    current_chunk_text = section_text
                    current_chunk_html = section_html
        
        # Add final chunk
        if current_chunk_text and self._estimate_tokens(current_chunk_text) >= self.min_chunk_tokens:
            chunk = self._create_chunk(
                page, chunk_index, current_chunk_text, current_chunk_html,
                current_offset, current_offset + len(current_chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_heading_sections(self, soup: BeautifulSoup) -> List[dict]:
        """Extract sections based on headings"""
        sections = []
        current_section = None
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li']):
            if element.name.startswith('h'):
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'heading': element.get_text().strip(),
                    'level': int(element.name[1]),
                    'text': element.get_text().strip(),
                    'html': str(element)
                }
            elif current_section:
                # Add content to current section
                text = element.get_text().strip()
                if text:
                    current_section['text'] += " " + text
                    current_section['html'] += str(element)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _chunk_by_paragraphs(self, page: DocumentPage) -> List[DocumentChunk]:
        """Fallback chunking by paragraphs when no headings found"""
        soup = BeautifulSoup(page.html_content, 'html.parser')
        paragraphs = soup.find_all(['p', 'div', 'li'])
        
        chunks = []
        current_chunk_text = ""
        current_chunk_html = ""
        current_offset = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_text = para.get_text().strip()
            if not para_text:
                continue
                
            para_tokens = self._estimate_tokens(para_text)
            
            # If adding this paragraph would exceed target, finalize current chunk
            if (current_chunk_text and 
                self._estimate_tokens(current_chunk_text + " " + para_text) > self.target_tokens):
                
                if self._estimate_tokens(current_chunk_text) >= self.min_chunk_tokens:
                    chunk = self._create_chunk(
                        page, chunk_index, current_chunk_text, current_chunk_html,
                        current_offset, current_offset + len(current_chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text + " " + para_text if overlap_text else para_text
                current_chunk_html = str(para)
                current_offset = current_offset + len(current_chunk_text) - len(overlap_text) - len(para_text)
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + para_text
                    current_chunk_html += str(para)
                else:
                    current_chunk_text = para_text
                    current_chunk_html = str(para)
        
        # Add final chunk
        if current_chunk_text and self._estimate_tokens(current_chunk_text) >= self.min_chunk_tokens:
            chunk = self._create_chunk(
                page, chunk_index, current_chunk_text, current_chunk_html,
                current_offset, current_offset + len(current_chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, page: DocumentPage, chunk_index: int, 
                     text: str, html: str, start_offset: int, end_offset: int) -> DocumentChunk:
        """Create a DocumentChunk from page data"""
        chunk_id = f"{page.id}_chunk_{chunk_index}"
        
        # Build anchor mapping for this chunk
        chunk_anchors = self._extract_chunk_anchors(html, start_offset, end_offset)
        for anchor, offset in chunk_anchors.items():
            anchor_key = f"{page.id}#{anchor}"
            self.anchor_map[anchor_key] = {
                'chunk_id': chunk_id,
                'offset': offset,
                'start_offset': start_offset,
                'end_offset': end_offset
            }
        
        return DocumentChunk(
            id=chunk_id,
            source=page.source,
            page_id=page.id,
            title=page.title,
            path=page.path,
            content=text,
            html_content=html,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be set later
            start_offset=start_offset,
            end_offset=end_offset,
            metadata={
                'original_page_title': page.title,
                'original_page_path': page.path,
                'word_count': len(text.split()),
                'char_count': len(text),
                'anchors': list(chunk_anchors.keys())
            }
        )
    
    def _extract_chunk_anchors(self, html: str, start_offset: int, end_offset: int) -> Dict[str, int]:
        """Extract anchors within a chunk and their relative offsets"""
        soup = BeautifulSoup(html, 'html.parser')
        anchors = {}
        
        for element in soup.find_all(attrs={'id': True}):
            anchor_id = element['id']
            # Calculate relative offset within the chunk
            element_text = str(element)
            element_pos = html.find(element_text)
            if element_pos != -1:
                anchors[anchor_id] = element_pos
        
        for element in soup.find_all(attrs={'name': True}):
            anchor_name = element['name']
            element_text = str(element)
            element_pos = html.find(element_text)
            if element_pos != -1:
                anchors[anchor_name] = element_pos
        
        return anchors
    
    def get_anchor_map(self) -> Dict[str, Dict]:
        """Get the anchor mapping"""
        return self.anchor_map.copy()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = words[-self.overlap_tokens//4:] if len(words) > self.overlap_tokens//4 else []
        return " ".join(overlap_words)

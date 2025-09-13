"""
Topic parser for CHM HTML files
"""

import os
import re
from typing import List, Dict, Set, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from .models import DocumentPage, SourceType


class TopicParser:
    """Parser for CHM HTML topic files"""
    
    def __init__(self, extraction_root: str, source: SourceType):
        self.extraction_root = Path(extraction_root)
        self.source = source
        self.chm_base_url = f"mk:@MSITStore:{source.value}.chm::/"
    
    def parse_all_topics(self) -> List[DocumentPage]:
        """Parse all HTML topics in the extraction directory"""
        pages = []
        
        # Find all HTML files
        html_files = list(self.extraction_root.rglob("*.html"))
        html_files.extend(self.extraction_root.rglob("*.htm"))
        
        for html_file in html_files:
            try:
                page = self._parse_topic_file(html_file)
                if page:
                    pages.append(page)
            except Exception as e:
                print(f"Error parsing {html_file}: {e}")
                continue
        
        return pages
    
    def _parse_topic_file(self, html_file: Path) -> Optional[DocumentPage]:
        """Parse a single HTML topic file"""
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'lxml')
        
        # Extract title
        title = self._extract_title(soup, html_file)
        
        # Extract text content
        text_content = self._extract_text_content(soup)
        
        # Extract anchors
        anchors = self._extract_anchors(soup)
        
        # Extract intra-doc links
        outlinks = self._extract_intra_doc_links(soup)
        
        # Generate page ID
        page_id = self._generate_page_id(html_file)
        
        # Get relative path
        relative_path = str(html_file.relative_to(self.extraction_root))
        
        return DocumentPage(
            id=page_id,
            source=self.source,
            title=title,
            path=relative_path,
            content=text_content,
            html_content=content,
            anchors=anchors,
            see_also=outlinks,
            metadata={
                'original_path': str(html_file),
                'word_count': len(text_content.split()),
                'has_code': bool(soup.find('code') or soup.find('pre')),
                'has_tables': bool(soup.find('table')),
                'has_images': bool(soup.find('img'))
            }
        )
    
    def _extract_title(self, soup: BeautifulSoup, html_file: Path) -> str:
        """Extract page title"""
        # Try different title sources
        title_selectors = [
            'h1',
            'title',
            '.title',
            '#title',
            'h2:first-of-type'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                return element.get_text().strip()
        
        # Fallback to filename
        return html_file.stem.replace('_', ' ').title()
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_anchors(self, soup: BeautifulSoup) -> List[str]:
        """Extract all anchor IDs and names"""
        anchors = []
        
        # Get all elements with id attribute
        for element in soup.find_all(attrs={'id': True}):
            anchors.append(element['id'])
        
        # Get all elements with name attribute
        for element in soup.find_all(attrs={'name': True}):
            anchors.append(element['name'])
        
        return list(set(anchors))  # Remove duplicates
    
    def _extract_intra_doc_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract intra-document links (links within the same CHM)"""
        outlinks = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip external links
            if href.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
                continue
            
            # Skip anchor-only links
            if href.startswith('#'):
                continue
            
            # Handle CHM-specific links
            if href.startswith(self.chm_base_url):
                # Extract the path part
                path = href[len(self.chm_base_url):]
                outlinks.append(path)
            elif not href.startswith('/') and not href.startswith('\\'):
                # Relative link
                outlinks.append(href)
        
        return list(set(outlinks))  # Remove duplicates
    
    def _generate_page_id(self, html_file: Path) -> str:
        """Generate a unique page ID"""
        # Use the source type and a cleaned version of the path
        relative_path = html_file.relative_to(self.extraction_root)
        clean_path = str(relative_path).replace('/', '_').replace('\\', '_').replace('.', '_')
        return f"{self.source.value}_{clean_path}"

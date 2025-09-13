from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class SourceType(str, Enum):
    ARXMGD = "arxmgd"
    ARXMGD_DEV = "arxmgd_dev"
    ARXDEV = "arxdev"
    ARXDOC = "arxdoc"
    ARXIOP = "arxiop"
    ARXMGR = "arxmgr"
    ARXREF = "arxref"
    READARX = "readarx"


class DocumentChunk(BaseModel):
    """A chunk of documentation with metadata"""
    id: str
    source: SourceType
    page_id: str
    title: str
    path: str
    anchor: Optional[str] = None
    content: str
    html_content: str
    chunk_index: int
    total_chunks: int
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any] = {}


class DocumentPage(BaseModel):
    """A complete documentation page"""
    id: str
    source: SourceType
    title: str
    path: str
    content: str
    html_content: str
    anchors: List[str] = []
    see_also: List[str] = []
    metadata: Dict[str, Any] = {}


class TOCNode(BaseModel):
    """Table of Contents node"""
    title: str
    path: str
    level: int
    children: List['TOCNode'] = []
    page_id: Optional[str] = None


class SearchResult(BaseModel):
    """Search result with ranking information"""
    id: str
    title: str
    path: str
    snippet: str
    score: float
    source: SourceType
    chunk_index: Optional[int] = None


class NeighborInfo(BaseModel):
    """Information about document neighbors"""
    parent: Optional[SearchResult] = None
    children: List[SearchResult] = []
    related: List[SearchResult] = []

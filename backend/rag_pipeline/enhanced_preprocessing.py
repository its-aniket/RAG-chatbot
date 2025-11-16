"""
Enhanced Document Preprocessing with Semantic Chunking
Improvements over basic preprocessing:
1. Semantic-aware chunking
2. Document structure preservation  
3. Adaptive chunk sizes
4. Better overlap strategies
"""

import PyPDF2
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    """Enhanced metadata for chunks"""
    chunk_id: str
    chunk_index: int
    document_section: str
    section_hierarchy: List[str]
    chunk_type: str  # 'paragraph', 'header', 'list', 'table'
    semantic_density: float
    word_count: int
    char_count: int
    has_numbers: bool
    has_urls: bool
    has_emails: bool

class EnhancedDocumentProcessor:
    def __init__(
        self, 
        base_chunk_size: int = 400,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        semantic_overlap_ratio: float = 0.15,
        preserve_structure: bool = True
    ):
        """
        Enhanced document processor with semantic awareness
        
        Args:
            base_chunk_size: Target chunk size in words
            min_chunk_size: Minimum allowed chunk size
            max_chunk_size: Maximum allowed chunk size  
            semantic_overlap_ratio: Ratio of overlap between chunks
            preserve_structure: Whether to preserve document structure
        """
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.semantic_overlap_ratio = semantic_overlap_ratio
        self.preserve_structure = preserve_structure
        
        # Patterns for document structure detection
        self.header_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$'  # Title case headers
        ]
        
    def extract_text_with_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text while preserving document structure
        
        Returns:
            Dict with text and structural information
        """
        try:
            text = ""
            page_structure = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Preserve page boundaries
                        page_marker = f"\n===== PAGE {page_num + 1} =====\n"
                        text += page_marker + page_text + "\n"
                        
                        page_structure.append({
                            "page_number": page_num + 1,
                            "text_length": len(page_text),
                            "has_content": bool(page_text.strip())
                        })
            
            return {
                "text": text,
                "page_structure": page_structure,
                "total_pages": len(page_structure)
            }
            
        except Exception as e:
            raise Exception(f"Error extracting text with structure: {str(e)}")
    
    def detect_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect document structure (headers, sections, paragraphs)
        
        Returns:
            List of structural elements with metadata
        """
        lines = text.split('\n')
        structure = []
        current_section = []
        section_hierarchy = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
                
            # Check for page markers
            if line.startswith("===== PAGE"):
                if current_section:
                    structure.append({
                        "type": "section",
                        "content": '\n'.join(current_section),
                        "line_range": (line_num - len(current_section), line_num),
                        "hierarchy": section_hierarchy.copy()
                    })
                    current_section = []
                continue
            
            # Check for headers
            is_header = False
            header_level = 0
            
            for pattern in self.header_patterns:
                match = re.match(pattern, line)
                if match:
                    is_header = True
                    header_level = self._determine_header_level(line, pattern)
                    
                    # Update section hierarchy
                    if header_level <= len(section_hierarchy):
                        section_hierarchy = section_hierarchy[:header_level-1]
                    section_hierarchy.append(line)
                    
                    # Save previous section
                    if current_section:
                        structure.append({
                            "type": "section", 
                            "content": '\n'.join(current_section),
                            "line_range": (line_num - len(current_section), line_num),
                            "hierarchy": section_hierarchy[:-1].copy()
                        })
                        current_section = []
                    
                    # Add header
                    structure.append({
                        "type": "header",
                        "content": line,
                        "level": header_level,
                        "line_range": (line_num, line_num + 1),
                        "hierarchy": section_hierarchy.copy()
                    })
                    break
            
            if not is_header:
                current_section.append(line)
        
        # Add final section
        if current_section:
            structure.append({
                "type": "section",
                "content": '\n'.join(current_section), 
                "line_range": (len(lines) - len(current_section), len(lines)),
                "hierarchy": section_hierarchy.copy()
            })
        
        return structure
    
    def _determine_header_level(self, line: str, pattern: str) -> int:
        """Determine header level based on pattern and formatting"""
        # Markdown headers
        if pattern.startswith(r'^#+'):
            return len(re.match(r'^(#+)', line).group(1))
        
        # ALL CAPS - assume level 1
        if pattern == r'^([A-Z][A-Z\s]+)$':
            return 1
        
        # Numbered sections
        if pattern.startswith(r'^(\d+'):
            match = re.match(r'^(\d+)', line)
            if match:
                return len(match.group(1).split('.'))
        
        # Default level
        return 2
    
    def calculate_semantic_density(self, text: str) -> float:
        """
        Calculate semantic density of text
        Higher density = more information-rich content
        """
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 3:
            return 0.0
        
        # Factors that increase semantic density
        density_score = 0.0
        
        # 1. Vocabulary diversity (unique words ratio)
        unique_words = len(set(word.lower() for word in words))
        diversity = unique_words / len(words)
        density_score += diversity * 0.3
        
        # 2. Technical terms (words with numbers, capital letters)
        technical_count = sum(1 for word in words if re.search(r'[A-Z]|\d', word))
        technical_ratio = technical_count / len(words)
        density_score += technical_ratio * 0.2
        
        # 3. Sentence complexity (average sentence length)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        complexity = min(avg_sentence_length / 20, 1.0)  # Normalize to max 1.0
        density_score += complexity * 0.2
        
        # 4. Information markers (numbers, lists, special chars)
        info_markers = len(re.findall(r'\d+|[•\-\*]|\([a-zA-Z]\)|[A-Z]{2,}', text))
        marker_ratio = min(info_markers / len(words), 0.5)
        density_score += marker_ratio * 0.3
        
        return min(density_score, 1.0)
    
    def adaptive_chunking(
        self, 
        structure: List[Dict[str, Any]], 
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create adaptive chunks based on document structure and semantic density
        """
        chunks = []
        
        for struct_elem in structure:
            if struct_elem["type"] == "header":
                # Headers become their own small chunks
                chunk = self._create_header_chunk(struct_elem, base_metadata)
                chunks.append(chunk)
                
            elif struct_elem["type"] == "section":
                # Sections are chunked based on content and density
                section_chunks = self._chunk_section_adaptively(struct_elem, base_metadata)
                chunks.extend(section_chunks)
        
        return chunks
    
    def _create_header_chunk(
        self, 
        header_elem: Dict[str, Any], 
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a chunk for a header element"""
        text = header_elem["content"]
        
        metadata = ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            chunk_index=len([]),  # Will be updated later
            document_section=text,
            section_hierarchy=header_elem.get("hierarchy", []),
            chunk_type="header",
            semantic_density=1.0,  # Headers are high importance
            word_count=len(text.split()),
            char_count=len(text),
            has_numbers=bool(re.search(r'\d', text)),
            has_urls=bool(re.search(r'http[s]?://', text)),
            has_emails=bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        )
        
        return {
            "text": text,
            "metadata": {**base_metadata, **metadata.__dict__},
            "structure_type": "header",
            "importance": "high"
        }
    
    def _chunk_section_adaptively(
        self, 
        section_elem: Dict[str, Any], 
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk a section based on semantic density and content"""
        text = section_elem["content"]
        hierarchy = section_elem.get("hierarchy", [])
        
        # Calculate semantic density
        density = self.calculate_semantic_density(text)
        
        # Adjust chunk size based on density
        if density > 0.7:  # High density content
            target_size = int(self.base_chunk_size * 0.8)
        elif density < 0.3:  # Low density content  
            target_size = int(self.base_chunk_size * 1.2)
        else:
            target_size = self.base_chunk_size
        
        # Ensure size bounds
        target_size = max(self.min_chunk_size, min(target_size, self.max_chunk_size))
        
        # Split into sentences for better boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence exceeds target size
            if current_word_count + sentence_words > target_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_adaptive_chunk(
                    chunk_text, hierarchy, density, base_metadata, len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk with semantic overlap
                overlap_size = int(len(current_chunk) * self.semantic_overlap_ratio)
                overlap_sentences = current_chunk[-overlap_size:] if overlap_size > 0 else []
                
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = self._create_adaptive_chunk(
                chunk_text, hierarchy, density, base_metadata, len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better boundary detection"""
        # Enhanced sentence splitting that preserves some context
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_adaptive_chunk(
        self, 
        text: str, 
        hierarchy: List[str], 
        density: float, 
        base_metadata: Dict[str, Any], 
        chunk_index: int
    ) -> Dict[str, Any]:
        """Create an adaptive chunk with enhanced metadata"""
        
        metadata = ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            chunk_index=chunk_index,
            document_section=hierarchy[-1] if hierarchy else "Unknown Section",
            section_hierarchy=hierarchy,
            chunk_type="paragraph", 
            semantic_density=density,
            word_count=len(text.split()),
            char_count=len(text),
            has_numbers=bool(re.search(r'\d', text)),
            has_urls=bool(re.search(r'http[s]?://', text)),
            has_emails=bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        )
        
        return {
            "text": text,
            "metadata": {**base_metadata, **metadata.__dict__},
            "structure_type": "content",
            "importance": "high" if density > 0.6 else "medium" if density > 0.3 else "low"
        }
    
    def process_document_enhanced(
        self, 
        file_path: str, 
        file_id: str, 
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Enhanced document processing pipeline
        
        Args:
            file_path: Path to document file
            file_id: Unique identifier for the document
            filename: Original filename
            
        Returns:
            List of processed chunks with enhanced metadata
        """
        try:
            # Step 1: Extract text with structure
            extraction_result = self.extract_text_with_structure(file_path)
            text = extraction_result["text"]
            page_structure = extraction_result["page_structure"]
            
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Step 2: Detect document structure 
            structure = self.detect_document_structure(text)
            
            # Step 3: Create base metadata
            base_metadata = {
                "file_id": file_id,
                "filename": filename,
                "file_path": file_path,
                "file_type": "pdf",
                "total_pages": extraction_result["total_pages"],
                "processing_version": "enhanced_v1.0",
                "processed_at": datetime.now().isoformat()
            }
            
            # Step 4: Adaptive chunking
            chunks = self.adaptive_chunking(structure, base_metadata)
            
            # Step 5: Validate and enhance chunks
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                chunk["metadata"]["chunk_index"] = i  # Update correct index
                if self._validate_enhanced_chunk(chunk):
                    valid_chunks.append(chunk)
            
            return valid_chunks
            
        except Exception as e:
            raise Exception(f"Enhanced processing failed for {filename}: {str(e)}")
    
    def _validate_enhanced_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Enhanced validation for chunks"""
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Basic validation
        if len(text.strip()) < 10:
            return False
        
        if len(text.split()) < 3:
            return False
        
        # Check for actual content (not just special characters)
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Headers can be shorter
        if metadata.get("chunk_type") == "header":
            return len(text.strip()) > 3
        
        return True
    
    def get_enhanced_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed statistics about enhanced processing"""
        if not chunks:
            return {"total_chunks": 0}
        
        # Basic stats
        word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
        densities = [chunk["metadata"]["semantic_density"] for chunk in chunks]
        
        # Type distribution
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk["metadata"].get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Section distribution
        sections = {}
        for chunk in chunks:
            section = chunk["metadata"].get("document_section", "unknown")
            sections[section] = sections.get(section, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_words_per_chunk": np.mean(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "avg_semantic_density": np.mean(densities),
            "high_density_chunks": sum(1 for d in densities if d > 0.6),
            "chunk_type_distribution": chunk_types,
            "section_distribution": sections,
            "total_words": sum(word_counts),
            "processing_version": "enhanced_v1.0"
        }


# Test function
def test_enhanced_processing():
    """Test enhanced processing with sample document"""
    processor = EnhancedDocumentProcessor()
    
    file_path = "uploads/mem.pdf"
    file_id = str(uuid.uuid4())
    filename = "mem.pdf"
    
    try:
        chunks = processor.process_document_enhanced(file_path, file_id, filename)
        stats = processor.get_enhanced_stats(chunks)
        
        print("=== Enhanced Processing Test ===")
        print(f"Processed {stats['total_chunks']} chunks")
        print(f"Average semantic density: {stats['avg_semantic_density']:.3f}")
        print(f"High density chunks: {stats['high_density_chunks']}")
        print(f"Chunk types: {stats['chunk_type_distribution']}")
        print("=== Test Complete ===")
        
        # Show first few chunks
        for i, chunk in enumerate(chunks[:3]):
            meta = chunk["metadata"]
            print(f"\nChunk {i}:")
            print(f"  Type: {meta['chunk_type']}")
            print(f"  Density: {meta['semantic_density']:.3f}")
            print(f"  Section: {meta['document_section']}")
            print(f"  Words: {meta['word_count']}")
            print(f"  Preview: {chunk['text'][:100]}...")
        
        return chunks
        
    except Exception as e:
        print(f"Enhanced processing test failed: {e}")
        return None

if __name__ == "__main__":
    test_enhanced_processing()
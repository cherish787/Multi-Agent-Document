import os
import logging
from typing import List, Dict, Any
from pypdf import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text page by page from PDF.
        Returns a list of dicts: [{"text": page_text, "metadata": {"page": page_number}}]
        """
        pages = []
        try:
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "text": text,
                        "metadata": {"page": idx + 1}
                    })
            logger.info(f"Extracted {len(pages)} pages from PDF: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            raise e
        return pages

    def extract_text_from_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from DOCX paragraph by paragraph.
        """
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text and para.text.strip():
                    full_text.append(para.text)
            
            # Combine paragraphs but treat as a single large document for docx
            combined_text = "\n\n".join(full_text)
            logger.info(f"Extracted text from DOCX: {file_path}")
            return [{"text": combined_text, "metadata": {"page": 1}}]
        except Exception as e:
            logger.error(f"Error parsing DOCX file {file_path}: {e}")
            raise e

    def extract_text_from_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from raw TXT file.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            logger.info(f"Extracted text from TXT: {file_path}")
            return [{"text": content, "metadata": {"page": 1}}]
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {e}")
            raise e

    def parse_and_chunk(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Extracts text according to file extension and splits it using RecursiveCharacterTextSplitter.
        Attaches metadata like document name and chunk index.
        """
        _, ext = os.path.splitext(filename.lower())
        
        # 1. Extract raw text with page metadata
        if ext == ".pdf":
            raw_pages = self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            raw_pages = self.extract_text_from_docx(file_path)
        elif ext == ".txt":
            raw_pages = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # 2. Chunk text
        chunks = []
        chunk_idx = 0
        for page_data in raw_pages:
            text = page_data["text"]
            page_meta = page_data["metadata"]
            
            page_chunks = self.splitter.split_text(text)
            for pc in page_chunks:
                if pc.strip():
                    chunks.append({
                        "text": pc,
                        "metadata": {
                            "filename": filename,
                            "page": page_meta["page"],
                            "chunk_index": chunk_idx
                        }
                    })
                    chunk_idx += 1
                    
        logger.info(f"Generated {len(chunks)} chunks for {filename}")
        return chunks

# Global Processor instance
document_processor = DocumentProcessor()

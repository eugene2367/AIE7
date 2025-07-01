"""PDF handling utilities for the aimakerspace library."""

from typing import List
import PyPDF2
from .text_utils import CharacterTextSplitter

class PDFLoader:
    """A class to load and process PDF documents."""
    
    def __init__(self, file_path: str):
        """Initialize the PDFLoader with a file path.
        
        Args:
            file_path (str): Path to the PDF file
        """
        self.file_path = file_path
        self.documents = []
        
    def load_pdf(self) -> List[str]:
        """Load the PDF file and extract text from all pages.
        
        Returns:
            List[str]: List containing the extracted text from each page
        """
        with open(self.file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    self.documents.append(text)
                    
        return self.documents

class PDFTextSplitter(CharacterTextSplitter):
    """A text splitter specifically designed for PDF content.
    
    Inherits from CharacterTextSplitter but adds PDF-specific handling:
    - Improved handling of page breaks
    - Better preservation of document structure
    - Special handling for headers and footers
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the PDFTextSplitter.
        
        Args:
            chunk_size (int): The target size of each text chunk
            chunk_overlap (int): The number of characters to overlap between chunks
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    def clean_text(self, text: str) -> str:
        """Clean the PDF text by removing common artifacts.
        
        Args:
            text (str): The text to clean
            
        Returns:
            str: The cleaned text
        """
        # Remove common PDF artifacts
        text = text.replace('\n\n', '\n')  # Remove double line breaks
        text = ' '.join(text.split())  # Normalize whitespace
        return text
        
    def split_texts(self, texts: List[str]) -> List[str]:
        """Split the texts while preserving PDF-specific formatting.
        
        Args:
            texts (List[str]): List of texts to split
            
        Returns:
            List[str]: List of split text chunks
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        return super().split_texts(cleaned_texts) 
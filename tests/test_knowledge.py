"""
Unit tests for PIMALUOS knowledge module.
"""

import pytest


class TestLLMAbstraction:
    """Tests for LLM abstraction layer."""
    
    def test_get_llm_mock(self):
        """Test mock LLM creation."""
        from pimaluos.knowledge import get_llm
        
        llm = get_llm('mock')
        
        assert llm is not None
        assert llm.name == 'mock/mock'
    
    def test_mock_generate(self, mock_llm):
        """Test mock LLM generation."""
        response = mock_llm.generate("What are the zoning constraints for R6?")
        
        assert response is not None
        assert 'max_far' in response
    
    def test_mock_embed(self, mock_llm):
        """Test mock LLM embedding."""
        embedding = mock_llm.embed("Test text")
        
        # MockLLM returns random embeddings - just check it's a list/array
        assert len(embedding) > 0
        assert all(isinstance(v, (int, float)) for v in embedding)
    
    def test_invalid_provider(self):
        """Test error handling for invalid provider."""
        from pimaluos.knowledge import get_llm
        
        with pytest.raises(ValueError):
            get_llm('invalid_provider')


class TestRAGPipeline:
    """Tests for RAG pipeline."""
    
    def test_document_creation(self):
        """Test Document creation."""
        from pimaluos.knowledge.rag import Document
        
        doc = Document(
            content="R6 districts allow 2.0 FAR",
            metadata={"zone": "R6"}
        )
        
        assert doc.content == "R6 districts allow 2.0 FAR"
        assert doc.metadata['zone'] == 'R6'
        assert doc.id is not None
    
    def test_text_splitter(self):
        """Test text splitting."""
        from pimaluos.knowledge.rag import TextSplitter, Document
        
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document(content="A" * 100, metadata={})
        
        chunks = splitter.split(doc)
        
        assert len(chunks) > 1
    
    def test_vector_store(self, mock_llm):
        """Test vector store operations."""
        from pimaluos.knowledge.rag import VectorStore, Document
        
        store = VectorStore()
        docs = [
            Document(content="R6 districts", metadata={}),
            Document(content="C4 commercial", metadata={}),
        ]
        
        embeddings = [mock_llm.embed(d.content) for d in docs]
        store.add_documents(docs, embeddings)
        
        assert len(store.documents) == 2


class TestConstraintParser:
    """Tests for constraint parser."""
    
    def test_extractor_init(self):
        """Test constraint extractor initialization."""
        from pimaluos.knowledge import ConstraintExtractor
        
        extractor = ConstraintExtractor()
        
        assert extractor.cache is not None
    
    def test_extract_default_constraints(self):
        """Test default constraint extraction."""
        from pimaluos.knowledge import ConstraintExtractor
        
        extractor = ConstraintExtractor()
        constraints = extractor.extract_for_zone('R6')
        
        assert constraints.zone_code == 'R6'
        assert constraints.bulk.max_far is not None
        assert constraints.bulk.max_height_ft is not None
    
    def test_validate_proposal_valid(self):
        """Test valid proposal validation."""
        from pimaluos.knowledge import ConstraintExtractor
        
        extractor = ConstraintExtractor()
        
        proposal = {'far': 1.5, 'height_ft': 50, 'use': 'residential'}
        is_valid, violations = extractor.validate_proposal('R6', proposal)
        
        assert is_valid
        assert len(violations) == 0
    
    def test_validate_proposal_invalid(self):
        """Test invalid proposal validation."""
        from pimaluos.knowledge import ConstraintExtractor
        
        extractor = ConstraintExtractor()
        
        proposal = {'far': 10.0, 'height_ft': 200}  # Exceeds R6 limits
        is_valid, violations = extractor.validate_proposal('R6', proposal)
        
        assert not is_valid
        assert len(violations) > 0


class TestZoningConstraintsModel:
    """Tests for Pydantic ZoningConstraints model."""
    
    def test_model_creation(self):
        """Test ZoningConstraints model creation."""
        from pimaluos.knowledge import ZoningConstraints, BulkRegulations
        
        constraints = ZoningConstraints(
            zone_code='R6',
            zone_type='residential',
            bulk=BulkRegulations(max_far=2.0, max_height_ft=65),
        )
        
        assert constraints.zone_code == 'R6'
        assert constraints.bulk.max_far == 2.0
    
    def test_model_serialization(self):
        """Test model JSON serialization."""
        from pimaluos.knowledge import ZoningConstraints
        
        constraints = ZoningConstraints(zone_code='C4')
        json_str = constraints.model_dump_json()
        
        assert 'zone_code' in json_str
        assert 'C4' in json_str

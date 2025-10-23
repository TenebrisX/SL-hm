"""
Django management command to index documents from the dataset.

Usage:
    python manage.py init

This command:
1. Loads documents from train.docs
2. Loads queries from train.titles.queries
3. Loads qrels from train.3-2-1.qrel
4. Generates embeddings for all documents
5. Stores everything in the database
"""
import os
import logging
from typing import List, Tuple

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.db import transaction

from search_api.models import Document, QueryRelevance, Query
from search_api.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Management command to index documents and load qrels."""
    
    help = 'Index documents from the dataset and load qrels'
    
    def __init__(self):
        super().__init__()
        self.embedding_service = EmbeddingService()
    
    def add_arguments(self, parser):
        """Add command-line arguments."""
        parser.add_argument(
            '--data-path',
            type=str,
            default=settings.DATASET_PATH,
            help='Path to the dataset directory'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Batch size for embedding generation'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before indexing'
        )
    
    def handle(self, *args, **options):
        """Execute the command."""
        data_path = options['data_path']
        batch_size = options['batch_size']
        clear_data = options['clear']
        
        self.stdout.write(self.style.SUCCESS('Starting indexing process...'))
        
        # Verify data directory exists
        if not os.path.exists(data_path):
            raise CommandError(
                f"Data directory not found: {data_path}\n"
                f"Please download and extract the dataset to this location."
            )
        
        # Clear existing data if requested
        if clear_data:
            self.stdout.write('Clearing existing data...')
            Document.objects.all().delete()
            QueryRelevance.objects.all().delete()
            Query.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Data cleared'))
        
        try:
            # Step 1: Load and index documents
            self.stdout.write('Step 1: Loading documents...')
            docs = self.load_documents(data_path)
            self.stdout.write(f'Loaded {len(docs)} documents')
            
            self.stdout.write('Step 2: Generating embeddings...')
            self.index_documents(docs, batch_size)
            self.stdout.write(self.style.SUCCESS(
                f'Indexed {len(docs)} documents'
            ))
            
            # Step 2: Load queries
            self.stdout.write('Step 3: Loading queries...')
            queries = self.load_queries(data_path)
            self.save_queries(queries)
            self.stdout.write(self.style.SUCCESS(
                f'Loaded {len(queries)} queries'
            ))
            
            # Step 3: Load qrels
            self.stdout.write('Step 4: Loading qrels...')
            qrels = self.load_qrels(data_path)
            self.save_qrels(qrels)
            self.stdout.write(self.style.SUCCESS(
                f'Loaded {len(qrels)} query-document relevance judgments'
            ))
            
            # Summary
            self.stdout.write(self.style.SUCCESS('\n=== Indexing Complete ==='))
            self.stdout.write(f'Documents indexed: {Document.objects.count()}')
            self.stdout.write(f'Queries loaded: {Query.objects.count()}')
            self.stdout.write(f'Qrels loaded: {QueryRelevance.objects.count()}')
            
        except Exception as e:
            logger.error(f'Indexing failed: {e}', exc_info=True)
            raise CommandError(f'Indexing failed: {e}')
    
    def load_documents(self, data_path: str) -> List[Tuple[str, str]]:
        """
        Load documents from train.docs file.
        
        Format: Each line is a document with format "doc_id\tdocument_text"
        
        Returns:
            List of (doc_id, text) tuples
        """
        docs_file = os.path.join(data_path, 'train.docs')
        
        if not os.path.exists(docs_file):
            raise CommandError(f"Documents file not found: {docs_file}")
        
        documents = []
        with open(docs_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split on first tab
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    logger.warning(
                        f"Skipping malformed line {line_num} in train.docs"
                    )
                    continue
                
                doc_id, text = parts
                
                # Skip empty documents
                if not text or not text.strip():
                    logger.warning(f"Skipping empty document: {doc_id}")
                    continue
                
                documents.append((doc_id, text))
        
        return documents
    
    def index_documents(self, documents: List[Tuple[str, str]], batch_size: int):
        """
        Generate embeddings and save documents to database.
        
        Args:
            documents: List of (doc_id, text) tuples
            batch_size: Number of documents to process at once
        """
        total = len(documents)
        total_batches = (total + batch_size - 1) // batch_size
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Extract texts and IDs
            doc_ids = [doc_id for doc_id, _ in batch]
            texts = [text for _, text in batch]
            
            # Progress update
            progress = (batch_num / total_batches) * 100
            self.stdout.write(
                f'Processing batch {batch_num}/{total_batches} ({progress:.1f}%)...'
            )
            
            try:
                # Generate embeddings for batch
                embeddings = self.embedding_service.embed_batch(texts)
                
                # Save to database
                with transaction.atomic():
                    for j, (doc_id, text) in enumerate(batch):
                        embedding_str = self.embedding_service.serialize_embedding(
                            embeddings[j]
                        )
                        
                        Document.objects.update_or_create(
                            doc_id=doc_id,
                            defaults={
                                'text': text,
                                'embedding': embedding_str
                            }
                        )
                
                self.stdout.write(self.style.SUCCESS(f'✓ Saved batch {batch_num}'))
                
            except Exception as e:
                logger.error(f'Failed to process batch {batch_num}: {e}', exc_info=True)
                self.stdout.write(
                    self.style.ERROR(f'✗ Failed to process batch {batch_num}: {e}')
                )
                raise
    
    def load_queries(self, data_path: str) -> List[Tuple[str, str]]:
        """
        Load queries from train.titles.queries file.
        
        Format: Each line is "query_id\tquery_text"
        
        Returns:
            List of (query_id, query_text) tuples
        """
        queries_file = os.path.join(data_path, 'train.titles.queries')
        
        if not os.path.exists(queries_file):
            raise CommandError(f"Queries file not found: {queries_file}")
        
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    logger.warning(
                        f"Skipping malformed line {line_num} in train.titles.queries"
                    )
                    continue
                
                query_id, query_text = parts
                queries.append((query_id, query_text))
        
        return queries
    
    def save_queries(self, queries: List[Tuple[str, str]]):
        """Save queries to database."""
        with transaction.atomic():
            for query_id, query_text in queries:
                Query.objects.update_or_create(
                    query_id=query_id,
                    defaults={'query_text': query_text}
                )
    
    def load_qrels(self, data_path: str) -> List[Tuple[str, str, int]]:
        """
        Load query relevance judgments from train.3-2-1.qrel file.
        
        Format: "query_id\tdoc_id\tunused\trelevance_score"
        
        Returns:
            List of (query_id, doc_id, relevance_score) tuples
        """
        qrels_file = os.path.join(data_path, 'train.3-2-1.qrel')
        
        if not os.path.exists(qrels_file):
            raise CommandError(f"Qrels file not found: {qrels_file}")
        
        qrels = []
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 4:
                    logger.warning(
                        f"Skipping malformed line {line_num} in train.3-2-1.qrel"
                    )
                    continue
                
                query_id = parts[0]
                doc_id = parts[2]
                relevance_score = int(parts[3])
                
                qrels.append((query_id, doc_id, relevance_score))
        
        return qrels
    
    def save_qrels(self, qrels: List[Tuple[str, str, int]]):
        """Save query relevance judgments to database."""
        with transaction.atomic():
            for query_id, doc_id, relevance_score in qrels:
                QueryRelevance.objects.update_or_create(
                    query_id=query_id,
                    doc_id=doc_id,
                    defaults={'relevance_score': relevance_score}
                )
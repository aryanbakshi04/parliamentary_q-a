import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings as SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from chromadb.config import Settings
import os
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParliamentaryQA:
    def __init__(self):
        self.persist_directory = "./chroma_db"
        self.collection_name = "parliament_qa"
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=self.embedding_model_name,
            cache_folder="./models"
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )

    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Add documents to the vector store in batches"""
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                ids = [str(doc['id']) for doc in batch]
                texts = [doc['text'] for doc in batch]
                metadatas = [doc['metadata'] for doc in batch]
                
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                logger.info(f"Added batch {i//batch_size + 1} to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def query_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store and return relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise

    def get_llm_response(self, query: str, context: List[str]) -> str:
        """Generate response using Gemini"""
        try:
            prompt = f"""
            Given the following parliamentary Q&A context and query, provide a detailed response:
            
            Context:
            {context}
            
            Query: {query}
            
            Provide a clear, factual response based on the given context.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
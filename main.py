import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from tqdm import tqdm
import pickle

# NLP and embedding libraries
from sentence_transformers import SentenceTransformer

# For vector storage - ChromaDB instead of FAISS
import chromadb
from chromadb.utils import embedding_functions

# For RAG generation - using a simple approach here
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JoshuaProjectDataProcessor:
    """Process Joshua Project CSV files for RAG."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.dataframes = {}
        self.file_mapping = {
            'countries': 'AllCountriesListing.csv',
            'languages': 'AllLanguageListing.csv',
            'peoples_across': 'AllPeoplesAcrossCountries.csv',
            'peoples_in_country': 'AllPeoplesInCountry.csv',
            'field_definitions': 'FieldDefinitions.csv',
            'people_country_lang': 'PeopleCtryLangListing.csv',
            'unreached_peoples': 'UnreachedPeoplesByCountry.csv'
        }
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files into dataframes.
        
        Returns:
            Dictionary of dataframes
        """
        logger.info("Loading Joshua Project data files...")
        
        for key, filename in self.file_mapping.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                # Skip the first two rows which contain metadata
                df = pd.read_csv(file_path, skiprows=2, encoding='utf-8')
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Basic data cleaning
                for col in df.columns:
                    # Convert string columns with numbers to numeric if possible
                    if df[col].dtype == 'object':
                        try:
                            # Handle columns with percentage values
                            if df[col].str.contains('%').any():
                                df[col] = df[col].str.replace('%', '').astype(float) / 100
                            else:
                                # Use try/except instead of errors='ignore' to avoid deprecation warning
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except ValueError:
                                    pass
                        except:
                            # If conversion fails, keep as is
                            pass
                
                # Store the dataframe
                self.dataframes[key] = df
                logger.info(f"Loaded {filename} with shape {df.shape}")
            
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
        
        return self.dataframes
    
    def create_documents(self) -> List[Dict]:
        """
        Convert dataframes to Chroma-ready documents with enhanced summaries and metadata.
        
        Returns:
            List of document dictionaries
        """
        logger.info("Creating documents for vector store...")
        documents = []

        for key, df in self.dataframes.items():
            logger.info(f"Processing: {key} ({len(df)} rows)")
            for idx, row in df.iterrows():
                fields = []
                summary = ""
                metadata = {"source": key, "row_id": idx}

                # Capture relevant raw fields
                for col, val in row.items():
                    if pd.notna(val):
                        fields.append(f"{col}: {val}")
                        if col in ["Ctry", "PeopNameInCountry", "PeopName", "Language", "PrimaryLanguageName"]:
                            metadata[col] = str(val)

                # Generate a contextual summary
                if key == "countries":
                    summary = (
                        f"Country: {row.get('Ctry', 'Unknown')} has "
                        f"{row.get('CntPeoples', 'an unknown number of')} people groups, "
                        f"of which {row.get('CntPeoplesLR', 'N/A')} are least-reached. "
                        f"Primary religion: {row.get('PrimaryReligion', 'N/A')}."
                    )

                elif key in ["peoples_in_country", "peoples_across"]:
                    name = row.get("PeopNameInCountry") or row.get("PeopName") or "Unnamed group"
                    summary = (
                        f"People group: {name} in {row.get('Ctry', 'unknown location')} "
                        f"has a population of {row.get('Population', 'N/A')} and "
                        f"speaks {row.get('PrimaryLanguageName', 'N/A')}. "
                        f"Religion: {row.get('PrimaryReligion', 'N/A')}."
                    )

                elif key == "languages":
                    summary = (
                        f"Language: {row.get('Language', 'N/A')}, "
                        f"Bible status: {row.get('BibleStatus', 'N/A')}."
                    )

                # Combine into document text
                doc_text = f"Source: {key}\nSummary: {summary}\n\n" + "\n".join(fields)

                documents.append({
                    "text": doc_text,
                    "metadata": metadata
                })

        logger.info(f"Created {len(documents)} documents.")
        return documents

class VectorStore:
    """Vector store for document embeddings using ChromaDB."""
    
    def __init__(self, embedding_model_name: str = 'paraphrase-MiniLM-L6-v2', persist_directory: str = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            persist_directory: Directory to persist ChromaDB
        """
        logger.info(f"Initializing vector store with model {embedding_model_name}")
        
        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        
        # If persist directory is provided, use PersistentClient instead
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="joshua_project",
            embedding_function=self.embedding_function,
            metadata={"description": "Joshua Project data for RAG"}
        )
        
        self.documents = []
        self.embedding_model_name = embedding_model_name
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector store and generate embeddings.
        
        Args:
            documents: List of document dictionaries
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.documents = documents
        
        # Prepare data for ChromaDB
        ids = [str(i) for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add documents in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            logger.info(f"Adding batch {i}-{end_idx} to ChromaDB")
            
            batch_ids = ids[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            # Add to collection
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
        
        logger.info(f"ChromaDB collection created with {self.collection.count()} vectors")
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save documents
        import pickle
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save embedding model name
        with open(os.path.join(directory, "embedding_model.txt"), "w") as f:
            f.write(self.embedding_model_name)
        
        # Note: If using PersistentClient, ChromaDB is already saved
        logger.info(f"Vector store saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """
        Load vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            VectorStore instance
        """
        # Load embedding model name
        with open(os.path.join(directory, "embedding_model.txt"), "r") as f:
            embedding_model_name = f.read().strip()
        
        # Create vector store with persistence
        vector_store = cls(embedding_model_name=embedding_model_name, persist_directory=directory)
        
        # Load documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            vector_store.documents = pickle.load(f)
        
        logger.info(f"Vector store loaded from {directory} with {vector_store.collection.count()} vectors")
        return vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        # Search in ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Process results
        retrieved_docs = []
        if results and results['documents'] and results['distances']:
            for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                # Get document from stored documents
                idx = int(doc_id)
                doc = self.documents[idx].copy()  # Copy to avoid modifying original
                
                # Convert distance to similarity score (ChromaDB returns L2 distance or cosine distance)
                # Normalize to 0-1 range where 1 is most similar
                similarity_score = 1.0 - min(1.0, distance)
                doc["score"] = float(similarity_score)
                
                retrieved_docs.append(doc)
        
        return retrieved_docs


class RAGModel:
    """RAG model for Joshua Project data."""
    
    def __init__(
        self, 
        vector_store: VectorStore
    ):
        """
        Initialize the RAG model.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        
        logger.info("RAG model initialized with vector store.")
    
    def generate(self, query: str, k: int = 5) -> Dict:
        """
        Generate a response for a query.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with query and retrieved documents
        """
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for query: {query}")
        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs
        }


def build_joshua_project_rag(data_dir: str, output_dir: str = "jp_rag_model") -> RAGModel:
    """
    Build a complete RAG model for Joshua Project data.
    
    Args:
        data_dir: Directory containing CSV files
        output_dir: Directory to save the model
        
    Returns:
        RAGModel instance
    """
    # Process data
    processor = JoshuaProjectDataProcessor(data_dir)
    dataframes = processor.load_data()
    documents = processor.create_documents()
    
    # Create vector store with persistence
    vector_store = VectorStore(persist_directory=output_dir)
    vector_store.add_documents(documents)
    
    # Save vector store (documents and embedding model name)
    vector_store.save(output_dir)
    
    # Create RAG model
    rag_model = RAGModel(vector_store)
    
    return rag_model


def load_joshua_project_rag(model_dir: str) -> RAGModel:
    """
    Load a saved RAG model.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        RAGModel instance
    """
    # Load vector store
    vector_store = VectorStore.load(model_dir)
    
    # Create RAG model
    rag_model = RAGModel(vector_store)
    
    return rag_model


def main():
    """Main function to demonstrate the RAG model."""
    import argparse

    parser = argparse.ArgumentParser(description="Joshua Project RAG Model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CSV files")
    parser.add_argument("--model_dir", type=str, default="./jp_rag_model", help="Directory to save/load model")
    parser.add_argument("--build", action="store_true", help="Build the model from scratch")
    parser.add_argument("--query", type=str, help="Query to test")

    args = parser.parse_args()

    try:
        if args.build:
            logger.info(f"Building model from data in {args.data_dir}...")
            rag_model = build_joshua_project_rag(args.data_dir, args.model_dir)
        else:
            logger.info(f"Loading model from {args.model_dir}...")
            rag_model = load_joshua_project_rag(args.model_dir)

        # Example queries
        example_queries = [
            "What are the primary people groups in Afghanistan?",
            "Which languages are spoken by the Tajik people?",
            "What is the religious composition of Albania?",
            "What are the unreached people groups in Central Asia?",
            "How many languages are spoken in India?",
            "Which people groups have the lowest percentage of Christians?"
        ]

        # Use provided query if available
        if args.query:
            example_queries = [args.query]

        # Generate responses
        for query in example_queries:
            print(f"\nQuery: {query}")
            result = rag_model.generate(query)

            print("\nTop retrieved documents:")
            for i, doc in enumerate(result['retrieved_documents'][:5]):  # Show only top 5
                print(f"  {i+1}. {doc['metadata'].get('source')} - Score: {doc['score']:.3f}")
                if 'Ctry' in doc['metadata']:
                    print(f"     Country: {doc['metadata']['Ctry']}")
                if any(key in doc['metadata'] for key in ['PeopNameInCountry', 'PeopName']):
                    people_name = doc['metadata'].get('PeopNameInCountry') or doc['metadata'].get('PeopName')
                    print(f"     People Group: {people_name}")
            print("-" * 80)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
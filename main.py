import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from tqdm import tqdm

# NLP and embedding libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# For vector storage
import faiss
import pickle

# For RAG generation - using a simple approach here
# The error was in using AutoModelForCausalLM with T5, which is not compatible
# Replacing with the correct model class
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
        Convert dataframes to text documents for embedding.
        
        Returns:
            List of document dictionaries
        """
        logger.info("Converting dataframes to documents...")
        documents = []
        
        for key, df in self.dataframes.items():
            logger.info(f"Processing {key}...")
            
            # Generate documents for each row
            for idx, row in df.iterrows():
                # Format row as text
                text = f"Source: {key}\n"
                
                # Add all fields
                for col, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        text += f"{col}: {value}\n"
                
                # For countries, add a summary
                if key == 'countries' and 'Ctry' in row:
                    text += f"\nSummary: Information about {row['Ctry']}, a country with "
                    if 'PoplPeoples' in row and pd.notna(row['PoplPeoples']):
                        text += f"population of {row['PoplPeoples']}. "
                    if 'CntPeoples' in row and pd.notna(row['CntPeoples']):
                        text += f"It has {row['CntPeoples']} people groups. "
                    if 'PrimaryReligion' in row and pd.notna(row['PrimaryReligion']):
                        text += f"The primary religion is {row['PrimaryReligion']}."
                
                # For people groups, add a summary
                if key in ['peoples_in_country', 'peoples_across']:
                    people_name = None
                    if 'PeopNameInCountry' in row and pd.notna(row['PeopNameInCountry']):
                        people_name = row['PeopNameInCountry']
                    elif 'PeopName' in row and pd.notna(row['PeopName']):
                        people_name = row['PeopName']
                    
                    if people_name:
                        text += f"\nSummary: Information about {people_name}, "
                        if 'Ctry' in row and pd.notna(row['Ctry']):
                            text += f"a people group in {row['Ctry']} "
                        
                        population = None
                        if 'Population' in row and pd.notna(row['Population']):
                            population = row['Population']
                        elif 'PopulationPGAC' in row and pd.notna(row['PopulationPGAC']):
                            population = row['PopulationPGAC']
                        
                        if population:
                            text += f"with population of {population}. "
                        
                        language = None
                        if 'PrimaryLanguageName' in row and pd.notna(row['PrimaryLanguageName']):
                            language = row['PrimaryLanguageName']
                        elif 'PrimaryLanguagePGAC' in row and pd.notna(row['PrimaryLanguagePGAC']):
                            language = row['PrimaryLanguagePGAC']
                        
                        if language:
                            text += f"Their primary language is {language}. "
                        
                        religion = None
                        if 'PrimaryReligion' in row and pd.notna(row['PrimaryReligion']):
                            religion = row['PrimaryReligion']
                        elif 'PrimaryReligionPGAC' in row and pd.notna(row['PrimaryReligionPGAC']):
                            religion = row['PrimaryReligionPGAC']
                        
                        if religion:
                            text += f"Their primary religion is {religion}."
                
                # For languages, add a summary
                if key == 'languages' and 'Language' in row:
                    text += f"\nSummary: Information about the {row['Language']} language. "
                    if 'BibleStatus' in row and pd.notna(row['BibleStatus']):
                        text += f"Bible translation status: {row['BibleStatus']}. "
                
                # Create document object
                doc = {
                    "text": text,
                    "metadata": {
                        "source": key,
                        "row_id": idx
                    }
                }
                
                # Add specific metadata fields if they exist
                for field in ['Ctry', 'ROG3', 'PeopNameInCountry', 'Language', 'PeopName']:
                    if field in row and pd.notna(row[field]):
                        doc["metadata"][field] = row[field]
                
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents")
        return documents


class VectorStore:
    """Vector store for document embeddings using FAISS."""
    
    def __init__(self, embedding_model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        logger.info(f"Initializing vector store with model {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.index = None
        self.embedding_size = self.model.get_sentence_embedding_dimension()
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector store and generate embeddings.
        
        Args:
            documents: List of document dictionaries
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.documents = documents
        
        # Extract texts for embedding
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_size)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(range(len(documents))))
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Vector store saved to {directory}")
    
    @classmethod
    def load(cls, directory: str, embedding_model_name: str = 'paraphrase-MiniLM-L6-v2') -> 'VectorStore':
        """
        Load vector store from disk.
        
        Args:
            directory: Directory to load from
            embedding_model_name: Name of the sentence transformer model
            
        Returns:
            VectorStore instance
        """
        vector_store = cls(embedding_model_name)
        
        # Load index
        vector_store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            vector_store.documents = pickle.load(f)
        
        logger.info(f"Vector store loaded from {directory} with {vector_store.index.ntotal} vectors")
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
        # embed and normalize the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        D, I = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Get results
        results = []
        for i, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx != -1:  # -1 means no result found
                doc = self.documents[idx].copy()  # Copy to avoid modifying original
                doc["score"] = float(1 - score/2)  # Convert L2 distance to similarity score
                results.append(doc)
        
        return results


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
    
    # Create vector store
    vector_store = VectorStore()
    vector_store.add_documents(documents)
    
    # Save vector store
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
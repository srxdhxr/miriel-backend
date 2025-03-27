import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def load_and_test_vectors():
    # Paths
    vector_dir = "vector_store/domain_1"
    index_path = os.path.join(vector_dir, "index.faiss")
    metadata_path = os.path.join(vector_dir, "index.pkl")
    
    print("\n=== Loading Vector Store ===")
    
    # Load FAISS index
    print("\nLoading FAISS index...")
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded with {index.ntotal} vectors")
    print(f"Vector dimension: {index.d}")
    
    # Load metadata
    print("\nLoading metadata...")
    with open(metadata_path, 'rb') as f:
        docstore, id_to_uuid = pickle.load(f)
    
    print(f"\nDocstore type: {type(docstore)}")
    print(f"Number of documents: {len(id_to_uuid)}")
    print(f"Sample UUID: {id_to_uuid[0]}")
    
    # Test search
    print("\n=== Testing Search ===")
    
    # Initialize embedding model
    print("\nInitializing embedding model...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test query
    query = "Credit card?"
    print(f"\nTest query: '{query}'")
    
    # Generate embedding
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Search
    k = 2  # Get top 2 results
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    print("\nSearch Results:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx != -1 and idx < len(id_to_uuid):
            uuid = id_to_uuid[idx]
            doc = docstore.search(uuid)
            
            print(f"\nResult {i+1}:")
            print(f"Index: {idx}")
            print(f"UUID: {uuid}")
            print(f"Distance: {dist}")
            print(f"Score: {1 / (1 + dist)}")
            print("\nDocument Content:")
            print("----------------")
            print(f"Page Content: {doc.page_content[:500]}...")  # Show first 500 chars
            print("\nMetadata:")
            print("----------------")
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    load_and_test_vectors()

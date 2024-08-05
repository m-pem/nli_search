from pinecone import Pinecone, ServerlessSpec
import itertools
import os
import logging

class DatabaseUtil:
    """
    A class that interacts with the Pinecone database. Allowing us to store and query image embeddings.
    """

    def __init__(self):
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index = self.pc.Index("nli-search")
            logging.info("Connected to Pinecone.")

    def upsert_image_embeddings(self, image_embeddings, namespace_id):
        """
        Save the image embeddings to the database.
        :param image_embeddings: A list of image embeddings from the clients library.
        image embedding looks like:  {"id": "A", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        """
        index = self.index
        # Upsert data with 100 vectors per upsert request
        for ids_vectors_chunk in self.chunks(image_embeddings):
            index.upsert(vectors=ids_vectors_chunk,
                         namespace=namespace_id)
    

    def query_vector_database(self, query_vector, k, namespace_id):
        """
        Return the id for the top k matches, .
        :param query_vector: A vector of the users search query.
        :param k: the number of matches to return.
        """
        index = self.index
        query_result = index.query(
             vector=query_vector,
             namespace=namespace_id,
             top_k=k,
             include_metadata=True,
             include_values=False
             )
        return (query_result)
    
    def create_vector_index(self, name, dimension):
        """
        Create our vector index 
        :param name: The name we want to call our index
        :param dimension: 
        :param metric: 
        """
        self.pc.create_index(
             name=name,
             dimension=dimension, 
             metric="cosine", 
             spec=ServerlessSpec(
                  cloud="aws",
                  region="us-west-2"
                  ) 
        )

    def chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
    
from dotenv import load_dotenv

load_dotenv()



# generate embeddings from images
from functions.embedding_utils import EmbeddingUtil
from functions.database_utils import DatabaseUtil

# Instanciate required classes
databaseUtil = DatabaseUtil()
embedding_util = EmbeddingUtil()

# Create a search embedding based off a sample query
query_text = "Beach with palm trees"
text_embedding = embedding_util.calculate_text_embedding(query_text=query_text)

# use the search embedding to query the database
query_result = databaseUtil.query_vector_database(query_vector=text_embedding[0].tolist(), k=3, namespace_id="barekind")
query_result["matches"][0]['id']


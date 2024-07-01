from dotenv import load_dotenv

load_dotenv()

# load images
import cv2
import os

def load_images_from_folder(folder):
    images = []
    image_ids = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            image_ids.append(filename)
    return images, image_ids


images, image_ids = load_images_from_folder("example_images/")

# generate embeddings from images
from functions.embedding_utils import EmbeddingUtil

embedding_util = EmbeddingUtil()
image_embeddings = embedding_util.calculate_image_embedding(images=images)

# need to label the embeddings so that we know what image they are associated with
image_embeddings = [{"id": "Bee_2", "values": image_embeddings[0].tolist()}]

labeled_image_embeddings = [{"id": id, "values": embedding.tolist()} for id, embedding in zip(image_ids, image_embeddings)]

# save embeddings to database
from functions.database_utils import DatabaseUtil
databaseUtil = DatabaseUtil()

databaseUtil.upsert_image_embeddings(image_embeddings=labeled_image_embeddings, namespace_id="barekind")

# create a search embedding based off a sample query
text_embedding = embedding_util.calculate_text_embedding(query_text="Bee socks")

# use the search embedding to query the database
query_result = databaseUtil.query_vector_database(query_vector=text_embedding[0].tolist(), k=3, namespace_id="barekind")
query_result["matches"][0]['id']


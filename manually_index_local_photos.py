from dotenv import load_dotenv
import cv2
import os

from functions.embedding_utils import EmbeddingUtil
from functions.database_utils import DatabaseUtil
from functions.image_processor_utils import ImageProcessor

# Instanciate required classes
databaseUtil = DatabaseUtil()
embedding_util = EmbeddingUtil()
path_to_images = "example_images/"
image_processor = ImageProcessor(path_to_images, batch_size=10)

load_dotenv()

# Process the images in batches and store the embeddings in the database
for images, metadata in image_processor.load_images_from_folder():
    # generate embeddings from images     
    image_embeddings = embedding_util.calculate_image_embedding(images=images)
    # need to label the embeddings so that we know what image they are associated with
    labeled_image_embeddings = [{"id": meta['id'], "metadata": meta, "values": embedding.tolist()} for meta, embedding in zip(metadata, image_embeddings)]
    # save embeddings to database
    databaseUtil.upsert_image_embeddings(image_embeddings=labeled_image_embeddings, namespace_id="barekind")

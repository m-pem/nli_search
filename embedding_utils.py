# This file will be called when the user uploads some photos. The task here will be to extract the image embeddings and then store them in the database.

# https://medium.com/red-buffer/diving-into-clip-by-creating-semantic-image-search-engines-834c8149de56


### This file needs to be updated to use image embeddings rather than the images themselves.

import logging
import time

from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import torch

logger = logging.getLogger("nli_search_server")

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_PREPROCESSOR = "openai/clip-vit-base-patch32"



class EmbeddingUtil:
    def __init__(self):
        self.visionModel = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL)
        self.textModel = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL)
        self.processor = AutoProcessor.from_pretrained(CLIP_PREPROCESSOR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Model and preprocessor loaded.")

    def calculate_image_embedding(self, images):
        """
        Extract and return the image embeddings 
        :param images: A named list of images and their id from the clients library.
        :return: The image embedding and the id of said image
        """
        model = self.visionModel.to(self.device)

        images_processed = self.processor(images=images, return_tensors="pt")
        
        # Obtain the text-image similarity scores
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**images_processed)
            image_embeddings = outputs.image_embeds  # this is the image-text similarity score
        end_time = time.time() - start_time
        logger.info(f"Image embeddings created in {end_time:.3f} seconds.")

        return (image_embeddings)

    def calculate_text_embedding(self, query_text):
        """
        Extract and return the image embeddings 
        :param images: A named list of images and their id from the clients library.
        :return: The image embedding and the id of said image
        """
        textModel = self.textModel.to(self.device)
        
        qyery_text_processed = self.processor(text=query_text, padding=True, return_tensors="pt")

        # inputs = tokenizer(query_text, padding=True, return_tensors="pt")
        # Obtain the text-image similarity scores
        with torch.no_grad():
            start_time = time.time()
            outputs = textModel(**qyery_text_processed)
            text_embeds = outputs.text_embeds
        end_time = time.time() - start_time
        logger.info(f"Query text embeddings created in {end_time:.3f} seconds.")

        return (text_embeds)











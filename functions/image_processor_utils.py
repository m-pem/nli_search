import os
import logging
import cv2
import numpy as np
import uuid
import imghdr
from PIL import Image
import tempfile

logger = logging.getLogger("__name__")

class ImageProcessor:
    """
    A class that processes images from a range of sources

    Args:
        folder (str): The path to the folder containing the images.
        batch_size (int): The number of images to process in each batch.

    Methods:
        get_image_metadata(filepath, img):
            Retrieves metadata for the given image.

        load_image_and_get_metadata(filepath):
            Loads an image from the specified filepath and retrieves its metadata.

        load_images_from_folder():
            Loads images from the specified folder in batches and yields the images and their metadata.

    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_image_metadata_from_filepath(self, filepath, img):
        """
        Retrieves metadata for the given image.

        Args:
            filepath (str): The path to the image file.
            img (PIL.Image.Image): The loaded image.

        Returns:
            dict: A dictionary containing the image metadata, including id, path, size_x, size_y, file_type, and file_size_MB.

        """
        size_x, size_y = img.size
        file_type = img.format
        file_size = os.path.getsize(filepath) 
        unique_id = str(uuid.uuid4())
        return {
            "id": unique_id,
            "name": "",
            "source": "local_file_system", 
            "image_height": size_x,
            "image_width": size_y,
            "file_type": file_type,
            "file_size": file_size,
            "microsoft_drive_file_id": "",
            "local_file_path": filepath
        }

    def load_image_and_get_metadata(self, filepath):
        """
        Loads an image from the specified filepath and retrieves its metadata.

        Args:
            filepath (str): The path to the image file.

        Returns:
            tuple: A tuple containing the loaded image and its metadata. If the file is not an image, returns (None, None).

        """
        if imghdr.what(filepath) is not None:  # Check if the file is an image
            img = cv2.imread(filepath)
            image_for_meta = Image.open(filepath)
            image = img
            metadata = self.get_image_metadata_from_filepath(filepath, image_for_meta)
            return image, metadata
        else:
            logger.warning(f'File is not an image: {filepath}')
            return None, None

    def load_images_from_folder(self, folder):
        """
        Loads images from the specified folder in batches and yields the images and their metadata.

        Yields:
            tuple: A tuple containing a list of loaded images and a list of their corresponding metadata.

        """
        for subdir, dirs, files in os.walk(folder):
            logger.info(f'Processing {len(files)} files in {subdir}') 
            print(f'Processing {len(files)} files in {subdir}') 
            for i in range(0, len(files), self.batch_size):
                batch_files = files[i:i+self.batch_size]
                images = []
                metadata = []
                for filename in batch_files:
                    filepath = os.path.join(subdir, filename)
                    image, image_metadata = self.load_image_and_get_metadata(filepath)
                    if image is not None and image_metadata is not None:
                        images.append(image)
                        metadata.append(image_metadata)
                    else:
                        print(f'Skipped file: {filepath}') 
                yield images, metadata
    
    def load_heic_image(self, heic_file_path):
        try:
            image = Image.open(heic_file_path)
            return image
        except Exception as e:
            logger.error(f'Failed to load HEIC image: {heic_file_path}', exc_info=True)
            return None
        
    def convert_to_jpeg(self, image):
        """
        Converts the given image to JPEG format.

        Args:
            image (PIL.Image.Image): The image to be converted.

        Returns:
            PIL.Image.Image: The converted image in JPEG format.

        """
        try:
            jpeg_image = image.convert("RGB")
            return jpeg_image
        except Exception as e:
            logger.error("Failed to convert image to JPEG", exc_info=True)
            return None

    def write_temp_image_file(self, image_array, quality):
    # Write a NumPy array as a temporary image file
        try:
            img = Image.fromarray(image_array.astype('uint8'))
            temp_file = None
            img_format = None
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(temp_file, format='PNG', optimize=True, quality=quality)
                img_format = 'PNG'
            else:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg")
                img.save(temp_file, format='JPEG', optimize=True, quality=quality)
                img_format = 'JPEG'
            return temp_file, img_format
        except Exception as e:
            logger.error(f"Error occurred while writing temporary image file: {e}")
            raise e
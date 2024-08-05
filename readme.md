# NatLang Image Search

## Description

Search your photos using natural language! Using open AI's CLIP model and a vector database we can use natural language to search images with no fine tuning needed.

The CLIP model uses contrastive learning on images and text to get some really impressive zero shot predictions on images. For info on this have a look at the openAI blog: https://openai.com/index/clip/

This project indexes your photos by running them through the model obtaining the resulting vectors and storing them in a vector database. With all of the photos processed you are then able to use a search phrase, which is converted to a vector by the model to search this database, returning the closest matching vector you have stored. With this you get the file name of the image you're after.

This is a small snippet of some functionality that was used in a web application built for a client and is meant as a prof of concept. If you are interested in searching images using natural language please reach out!

## Installation

This project is built using python and I strongly recommend you use a virtual environment to manage the packages for it.

**Vector DB requirements**
- Create a pinecone account at: https://www.pinecone.io/
- Create a serverless API key so you are able to access your account
- Make note of that key as we will need to add it to our .env file in our local project

**Local requirements**
- Clone the project repo
- Create a python virtual environment within root project folder
- Activate the virtual environment and install the dependencies using 'pip install -r requirements.txt'
- Create a .env file, you can use the example_env.txt file as a template, just replace the X's with the API key from your pinecone account

### Dependencies

- Vector database (I've used pinecone)

## Usage

With everything set up you can get started
- Start by creating a vector index on your pinecone server, use the "manually_create_vector_index.py" script to do this. This will create a database for the results of the image processing
- With this created we can now index your photos. Use the "manually_index_local_photos.py" script to vectorise all your photos at the given path
- Finally we can search these photos using the "search_photos.py" script, you will receive the file name of the top n matching photos which you can then search in your folder


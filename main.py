from glob import glob
import os

import argparse

import chromadb

from describe import describe_image

DATA_PREFIX = "./chromadb-data"

def get_client(persistent=False):
    if persistent:
        return chromadb.PersistentClient(path=DATA_PREFIX)
    else:
        return chromadb.Client()

class ImageIndex(object):

    def __init__(self, collection_name):
        self.client = None
        self.collection_name = collection_name

    def __enter__(self):
        self.client = get_client(persistent=True)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def index_image(self, id, image_data, prompt='describe this image', model='llava-phi3'):
        description = describe_image(image_data, prompt=prompt, model=model)
        self.collection.add(ids=id, documents=description)
        return description
    
    def search_images(self, query):
        return self.collection.query(query_texts=[query])
    

def main_index(directory):
    with ImageIndex('images') as index:
        pngs = glob(os.path.join(directory, '*.png'))
        jpgs = glob(os.path.join(directory, '*.jpg'))
        image_paths = [*pngs, *jpgs]
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            try:
                description = index.index_image(image_path, image_data)
                print(f"Indexed {image_path}: {description}")
            except:
                print(f"Failed to index {image_path}")

def main_query(query):
    with ImageIndex('images') as index:
        results = index.search_images(query)
        top_hit_id = results['ids'][0][0]
        top_hit_document = results['documents'][0][0]
        print(f'Top hit for query "{query}" is {top_hit_id}')
        print(f'Description:\n{top_hit_document}')

def main(directory, query='image'):
    if directory:
        main_index(directory)
    main_query(query)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory containing images to index', default=None)
    parser.add_argument('--query', type=str, help='Query to search for', default='image')
    args = parser.parse_args()
    main(args.directory, args.query)
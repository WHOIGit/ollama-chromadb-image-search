from glob import glob
import os
import json
import sys
from tqdm import tqdm
from PIL import Image

import argparse

import chromadb

from describe import describe_image
from prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


DATA_PREFIX = "./chromadb-data"

def get_client(persistent=False):
    if persistent:
        return chromadb.PersistentClient(path=DATA_PREFIX)
    else:
        return chromadb.Client()

class ImageIndex(object):

    def __init__(self, collection_name, prompt=DEFAULT_USER_PROMPT, system=DEFAULT_SYSTEM_PROMPT, model='llava-phi3'):
        self.client = None
        self.collection_name = collection_name
        self.prompt = prompt
        self.system = system
        self.model = model

    def __enter__(self):
        self.client = get_client(persistent=True)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def index_image(self, id, image_data):
        description = describe_image(image_data, prompt=self.prompt, system=self.system, model=self.model)
        self.collection.add(ids=id, documents=description)
        return description
    
    def search_images(self, query):
        return self.collection.query(query_texts=[query])
    

def main_index(directory, prompt=DEFAULT_USER_PROMPT, system=DEFAULT_SYSTEM_PROMPT, model='llava-phi3'):
    with ImageIndex('images', prompt=prompt, system=system, model=model) as index:
        pngs = glob(os.path.join(directory, '*.png'))
        jpgs = glob(os.path.join(directory, '*.jpg'))
        image_paths = [*pngs, *jpgs]
        for image_path in tqdm(image_paths):
            with open(image_path, 'rb') as f:
                image_data = f.read()
            try:
                description = index.index_image(image_path, image_data)
                # print(f"Indexed {image_path}: {description}")
            except:
                # print(f"Failed to index {image_path}")
                pass

def main_query(query):
    with ImageIndex('images') as index:
        results = index.search_images(query)
        # json.dump(results, sys.stdout, indent=2)
        top_hit_id = results['ids'][0][0]
        top_hit_document = results['documents'][0][0]
        print(f'Top hit for query "{query}" is {top_hit_id}')
        print(f'Description:\n{top_hit_document}')
        if os.path.exists(top_hit_id):
            image = Image.open(top_hit_id)
            image.save('./top_hit.jpg')

def main(directory, query='image', prompt=DEFAULT_USER_PROMPT, system=DEFAULT_SYSTEM_PROMPT, model='llava-phi3'):
    if directory:
        main_index(directory, prompt=prompt, system=system, model=model)
    main_query(query)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory containing images to index', default=None)
    parser.add_argument('--query', type=str, help='Query to search for', default='image')
    parser.add_argument('--model', type=str, help='Model to use for description', default='llava-phi3')
    parser.add_argument('--prompt', type=str, help='Prompt to use for description', default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument('--system', type=str, help='System prompt to use for description', default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()
    main(args.directory, args.query)
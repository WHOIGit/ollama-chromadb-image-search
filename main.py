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

DEFAULT_MODEL = 'qwen2.5vl'

def get_client(persistent=False):
    if persistent:
        return chromadb.PersistentClient(path=DATA_PREFIX)
    else:
        return chromadb.Client()

class ImageIndex(object):

    def __init__(self, collection_name, prompt=DEFAULT_USER_PROMPT, system=DEFAULT_SYSTEM_PROMPT, skip_existing=True, model=DEFAULT_MODEL):
        self.client = None
        self.collection_name = collection_name
        self.prompt = prompt
        self.system = system
        self.model = model
        self.skip_existing = skip_existing

    def __enter__(self):
        self.client = get_client(persistent=True)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def exists(self, id):
        docs = self.collection.get(ids=[id], include=['documents'])['documents']
        return len(docs) > 0
    
    def index_image(self, id, image_data):
        description = describe_image(image_data, prompt=self.prompt, system=self.system, model=self.model)
        self.collection.add(ids=id, documents=description)
        return description
    
    def search_images(self, query):
        return self.collection.query(query_texts=[query])
    
    def index_directory(self, directory):
        pngs = glob(os.path.join(directory, '*.png'))
        jpgs = glob(os.path.join(directory, '*.jpg'))
        image_paths = [*pngs, *jpgs]
        for image_path in tqdm(image_paths):
            if self.skip_existing and self.exists(image_path):
                continue
            with open(image_path, 'rb') as f:
                image_data = f.read()
            try:
                description = self.index_image(image_path, image_data)
                print(f"Indexed {image_path}: {description}")
            except:
                print(f"Failed to index {image_path}")
                pass
    
    def search(self, query):
        results = self.search_images(query)
        # json.dump(results, sys.stdout, indent=2)
        top_hit_id = results['ids'][0][0]
        top_hit_document = results['documents'][0][0]
        print(f'Top hit for query "{query}" is {top_hit_id}')
        print(f'Description:\n{top_hit_document}')
        if os.path.exists(top_hit_id):
            image = Image.open(top_hit_id)
            # remove alpha channel if present
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save('./top_hit.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory containing images to index', default=None)
    parser.add_argument('--query', type=str, help='Query to search for', default=None)
    parser.add_argument('--model', type=str, help='Model to use for description', default=DEFAULT_MODEL)
    parser.add_argument('--prompt', type=str, help='Prompt to use for description', default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument('--system', type=str, help='System prompt to use for description', default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument('--skip-existing', action='store_true', help='Skip images that are already indexed')
    args = parser.parse_args()

    with ImageIndex('images', prompt=args.prompt, system=args.system, model=args.model, skip_existing=args.skip_existing) as index:
        if args.directory:
            index.index_directory(args.directory)
        elif args.query:
            index.search(args.query)
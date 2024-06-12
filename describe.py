from ollama import generate

def describe_image(image_data, prompt='describe this image', model='llava-phi3'):
    result = generate(model=model, prompt=prompt, images=[image_data])
    return result['response']

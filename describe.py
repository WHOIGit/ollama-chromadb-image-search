from ollama import generate

from prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

def describe_image(image_data, prompt=DEFAULT_USER_PROMPT, system=DEFAULT_SYSTEM_PROMPT, model='llava-phi3'):
    result = generate(model=model, prompt=prompt, images=[image_data])
    return result['response']

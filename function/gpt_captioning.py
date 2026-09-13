from openai import OpenAI
import os
import base64
from dotenv import load_dotenv
from tqdm import tqdm
from function.gpt_models import completion_kwargs

default_prompt = "Generate a descriptive caption for the given image. Please respond only with the caption!"
default_prompt_2 = ("Generate a caption to the given image that is descriptive and precise."
                    "Focus on identifying the key elements or actions visible in the image using very simple language"
                    "without adding interpretations, subjective opinions, or unnecessary details."
                    "Please respond only with the caption!")
default_prompt_3 = ("Generate a caption to the given image that is descriptive and precise."
                    "Focus on identifying the key elements or actions visible in the image without adding interpretations, subjective opinions, or unnecessary details."
                    "Respond in simple language as if you are explaining it to a 5 year old child."
                    "Please respond only with the caption!")

default_prompt_4 = ("Generate a caption to the given image that is descriptive and precise."
                    "Focus on identifying the key elements or actions visible in the image without adding"
                    "interpretations, subjective opinions, or unnecessary details."
                    "The caption will be used to identify potential bias keywords within the miss-classified images"
                    "Please respond only with the caption!")


_client = None


def _get_client():
    # Lazy init: only touches OPENAI_API_KEY when a gpt-* option is actually used,
    # so importing this module doesn't require a key.
    global _client
    if _client is None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_script_dir, "..", ".env")
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file")
        _client = OpenAI(api_key=api_key)
    return _client


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



def construct_message(prompt, image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    message = [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt,
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
            },
            },
        ],
        }
    ]
    return message


def generate_gpt_caption(img_path, prompt=default_prompt_2, model="gpt-4o-mini"):
    message = construct_message(prompt, img_path)
    chat_completion = _get_client().chat.completions.create(
        messages=message,
        model=model,
        **completion_kwargs(model),
    )
    if chat_completion.choices[0].finish_reason == "length":
        tqdm.write(f"WARNING: {model} hit the output token limit for '{img_path}'; the caption was cut off.")
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

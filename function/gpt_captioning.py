from openai import OpenAI
from dotenv import load_dotenv
import base64

default_prompt = "Generate a descriptive caption for the given image"


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # load api_key from .dotenv file
    # api_key="...",
)


def configure():
    load_dotenv()


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
    ],
    return message.choices[0]


def generate_gpt_caption(img_path, prompt=default_prompt, model="gpt-4o-mini"):
    message = construct_message(prompt, img_path)
    chat_completion = client.chat.completions.create(
        messages=message,
        model=model,
        # max_tokens=300,
    )
    return chat_completion

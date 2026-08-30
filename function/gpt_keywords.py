from openai import OpenAI
import os
from dotenv import load_dotenv
import ast
import re

default_prompt = ("Analyse the following captions which belong to misclassified images of a single class. Please find 10 to 20 potential bias keywords!"
                  "captions = {captions}"
                  "Please respond only with the 10 to 20 bias keywords in a python list!")


# Load the .env file and initialize the api endpoint
# Get the absolute path of the .env file
current_script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_script_dir, "..", ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("OPENAI_API_KEY")  # load api_key from .env file
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")
client = OpenAI(api_key=api_key)


def construct_message(prompt, captions):
    prompt = prompt.replace("{captions}", str(captions))
    message = [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt,
            },
        ],
        }
    ]
    return message


def extract_list_from_string(input_string):
    # Use a regular expression to extract the portion that looks like a Python list
    match = re.search(r'\[.*?\]', input_string, re.DOTALL)
    if match:
        # Safely evaluate the extracted string as a Python list
        extracted_list = ast.literal_eval(match.group(0))
        return extracted_list
    else:
        raise ValueError("No valid list found in the input string.")



def extract_gpt_keywords(captions, prompt=default_prompt, model="gpt-4o-mini"):
    message = construct_message(prompt, captions)
    chat_completion = client.chat.completions.create(
        messages=message,
        model=model,
        max_tokens=300,
    )
    response = chat_completion.choices[0].message.content
    print(response)
    try:
        # Attempt to extract the list from the input string
        result = extract_list_from_string(response)
        print("Extracted list:", result)
        return result
    except ValueError as e:
        # Handle the case where no valid list is found
        print("Error:", e)
        return None
    except Exception as e:
        # Handle any other unexpected errors
        print("Unexpected error:", e)
        return None

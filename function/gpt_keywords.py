from openai import OpenAI
import os
from dotenv import load_dotenv
from function.gpt_models import completion_kwargs
import ast
import re

default_prompt = ("Analyse the following captions which belong to misclassified images of a single class. Please find 10 to 20 potential bias keywords!"
                  "captions = {captions}"
                  "Please respond only with the 10 to 20 bias keywords in a python list!")


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
    chat_completion = _get_client().chat.completions.create(
        messages=message,
        model=model,
        **completion_kwargs(model),
    )
    if chat_completion.choices[0].finish_reason == "length":
        # A cut-off list can't be parsed, and an empty keyword list invalidates the whole run.
        raise RuntimeError(f"{model} hit the output token limit while extracting keywords; "
                           "the response was cut off. Raise max_tokens in completion_kwargs.")
    response = chat_completion.choices[0].message.content
    #print(response)
    try:
        # Attempt to extract the list from the input string
        result = extract_list_from_string(response)
        #print("Extracted list:", result)
        return result
    except ValueError as e:
        # Handle the case where no valid list is found
        print("Error:", e)
        return []
    except Exception as e:
        # Handle any other unexpected errors
        print("Unexpected error:", e)
        return []

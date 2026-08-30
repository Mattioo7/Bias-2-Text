from collections import Counter
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv


### IGNORE THIS SCRIPT FOR NOW ###


# Load the .env file and initialize the api endpoint
# Get the absolute path of the .env file
current_script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_script_dir, "..", ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("OPENAI_API_KEY")  # load api_key from .env file
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")
client = OpenAI(api_key=api_key)


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


def query_gpt_as_vqa(image, keywords, model):
    message = construct_message(prompt, img_path)
    print("Input: ", message)
    chat_completion = client.chat.completions.create(
        messages=message,
        model=model,
        max_tokens=300,
    )
    print("Output: ", chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content


def query_vqa_model(image_dir, image, keywords, model="gpt-4o-mini"):
    """
    Simulate querying a Visual Question Answering model for keyword presence in an image.
    This function should be replaced with the actual model inference logic.
    Args:
        image: The image to query.
        keywords: A list of keywords to check in the image.
    Returns:
        A binary vector (list of 0s and 1s) indicating whether each keyword is present in the image.
    """
    # Mock response: Replace with actual VQA model call
    # Here, we'll just randomly simulate results for demonstration purposes.
    import random
    return [random.choice([0, 1]) for _ in keywords]


def calculate_keyword_occurrences(image_dir, images, keywords):
    """
    Calculate keyword occurrences for a list of images.
    Args:
        image_dir: Path to image directory
        images: List of images to query.
        keywords: List of keywords to check.
    Returns:
        A Counter object with counts of each keyword.
    """
    keyword_occurrences = Counter()
    for image in images:
        # Get the binary vector for keyword presence
        binary_vector = query_vqa_model(image_dir, image, keywords)
        # Update the counter with the presence of keywords
        for keyword, presence in zip(keywords, binary_vector):
            if presence:
                keyword_occurrences[keyword] += 1
    return keyword_occurrences


def calculate_vqs_score(image_dir, corr_images, wrong_images, keywords):
    # This methods is called for each class individually
    # It gets a set of correctly and incorrectly classified images and also potential bias keywords for this class
    # Then a visual questioning answering model is called for each image and it should respond which of the keywords are present in the image
    # (respond with a binary vector '1' occurs, '0' doesnt occur) or (respond with a keyword list that occur in the image and build the binary vector myself)
    # Then sum the occurences of each word, separately for correctly and wrongly classified images
    # Then two variables store the ratio of occurences for each keyword over the number of all keywords (percentage). This is done for correctly and wrong classified images
    # The idea is that you can identify inbalances in the word occurences
    # These two variables are returned
    """
    Calculate the VQS score based on correctly and incorrectly classified images.
    Args:
        image_dir: Path to image directory
        corr_images: List of correctly classified images.
        wrong_images: List of incorrectly classified images.
        keywords: List of bias-related keywords.
    Returns:
        Two dictionaries representing the percentage of keyword occurrences
        in correctly and incorrectly classified images.
    """
    # Calculate keyword occurrences
    correct_keyword_occurrences = calculate_keyword_occurrences(image_dir, corr_images, keywords)
    wrong_keyword_occurrences = calculate_keyword_occurrences(image_dir, wrong_images, keywords)

    # Total number of images
    total_correct = len(corr_images)
    total_wrong = len(wrong_images)

    # Calculate percentages for each keyword
    correct_ratios = {kw: (correct_keyword_occurrences[kw] / total_correct * 100) if total_correct > 0 else 0 for kw in
                      keywords}
    wrong_ratios = {kw: (wrong_keyword_occurrences[kw] / total_wrong * 100) if total_wrong > 0 else 0 for kw in
                    keywords}

    return correct_ratios, wrong_ratios

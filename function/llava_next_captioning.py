from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

default_prompt = "Generate a descriptive caption for the given image. Please respond only with the caption!"

# Initialize the processor and model
# torch.cuda.empty_cache()  # i thought that it will help
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    load_in_4bit=True # Remove to get error: torch.OutOfMemoryError: CUDA out of memory.
)
model.to("cuda:0")

def generate_llava_next_caption(img_path, prompt=default_prompt, max_new_tokens=100):

    # Load the image
    print("Loading image...")
    image = Image.open(img_path)

    # Define the conversation format for LLaVA-Next
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    # Apply the chat template to format the prompt
    print("Applying chat template...")
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare the inputs for the model
    print("Processing inputs...")
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to("cuda:0")

    # Generate the output
    print("Generating caption...")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode and return the generated caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    print(caption)
    return caption

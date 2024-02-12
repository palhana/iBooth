import os
import openai
import logging

logging.basicConfig(level=logging.ERROR)

def get_openai_chat_completion(prompt_text):
    """
    Requests a chat completion from the OpenAI API based on the given prompt text.

    Parameters:
    - prompt_text: A string containing the prompt to send to the API.

    Returns:
    - The chat completion object returned by the OpenAI API.
    
    Raises:
    - ValueError: If the OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set this variable with your OpenAI API key.")

    openai.api_key = api_key
    try:
        # Adjust the function call according to the latest OpenAI Python client library
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4-0613",  # Ensure this model supports chat, and use the correct identifier
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        return chat_completion
    except openai.error.OpenAIError as e:
        logging.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    prompt_text = "what is the curcumference of the earth in km?"
    completion = get_openai_chat_completion(prompt_text)
    if completion:
        # Adjust according to the actual structure of the chat completion response
        print(completion['choices'][0]['message']['content'].strip())
    else:
        print("Failed to get a chat completion.")


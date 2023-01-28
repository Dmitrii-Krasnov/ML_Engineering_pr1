"""
Functions for text generation
"""

import torch
from streamlit import cache
from transformers import pipeline

cuda_device = 0 if torch.cuda.is_available() else -1


@cache(allow_output_mutation=True)
def load_model(max_length: int):
    """
    Load the ML model with specified max length

    :param max_length: output length of generated text
    :return: pipeline with model for text generation
    """

    if max_length <= 8:
        return None

    return pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        max_length=max_length,
        device=cuda_device
    )


def generate_text(generation_text: str, generation_len: int):
    """
    Generate text until the output length (which includes the context length) reaches generation_len

    :param generation_text (str): source text to be processed
    :param generation_len (int): output length
    :return: generated text
    """

    if len(generation_text) < 10:
        raise ValueError("Source text is too short")
    if generation_len <= 8:
        raise ValueError("Max length is less than 8")

    word_count = len(str.split(generation_text))
    if word_count <= generation_len:
        raise ValueError("Source text is too short")

    generator = load_model(generation_len)

    if generator:
        return generator(generation_text)[0]['summary_text']

    raise SystemError("Generation is impossible!")

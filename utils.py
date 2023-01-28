import torch
from streamlit import cache
from transformers import pipeline

cuda_device = 0 if torch.cuda.is_available() else -1


@cache(allow_output_mutation=True)
def load_model(txt_length):
    """Load the AI model and check text length"""
    if txt_length <= 8:
        return None

    return pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        max_length=txt_length,
        device=cuda_device
    )


def generate_text(generation_text, generation_len):
    """Function where AI generate short text"""
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

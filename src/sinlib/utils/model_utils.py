import torch
from pathlib import Path
from os import path
from sinlib.utils.models.transliterator_model import BiLSTMTranslator
from sinlib.utils.dataset_utils import load_tokenizer

CURRENT_PATH = path.dirname(path.abspath(__file__))
MODELS_PATH = path.join(CURRENT_PATH, "models")
CHECKPOINT_NAME = "transliterator-checkpoint.pth"
HIDDEN_SIZE = 128


def detect_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transliterator_model():
    tokenizer = load_tokenizer()
    input_size = len(tokenizer)
    output_size = len(tokenizer)
    hidden_size = HIDDEN_SIZE
    filepath = Path(MODELS_PATH) / CHECKPOINT_NAME

    device = detect_device()
    model = BiLSTMTranslator(input_size, hidden_size, output_size).to(device)
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def inference(model, tokenizer, input_text):
    model.eval()
    device = detect_device()
    tokens_to_ignore = [tokenizer.vocab_map[tok] for tok in tokenizer.special_tokens]

    # Tokenize and encode input text
    input_encoded = tokenizer(input_text)
    input_tensor = (
        torch.tensor(input_encoded).unsqueeze(0).to(device)
    )  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        predicted = output.argmax(dim=-1)

        # Remove special tokens and decode
        pred = [p for p in predicted[0].tolist() if p not in tokens_to_ignore]
        translated_text = tokenizer.decode(pred)

    return translated_text

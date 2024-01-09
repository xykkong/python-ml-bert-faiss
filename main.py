import speech_recognition as sr
from pydub import AudioSegment
from transformers import BertTokenizer, BertModel
import faiss
import torch
import os
import pandas as pd
import typer
import ast
from loguru import logger
from typing_extensions import Annotated
from tabulate import tabulate

app = typer.Typer()

DATA_PATH = "./data"


def convert_mp4_to_wav(video_file: str, output_file: str) -> None:
    logger.info("Started convert_mp4_to_wav")
    video = AudioSegment.from_file(f"{DATA_PATH}/{video_file}", format="mp4")
    audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(f"{DATA_PATH}/{output_file}", format="wav")


def transcript_audio(audio_file: str, output_file: str) -> None:
    logger.info("Started transcript_audio")
    # Initialize recognizer
    recognizer = sr.Recognizer()

    with sr.AudioFile(f"{DATA_PATH}/{audio_file}") as source:
        # read the entire audio file
        audio_text = recognizer.record(source)

    transcript = recognizer.recognize_google(audio_text)

    # Save the transcript
    with open(f"{DATA_PATH}/{output_file}", "w") as file:
        file.write(transcript)


def split_transcript_in_sentences(file_path: str, chunk_size: int = 1024) -> list[str]:
    logger.info("Started split_transcript_in_sentences")
    sentences = []
    buffer = ''
    with open(f"{DATA_PATH}/{file_path}", 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            text = (buffer + chunk).split('.')
            for sentence in text[:-1]:
                sentences.append(sentence.strip() + ".")
            buffer = text[-1]  # Incomplete sentence, stored for the next iteration

    if buffer:  # Process any remaining incomplete sentence at the end of the file
        sentences.append(buffer.strip() + ".")
 
    return sentences


def encoding_sentences(sentences: list[str]) -> tuple[str, str]:
    logger.info("Started encoding_sentences")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoding = tokenizer.batch_encode_plus(
        sentences,                 # List of input texts
        padding=True,              # Pad to the maximum sequence length
        truncation=True,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',       # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP
    )
    input_ids = encoding['input_ids']           # Token IDs
    attention_mask = encoding['attention_mask']  # Attention mask
    return input_ids, attention_mask


def generate_embeddings(input_ids: list[int], attention_mask: list[int]) -> list[float]:
    logger.info("Started generate_embeddings")
    model = BertModel.from_pretrained('bert-base-uncased')
    outputs = None

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # This contains the embeddings

    return embeddings


def store_in_faiss(sentences: list[str], embeddings: list[int], index_file: str) -> None:
    logger.info("Started store_in_faiss")
    dimension = embeddings.shape[1]

    # Index that performs a brute-force L2 distance search
    # It can be very slow with a large dataset as it scales linearly with the number of indexed vectors
    index = faiss.IndexFlatL2(dimension)  # BERT embedding size
    faiss.normalize_L2(embeddings.numpy())
    index.add(embeddings)
    faiss.write_index(index, f'{DATA_PATH}/{index_file}')

    with open(f'{DATA_PATH}/{index_file}.raw', 'w') as file:
        for sentence in sentences:
            file.write(sentence + "\n")


# Step 4: CLI for Querying
def search_top_k(sentences_df, index, embeddings, k: int = 3):
    logger.info("Started search_top_k")
    distances, ann = index.search(embeddings.numpy(), k)  #ann is the approximate nearest neighbour
    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
    merge = pd.merge(results, sentences_df, left_on='ann', right_index=True)

    print(tabulate(merge, tablefmt='fancy_grid'))


@app.command()
def index(video_file: str, skip_transcription: Annotated[bool, typer.Option("--skip")] = False):
    logger.info("Started index command")
    prefix = video_file.split('.')[0]
    audio_file = f"{prefix}_audio.wav"
    transcript_file = f"{prefix}_transcript.txt"
    index_file = f"{prefix}.index"
    if not skip_transcription:
        convert_mp4_to_wav(video_file, output_file=audio_file)
        transcript_audio(audio_file=audio_file, output_file=transcript_file)
    sentences = split_transcript_in_sentences(transcript_file)
    input_ids, attention_mask = encoding_sentences(sentences)
    embeddings = generate_embeddings(input_ids, attention_mask)
    store_in_faiss(sentences, embeddings, index_file)


@app.command()
def search(query: Annotated[str, typer.Option("--query", "-q")], index_file: Annotated[str, typer.Option("--index", "-i")], k: Annotated[int, typer.Option("--k", "-k")]):
    logger.info("Started search command")
    input_ids, attention_mask = encoding_sentences([query + "."])
    embeddings = generate_embeddings(input_ids, attention_mask)

    sentences = []
    with open(f'{DATA_PATH}/{index_file}.raw') as file:
        for line in file:
            sentences.append(line.strip())
    sentences_df = pd.DataFrame(sentences)

    index = faiss.read_index(f"{DATA_PATH}/{index_file}")

    return search_top_k(sentences_df, index, embeddings, k)


if __name__ == "__main__":
    app()


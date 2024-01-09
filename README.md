# python-bert-faiss

The high level goal is to take a series of video files, transcribe it and make the transcript searchable using embedding based search.
● Take input as a set of video files (.mp4), run it through Speech to text engine (S2T).
● Chunk the output transcript, vectorize it using open source text embedding models (BERT).
● Store the embeddings of the transcript chunks in an open source graph database which can support vector search (FAISS).
● On the querying side, the user input is a text query, which should be used to search the top K results against the vector DB.

# Installation

## Creating and activating your virtual env
```
  python -m venv venv
  source venv/bin/activate
```

## Installing dependencies
```
    pip install -r requirements.txt
```

## Indexing a video. This can take a long time to run, specially if you are using a free license
```
    python main.py index video1.mp4
```

# Running

## In case you already have the transcription of the video, you can skip transcription step
```
    python main.py index video1.mp4 --skip 
```

## Querying a string
```
    python main.py search -q "united states" -i video1.index -k 3
```

## Using help
```
    python main.py search --help
```

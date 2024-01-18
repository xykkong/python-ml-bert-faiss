# python-ml-bert-faiss

The high-level goal is to take a series of video files, transcribe it, and make the transcript searchable using embedding-based search.
- Take input as a set of video files (.mp4), and run it through Speech the text engine (S2T).
- Chunk the output transcript, and vectorize it using open-source text embedding models (BERT).
- Store the embeddings of the transcript chunks in an open-source graph database that can support vector search (FAISS).
- On the querying side, the user input is a text query, which should be used to search the top K results against the vector DB.

# Running with Docker

## Building Image
```
    docker build -t mlsearch .
```

## Generate transcript from video. This can take a long time to run, specially using a free license.
```
    docker run -v .:/app/ mlsearch transcript YOUR_VIDEO.mp4
```

## Indexing a transcript.
```

    docker run -v .:/app/ mlsearch index sample.txt 
```

## Querying a string
```
    docker run -v .:/app/ mlsearch search -q "united states" -i sample.index -k 3
```

## Using help
```
    docker run -v .:/app/ mlsearch search --help
``````

# Running locally

## Creating and activating your virtual env
```
  python -m venv venv
  source venv/bin/activate
```

## Installing dependencies
```
    pip install -r requirements.txt
```

## Generate transcript from video. This can take a long time to run, specially using a free license.
```
    python main.py transcript YOUR_VIDEO.mp4
```

## Indexing a transcript.
```
    python main.py index sample.txt
```

## Querying a string
```
    python main.py search -q "united states" -i sample.index -k 3
```

## Using help
```
    python main.py search --help
```


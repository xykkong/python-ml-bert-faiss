# video-transcription-bert-faiss

# Installation

##Creating and activating your virtual env
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

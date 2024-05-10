# RAG POC

This project is a simple POC for RAG. Its focussed on the domain of medical/EEG clinical reports.

These reports come from the public domain Temple University Hospital EEG Corpus and are included in the `tuh_eeg` folder.

# Python
Scripts have been running under python 3.11. Its advised to create a virtual environment using pyenv-virtualenv - https://realpython.com/intro-to-pyenv/ 

When you have your virtualenv created and activated, you can install the needed python dependencies via pip

```bash
pip install -r requirements.txt
```

# Indexing
There is a script to index the text files, using a selected embedding vector and storing the documents in a postgres database using 
the pg_vector vector extensions.

Docker configuration is provided to run the postgres database locally. It can be brought up by using docker compose by simply issuing
```bash
docker-compose up -d
```

Once this is up, the scripts are already configured to work with it.

The files can be indexed by using the script as follows:

```bash
python rag/index.py -p tuh_eeg -e <EMBEDDINGS>
```

The `<EMBEDDINGS>` can be `openai` or `ollama`. 

There is some poor quality data in the `tuh_eeg/full` dataset. You may want to skip indexing this dataset until
some data cleansing is added. Skipping this dataset is simply a case of indexing the other sets as follows

```bash
python rag/index.py -p tuh_eeg/abnormal -e <EMBEDDINGS>
python rag/index.py -p tuh_eeg/seizure -e <EMBEDDINGS>
python rag/index.py -p tuh_eeg/slowing -e <EMBEDDINGS>
```


At this stage openai embeddings seem to give better search results. When using openai
the environment variable `OPENAI_API_KEY` must be set to a valid api key. 

If using ollama the environment variable `OLLAMA_BASE_URL` must be set to point to where your ollama instance is.

# Retrieval/Search

Once documents are indexed, they can be searched and summarised using the search script. This script can be used as follows

```bash
python rag/search.py -s "Some search term" -e <EMBEDDINGS> -l <LLM>
```

As before the `<EMBEDDINGS>` can be `openai` or `ollama`. Similarly the `<LLM>` can be `openai` or `ollama`.

All retrieved documents will be printed to the console and then the LLM summary with be printed after.

# Resetting the index

The simplest way to (clear) reset the index is to bring down post gres and then delete it data directory

i.e.

```bash
docker-compose down
rm -rf postgres/data

docker-compose up -d
```

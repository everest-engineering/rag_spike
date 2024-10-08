# RAG POC

This project is a simple POC for RAG. Its provides sample data in the domain of medical/EEG clinical reports, but should work with most specific domains.

The sample data are reports that come from the public domain Temple University Hospital EEG Corpus and are included in the `tuh_eeg` folder.

# Python
Scripts have been running under python 3.11. Its advised to create a virtual environment using pyenv-virtualenv - https://realpython.com/intro-to-pyenv/ 

When you have your virtualenv created and activated, you can install the needed python dependencies via pip

```bash
pip install -r requirements.txt
```

# Environment setup
## Prepare your .env file
There are example `.env` files `.env.*.example`. Copy one of these files to `.env` with the appropriate environment variables set. 

This POC can work with openai, ollama and AWS bedrock for both the embeddings model and the LLM. If you are using a combination of these (e.g. openai for LLM, ollama for embeddings)
you will need to include configuration for each in your `.env` file.

Each of these have some specific instructions.

## OpenAI
OpenAI provides an embedding model and a variety of LLMs that can be used. You will need a subscription to OpenAI to get api access. The following settings will need to be present to use
OpenAI.

```
OPENAI_API_KEY=your key
OPENAI_MODEL=gpt-4o
```

Strictly speaking if you are just using OpenAI's embedding model, you do not need to set `OPENAI_MODEL`

## Ollama
OLlama is an opensource server that hosts a variety of freely available models. This can be run on your own hardware. 
To use ollama you will need it installed and the models that you plan to use installed. You will need the URL for ollama
configured to use it. (Go to https://ollama.com/ for details regarding ollama including installation)

```
OLLAMA_BASE_URL=http://localhost:11434
```

Depending on the choice of model and the hardware you are running it on, ollama can be quite slow. GPUs generally
can give better performance.

## AWS Bedrock
AWS bedrock is supported, and will require your console session to be authenticated with AWS so that its required environment is set up. Bedrock can be used for either the LLM or to compute embeddings.

You will need the bedrock model that is set up in your AWS account configured

```
BEDROCK_MODEL=a model you have set up in AWS bedrock
```

# Indexing
There is a script to index the text files, using a selected embedding vector and storing the documents in a postgres database using 
the pg_vector vector extensions.

Docker configuration is provided to run the postgres database locally. It can be brought up by using docker compose by simply issuing
```bash
docker-compose up -d
```

Once postgres is up, the scripts are already configured to work with it.

Text files can be indexed by using the indexing script as follows:

```bash
python rag/index.py -p <SEARCH PATH> -e <EMBEDDINGS HOST> -m <EMBEDDING MODEL> -cs <CHUNK SIZE> -co <CHUNK OVERLAP> -st <CHUNKING STRATEGY> -c <COLLECTION>
```
* `<SEARCH PATH>` is the path in which to search for `.txt` files to add to the index. The repo has medical reports that can be used at `tuh_eeg/full`
* `<EMBEDDINGS HOST>` can be `openai` or `ollama`. The default is `ollama`.
* `<EMBEDDINGS MODEL>` only needs to be specified if `ollama` is the embeddings host. The chosen model must be installed in the ollama instance, with the default being `nomic-embed-text`.
* `<CHUNK SIZE>` is the maximum size (in characters) to chunk up the document. If not specified a default of `4096` is used.
* `<CHUNK OVERLAP>` is the size (in characters) to overlap chunks of the document. The default is `20`.
* `<CHUNKING STRATEGY>` is the strategy used to chunk up documents. Valid strategies are `recursive` (default) and `semantic`. Recursive looks for valid text breaks such as paragraph end, sentence end then word end for where to split the text. Semantic groups semantically similar sentences together in chunks.
* `<COLLECTION>` is the name of the collection where to store the index in postgres. Under the covers this is put together with the name of the embedding model.

# Retrieval/Search

### Searching and producing summaries of documents

Once documents are indexed, they can be searched and summarised using the search script. This script can be used as follows

```bash
python rag/search.py -s <SEARCH TERM> -q <QUESTION> -e <EMBEDDINGS HOST> -em <EMBEDDINGS MODEL> -d <DOMAIN> -l <LLM HOST> -n <NUMBER OF RESULTS TO SUMMARISE> -ov <OVERSAMPLE RATIO> -rr
```


* `<SEARCH TERM>` is what to search on and subsequently use to summarise. It needs to be in quotes if it has more than one word.
* `<QUESTION>` is the question to ask of the results matching the search terms.
* `<EMBEDDINGS HOST>` can be `openai` or `ollama` (default).
* `<EMBEDDINGS MODEL>` only needs to be specified if ollama is the embeddings host. The chosen model must be installed in the ollama instance, with the default being `nomic-embed-text`. This should be the same embedding model used when building the index.
* `<COLLECTION>` is the name of the collection where to store the index in postgres. Under the covers this is put together with the name of the embedding model.
* `<DOMAIN>` is the name of the domain to be used in the prompt. This can be anything but should be relevant. For example for the sample patient reports `patients` would be an appropriate domain.
* `<LLM HOST>` is the name of the service hosting the LLM. Valid selections are `ollama` (default), `openai` and `bedrock`. 
  * Bedrock requires your terminal session to be authenticated with AWS. 
  * When using openai the environment variable `OPENAI_API_KEY` must be set to a valid api key. 
  * If using ollama, the environment variable `OLLAMA_BASE_URL` must be set to point to where your ollama instance is.
* `<NUMBER OF RESULTS TO SUMMARISE>` is the number of results that will be given to the LLM to produce the summary. If this is set too high it may lead to exceeding token limits for particular models.
* `<OVERSAMPLE RATIO>` is a multiplier of the number of results to summarise. When reranking this will be applied to how many results to fetch for the reranking process. eg if you ask for 10 results and have an oversampling ratio of 20 then 200 results will be retrieved from the index and then reranked. The top 10 results from the reranked set will then be sent to the LLM to answer the question that you supply.
* `-rr` is optional when specified on the commandline causes the reranking process to be used. If not specified the raw matches from the index are used

All retrieved documents will be printed to the console and then the LLM summary with be printed after. Results will also be written to a text file that is placed in the `tmp` directory in the project.

### Searching just for matching documents
You can simply get a list of documents that match a given search along with their scores that are assigned by the search algorithms. Note the closer a score is to zero, the better a match it is considered to be. This is useful for getting an idea of the quality of a search, without needing to run the generation step.
```bash
python rag/search_matching_documents.py -s <SEARCH TERM> -e <EMBEDDINGS HOST> -em <EMBEDDINGS MODEL> -n <NUMBER OF RESULTS TO SUMMARISE> -ov <OVERSAMPLE RATIO> -rr
```

The paths to the matching documents and their scores are simply printed to the console.

# Resetting the index

The simplest way to (clear) reset the index is to bring down postgres and then delete its data directory (`postgres/data` in the project). This directory is mapped from the project in to the running docker container.

i.e.

```bash
docker-compose down
rm -rf postgres/data

docker-compose up -d
```

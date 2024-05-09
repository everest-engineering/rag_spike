import glob
import os
from argparse import ArgumentParser
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

from constants import OLLAMA_MODEL, COLLECTION_BASE_NAME, CONNECTION_STRING


def main():
    parser = ArgumentParser(description='Index and store text files')
    parser.add_argument('--path', '-p', required=True, help='Path to search for .txt files in')
    parser.add_argument('--embeddings', '-e',
                        default="ollama", choices=("ollama", "openai"),
                        help='Embeddings to use')

    args = parser.parse_args()
    print(f"Using {args.embeddings} embeddings.")
    if args.embeddings == "openai":
        # only import if we really want it as it gets whiny about needing openai keys set
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        model = "gpt-4"
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        model = OLLAMA_MODEL

        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL,
                                      base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    docs = []
    for filepath in glob.glob(f"{args.path}/**/*.txt", recursive=True):
        print(f"Indexing: {filepath}")
        loader = UnstructuredFileLoader(filepath)
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(raw_documents)
        docs.extend(documents)

    print(f"Indexing {len(docs)} from {args.path}")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=f"{COLLECTION_BASE_NAME}-{args.embeddings}-{model.replace(':', '_')}",
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    for doc in docs:
        print(f"Adding: {doc.metadata['source']}")
        vectorstore.add_documents([doc])


if __name__ == "__main__":
    main()
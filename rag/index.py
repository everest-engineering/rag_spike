import glob
import os
from argparse import ArgumentParser
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres.vectorstores import PGVector

from constants import COLLECTION_BASE_NAME, CONNECTION_STRING
from utils import full_collection_name


def main():
    parser = ArgumentParser(description='Index and store text files')
    parser.add_argument('--path', '-p', required=True, help='Path to search for .txt files in')
    parser.add_argument('--ollama_model', '-m',
                        default="llama3",
                        choices=("llama3",
                                 "llama3:instruct",
                                 "mistral:7b",
                                 "mixtral:8x7b",
                                 "nomic-embed-text",
                                 "mxbai-embed-large"),
                        help='Model to use with ollama (must be installed on server)')
    parser.add_argument('--embeddings', '-e',
                        default="ollama", choices=("ollama", "openai"),
                        help='Embeddings to use')

    parser.add_argument('--collection', '-c',
                        default=COLLECTION_BASE_NAME,
                        help='Name of collection to store embedding vectors')

    parser.add_argument('--chunk_size', '-cs',
                        default=4096,
                        type=int,
                        help='Size for chunks')

    parser.add_argument('--chunk_strategy', '-st',
                        default='recursive',
                        choices=("recursive", "semantic"),
                        help='Strategy for chunking')

    parser.add_argument('--chunk_overlap', '-co',
                        default=20,
                        type=int,
                        help='Overlap for chunks')

    args = parser.parse_args()
    print(f"Using {args.embeddings} embeddings.")
    if args.embeddings == "openai":
        # only import if we really want it as it gets whiny about needing openai keys set
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        model = "gpt-4"
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        model = args.ollama_model

        embeddings = OllamaEmbeddings(model=args.ollama_model,
                                      base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    docs = []
    for filepath in glob.glob(f"{args.path}/**/*.txt", recursive=True):
        print(f"Chunking: {filepath}")
        loader = UnstructuredFileLoader(filepath)
        raw_documents = loader.load()
        if args.chunk_strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size,
                                                           chunk_overlap=args.chunk_overlap,
                                                           length_function=len,
                                                           separators=["\n\n", "\n", ".", ",", " ", ""]
                                                           )
        else:
            text_splitter = SemanticChunker(embeddings=embeddings)
        documents = text_splitter.split_documents(raw_documents)
        docs.extend(documents)

    print(f"Indexing {len(docs)} from {args.path}")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=full_collection_name(collection=args.collection, embeddings=args.embeddings, model=model),
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    for doc in docs:
        print(f"Adding: {doc.metadata['source']}")
        vectorstore.add_documents([doc])


if __name__ == "__main__":
    main()

from argparse import ArgumentParser, BooleanOptionalAction

from langchain_postgres.vectorstores import DistanceStrategy

from constants import COLLECTION_BASE_NAME
from utils import search_for_docs


def main():
    parser = ArgumentParser(description='Search index for text files and print list with scores')
    parser.add_argument('--search_text', '-s', required=True, help='Text query')
    parser.add_argument('--search_type', '-t', default="similarity", help='Type of search')
    parser.add_argument('--number_to_summarise', '-n', default=10, type=int, help='Number of results')
    parser.add_argument('--embeddings', '-e',
                        default="ollama", choices=("ollama", "openai"),
                        help='Embeddings to use')
    parser.add_argument('--embeddings_model', '-em',
                        default="llama3",
                        choices=("llama3", "llama3:instruct", "mistral:7b", "mixtral:8x7b", "nomic-embed-text"),
                        help='Embeddings model to use')
    parser.add_argument('--collection', '-c',
                        default=COLLECTION_BASE_NAME,
                        help='Name of collection storing embedding vectors')
    parser.add_argument('--distance_strategy', '-ds',
                        choices=(DistanceStrategy.COSINE, DistanceStrategy.EUCLIDEAN, DistanceStrategy.MAX_INNER_PRODUCT),
                        default=DistanceStrategy.COSINE)

    parser.add_argument('--rerank', '-rr',
                        action=BooleanOptionalAction)

    parser.add_argument('--oversample_times', '-ov',
                        default=10,
                        type=int)

    args = parser.parse_args()

    results = search_for_docs(embeddings_server=args.embeddings,
                              search_text=args.search_text,
                              embeddings_model=args.embeddings_model,
                              collection=args.collection,
                              distance_strategy=args.distance_strategy,
                              number_to_summarise=args.number_to_summarise,
                              oversample_times=args.oversample_times,
                              rerank=args.rerank)

    print("\n".join([f"{score:.2f}: {path} ({num})" for num, (path, score) in enumerate(sorted(results, reverse=True, key=lambda pair: pair[1]))]))


if __name__ == "__main__":
    main()

import os
from argparse import ArgumentParser

from langchain_postgres.vectorstores import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

from constants import COLLECTION_BASE_NAME, CONNECTION_STRING
from utils import full_collection_name


def main():
    parser = ArgumentParser(description='Search index for text files')
    parser.add_argument('--search_text', '-s', required=True, help='Text query')
    parser.add_argument('--search_type', '-t', default="similarity", help='Type of search')
    parser.add_argument('--number_to_summarise', '-n', default=10, type=int, help='Number of results to summarise')
    parser.add_argument('--embeddings', '-e',
                        default="ollama", choices=("ollama", "openai"),
                        help='Embeddings to use')
    parser.add_argument('--llm', '-l',
                        default="ollama", choices=("ollama", "openai"),
                        help='LLM Service to use')
    parser.add_argument('--embeddings_model', '-em',
                        default="llama3",
                        choices=("llama3", "llama3:instruct", "mistral:7b", "mixtral:8x7b", "nomic-embed-text"),
                        help='Embeddings model to use')
    parser.add_argument('--collection', '-c',
                        default=COLLECTION_BASE_NAME,
                        help='Name of collection storing embedding vectors')
    parser.add_argument('--ollama_model', '-m',
                        default="llama3",
                        choices=("llama3", "llama3:instruct", "mistral:7b", "mixtral:8x7b"),
                        help='Model to use with ollama (must be installed on server)')

    parser.add_argument('--domain', '-d',
                        required=True,
                        help='Domain of data being searched')

    args = parser.parse_args()

    if args.embeddings == "openai":
        # only import if we really want it as it gets whiny about needing openai keys set
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        model = "gpt-4"
        embeddings_model = "gpt-4"
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        model = args.ollama_model
        embeddings_model = args.embeddings_model
        embeddings = OllamaEmbeddings(model=args.embeddings_model,
                                      base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=full_collection_name(collection=args.collection,
                                             embeddings=args.embeddings,
                                             model=embeddings_model),
        connection=CONNECTION_STRING,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )

    #results = vectorstore.search(args.search_text, args.search_type, k=args.number_to_summarise)
    # results = vectorstore.max_marginal_relevance_search(args.search_text,
    #                                                     k=args.number_to_summarise,
    #                                                     fetch_k=100,
    #                                                     lambda_mult=0.8)

    results = vectorstore.max_marginal_relevance_search_with_score(args.search_text,
                                                                   k=args.number_to_summarise,
                                                                   fetch_k=100,
                                                                   lambda_mult=0.8)
    messages = list([{"role": "system", "content": f"SCORE: {score}\nCONTENT: {doc.page_content}"} for (doc, score) in results])

    messages.append({"role": "system", "content": f"The previous messages are a selection of {args.number_to_summarise} {args.domain} to be used to answer questions. They have a score marked by \"SCORE:\". The score is indicating a calculated relevance and ranges from 0 (low relevance) to 1 (high relevance). They also have content marked by \"CONTENT:\". Use only this information to answer questions."})
    messages.append({"role": "user", "content": f"Please summarise the most common factors seen in {args.domain} regarding \"{args.search_text}\" in two paragraphs. After these paragraphs include a count of the number of relevant messages versus the total number of messages used. Provide a brief summary of the messages that were irrelevant and how well their score related to their relevance."})

    print("\n\n >>>><<<< \n\n".join([doc.page_content for (doc, _) in results]))

    print("------------------------------------------------------------------------------------\n\n")

    if args.llm == "openai":
        import openai
        completion = openai.chat.completions.create(
            model=model,
            messages=messages)
        response = completion.choices[0].message.content
    else:
        import ollama
        ollama_response = ollama.chat(
            model=model,
            messages=messages)
        response = ollama_response['message']['content']
    print(response)


if __name__ == "__main__":
    main()

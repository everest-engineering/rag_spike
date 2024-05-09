import os
from argparse import ArgumentParser

from langchain_postgres.vectorstores import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

from constants import OLLAMA_MODEL, COLLECTION_BASE_NAME, CONNECTION_STRING


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
                        help='Embeddings to use')

    args = parser.parse_args()

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

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=f"{COLLECTION_BASE_NAME}-{args.embeddings}-{model.replace(':', '_')}",
        connection=CONNECTION_STRING,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )

    #results = vectorstore.search(args.search_text, args.search_type, k=args.number_to_summarise)
    results = vectorstore.max_marginal_relevance_search(args.search_text, k=args.number_to_summarise, lambda_mult=0.8)

    messages = list([{"role": "system", "content": doc.page_content} for doc in results])

    messages.append({"role": "system", "content": f"The previous messages are a selection of {args.number_to_summarise} patient clinical summaries to be used to answer questions. Use only this information to answer questions."})
    messages.append({"role": "user", "content": f"Please summarise the most common factors seen in patients regarding \"{args.search_text}\" in four paragraphs."})

    print("\n\n >>>><<<< \n\n".join([doc.page_content for doc in results]))

    print("------------------------------------------------------------------------------------\n\n")

    if args.llm == "openai":
        import openai
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages)
        response = completion.choices[0].message.content
    else:
        import ollama
        ollama_response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages)
        response = ollama_response['message']['content']
    print(response)


if __name__ == "__main__":
    main()

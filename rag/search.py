from argparse import ArgumentParser
from statistics import mean

from langchain_postgres.vectorstores import DistanceStrategy

from constants import COLLECTION_BASE_NAME
from utils import read_file, search_for_docs


def main():
    parser = ArgumentParser(description='Search index for text files')
    parser.add_argument('--search_text', '-s', required=True, help='Text query')
    parser.add_argument('--search_type', '-t', default="similarity", help='Type of search')
    parser.add_argument('--number_to_summarise', '-n', default=10, type=int, help='Number of results to summarise')
    parser.add_argument('--number_to_process', '-p', default=None, type=int, help='Number of results to process (top N send to the LLM)')
    parser.add_argument('--relevance_threshold', '-r', default=0.5, type=float, help='Threshold for filtering out irrelevant results')
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
                        choices=("llama3", "llama3:instruct", "mistral:7b", "mistral:instruct", "mixtral:8x7b", "dolphin-llama3:8b", "orca-mini", "orca-mini:7b"),
                        help='Model to use with ollama (must be installed on server)')
    parser.add_argument('--distance_strategy', '-ds',
                        choices=(DistanceStrategy.COSINE, DistanceStrategy.EUCLIDEAN, DistanceStrategy.MAX_INNER_PRODUCT),
                        default=DistanceStrategy.COSINE)

    parser.add_argument('--lambda_mult', '-lm',
                        default=0.5,
                        type=float)

    parser.add_argument('--domain', '-d',
                        required=True,
                        help='Domain of data being searched')

    args = parser.parse_args()

    results = search_for_docs(embeddings_server=args.embeddings,
                              search_text=args.search_text,
                              embeddings_model=args.embeddings_model,
                              collection=args.collection,
                              distance_strategy=args.distance_strategy,
                              number_to_summarise=args.number_to_summarise,
                              lambda_mult=args.lambda_mult,
                              relevance_threshold=args.relevance_threshold)

    results_content = [(read_file(doc), score) for (doc, score) in results]

    messages = list([{"role": "system", "content": f"SCORE: {score}\nCONTENT: {text}"} for (text, score) in results_content])

    messages.append({"role": "system", "content": f"The previous {args.domain} records are a selection of {len(results_content)} {args.domain} to be used to answer questions. They have a relevance score marked by \"SCORE:\".They also have content marked by \"CONTENT:\". Use only this information to answer questions."})
    messages.append({"role": "user", "content": f"Please summarise what is common in {args.domain} regarding \"{args.search_text}\" in around 400 word. Only consider the records that are relevant to \"{args.search_text}\" and make all inferences from the information in these records. The summary should not refer to individual records. After the summary include a count of the number of relevant messages versus the total number of messages used. Provide a brief summary of the messages that were irrelevant and how well their score related to their relevance."})

    print("\n\n >>>><<<< \n\n".join([f"SCORE:{score} \nCONTENT:\n{text}" for (text, score) in results_content]))

    print("------------------------------------------------------------------------------------\n\n")
    scores = [score for (_, score) in results_content]

    print(f"Total messages processed {len(results)}, max score: {max(scores)}, min score: {min(scores)}, mean score: {mean(scores)}")

    if args.llm == "openai":
        import openai
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages)
        response = completion.choices[0].message.content
    else:
        import ollama
        ollama_response = ollama.chat(
            model=args.ollama_model,
            messages=messages)
        response = ollama_response['message']['content']
    print(response)


if __name__ == "__main__":
    main()

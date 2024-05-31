import json
from argparse import ArgumentParser, BooleanOptionalAction
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
                        default="ollama", choices=("ollama", "openai", "bedrock"),
                        help='LLM Service to use')
    parser.add_argument('--embeddings_model', '-em',
                        default="llama3",
                        choices=("llama3",
                                 "llama3:instruct",
                                 "mistral:7b",
                                 "mixtral:8x7b",
                                 "nomic-embed-text",
                                 "mxbai-embed-large"),
                        help='Embeddings model to use')
    parser.add_argument('--collection', '-c',
                        default=COLLECTION_BASE_NAME,
                        help='Name of collection storing embedding vectors')
    parser.add_argument('--ollama_model', '-m',
                        default="llama3",
                        choices=("llama3",
                                 "llama3:instruct",
                                 "mistral:7b",
                                 "mistral:instruct",
                                 "mixtral:8x7b",
                                 "dolphin-llama3:8b",
                                 "orca-mini",
                                 "orca-mini:7b",
                                 "zephyr",
                                 "mistrallite"),
                        help='Model to use with ollama (must be installed on server)')
    parser.add_argument('--distance_strategy', '-ds',
                        choices=(DistanceStrategy.COSINE, DistanceStrategy.EUCLIDEAN, DistanceStrategy.MAX_INNER_PRODUCT),
                        default=DistanceStrategy.COSINE)

    parser.add_argument('--rerank', '-rr',
                        action=BooleanOptionalAction)

    parser.add_argument('--oversample_times', '-ov',
                        default=10,
                        type=int)

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
                              rerank=args.rerank,
                              oversample_times=args.oversample_times)

    results_content = [(read_file(doc), score) for (doc, score) in results]

    messages = []
    messages.append({"role": "system", "content": f"The following {args.domain} records are a "
                                                  f"selection of {len(results_content)} {args.domain} "
                                                  f"to be used to answer questions. "
                                                  f"They are in json format with a score, number and content. We will refer to the number as \"record number\""
                                                  # f"The closer to zero the score is, the more relevant the search has determined it is."
                    })
    messages.extend(list([{"role": "system", "content": json.dumps({"number": num + 1,
                                                                    "score": score,
                                                                    "content": text})} for num, (text, score) in enumerate(results_content)]))

    messages.append({"role": "user", "content": f"Please summarise what is common in {args.domain} "
                                                f"regarding \"{args.search_text}\" in 3 to 4 paragraphs. "
                                                f"When making generalisations please "
                                                f"reference the records by their record number. "
                                                f"Use only the information in the records to make generalisations. Dont mention the specifics of any records when "
                                                f"making a summary, instead just referencing records by number in which what has been summarised is displayed."                                                
                                                f"Only consider the records that mention all key terms in \"{args.search_text}\" or something highly related as relevant. "
                                                f"Dont assume those records that dont mention these key terms or something highly related are relevant. "
                                                f"Be careful not to interpret correlation in the reports as causation. "
                                                f"After the summary, draw any conclusions with regards to \"{args.search_text}\" based on the information summarised. "
                                                f"Only make conclusions if there are more than three relevant records."
                                                f"After the conclusion include a count of the number "
                                                f"of relevant records versus the total number of records used. "
                                                f"Please list the relevant records by their number."
                                                f"Additionally add a sentence summary of the three most relevant records, referring to the record by number."
                                                f"Also list together any records that look like duplicates or very similar."
                                                f"Also summarise records that were determined to be irrelevant but may be marginally relevant."
                                                f"Provide a brief summary of the records that were irrelevant."})

    print("\n\n >>>><<<< \n\n".join([f"NUMBER:{num + 1} \nSCORE:{score} \nCONTENT:\n{text}" for num, (text, score) in enumerate(results_content)]))

    print("------------------------------------------------------------------------------------\n\n")
    scores = [score for (_, score) in results_content]

    print(f"Total messages processed {len(results)}, max score: {max(scores)}, min score: {min(scores)}, mean score: {mean(scores)}")

    if args.llm == "openai":
        print("Summarising using openai")
        import openai
        completion = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages)
        response = completion.choices[0].message.content
    elif args.llm == "bedrock":
        from langchain_aws.chat_models import ChatBedrock
        llm = ChatBedrock(model_id="mistral.mistral-large-2402-v1:0",
                          model_kwargs={"temperature": 0.1, "top_k": 50, "top_p": 0.9},)
        bedrock_response = llm.invoke(messages)
        response = bedrock_response.content
    else:
        import ollama
        ollama_response = ollama.chat(
            model=args.ollama_model,
            messages=messages)
        response = ollama_response['message']['content']
    print(response)


if __name__ == "__main__":
    main()

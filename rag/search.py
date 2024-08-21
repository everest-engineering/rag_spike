import json
import os
import re

from argparse import ArgumentParser, BooleanOptionalAction
from dotenv import load_dotenv
from statistics import mean

from langchain_postgres.vectorstores import DistanceStrategy

from constants import COLLECTION_BASE_NAME
from utils import read_file, search_for_docs


NON_ALPHA_NUMERIC = re.compile(r'[^a-zA-Z\d]')


def user_prompt(domain, search_text, question):
    result = f"""
     Produce a summary of the {domain} records, using the following instructions.
    
     * Dont assume the records that dont mention the key terms "{search_text}" or something highly related are relevant.
     ** Be careful not to interpret correlation in the reports as causation.
     ** After the summary, draw any conclusions with regards to the question "{question}" 
     based on the information summarised.

     * Summarise what is common in relevant {domain} records regarding the question "{question}" 
     in 3 to 4 paragraphs. 
     ** When making generalisations please reference the records by their record number.
     ** When making generalisation include the number and percentage of relevant records that display the generalised feature.
     ** Use only the information in the records to make generalisations. 
     ** Dont mention the specifics of any records when making a summary, instead just referencing 
     records by number in which what has been summarised is displayed.
     ** Only consider the records that mention all key terms in "{search_text}" or something highly related as
     relevant.

     * Only make conclusions if there are more than three relevant records.
     ** After the conclusion include a count of the number
     of relevant records versus the total number of records used.
     ** Provide a list of key features and the number of records that display this feature and and percentage 
     of the records displaying this feature to the total number of relevant records

     * Please list the relevant records by their number.
     ** Add a sentence summary of the three most relevant records, referring to the record by number.

     * List together any records that look like duplicates or very similar.

     * Summarise records that were determined to be irrelevant but may be marginally relevant.
     
     * Provide a brief summary of the records that were irrelevant.
     """

    return result


def system_prompt(domain, num_records):
    result = f"""The following {num_records} messages contain one record each and are a 
    selection of {domain} records to be used to answer questions.
    They are in json format with score, number and content properties.
    The content property contains the text of the record.
    We will refer to the number as \"record number\". Use only the content in these {domain} records
    to answer questions.
    """

    return result


def main():
    load_dotenv()

    parser = ArgumentParser(description='Search index for text files')
    parser.add_argument('--search_text', '-s', required=True, help='Text query')
    parser.add_argument('--question', '-q',
                        required=True,
                        help='The question to be answered using the retrieved records')
    parser.add_argument('--search_type', '-t', default="similarity", help='Type of search')
    parser.add_argument('--number_to_summarise', '-n', default=10, type=int, help='Number of results to summarise')
    parser.add_argument('--number_to_process', '-p', default=None, type=int,
                        help='Number of results to process (top N send to the LLM)')
    parser.add_argument('--relevance_threshold', '-r', default=0.5, type=float,
                        help='Threshold for filtering out irrelevant results')
    parser.add_argument('--embeddings', '-e',
                        default="ollama", choices=("ollama", "openai", "bedrock"),
                        help='Embeddings to use')
    parser.add_argument('--llm', '-l',
                        default="ollama", choices=("ollama", "openai", "bedrock"),
                        help='LLM Service to use')
    parser.add_argument('--embeddings_model', '-em',
                        default="llama3",
                        help='Embeddings model to use')
    parser.add_argument('--collection', '-c',
                        default=COLLECTION_BASE_NAME,
                        help='Name of collection storing embedding vectors')
    parser.add_argument('--ollama_model', '-m',
                        default="llama3",
                        help='Model to use with ollama (must be installed on server)')
    parser.add_argument('--distance_strategy', '-ds',
                        choices=(
                            DistanceStrategy.COSINE, DistanceStrategy.EUCLIDEAN, DistanceStrategy.MAX_INNER_PRODUCT),
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

    results_content = [(read_file(doc.metadata['source']), score) for (doc, score) in results]
    records = [
        {
            "role": "system",
            "content": json.dumps({"number": num + 1,
                                   "score": score,
                                   "content": text})
        }
        for num, (text, score)
        in enumerate(results_content)
    ]

    messages = [
        {
            "role": "system",
            "content": system_prompt(domain=args.domain, num_records=len(records))
        },
        *records,
        {
            "role": "user",
            "content": user_prompt(domain=args.domain,
                                   search_text=args.search_text,
                                   question=args.question)
        }
    ]

    results_summary = "\n\n >>>><<<< \n\n".join([f"NUMBER: {num + 1} \nSCORE: {score} \nCONTENT: \n{text}"
                                                 for num, (text, score)
                                                 in enumerate(results_content)])
    print(results_summary)

    print("------------------------------------------------------------------------------------\n\n")
    scores = [score
              for (_, score)
              in results_content]

    print(f"Total messages processed {len(results)}, "
          f"max score: {max(scores)}, "
          f"min score: {min(scores)}, mean score: {mean(scores)}")

    if args.llm == "openai":
        print("Summarising using openai")
        import openai
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages)
        response = completion.choices[0].message.content
    elif args.llm == "bedrock":
        print("Summarising using bedrock")
        from langchain_aws.chat_models import ChatBedrock
        llm = ChatBedrock(model_id=os.environ.get("BEDROCK_MODEL", "mistral.mistral-large-2402-v1:0"),
                          model_kwargs={"temperature": 0.1, "top_k": 50, "top_p": 0.9}, )
        bedrock_response = llm.invoke(messages)
        response = bedrock_response.content
    else:
        print("Summarising using ollama")
        import ollama
        ollama_client = ollama.Client(host=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        ollama_response = ollama_client.chat(
            model=args.ollama_model,
            messages=messages)
        response = ollama_response['message']['content']
    print(response)

    os.makedirs("tmp", exist_ok=True)
    filename = f"tmp/{re.sub(NON_ALPHA_NUMERIC, '_', args.search_text)}-" \
               f"{re.sub(NON_ALPHA_NUMERIC, '_', args.question)}-{args.llm}.txt"
    with open(filename, "w") as outfile:
        outfile.write("--------------\n")
        outfile.write(f"Search terms: {args.search_text}\n")
        outfile.write(f"Question asked of matching records: {args.question}\n")
        outfile.write("--------------\n\n")
        outfile.write(response)
        outfile.write("\n\n--------------\n\n")
        outfile.write(results_summary)


if __name__ == "__main__":
    main()

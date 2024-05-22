import os
from langchain_postgres.vectorstores import PGVector

from constants import CONNECTION_STRING


def full_collection_name(collection, embeddings, model):
    return f"{collection}-{embeddings}-{model.replace(':', '_')}"


def read_file(doc_source):
    print(f"Reading {doc_source}")
    with open(doc_source, "r", errors="ignore") as text_file:
        return text_file.read()


# Remove duplicate pointers to doc and assign the lowest score
def normalize_results(results):
    results_norm = {}

    for (doc, score) in results:
        source = doc.metadata["source"]
        current_score = results_norm.get(source, 10000000000)

        if score < current_score:
            results_norm[source] = score

    return list(results_norm.items())


def search_for_docs(embeddings_server,
                    search_text,
                    embeddings_model,
                    collection,
                    distance_strategy,
                    number_to_summarise,
                    lambda_mult,
                    relevance_threshold=0.):
    if embeddings_server == "openai":
        # only import if we really want it as it gets whiny about needing openai keys set
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        embeddings_model = "gpt-4"
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings_model = embeddings_model
        embeddings = OllamaEmbeddings(model=embeddings_model,
                                      base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=full_collection_name(collection=collection,
                                             embeddings=embeddings_server,
                                             model=embeddings_model),
        connection=CONNECTION_STRING,
        distance_strategy=distance_strategy,
        use_jsonb=True,
    )

    # results = vectorstore.max_marginal_relevance_search_with_score(search_text,
    #                                                                k=number_to_summarise,
    #                                                                fetch_k=number_to_summarise * 10,
    #                                                                lambda_mult=lambda_mult)
    results = vectorstore.similarity_search_with_score(search_text,
                                                       k=number_to_summarise)
    # we take the abs of score because inner product may give negatives
    return normalize_results(list([(doc, abs(score)) for (doc, score) in results if abs(score) >= relevance_threshold]))

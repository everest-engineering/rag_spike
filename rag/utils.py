import os
import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres.vectorstores import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from constants import CONNECTION_STRING, BEDROCK_EMBEDDING_MODEL, BEDROCK_EMBEDDING_COLLECTION_NAME, \
    GPT_EMBEDDING_COLLECTION_NAME


def full_collection_name(collection, embeddings, model):
    return f"{collection}-{embeddings}-{model.replace(':', '_')}"


def read_file(doc_source):
    print(f"Reading {doc_source}")
    with open(doc_source, "r", errors="ignore") as text_file:
        return text_file.read()


def _full_doc(source):
    result = Document(page_content=read_file(source))
    result.metadata["source"] = source

    return result


class ResultsRetriever(BaseRetriever):
    vectorstore: Any
    num_to_sample: Any
    transformer: Any

    def _get_relevant_documents(self, query, *, run_manager):
        results = self.vectorstore.similarity_search_with_score(query,
                                                                k=self.num_to_sample)
        transformed = self.transformer(list([(doc, score) for (doc, score) in results]))
        sorted_results = sorted([(doc, score) for (doc, score) in transformed], key=lambda doc_score: doc_score[1])

        return list([_full_doc(doc.metadata["source"]) for doc, _ in sorted_results])


# Remove duplicate pointers to doc and assign the lowest score
def normalize_results(results):
    results_norm = {}
    hashes = {}

    for (doc, score) in results:
        source = doc.metadata["source"]

        # we look for duplicate files by this hash
        content_hash = hashes.get(source)

        if content_hash is None:
            with open(source, encoding="utf-8", errors="ignore") as content_file:
                content = content_file.read()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                hashes[source] = content_hash

        _, current_score = results_norm.get(content_hash, (None, 10000000000))

        if score < current_score:
            results_norm[content_hash] = (doc, score)

    return list(results_norm.values())


def raw_results(results):
    return [(doc, score) for (doc, score) in results]


def search_for_docs(embeddings_server,
                    search_text,
                    embeddings_model,
                    collection,
                    distance_strategy,
                    number_to_summarise,
                    rerank,
                    oversample_times=10,
                    normalize=True):
    transform = normalize_results if normalize else raw_results
    if embeddings_server == "openai":
        # only import if we really want it as it gets whiny about needing openai keys set
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        embeddings_model = GPT_EMBEDDING_COLLECTION_NAME
    elif embeddings_server == "bedrock":
        from langchain_aws.embeddings import BedrockEmbeddings

        embeddings = BedrockEmbeddings(model_id=BEDROCK_EMBEDDING_MODEL)
        embeddings_model = BEDROCK_EMBEDDING_COLLECTION_NAME
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

    num_to_sample = number_to_summarise * oversample_times
    if rerank:
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=number_to_summarise)
        results_retriever = ResultsRetriever(vectorstore=vectorstore,
                                             num_to_sample=num_to_sample,
                                             transformer=transform)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=results_retriever
        )

        results = compression_retriever.invoke(search_text, k=num_to_sample)

        return transform(list([(doc, num) for num, doc in enumerate(results)]))
    else:
        results = vectorstore.similarity_search_with_score(search_text,
                                                           k=num_to_sample)

        transformed = transform(list([(doc, score) for (doc, score) in results]))
        sorted_results = sorted([(doc, score) for (doc, score) in transformed], key=lambda doc_score: doc_score[1])

        return sorted_results[:number_to_summarise]

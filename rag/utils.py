def full_collection_name(collection, embeddings, model):
    return f"{collection}-{embeddings}-{model.replace(':', '_')}"


def read_file(doc_source):
    print(f"Reading {doc_source}")
    with open(doc_source, "r", errors="ignore") as text_file:
        return text_file.read()


# Remove duplicate pointers to doc and assign the highest score
def normalize_results(results):
    results_norm = {}

    for (doc, score) in results:
        source = doc.metadata["source"]
        current_score = results_norm.get(source, 0.)

        if score > current_score:
            results_norm[source] = score

    return list(results_norm.items())
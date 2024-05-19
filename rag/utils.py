def full_collection_name(collection, embeddings, model):
    return f"{collection}-{embeddings}-{model.replace(':', '_')}"

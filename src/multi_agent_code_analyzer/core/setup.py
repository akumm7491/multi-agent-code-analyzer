from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


def setup_milvus():
    """Initialize Milvus collections for domain concept storage."""
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host="milvus",
            port=19530
        )

        # Define collection schema
        fields = [
            FieldSchema(name="concept_id", dtype=DataType.VARCHAR,
                        is_primary=True, max_length=100),
            # all-MiniLM-L6-v2 dimension
            FieldSchema(name="embeddings",
                        dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(
            fields=fields, description="Domain concepts embeddings")

        # Create collection if it doesn't exist
        if "domain_concepts" not in utility.list_collections():
            collection = Collection(name="domain_concepts", schema=schema)

            # Create index for vector similarity search
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="embeddings", index_params=index_params)
            collection.load()

        return True
    except Exception as e:
        print(f"Error setting up Milvus: {str(e)}")
        return False


if __name__ == "__main__":
    setup_milvus()

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time
import os


def setup_milvus(max_retries=5):
    """Initialize Milvus collections for domain concept storage."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=os.getenv("MILVUS_HOST", "standalone"),
                port=int(os.getenv("MILVUS_PORT", "19530")),
                timeout=30  # Increased timeout
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

            print("Successfully set up Milvus collection")
            return True

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(
                    f"Error setting up Milvus after {max_retries} attempts: {str(e)}")
                return False
            print(
                f"Attempt {retry_count}/{max_retries} failed. Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    setup_milvus()

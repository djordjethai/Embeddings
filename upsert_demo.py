import itertools
from pinecone import Pinecone

# Initialize the client with pool_threads=30. This limits simultaneous requests to 30.
pc = Pinecone(api_key="YOUR_API_KEY", pool_threads=30)
index = pc.Index("pinecone-index")

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# vector_dim = 128
# vector_count = 10000

# example_data_generator = map(lambda i: (f'id-{i}', [random.random() for _ in range(vector_dim)]), range(vector_count))
example_data_generator = ""

# Upsert data with 100 vectors per upsert request asynchronously
# - Pass async_req=True to index.upsert()
with pc.Index('pinecone-index', pool_threads=30) as index:
    # Send requests in parallel
    async_results = [
        index.upsert(vectors=ids_vectors_chunk, async_req=True)
        for ids_vectors_chunk in chunks(example_data_generator, batch_size=100)
    ]
    # Wait for and retrieve responses (this raises in case of error)
    [async_result.get() for async_result in async_results]


# using grpc i async

from pinecone.grpc import PineconeGRPC as Pinecone
def chunker(seq, batch_size):
  return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

async_results = [
  index.upsert(vectors=chunk, async_req=True)
  for chunk in chunker(data, batch_size=100)
]

# Wait for and retrieve responses (in case of error)
[async_result.result() for async_result in async_results]


# using dataframe

from pinecone import Pinecone, ServerlessSpec
from pinecone_datasets import list_datasets, load_dataset

pc = Pinecone(api_key="API_KEY")

dataset = load_dataset("quora_all-MiniLM-L6-bm25")

pc.create_index(
  name="my-index",
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)

index = pc.Index("my-index")

index.upsert_from_dataframe(dataset.drop(columns=["blob"]))


pc = Pinecone(api_key='YOUR_API_KEY')  # This is gRPC client aliased as "Pinecone"
index = pc.Index('example-index')



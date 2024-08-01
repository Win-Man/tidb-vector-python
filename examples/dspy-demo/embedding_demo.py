import os
import sys

from tidb_vector.integrations import TiDBVectorClient
import ollama
from FlagEmbedding import FlagModel

model = FlagModel('/mnt/d/004-Workspace/llm-workspace/bge-large-zh-v1.5',query_instruction_for_retrieval="为这个句子生成表示用于检索相关文章:",use_fp16=True)

TIDB_DATABASE_URL=os.getenv("TIDB_DATABASE_URL")

TABLE_NAME="sg_embedded_test"
SAMPLE_DATA_FILE="sample_data_cn.txt"

def print_embedding_data():
    print("Loading sample data...")
    with open(SAMPLE_DATA_FILE, 'r') as f:
        # I prepare a small set of data for speeding up embedding, you can replace it with your own data.
        print("{} found.".format(SAMPLE_DATA_FILE))
        sample_data = f.read()
    print("Sample data loaded successfully.")
    #print("Debug sample_data:{}".format(sample_data))

    # ollama embeddings 模型介绍：https://ollama.com/blog/embedding-models
    # all-minilm | nomic-embed-text | mxbai-embed-large

    print("Embedding sample data...")
    documents = []
    for idx, passage in enumerate(sample_data.split('\n')):
        #embedding = ollama.embeddings(model="all-minilm",prompt=passage)
        embedding = model.encode(passage)
        print(idx, passage[:10], embedding)
        if len(passage) == 0:
            continue
        documents.append({
            "id": str(idx),
            "text": passage,
            "embedding": embedding,
            "metadata": {"category": "album"},
        })
    print("Sample data embedded successfully.")
    print("Sample data number:", len(documents))

def embedding_tidb_document():
    tidb_vector_client = TiDBVectorClient(
    table_name=TABLE_NAME,
    connection_string=TIDB_DATABASE_URL,
    drop_existing_table=True,
    )
    print("Connected to TiDB.")
    print("describe table:", tidb_vector_client.execute("describe {};".format(TABLE_NAME)))

    print("Loading sample data...")
    with open(SAMPLE_DATA_FILE, 'r') as f:
        # I prepare a small set of data for speeding up embedding, you can replace it with your own data.
        print("{} found.".format(SAMPLE_DATA_FILE))
        sample_data = f.read()
    print("Sample data loaded successfully.")
    #print("Debug sample_data:{}".format(sample_data))

    # ollama embeddings 模型介绍：https://ollama.com/blog/embedding-models
    # all-minilm | nomic-embed-text | mxbai-embed-large

    print("Embedding sample data...")
    documents = []
    for idx, passage in enumerate(sample_data.split('\n')):
        #embedding = ollama.embeddings(model="all-minilm",prompt=passage)
        embedding = model.encode(passage)
        print(idx, passage[:10], embedding)
        if len(passage) == 0:
            continue
        documents.append({
            "id": str(idx),
            "text": passage,
            "embedding": embedding,
            "metadata": {"category": "album"},
        })
    print("Sample data embedded successfully.")
    print("Sample data number:", len(documents))

    print("Inserting documents into TiDB...")
    tidb_vector_client.insert(
        ids=[doc["id"] for doc in documents],
        texts=[doc["text"] for doc in documents],
        embeddings=[doc["embedding"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
    )
    print("Documents inserted successfully.")


if __name__ == '__main__':
    print("Start ")
    while True:
        user_input = input("input:")
        if user_input.lower() == "help":
            print("load_document: load sample data to tidb")
        elif user_input.lower() == "/load_document":
            embedding_tidb_document()
        elif user_input.lower() == "/print_embedding":
            print_embedding_data()
        elif user_input.lower() == "exit":
            sys.exit(0)
        else:
            print("Start to query")
            tidb_vector_client = TiDBVectorClient(
                table_name=TABLE_NAME,
                connection_string=TIDB_DATABASE_URL,
                drop_existing_table=False,
                )
            #queryres=tidb_vector_client.query(ollama.embeddings(model="all-minilm",prompt="hello")["embedding"],k=3)
            queryres=tidb_vector_client.query(model.encode(user_input),k=3)
            idx=1
            for res in queryres:
                print("========================================")
                print("Top {}:".format(idx))
                print("id:{}".format(res.id))
                print("docuemnt:{}".format(res.document))
                print("meta:{}".format(res.metadata))
                print("distance:{}".format(res.distance))
                idx+=1    
    
    
    

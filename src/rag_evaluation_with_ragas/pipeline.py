from create_document import GetData
from chunker import Chunker
from embedding import StoreData


def main():
    # 1. Load documents
    data_loader = GetData(data_type="training")
    documents = data_loader.get_documents()

    # 2. Create chunks
    chunker = Chunker()
    chunks = chunker.create_chunks(documents)

    # 3. Build vector store
    store = StoreData()
    store.build_store()

    # 4. Add chunks to vector store
    store.add_to_store(chunks)

    # 5. Create retriever
    retriever = store.as_retriever()

    # 6. Query
    query = "What was Entergy's long-term debt maturity in 2022?"

    results = retriever.invoke(query)

    # 7. Print retrieved results
    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Content:")
        print(result.page_content)
        print("\nMetadata:")
        print(result.metadata)


if __name__ == "__main__":
    main()
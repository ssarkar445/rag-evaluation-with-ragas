from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


from create_document import GetData
from chunker import Chunker
from config import settings


class StoreData:

    def __init__(self) -> None:
        self.embedding = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        self.vector_store: Chroma | None = None

    def build_store(self) -> Chroma:
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embedding,
            persist_directory=str(settings.chroma_persist_directory),
        )

        return self.vector_store

    def add_to_store(self, chunks: list[Document]) -> None:
        if self.vector_store is None:
            raise RuntimeError(
                "Vector store is not initialized. Call build_store() first."
            )

        self.vector_store.add_documents(chunks)

    def as_retriever(self):
        if self.vector_store is None:
            raise RuntimeError(
                "Vector store is not initialized. Call build_store() first."
            )

        return self.vector_store.as_retriever(
            search_kwargs={"k": settings.top_k}
        )


if __name__=="__main__":
    docclass = GetData()
    documents = docclass.get_documents()
    chunker = Chunker()
    chunks = chunker.create_chunks(documents)
    print(type(chunks))
    store = StoreData()
    store.build_store()
    store.add_to_store(chunks)
    retriever = store.as_retriever()
    results = retriever.invoke("What were Entergy's annual long-term debt maturities for 2018 through 2022?")

    for idx,result in enumerate(results):
        print(f"\n Content Number={idx+1}")
        print(result.page_content[:200])
        print(result.metadata)
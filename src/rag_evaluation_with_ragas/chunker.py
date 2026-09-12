from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from create_document import GetData



class Chunker:
     def create_chunks(self,documents:list[str])->list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

if __name__=="__main__":
    docclass = GetData()
    documents = docclass.get_documents()
    print(type(documents))
    chunker = Chunker()
    chunks = chunker.create_chunks(documents)
    print(type(chunks))

    for i,chunk in enumerate(chunks):
            if i==2:
                break
            else:
                print(f"\n ChunkNumber={i+1}")
                print(chunk.page_content[:200])
                print(chunk.metadata)
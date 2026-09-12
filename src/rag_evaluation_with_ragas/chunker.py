from langchain_text_splitters import RecursiveCharacterTextSplitter


# Initialize the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)

# Chunk documents
chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))

for i,chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print("Content:", chunk.page_content[:500])
    print("Metadata:", chunk.metadata)
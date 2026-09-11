from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
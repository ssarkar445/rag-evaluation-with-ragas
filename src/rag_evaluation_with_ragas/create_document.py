import json
from pathlib import Path
from langchain_core.documents import Document
from config import settings


class GetData:

    def __init__(self, data_type: str = "training"):
        self.data_type = data_type

    def get_documents(self) -> list[Document]:
        documents = []
        if self.data_type == "training":
            for json_file in settings.corpus.glob("*.json"):
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                documents.append(
                    Document(
                        page_content=data["text"],
                        metadata={
                            "document_id": data.get("document_id"),
                            **data.get("metadata", {}),
                        },
                    )
                )
            print("Training documents loaded:", len(documents))
            return documents
        raise ValueError(
            f"Unsupported data_type: {self.data_type}"
        )

    def get_evaluation_data(self) -> list[dict]:
        evaluation_data = []
        for json_file in settings.golden.glob("*.jsonl"):
            with json_file.open("r", encoding="utf-8") as f:
                evaluation_data.extend(
                    json.loads(line)
                    for line in f
                    if line.strip()
                )
        print("Evaluation records loaded:", len(evaluation_data))
        return evaluation_data

if __name__=="__main__":
    dataclass = GetData()
    data = dataclass.get_evaluation_data()
from datasets import load_dataset
from pathlib import Path
import json

# Config
NUM_DOCUMENTS = 200
NUM_GOLDEN = 50

OUTPUT_DIR = Path("finance_rag_data")
CORPUS_DIR = OUTPUT_DIR / "corpus"
GOLDEN_DIR = OUTPUT_DIR / "golden"

CORPUS_DIR.mkdir(parents=True,exist_ok=True)
GOLDEN_DIR.mkdir(parents=True,exist_ok=True)

# Load Data
dataset = load_dataset(
    "G4KMU/t2-ragbench",
    "FinQA"
)

# Use test set for evaluation examples
test_data = dataset["test"]

golden_records = test_data.select(
    range(min(NUM_GOLDEN, len(test_data)))
)


document_ids = []

for item in golden_records:
    context_id = golden_records['context_id']
    if context_id not in document_ids:
        document_ids.append(context_id)

for item in test_data:
    context_id = test_data['context_id']
    if context_id not in document_ids:
        document_ids.append(context_id)

    if len(document_ids)>=NUM_DOCUMENTS:
        break


document_ids = document_ids[:NUM_DOCUMENTS]


documents = {}

for item in test_data:

    context_id = item["context_id"]

    if context_id not in document_ids:
        continue

    if context_id in documents:
        continue

    documents[context_id] = {
        "document_id": context_id,
        "text": item["context"],
        "metadata": {
            "source": "T2-RAGBench",
            "dataset": "FinQA",
            "company_name": item.get("company_name"),
            "company_symbol": item.get("company_symbol"),
            "report_year": item.get("report_year"),
            "company_sector": item.get("company_sector"),
            "company_industry": item.get("company_industry"),
            "file_name": item.get("file_name"),
            "page_number": item.get("page_number"),
        }
    }

for document_id, document in documents.items():
    print(document_id,document)
    break

    # path = CORPUS_DIR / f"{document_id}.json"

    # with open(path, "w", encoding="utf-8") as f:
    #     json.dump(
    #         document,
    #         f,
    #         indent=2,
    #         ensure_ascii=False
    #     )
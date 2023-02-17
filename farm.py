from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.utils import print_answers
from haystack.utils import convert_files_to_docs

# Create an in-memory document store
document_store = InMemoryDocumentStore()

# Convert and add all files from the specified directory into the document store
datapath = "./data"
all_docs = convert_files_to_docs(dir_path=datapath, split_paragraphs=True)
document_store.write_documents(all_docs)

# Create an instance of the FARMReader class and use it to extract answers to a given question
reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2", use_gpu=False)
while True:
    question = input("Q: ")
    predictions = reader.predict(question=question, documents=document_store.get_all_documents(), top_k=3)
    print_answers(predictions, details="minimal")
#First, install the "farm-haystack" library and its "faiss" module.
# pip install 'farm-haystack[faiss]'

'''
Then, it imports the necessary modules from the library
and sets the path to the directory containing the data.
'''
from haystack.utils import convert_files_to_docs
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline


datapath = "./data"
index_path = "./idx.path"
initialRun = input("first run? y/n: ")
document_store = None

if initialRun == "y":
    document_store = FAISSDocumentStore.load(index_path=index_path)
else:
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", similarity="cosine")

'''
    Next, it converts the files in the directory into Haystack documents 
    using the "convert_files_to_docs" function, and writes the documents to the document store.
    '''
all_docs = convert_files_to_docs(dir_path=datapath, split_paragraphs=True)
document_store.write_documents(all_docs)

'''
The "EmbeddingRetriever" class retrieves the documents using sentence embeddings 
generated from a pre-trained model. It initializes an embedding model, 
and sets the number of documents to retrieve using the "top_k" parameter.
'''
model = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
retriever = EmbeddingRetriever(document_store=document_store, use_gpu=False, embedding_model=model, top_k=3)

'''
The document store's embeddings are updated with the retriever's 
embeddings using the "update_embeddings" method. This only needs to be done once, 
so it is commented out.
'''
print(f"Number of documents: {len(all_docs)}")
print(f"Number of embeddings: {document_store.get_embedding_count()}")

if initialRun == "y":
    # If the embeddings are missing, try rebuilding the index
    document_store.update_embeddings(retriever=retriever)
    print(f"Number of embeddings: {document_store.get_embedding_count()}")
    document_store.save(index_path=index_path)

'''
The "DocumentSearchPipeline" is created using the retriever. 
The pipeline takes a query as input and returns a dictionary containing 
the most relevant documents and their scores.
'''
semantic_search_pipeline = DocumentSearchPipeline(retriever=retriever)

'''
Finally, the user is prompted to input a query, 
and the pipeline is run on the query. The output is printed to the console, 
with the document ID and its relevance score
'''
while True:
    question = input("Q: ")
    if question == "quit":
        break
    prediction = semantic_search_pipeline.run(query=question)
    print(prediction['documents'])
    for i, pred in enumerate(prediction['documents']):
        print(i, pred.content[:100] + '\n')

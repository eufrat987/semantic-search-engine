'''
pip install --upgrade pip
pip install farm-haystack[colab]
'''
import logging
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint
from haystack.utils import print_answers

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

'''
We’ll start creating our question answering system by initializing a DocumentStore. 
A DocumentStore stores the Documents that the question answering system uses 
to find answers to your questions. In this tutorial, we’re using the InMemoryDocumentStore, 
which is the simplest DocumentStore to get started with. It requires no external dependencies 
and it’s a good option for smaller projects and debugging. But it doesn’t scale up 
so well to larger Document collections, so it’s not a good choice for production systems. 
To learn more about the DocumentStore and the different types of external databases 
that we support, see DocumentStore.

Let’s initialize the the DocumentStore:
'''
document_store = InMemoryDocumentStore(use_bm25=True)
doc_dir = "data/"

'''
Use TextIndexingPipeline to convert the files you just downloaded into Haystack Document 
objects and write them into the DocumentStore:

The code in this tutorial uses the Game of Thrones data, 
but you can also supply your own .txt files and index them in the same way.

As an alternative, you can cast you text data into Document objects 
and write them into the DocumentStore using DocumentStore.write_documents().
'''
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

'''
Our search system will use a Retriever, so we need to initialize it. 
A Retriever sifts through all the Documents and returns only the ones relevant 
to the question. This tutorial uses the BM25 algorithm. 
For more Retriever options, see Retriever.

Let’s initialize a BM25Retriever and make it use 
the InMemoryDocumentStore we initialized earlier in this tutorial:
'''
retriever = BM25Retriever(document_store=document_store)

'''
A Reader scans the texts it received from the Retriever 
and extracts the top answer candidates. Readers are based on powerful 
deep learning models but are much slower than Retrievers at processing 
the same amount of text. In this tutorial, we’re using a FARMReader 
with a base-sized RoBERTa question answering model called deepset/roberta-base-squad2. 
It’s a strong all-round model that’s good as a starting point. 
To find the best model for your use case, see Models.

Let’s initialize the Reader:
'''
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

'''
In this tutorial, we’re using a ready-made pipeline called ExtractiveQAPipeline. 
It connects the Reader and the Retriever. The combination 
of the two speeds up processing because the Reader only processes 
the Documents that the Retriever has passed on. 
To learn more about pipelines, see Pipelines.

To create the pipeline, run:
'''
pipe = ExtractiveQAPipeline(reader, retriever)

'''
Use the pipeline run() method to ask a question. 
The query argument is where you type your question. 
Additionally, you can set the number of documents you want 
the Reader and Retriever to return using the top-k parameter. 
To learn more about setting arguments, see Arguments. 
To understand the importance of the top-k parameter, 
see Choosing the Right top-k Values.
'''
while True:
    prediction = pipe.run(
        query=input("Q: "),
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
    )

    pprint(prediction)

    print_answers(
        prediction,
        details="minimum"  # Choose from `minimum`, `medium`, and `all`
    )

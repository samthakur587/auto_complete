from llama_index.core import PromptTemplate
import openai
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

from llama_index.core.query_engine import FLAREInstructQueryEngine

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

######### Prompt Templates #########
text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " both the context information and also using your own knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own.\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer  with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing both the new"
    " context and your own knowledge, update or repeat the existing answer.\n"
)
refine_template = PromptTemplate(refine_template_str)

######### OpenAI LLM and indexing  #########

llm = OpenAI(model="gpt-3.5-turbo")

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)


from llama_index.core.composability import QASummaryQueryEngineBuilder

query_engine_builder = QASummaryQueryEngineBuilder(
    llm=llm,
)
query_engine = query_engine_builder.build_from_documents(documents)
flare_query_engine = FLAREInstructQueryEngine(
    query_engine=index.as_query_engine(similarity_top_k=3),
    max_iterations=3,
    verbose=True,
)

######### FastAPI #########

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/complete")
def complete(query_inp: str):
    # response = flare_query_engine.query(query_inp)
    response2 = query_engine.query(query_inp)
    return response2


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0",port=8000)
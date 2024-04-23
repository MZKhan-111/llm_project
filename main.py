from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import dask.dataframe as dd
import uvicorn
from fastapi import Form

app = FastAPI()
templates = Jinja2Templates(directory="templates")




class Item(BaseModel):
    query: str

model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

pandas_df = pd.read_csv("/home/mzkhan/Desktop/patent_code/1_9_24/bigPatent_curated_550k.csv")
sentence_embeddings = np.load("/home/mzkhan/Desktop/patent_code/1_9_24/BigPatent_data_embeddings_550k.npy")


@app.get("/")
async def get_query_form(request: Request):
    return templates.TemplateResponse("query_form.html", {"request": request})


@app.post("/similarity")
async def calculate_similarity(request: Request, query: str = Form(...)):
    query_embedding = model.encode([query])[0]
    cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    result_df = pd.DataFrame({
        'patent_id': pandas_df['patent_id'],
        'text': pandas_df['text'],
        'similarity_score': cosine_scores.tolist()
    })
    result_df = result_df.sort_values(by='similarity_score', ascending=False)
    final_df = result_df.nlargest(10, ['similarity_score'])
    return templates.TemplateResponse("results.html", {"request": request, "results": final_df.to_dict(orient='records')})

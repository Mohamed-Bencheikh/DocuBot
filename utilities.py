import base64
import pdfplumber
import textwrap
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
G_API_KEY = os.environ['GOOGLE_API_KEY']
genai.configure(api_key=G_API_KEY)
MODEL = "models/embedding-001"
# Function to extract text from a PDF and summarize it
def get_pdf_text(pdf_file):
    text = ""
    # Open the PDF file and extract text
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

def display_pdf(file_path):
  # Read the PDF file
  with open(file_path, "rb") as f:
      data = f.read()
  # Convert PDF content to base64
  base64_pdf = base64.b64encode(data).decode("utf-8")
  # Create an iframe to display the PDF
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
  return pdf_display

def split_text(text, max_length=200):
  """Split text into smaller chunks based on a specified length."""
  words = text.split()
  chunks = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
  return chunks


def get_embeddings(text,task_type):
  embs = genai.embed_content(model=MODEL,content=text, task_type=task_type)
  return embs["embedding"]

def chunks_to_dataframe(chunks):
  df = pd.DataFrame(chunks,columns=['text'])
  df['embeddings'] = df.apply(lambda row : get_embeddings(row['text'], task_type='retrieval_document'), axis=1)
  return df

def get_relevant_context(query, df):
  query_embeddings = get_embeddings(query, task_type='retrieval_query')
  df_embeddings = np.stack(df['embeddings'])
  dot_products = np.dot(df_embeddings, query_embeddings)
  idx = np.argmax(dot_products)
  relevant = df.iloc[idx]['text'] # Return text from index with max value
  return relevant

## PROMPT
def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent(f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'
    ANSWER:
  """)
  return prompt

def get_response(content,query):
  chunks = split_text(content)
  chunks_df = chunks_to_dataframe(chunks)
  relevant_passage = get_relevant_context(query,chunks_df)
  prompt = make_prompt(query,relevant_passage)
  model = genai.GenerativeModel(model_name='gemini-pro')
  response = model.generate_content(prompt)
  return response.text



def summarize(text,max_length):
#   # summarizer = pipeline(task="summarization", model='facebook/bart-large-cnn')
#   text_chunks = split_text(text, max_length=max_length)  # Split into chunks of 500 words
#   # Summarize each chunk and combine the results
#   summaries = [summarizer(chunk)[0]['summary_text'] for chunk in text_chunks]
#   # Combine the summaries into a final summary
#   final_summary = ' '.join(summaries)
#   return final_summary
  return "This is a summary of the document."
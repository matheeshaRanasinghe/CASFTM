import numpy as np
from google import genai
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
import wikipedia 
import requests
from rag import main

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
answer the question based on the folowing code snipets:

{context}

---

question is: {question}
"""


def chat():
	API_KEY = input("type in the google-gemini-api KEY:  ")
	query_text = input("input: ")
	
	print("generating embedings...")
	embedding_function = get_embedding_function()
	print("finish generating embeddings")
	print("serching simlarities...")
	db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
	
	results = db.similarity_search_with_score(query_text, k=8)

	context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])


	prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
	prompt = prompt_template.format(context=context_text, question=query_text)
    
	client = genai.Client(api_key=API_KEY)
	response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
	return response.text



print("""
 .---------------------------------------------------------------------.
| .------------------------------------------------------------------. |
| |  _________      ______      _______      _______     ___    ____ | |
| | |  _______|    /      \    /  ___  |    |  ____|   |_   \  /   _|| |
| | | |           /   /\   \  |  (__ \_|    | |____      |   \/   |  | |
| | | |           |  /__\  |    '.___`-.    | _____|     | |\  /| |  | |
| | | |_______    |  ____  |   |`\____) |   | |         _| |_\/_| |_ | |
| | |_________|   |_|    |_|   |_______.'   |_|        |____________|| |
| | by synapsnex ass. with colabaration of mr.brain cyberzer         | |
| '------------------------------------------------------------------' |
 '---------------------------------------------------------------------'
""")
print("")
print("Welcome to C.A.S.F.M. multi-languege code analzing software.")
print("powerd by gemini 1.5 flash")
print("")
print("1) create vector database")
print("2) run analizer")
op = input("choose an option: ")
print("")
if op == "1":
	main()
	
if op == "2":
	terminated = ""
	while terminated != "exit":
		terminated = input("press any key to continue or type exit to close the app:  ")
		if terminated == "exit":
			break
		respone = chat()
		print("this is the response")
		print("")
		print(respone)
	
	
print("thankyou for using the softwear")




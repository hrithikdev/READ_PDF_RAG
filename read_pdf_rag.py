import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
# from google import genai
# client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# for model in client.models.list(): 
#     print(model.name)
loader = PyPDFLoader('https://fairbanks.indianapolis.iu.edu/doc/10-Tips-Healthy-Lifestyle.pdf')
pages = loader.load_and_split()
print(len(pages)," pages loaded.")

#Set up the LLM Object
llm = ChatGoogleGenerativeAI(model='gemini-4-hydrogen',  google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0.5)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=os.getenv('GOOGLE_API_KEY'))

#Convert the pages into Embeddings and store into FAISS Vector DB
vector = FAISS.from_documents(pages, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following the question based only on the provided context:
<context>
{context}
</context>

Question: {input}""")


#Combines the LLM and the prompt. "Stuff" means it will take all retrieved text and "stuff" it directly into the {context} placeholder.
document_chain = create_stuff_documents_chain(llm, prompt)      

# Turns the FAISS database into a search engine.
retriever = vector.as_retriever() 

# Connects the search engine (retriever) to the AI (document_chain).
retrieval_chain = create_retrieval_chain(retriever, document_chain)         


# Behind the scenes, the retriever searches the PDF pages in the FAISS database for text matching the question, 
# inserts those pages into the prompt's {context} (overwriting the "Benefits of Healthy Body." 
# string passed in invoke), sends it to Gemini, and prints the final summarized answer.
response = retrieval_chain.invoke({"context": "Benefits of Healthy Body.",
                                   "input":"Give 11 tips to maintain a healthy lifestyle in short."})

print(response["answer"])
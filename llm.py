from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(model='openai/gpt-oss-120b',
               temperature=0.0
               )



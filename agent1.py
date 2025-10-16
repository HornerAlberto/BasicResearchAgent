import wikipedia
# from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI


class Agent:
    def __init__(self, max_tokens=500, max_info_tokens=2000, temperature=0.7):
        self.llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
        self.max_info_tokens = max_info_tokens
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
        )
        self.knowledge_base = None

    def gather_information(self, topic: str) -> str:
        return wikipedia.page(topic).content[: self.max_info_tokens]

    def analyze_information(self, info: str):
        prompt = f"Summarize key points from this text: \n\n{info}\n\n"
        return self.llm.invoke(prompt)

    def generate_summary(self, analysis: str):
        prompt = (
            f"Based on the following analysis, "
            f"generate a concise summary: \n\n{analysis}"
        )
        return self.llm.invoke(prompt)

    def enrich_knowledge_base(self, text: str):
        if self.knowledge_base:
            self.knowledge_base.add_texts(self.text_splitter.split_text(text))
        else:
            self.knowledge_base = FAISS.from_texts(
                self.text_splitter.split_text(text),
                embedding=OpenAIEmbeddings(),
            )

    def query_knowledge_base(self, query):
        return self.knowledge_base.similarity_search(query, k=1)[0].page_content

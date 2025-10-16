import asyncio
import arxiv
import wikipedia
from langchain_text_splitters import CharacterTextSplitter
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

    async def gather_wikipedia_information(self, topic: str) -> str:
        return wikipedia.page(topic).content[: self.max_info_tokens]

    async def gather_arxiv_information(self, topic: str) -> str:
        search_query = "abs:" + " AND abs:".join(topic.split())
        search = arxiv.Search(
            search_query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = arxiv.Client().results(search)
        info = "\n========\n".join(
            ["Title: " + i.title + "\n" + "Abstract: " + i.summary
             for i in results]
            )
        return info[: self.max_info_tokens]

    async def gather_information(self, topic: str) -> str:
        task1 = asyncio.create_task(
            self.gather_wikipedia_information(topic)
        )
        task2 = asyncio.create_task(
            self.gather_arxiv_information(topic)
        )
        wiki_info = await task1
        arxiv_info = await task2
        if (len(wiki_info) > 0) and (len(arxiv_info) > 0):
            return ("Wikipedia:\n"
                    + wiki_info[: self.max_info_tokens]
                    + "\nArxiv:\n"
                    + arxiv_info[: self.max_info_tokens])
        else:
            return ("Wikipedia:\n"
                    + wiki_info
                    + "\nArxiv:\n"
                    + arxiv_info)

    def analyze_information(self, info: str):
        prompt = (
            f"Identify key points from each source in this text:"
            f"\n\n{info}\n\n"
            )
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
        content = self.knowledge_base.similarity_search(
            query, k=1
        )[0].page_content
        return self.llm.invoke(
            "Please, answer the following question with the given information:"
            f"\nQuestion:\n***{query}***"
            f"\nInformation:\n***{content}***"
        )
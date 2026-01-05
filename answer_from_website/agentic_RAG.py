from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent


load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# retriever = vector_store.as_retriever(search_kwargs={"k": 3})


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information from wikipedia about to help answer a query."""

    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs



tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a wikipedia page. "
    "Use the tool to help answer user queries."
)
agent = create_agent(llm, tools, system_prompt=prompt)


query = (
    "which current ThinkPad models has dedicated GPU? don't ask me follow-up questions."
)

# response = agent.invoke({"messages": [{"role": "user", "content": query}]})

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    print("=" * 80)
    event["messages"][-1].pretty_print()
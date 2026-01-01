import warnings
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch


class Company(BaseModel):
    """Schema for a company used by the agent"""

    company_name: str = Field(description="Company Name")
    description: str = Field(description="Company Description")
    size: str = Field(description="Company Size of employees")
    county_of_origin: str = Field(description="Company County of Origin")
    office_location: str = Field(description="Company Office Location in Canada")
    website: str = Field(description="Company Website url")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and companies"""

    companies: List[Company] = Field(
        default=None, description="List of companies "
    )
    

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Tool to search the internet and return results.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


def main():
    llm = ChatOpenAI()
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    
    results = agent.invoke({"messages": [HumanMessage(
        content="Find top 5 healthcare IT companies in Ontario by size, find their size in employees, county of origin, office location in canada, and website")]})
    print(results['structured_response'])
    

if __name__ == "__main__":
    main()

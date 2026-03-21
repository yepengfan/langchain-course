from datetime import date
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str):
    """
    Tool that searches over the Internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for a response from the agent with answer and sources."""

    answer: str = Field(description="The answer to the user's query")
    sources: list[Source] = Field(
        default_factory=list, description="The sources used to generate the answer"
    )


llm = ChatOpenAI(model="gpt-4o")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    query = "What is the weather in Tokyo tomorrow?"
    result = agent.invoke({"messages": [
        SystemMessage(content=f"You are a helpful assistant. Today's date is {date.today()}."),
        HumanMessage(content=query),
    ]})
    answer, sources = (
        result["structured_response"].answer,
        result["structured_response"].sources,
    )
    print(f"answer: {answer}")
    print(f"sources:\n{'\n'.join(s.url for s in sources)}")


if __name__ == "__main__":
    main()

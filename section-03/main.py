from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# from langchain_tavily import TavilySearch
from tavily import TavilyClient

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


llm = ChatOpenAI(model="gpt-5")
# tools = [TavilySearch()]
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    query = "What is the weather in Tokyo tomorrow?"
    print(f"{query}")
    result = agent.invoke({"messages": HumanMessage(content=query)})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


information = """
Elon Musk is a businessman and investor known for leading several high-profile technology
companies. Born on June 28, 1971, in Pretoria, South Africa, he moved to Canada at age 17
and later to the United States, where he attended the University of Pennsylvania.

Musk co-founded PayPal, which was sold to eBay in 2002 for $1.5 billion. He then founded
SpaceX in 2002 with the goal of reducing space transportation costs and enabling the
colonization of Mars. SpaceX developed the Falcon and Starship rockets and became the first
private company to send astronauts to the International Space Station.

In 2004, Musk joined Tesla Motors as chairman and later became CEO, driving the company to
become the world's most valuable automaker. Tesla produces electric vehicles, battery energy
storage systems, and solar panels.

Musk also co-founded Neuralink, a neurotechnology company developing brain-computer
interfaces, and The Boring Company, which focuses on tunnel construction and infrastructure.
In 2022, he acquired Twitter and rebranded it as X. He has been involved in artificial
intelligence through xAI, which developed the Grok chatbot.

Musk is one of the wealthiest individuals in the world and remains a polarizing public figure
known for his ambitious vision of humanity's future.
"""


def main():
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()

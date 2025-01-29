from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    llm = ChatOpenAI(
        model="gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    agent = Agent(
        task="Go to Reddit, search for 'browser-use', click on the first post and return the first comment.",
        llm=llm
    )
    
    result = await agent.run()
    print(result)

asyncio.run(main())
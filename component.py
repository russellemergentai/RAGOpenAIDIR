from langchain.agents import AgentExecutor
from langchain_core.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
secret_key = "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"

# https://platform.openai.com/usage
# https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents
# https://platform.openai.com/docs/overview


try:

    # loader = TextLoader("target/data.txt", encoding = 'UTF-8')

    loader = DirectoryLoader("target/")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=secret_key)
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever()


    tool = create_retriever_tool(
        retriever,
        "search_data",
        "Searches and returns documents from data file.",
    )
    tools = [tool]


    # default model: Gpt-3.5-turbo-0613
    llm = ChatOpenAI(openai_api_key=secret_key)

    # This is needed for both the memory and the prompt
    memory_key = "history"


    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)


    system_message = SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)


    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    result = agent_executor({"input": "hi, im bob"})

    print(result["output"])

    result = agent_executor(
        {
            "input": "briefly summarize the story 'the moon thing'. include the name of the protagonist"
        }
    )

    print(result["output"])

except SyntaxError as e:
    print(e.msg)

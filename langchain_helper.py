import os
import json
from dotenv import load_dotenv

# LangChain Packages
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType, create_sql_agent

load_dotenv() # For environment variables

def generate_retriever_tool():

    embeddings = OpenAIEmbeddings()

    with open('few_shots.json','r') as file:
        few_shots = json.load(file)
    
    few_shot_docs = [
        Document(page_content=question, metadata={'sql_query': few_shots[question]})
        for question in few_shots.keys()
    ]
    
    vector_db = FAISS.from_documents(few_shot_docs, embeddings)
    retriever = vector_db.as_retriever()

    tool_description = """
    This tool will help you understand similar examples to adapt them to the user question.
    Input to this tool should be the user question.
    """

    retriever_tool = create_retriever_tool(
        retriever, name="sql_get_similar_examples", description=tool_description
    )

    custom_tool_list = [retriever_tool]

    return custom_tool_list

def langchain_sql_agent():

    custom_tool_list = generate_retriever_tool()
    
    snowflake_url = f"snowflake://{os.getenv('username')}:{os.getenv('password')}@{os.getenv('snowflake_account')}/{os.getenv('database')}/{os.getenv('schema')}?warehouse={os.getenv('warehouse')}&role={os.getenv('role')}"
  
    db = SQLDatabase.from_uri(snowflake_url,sample_rows_in_table_info=1, include_tables=['census_data_zip_codes','home_value_zillow_zip_codes'])
    llm = ChatOpenAI(temperature=0, verbose=True, model="gpt-4-1106-preview", openai_api_key = os.getenv('OPENAI_API_KEY'))
  
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
  
    custom_suffix = """
                    I should first get the similar examples I know.
                    If the examples are enough to construct the query, I can build it.
                    Otherwise, I can then look at the tables in the database to see what I can query.
                    Then I should query the schema of the most relevant tables
                    """
    
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=custom_tool_list,
        suffix=custom_suffix,
    )
  
    return agent

def invoke_llm(agent, query):

    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            Do not apply any LIMIT in the query.

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Please ensure that the response is in json format without getting issue of formatting while applying json.loads()

            Below is the query.
            Query: 
            """
        + query
    )

    response = agent.run(prompt)

    return response.__str__()
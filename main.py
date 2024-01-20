import langchain_helper as lch
import streamlit as st
import json
import pandas as pd
import altair as alt

# Creating the LangChain SQL Agent
llm_agent = lch.langchain_sql_agent()

st.title("Questions with LLM powered by Snowflake")

query = st.text_area("Insert your query")

if st.button("Run"):
    
    response = lch.invoke_llm(llm_agent, query)
    
    # Special data cleansing of the response (just in case...)
    response_cleaned = response.replace('```json\n','').replace('\n```','')
    json_response = json.loads(response_cleaned)

    main_key = list(json_response.keys())[0]

    if main_key == 'answer':
        st.write("API Response (Answer):", json_response['answer'])

    elif main_key == 'line':
        st.write("API Response (Line Chart):")
        
        # Extracting columns and data
        line_data = json_response['line']['data']
        columns = json_response['line']['columns']
        values = [item[1] for item in json_response['line']['data']] 

        # Create DataFrame
        df_line = pd.DataFrame(line_data, columns=[columns[0], columns[1]])

        # Create the chart
        chart = alt.Chart(df_line).mark_line().encode(
            x=columns[0], 
            y=alt.Y(columns[1], scale=alt.Scale(domain=(min(values), max(values))))
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

    elif main_key == 'table':
        st.write("API Response (Table):")
        data = json_response['table']['data']
        columns = json_response['table']['columns']
  
        # Convert numeric columns to float
        for i, column in enumerate(columns):
            try:
                data = [[row[0], float(row[1])] if i == columns.index('numeric_column_name') else row for row in data]
            except ValueError:
                pass
        
        df_data = pd.DataFrame(data, columns = json_response['table']['columns'])
        st.write(df_data)
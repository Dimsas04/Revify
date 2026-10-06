from typing import Annotated
from langchain_google_community import GoogleSearchResults
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import requests
from dotenv import load_dotenv
import os
from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from IPython.display import Image, display
import os
import cv2
import numpy as np
from old_code.scrapper import scraping, is_valid_url


# Load environment variables
load_dotenv()

os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    


graph_builder = StateGraph(State)

# llm = ChatGoogleGenerativeAI()
# print(llm.list_models())
# exit(0)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)


def u_chatbot1(state: State):
    msg = ""
    for i in state["messages"][0].content:
        msg += i + "\n"
    msg +=  "\n The data give is the reviews of the product. Perform sentiment analysis on the reviews to give an overall analysis of the product."
    msg += "\n Then from the original data extract the key features of the product and categorize them."
    msg += "\n Finally, give a summary of the product based on the sentiment analysis and key features."
    msg += "\n Do not assume anything or make any assumptions whatsoever. Just use the data given."
    val = llm.invoke(str(msg))
    return {"messages": [val]}

def u_chatbot2(state: State):
    msg = ""
    for i in state["messages"][0].content:
        msg += i + "\n"
    msg +=  "\n The data give is the reviews of the product. Perform sentiment analysis on the reviews to give an overall analysis of the product."
    msg += "\n Then from the original data extract the key features of the product and categorize them."
    msg += "\n Finally, give a summary of the product based on the sentiment analysis and key features."
    msg += "\n Do not assume anything or make any assumptions whatsoever. Just use the data given."
    # msg = HumanMessage(content = str(msg)) 
    print(msg)
    print(type(msg))
    print("OK")
    val = llm.invoke(str(msg))
    print("OKOKOKK")
    return {"messages": [val]}

def v_chatbot1(state: State):
    msg = ""
    for i in state["messages"][0].content:
        msg += i + "\n"
    msg += "\n\n\nThis is the oiginal Data.\n\n\n\n"+ "-----------------------------\n"  + state["messages"][-1].content  + "\n-----------------------------\n" 
    print("check2")
    msg += "This is the analytics generated from the data given. The data is the reviews of the product. Validate if the data produced is accurate and based solely on the data given. Don't make any assumptions of your own."
    msg += "If the reviews are accurate, then just return the original reviews. Else return the reviews and give the feedback on where you think its wrong."
    print("check3")
    # print(state["messages"][0].content)
    print("OK1")
    # print(msg)
    print("OK2")
    # msg = HumanMessage(content = str(msg)) 
    print("OK3")
    # print(msg)
    val = llm.invoke(str(msg))
    print("\n======================\n")
    print("LLM1: ", val)
    print("\n======================\n")
    return {"messages": [val]}

def v_chatbot2(state: State):
    msg = ""
    for i in state["messages"][0].content:
        msg += i + "\n"
    msg += "This is the oiginal Data.\n\n\n\n"+ "-----------------------------"  + state["messages"][-1].content  + "-----------------------------" 
    msg += "This is the analytics generated from the data given. The data is the reviews of the product. Validate if the data produced is accurate and based solely on the data given. Don't make any assumptions of your own."
    # print(state["messages"][0].content)
    msg = HumanMessage(content = str(msg)) 
    val = llm.invoke(msg)
    print("\n======================\n")
    print("LLM2: ", val)
    print("\n======================\n")
    return {"messages": [val]}

def abc():
    # def search(state: State):
    #     user_query = state["messages"][-1].content  
    #     search_results = search_tool.run(user_query) 
    #     # print("OK1")
    #     # If the result is a string, wrap it in a list
    #     if isinstance(search_results, str):
    #         search_results = [search_results]
    #     return {"messages": state["messages"] + [AIMessage(content="\n".join(search_results))]}   


    # def scrape_reviews(state: State):
    #     url = state["messages"][-1].content
    #     reviews = scraping(url)

    #     return {"messages": state["messages"] + [AIMessage(content="\n".join(reviews))]}



    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    # graph_builder.add_node("chatbot3", u_chatbot3)
    # graph_builder.add_node("search", search)

    # Set the entry point of the graph
    # graph_builder.set_entry_point("router")
    # graph_builder.add_edge("chatbot1", "chatbot2")

    # graph_builder.add_edge("router", "chatbot")
    # graph_builder.add_edge("router", "search")

    # graph_builder.add_edge("s", END)
    # graph_builder.add_edge("chatbot", 'search')
    # graph_builder.add_edge("chatbot1", END)
    print()



graph_builder.add_node("chatbot1", u_chatbot1)
# graph_builder.add_node("chatbot2", u_chatbot2)
graph_builder.add_node("chatbot3", v_chatbot1)
# graph_builder.add_node("chatbot4", v_chatbot2)


graph_builder.add_edge(START, "chatbot1")

# graph_builder.add_edge(START, "chatbot2")

graph_builder.add_edge("chatbot1", "chatbot3")
# graph_builder.add_edge("chatbot2", "chatbot4")

graph_builder.add_edge("chatbot3", END)
# graph_builder.add_edge("chatbot4", END)




graph = graph_builder.compile()



# try:
#     # Generate the Mermaid diagram and save it as a PNG
#     graph_image = graph.get_graph().draw_mermaid_png()
    
#     # Save the PNG data to a file
#     temp_file = "graph.png"
#     with open(temp_file, "wb") as f:
#         f.write(graph_image)
    
#     # Read and display the image using OpenCV
#     img = cv2.imread(temp_file)
#     cv2.imshow('Graph Visualization', img)
#     cv2.waitKey(0)  # Wait for any key press
#     cv2.destroyAllWindows()  # Close the window
    
#     # Cleanup the temporary file
#     os.remove(temp_file)
    
# except Exception as e:
#     print(f"Failed to display graph: {e}")
#     pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # print("Assistant:", value["messages"][-1].content)
            print()


while True:
    try:
        user_input = input("User: ")
        user_input = scraping(user_input)
        
        if user_input in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # user_input = "Reviews: " + user_input
        
        stream_graph_updates(user_input)

    except Exception as e:
        print(f"Error: {e}")    
        break

# Agentic-AI-Skills

**Table of Contents:**

**A) UV for creating Virtual Environment**

**B) Getting Started with LangGraph - Creating the Environment**

**C) Setting up OpenAI API Key**

**D) Setting up GROQ API Key**

**E) Getting up with LangSmith API Key**

**F) Creating a Simple Graph or Workflow using LangGraph - Building Nodes and Edges**

**G) Building Simple Graph StateGraph and Graph Compiling**

**H) Developing LLM Powered Simple Chatbot using LangGraph**

**AA) Demo of MCP with Claude Desktop**

 #### Always create .env file in the venv - It is for LLM Models, OpenAI Keys, GroQ Keys, LangSmith keys [All those Environment variables and keys we will be storing there]

**A) UV for creating Virtual Environment**

uv - 10 - 100 times faster than pip

**pip install uv; uv init demouv; cd demouv (We have to go inside the demouv folder); uv venv (To create Virtual Envrionment)**

****.venv\Scripts\activate - To activate the Virtual Envrionment****

**To Install Libraries - uv add pandas (It was installed very very fast when compared to pip) [pyproject.toml file will get updated with dependencies for pandas, to confirm if pandas got installed]**

**In Krish Course, we use Conda for creating virtual environments because, it helps to manage multiple virtual environments to manage and switch among them**

**B) Getting Started with LangGraph**

**Installing the following Libraries from Requirements.txt**

Langchain, LangGraph, Langchain-core, Langchain-Community

**We also need ipykernel; Without that we can't run Jupyter Notebooks; We can also install in Terminal (Command Prompt)**

## Also created .env file in the venv - It is for LLM Models, OpenAI Keys, GroQ Keys, LangSmith keys [All those Environment variables and keys we will be storing there]

**C) Setting up OpenAI API Key**

#### OpenAI Keys - To use LLM Models (Which are Paid Models)

**OpenAI API Key - Paid Models; Most of the GenAI and Agentic AI Applications in the Industry uses OpenAI API Key's; So we will get an Idea on how it is happening in the Industry**

## We can use GroQ API Keys to call Open Source Available models; Groq provides access to LLM's for number of request

For OpenAI API Keys, we need to put some credit cards, and load some amount of Money in OpenAI API

**Steps to do:**

1. Search for OpenAI API in Google; It takes to "platform.openai.com"

2. Login if not; We can play with any models in "Playground"; OpenAI also provides codes say Python, Java to use those Models

3. **To get OpenAI API Key - Go to Dashboard --> API Keys --> Create a new API Key and create a secret API Key**

4. Copy that Secret key and paste in VS Code --> .env under "OPENAI_API_KEY=YOUR_API_KEY"

**The key will work only when we have a sufficient amount of Money credited in OpenAI account**

**We can see Credit Balance in Billing and also see Limits for every model in "Limits" section**

**D) Setting up GROQ API Key**

**If we can't afford OpenAI API key because of cost, we can use GROQ API key for using Open Source LLM Models. They have multiple open source LLM Models which they have hosted**

**Groq is an FAST AI INFERENCE. It provides access to open source LLM Models for some amount of tokens completely for free**

**Speed is an amazing thing about Groq. It is fast due to Language Processing Unit**

**Open Source Models such as LLAMA (From Meta), MIXTRAL (From Mistral AI), GEMMA (From Google), WHISPER (From OpenAI); It has Quint 2.5 - 32B parameters, Quint 2.5, Conda - 32B Parameters, DeepSeek Models, Google Models, Whisper, Meta**

**Steps to do:**

1. Click on Dev Console and do the sign up

2. It also has the Playground to test the models

3. Go to API Keys and create a new API Key

4. Copy the API Key and paste in VS Code on .env file with "GROQ_API_KEY=YOUR_API_KEY"

**E) Getting up with LangSmith API Key**

### We can name as LangChain API Key or LangSmith API Key

**Steps to do:**

1. Go to Langchain.com and Sign Up

#### 2. Once we directly sign up it goes to "smith.langchain.com". Langsmith helps us with a lot such as Debug, Playground, Prompt Management, Annotation, Testing, etc..

#### 3. We will try to use free version which has around 5000 request; There is also a paid version

#### 4. We are creating an Application and going to Monitor it; So we need an API Key; So we will use the LangSmith API Key; Whenever we develop an application with help of those API Key, it will send the send the data to LangSmith and there we can display/see everything (We can see all the runs, requests, Human Input, AI Response, Cost of Token, Which OpenAI is used, which model is used

#### This kind of Tracing is helpful when we are developing an Application, because it will help us to debug the part; All these information will be available in Langsmith; It will provide us an opportunity to monitor the entire application in the right way

### There will be option of "Prompts", "LangGraph Platform" --> "LangGraph Studio" (Any Application we run locally we can debug using LangGraph Studi); We have these options in LangSmith

5. Inside Settings in LangSmith --> Click on API Keys to create an API Key;

**Before that Make sure to choose "Developer - Free** option from **"Usage and Billing**, which has around 5000 free traces per month; We also have option to select "Startup", "Plus" option in Usage and Billing option in LangSmith

6. Once created an API Key, copy it and paste in VS Code on .env on "LANGCHAIN_API_KEY=YOUR_API_KEY"

#### Whenever we create any applications with help of LangGraph, once we run, all the Logs, all the monitoring, should happen in LangGraph Platform

**F) Creating a Simple Graph or Workflow using LangGraph - Building Nodes and Edges**

We don't use any LLM's here; We just use Nodes and Edges

**Coding Part:**

from typing_extensions import TypedDict

class State(TypedDict):
    graph_info:strf

**The state will be holding information in graph_info**

**If our Graph wants to return some information, we use from typing_extensions import TypedDict; TypeDict creates a dictionary type such that type checker will expect all instances to have a certain set of keys, where each key is associated with a value of consistent type. This expectation is not checked at the runtime; We can define all of them in a particular variable later; If we add that in class, we can represent as Dictionaries later**

**graph_info - Whenever we move from one node to another, this will be the information we will be sharing**

**State - First, define the State of the graph. The State schema serves as the input schema for all Nodes and Edges in the graph. Let's use the TypedDict class from python's typing module as our schema, which provides type hints for the keys.**

**Node - Nodes are just python functions. The first positional argument is the state, as defined above. Because the state is a TypedDict with schema as defined above, each node can access the key, graph_state, with state['graph_state']. Each node returns a new value of the state key graph_state. By default, the new value returned by each node will override the prior state value.**

def start_play(state:State):
    print("Start_Play node has been called")
    return {"graph_info":state['graph_info'] + " I am planning to play"}

#### **As soon as we start the execution of the flow, graph_info is nothing but the variable inside the class, which is acting as a schema. Every node which we execute should over write that particular value**

#### It should have the updated value as soon as the every node is executed; That is the reason we are returing as a key-value pair in the form of a dictionary like a graph_info=state graph_info and we added + "I am planning to play". This is the first Node that is getting executed in this Workflow

#### As soon as the start flow is getting executed, we are taking the previous information of whatever graph value was there and adding one more statement "I am planning to play", just to say that my node has got executed successfully. That is the first node;

def cricket(state:State):
    print("My Cricket node has been called")
    return {"graph_info":state['graph_info'] + " Cricket"}

### state['graph_info'] will already have the info that "I am planning to play", if this node gets executed, we will get "Cricket" gets printed along with that; This is what happens when we get Cricket Node gets executed

def badminton(state:State):
    print("My badminton node has been called")
    return {"graph_info":state['graph_info'] + " Badminton"}

### Here instead of Cricket, we have Badmiton node gets executed; Two edges are called over there for the conditional check; This is what three nodes got created

### Now we will create edges to connect the nodes; How to know, to go to which edge, we need to explicitly define some condition

import random
from typing import Literal

def random_play(state:State)-> Literal['cricket','badminton']:
    graph_info=state['graph_info']
    if random.random()>0.5:
        return "cricket"
    else:
        return "badminton"

**We created one more function; Literal means constant; This above function will be responsible on deciding which edge we should go**

**State will be of state; Return type will be Literal, means constant, it should be of two options only; It should either return cricket or it should return Badmiton; If we have football, we can go ahead and write that too**

**We are getting previous graph_info; if random.random>0.5, return cricket, else return Badmiton; It just selects only random number, if it is greater than 0.5 it returns cricket or it returns badmiton**

### Start Play is the node and from there we need to call the Function, decide play and it will either go in left or in right direction; Reason for hardcoding is those are the name of the nodes; Cricket and Badmiton should be same in the codes

**We have created nodes and also created the edge conditions, now we need to create our Graph**

#### Graph Construction

Now, we build the graph from our components defined above.

The StateGraph class is the graph class that we can use.

First, we initialize a StateGraph with the State class we defined above.

Then, we add our nodes and edges.

We use the START Node, a special node that sends user input to the graph, to indicate where to start our graph.

The END Node is a special node that represents a terminal node.

Finally, we compile our graph to perform a few basic checks on the graph structure.

We can visualize the graph as a Mermaid diagram.

**G) Building Simple Graph StateGraph and Graph Compiling**

**State Schema is created and now we need to create our Graph**

**For constructing our Graph, we will be using StateGraph; It is a Class provided by LangGraph itself; With help of it we can define entire structure of the Graph and which node will be connected to what other nodes and what will be the flow of executions; Everything will be able to decide with help of this StateGraph**

**Coding Part:**

from IPython.display import Image,display **To display the Graph itself**

from langgraph.graph import StateGraph,START,END **StateGraph is responsible in creating the entire graph or graph of the Workflow**

## Build Graph

graph=StateGraph(State) **Initializing the StateGraph to define the workflow**

## Adding the nodes

graph.add_node("start_play",start_play)

graph.add_node("cricket",cricket)

graph.add_node("badminton",badminton)

## Schedule the flow of the graph (As we planned in the Image)

graph.add_edge(START,"start_play")

### In LangGraph we have add_conditional_edges, to decid the condition 

graph.add_conditional_edges("start_play",random_play) **After we start, we need to call the function, which decide the game to play; This random_play will decide which node to call; If cricket automatically cricket node will be called; If Badmiton automatically Badmiton node will be called**

graph.add_edge("cricket",END) **To call cricket node**
 
graph.add_edge("badminton",END) **To call badmiton node**

## Compile the graph

graph_builder=graph.compile() **Compiling the Graph to visualize it**

## Viewing the Graph

display(Image(graph_builder.get_graph().draw_mermaid_png()))

**First we added all the nodes; Names can be different but make sure to provide the same name as the function name so there won't be any confusion; Then we added start play, then random_play which decides which node to call, cricket or badmiton, whichever it calls, that node gets executed; Then added edges for Cricket and Badmiton; Random play is calling cricket or badmiton**

### GRAPH Invocation - To invoke the Graph

graph_builder.invoke({"graph_info":"Hey My name is Krish"})

**Once we give this, it will go and start after adding My Name is "I am planning to play", then it will call the function random_play, then decide which node to call, whichever node it calls that node gets executed; That message will keep on getting appended**

**We will get output as sometimes I play cricket and sometimes I play Badmiton**

**Outputs were :Start_Play node has been called
My badminton node has been called; {'graph_info': 'Hey My name is Ashwath I am planning to play Badminton'}**

**We gave output for "Start_Play node has been called" because to debug what has happened**

**We didn't use any LLM here; Next we will use same workflow and create a chatbot; As we go ahead we will create more complex workflows and chatbot will be able to do multiple tasks**

**We will use LLM for nodes; LLM's can generate blogs, poems, jokes, anything it can do**

**H) Developing LLM Powered Simple Chatbot using LangGraph**

**Flow:** Start --> Superbot (Chatbot) [Can be OpenAI LLM or Groq Open Source LLM] --> End

1. **Reducers** - from typing import Annotated; from langgraph.graph.message import add_messages

add_messages - Merge Two list of messages, updating existing messages by ID, State is append only, unless new message has the same ID as existing message

Annotated - It will go and annotate with respect to every messages; It is also a List; Human Message, AI Message will be there; we need to annotate it with help of Annotator; We will annotate for the list of messages

### Add Message is used here to append all the list of messages, keeping it a track; As we have conversation with SuperBot; It is a reducer here

It will be a Key value pair

#### Very Important below to Import Models to Environment

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model="gpt-4o")

#### Invoking the Model: Start the convo and gives reply like "Hi, How can I help you"; llm.invoke("Hello")

**Creating the Grpah:**

def superbot(state:State):
    return {"messages":[llm_groq.invoke(state['messages'])]}

graph=StateGraph(State)

## node
graph.add_node("SuperBot",superbot)
## Edges

graph.add_edge(START,"SuperBot")
graph.add_edge("SuperBot",END)


graph_builder=graph.compile()


## Display
from IPython.display import Image, display
display(Image(graph_builder.get_graph().draw_mermaid_png()))

## Invocation - graph_builder.invoke({'messages':"Hi,My name is Krish And I like cricket"})

### If we don't use the Reducer, it won't get appended in Invocation; User Message and the AI Message; We will get only the recent message

# Streaming the Responses - for event in graph_builder.stream({"messages":"Hello My name is KRish"}): print(event)

#### If we want to have only the recent message we can use this; Streaming will help, when we develop end to end applications; By default, "stream_mode" is "update", so we will get only AI Message; We will learn about "stream_mode=values" later

#### If we use "stream_mode="values" or "updates"", there is a possibility that we can get both human and AI Reponse; But that Krish will cover later in the course





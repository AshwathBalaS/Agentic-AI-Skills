# Agentic-AI-Skills

**Table of Contents:**

**I)  Installation Of Anaconda And VS Code IDE**

**A) UV for creating Virtual Environment**

**II) - Python Prerequisites (No Need of Notes)**

**III) Getting Started With Pydantic In Python**

**A) Pydantic Practical Implementation**

**IV to VII - Already Covered in GenAI Krish Course; VIII - Theoritical (Please refer to Notes)**

**IX) Getting Started with LangGraph**

**B) Getting Started with LangGraph - Creating the Environment**

**C) Setting up OpenAI API Key**

**D) Setting up GROQ API Key**

**E) Getting up with LangSmith API Key**

**F) Creating a Simple Graph or Workflow using LangGraph - Building Nodes and Edges**

**G) Building Simple Graph StateGraph and Graph Compiling**

**H) Developing LLM Powered Simple Chatbot using LangGraph**

**X) LangGraph Components:**

**A) State Schema With DataClasses**

**B) Pydantic**

**C) Chain In LangGraph**

**D) Routers In LangGraph**

**E) Tools And ToolNode With Chain Integration- Part 1**

**F) Tools And Tool Node With Chain Integration-Part 2**

**G) Building Chatbot With Multiple Tools Integration- Part 1**

**H) Building Chatbot With Multiple Tools Integration-Part 2**

**I) Introduction To Agents And ReAct Agent Architecture In LangGraph**

**J) ReAct Agent Architecture Implementation**

**K) Agent With Memory In LangGraph**

**L) Streaming In LangGraph**

**M) Streaming using astream events Using Langgraph**

**XI) Debugging LangGraph Application With LangSmith**

**A) LangGraph Studio**

**XII) Different Workflows In LangGraph**

**A) Prompt Chaining**

**B) Prompt Chaining Implementation With Langgraph**

**C) Parallelization**

**D) Routing**

**E) Orchestrator-Worker**

**F) Orchestrator Worker Implementation**

**G) Evaluator-optimizer**

**XIII) Human in the Loop in LangGraph**

**A) Human In The Loop With LangGraph Workflows**

**B) Human In the Loop Continuation**

**C) Editing Human Feedback In Workflow**

**D) Runtime Human Feedback In Workflow**

**XIV) RAG with LangGraph**

**A) Agentic RAG Theoretical Understanding**

**B) Agentic RAG Implementation- Part 1**

**C) Agentic RAG Implementation-Part 2**

**D) Corrective RAG Theoretical Understanding**

**E) Corrective RAG Practical Implementation**

**F) Adaptive RAG Theoretical Understanding**

**G) Adaptive RAG Implementation**

**XV) End To End Agentic AI Projects With LangGraph**

**A) Introduction And Overview**

**B) Project Set Up With VS Code**

**C) Setting up The Github Repository**

**D) Setting Up The Project Structure**

**E) Designing The Front End Using streamlit**

**F)  Implementing The LLM Module In Graph Builder**

**G) Implementing The Graph Builder Module**

**H) Implementing The Node Implementation**

**I) Integrating the Entire Pipeline With Front End**

**J) Testing The End To End Agentic Application**

**XVI) End To End Agentic Chatbot With Web Search Functionality**

**A) Introduction To The Project**

**B) Implementing The Front End With Streamlit**

**C) Implementing GraphBuilder and Search Tools Pipeline**

**D) Implementing Node Functionality With End To End Agentic Pipeline**

**XVII) AI News Summarizer End To End Agentic AI Projects**

**A) Project Introduction**

**B) Building the Front End With Streamlit**

**C) Building The AI News State Graph Builder**

**D) Tavily Client Search Fetch News Node Implementation**

**E) AI News Summarize Node Functionality Implementation**

**F) Save Results Node Functionality Implementation**

**G) Running The Entire AINEWS Agentic Workflow**

**XVIII) End To End Blog Generation Agentic AI App**

**A) Introduction And Project Demo**

**B) Building Project Structure Using UV Package**

**C) Blog Generation Grpah Builder And State Implementation**

**D) Blog Generation Node Implementation Definition**

**E) Creating Blog Generating API Using FAST API**

**F) Integrating Langgraph Studion For Debugging**

**G) Blog Generation And Translation With Language**

**H) Building Blog Generation And Translation Graph Builder**

**I) Blog Generation And Translation Node Implementation**

**J) Testing In Postman And Langgraph Studio**

**XIX) Model Context Protocol**

**A) Demo of MCP with Claude Desktop**

**B) Cursor IDE Installation**

**C) Getting Started With Smithery AI**

**D) Building MCP Servers With Tools And Client From Scratch Using Langchain**

 #### Always create .env file in the venv - It is for LLM Models, OpenAI Keys, GroQ Keys, LangSmith keys [All those Environment variables and keys we will be storing there]

**A) UV for creating Virtual Environment**

uv - 10 - 100 times faster than pip

**pip install uv; uv init demouv; cd demouv (We have to go inside the demouv folder); uv venv (To create Virtual Envrionment)**

****.venv\Scripts\activate - To activate the Virtual Envrionment****

**To Install Libraries - uv add pandas (It was installed very very fast when compared to pip) [pyproject.toml file will get updated with dependencies for pandas, to confirm if pandas got installed]**

**In Krish Course, we use Conda for creating virtual environments because, it helps to manage multiple virtual environments to manage and switch among them**

# **III) Getting Started With Pydantic In Python**

**A) Pydantic Practical Implementation**

So now let me go ahead and quickly open my VS Code. In this specific VS Code, what we are basically going to do is that I have already created an environment. I’ll create my first module that is called as pedantic. All these materials will be given in the description or in the register in the link that is given in the description of this particular video. So in Krishna Academy we have kept this entirely as a free course. You can go ahead and download it from there.

Just to start with, I will go ahead and write "intro.ipynb". Okay, so I’ll start with ipynb. But before I go ahead, I will install "ipykernel", because we will be requiring this. Once I go ahead and install the ipykernel here, you will be able to see that my installation will probably happen. And then we should be able to select a kernel for this particular Jupyter notebook. Step by step, we will start with how this Python module can be used for data validation. We’ll create a simple model. When I say models, I’m basically talking about a class, how we can inherit from a base model, how the entire data validation will happen, and many more things.

So once the ipykernel installation is done, we will go ahead and select the kernel. I will go ahead and select the Python environment. Let me go ahead and create a markdown, and for this markdown, I’ll write some information. The first thing that we are going to discuss is “Pydantic basics: creating and using models.” By Pydantic, models are the foundation of data validation in Python. They use Python type annotations to define the structure and validate data at runtime.

Here is a detailed explanation of basic model creation with several examples. First of all, in order to use Pydantic in Python, we need to import "from pydantic import BaseModel". Right now you can see that Pydantic is not coming up, so what we’ll do is create one "requirements.txt", because I’ll be requiring some libraries. Inside "requirements.txt", I’ll write "pydantic". Then I’ll open the terminal and run "pip install -r requirements.txt". Once we do this, the entire Pydantic module will get installed.

Now I’ll write "from pydantic import BaseModel". As soon as we import this BaseModel, you’ll see that this is basically a base class for creating Pydantic models which performs data validation. What we are going to do first is create a simple model inheriting this BaseModel with some required fields, and see how validation happens.

Let’s create a class: "class Person(BaseModel): name: str; age: int; city: str". What I’m saying is that name should be of type string, age should be integer, and city should be string. Now let’s create an instance: "person = Person(name='Chris', age=35, city='Bangalore')". If I print person, you can see all the specific information. If I check "type(person)", it shows "__main__.Person".

Now, some might say, “Krish, with dataclasses we can do the same.” True, if I import "from dataclasses import dataclass" and use "@dataclass" on Person, without inheriting from BaseModel, I can still print the values. But what is the difference? Let me show you.

If I create "person1 = Person(name='Chris', age=35, city=12)", where instead of a string I gave an integer for city, Pydantic will raise a validation error: "1 validation error for Person → city: Input should be a valid string". But with a dataclass, if I give "city=235", it will not raise any error — it will just accept it. That is the magic of Pydantic: real data validation.

Now, let’s go deeper. Suppose we want optional fields. For that, we import "from typing import Optional". Then we write "class Employee(BaseModel): id: int; name: str; department: str; salary: Optional[float] = None; is_active: Optional[bool] = True". Here salary and is_active are optional fields with default values. If I create "emp1 = Employee(id=1, name='John', department='IT')", then salary will default to None and is_active to True. If I provide "salary=6000, is_active=False", it will take those values. If I give "salary=5" as integer, it auto typecasts to float. That is the beauty of Pydantic.

Similarly, we can have list fields. Suppose "class Classroom(BaseModel): room_number: str; students: list[str]; capacity: int". Now I can create "classroom = Classroom(room_number='101', students=['Alice','Bob','Charlie'], capacity=30)". This works. If I provide a tuple instead of list, Pydantic will still typecast it to list. But if I provide a dictionary instead of a string inside, it throws validation error.

Let’s test error handling. If I try "invalid = Classroom(room_number='A1', students=['Krish',123], capacity=30)", Pydantic throws: "1 validation error for Classroom → students.1: Input should be a valid string (input type=int)". This is happening because of Pydantic’s validation. We can catch it with try-except block: "try: ... except ValueError as e: print(e)".

Now let’s see nested models. Suppose "class Address(BaseModel): street: str; city: str; zip: int". Then "class Customer(BaseModel): name: str; address: Address". If I create "cust1 = Customer(name='Krish', address={'street':'123 Lane','city':'Bangalore','zip':'560001'})", Pydantic automatically typecasts the zip string into integer. This shows how nested models work with validation.

Next, let’s see field constraints. For that, "from pydantic import BaseModel, Field". Suppose "class Item(BaseModel): name: str = Field(..., min_length=2, max_length=50); price: float = Field(..., gt=0, lt=1000); quantity: int = Field(..., gt=0, le=1000)". Here, name must have at least 2 characters and max 50, price must be >0 and <1000, quantity >0 and <=1000. If I create "item = Item(name='Laptop', price=500, quantity=10)", it works fine. But if I set "price=-1", it raises a validation error: "Input should be greater than 0".

We can also set default values with Field. Example: "class User(BaseModel): username: str = Field(..., description='Unique username for the user'); age: int = Field(18, description='User age'); email: str = Field(default_factory=lambda:'user@gmail.com')" . If I create "u = User(username='Alice')", then age defaults to 18 and email defaults to "user@gmail.com".

Now, why is this useful? Because we can generate schemas. If I run "User.model_json_schema()", I get a full JSON schema with field types, descriptions, defaults, etc. This makes it super easy for developers to understand what kind of API request should be made.

So yes, this was it from my side. We saw how Pydantic models are created, optional fields, list fields, nested models, field constraints, and schema generation. This is why Pydantic is very powerful for data validation. I will see you all in the next video. Have a great day ahead. Thank you.

# **IX) Getting Started with LangGraph**

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

# **X) LangGraph Components:**

**A) State Schema With DataClasses**

So we are going to continue the discussion with respect to our Landgraf series. In this specific video, we are going to discuss about state schema with data classes. Already in our previous video, we have seen how to create a simple graph, how do we probably go ahead and create a chatbot with the help of a graph, and we also discussed about state schema, state graph, how to compile the graph, and how to invoke the graph during runtime. Now in this particular video, we will try to see what are the different ways of creating this kind of schema with the help of data classes.

If you remember till now, how did we go ahead and create our data class? We basically used TypeDict and then we used a reducer function if you remember it. So here if I go ahead and open in the chatbot, you will be able to see that when we were creating this particular state, we used TypeDict. TypeDict basically means this class will return the values in the form of dictionaries. And this is the variable that will keep on getting appended as each and every node will get executed. And the add_messages is just like a reducer, and the importance of reducer is that it is appending inside this particular variable instead of overriding.

Now we are going to talk about data classes. We will still see more examples with respect to reducer, but let’s first of all see how do we create some different types of state schemas considering TypeDict and along with this, data classes. Let me consider one more example. First of all, I will go ahead and from typing_extensions import TypedDict, so "from typing_extensions import TypedDict". Along with that, from typing I’m going to import Literal i.e., "from typing import Literal". Literal is just to define the constants.

Now let’s consider how to create a state, a normal state, and remember the state will be an input for the entire graph, and each and every node, when it is executed, the state variables will get updated. So here I will just go ahead and create my class called TypedDictState. This will basically be inheriting from TypedDict. So "class TypedDictState(TypedDict):" and then inside I will create two variables, "name: str" and "game: Literal['cricket','badminton']". This means the state will have a name of type string and game which can only be either cricket or badminton.

Now let me create my node function "def play_game(state: TypedDictState):". Inside this I will just print "print('play game node has been called')" and then I will return a key-value pair like {"name": state["name"], "message": f"{state['name']} wants to play {state['game']}"}. Similarly, let me create two more node functions. One is "def cricket(state: TypedDictState):" where I print "print('cricket node has been called')" and return {"game": "cricket"}. The other one is "def badminton(state: TypedDictState):" where I print "print('badminton node has been called')" and return {"game": "badminton"}. So now I have three node functions ready.

Next, we will create our builder graph. For this, I will import "import random", "from IPython.display import Image, display", and from "langgraph.graph import StateGraph, START, END". Then I will build the graph by writing "builder = StateGraph(TypedDictState)". After that, I add nodes one by one: "builder.add_node('play_game', play_game)", "builder.add_node('cricket', cricket)", and "builder.add_node('badminton', badminton)".

Now let’s define the flow of the graph. I will add an edge from START to play_game: "builder.add_edge(START, 'play_game')". But after play_game, we need to decide whether to go to cricket or badminton, so we use a conditional edge. I will define a function "def decide_play(state: TypedDictState) -> Literal['cricket','badminton']:" where we return either "cricket" or "badminton" based on a random condition like "if random.random() < 0.5: return 'cricket' else: return 'badminton'". Then I add edges like "builder.add_conditional_edges('play_game', decide_play, {'cricket':'cricket','badminton':'badminton'})". Finally, I connect both cricket and badminton nodes to END: "builder.add_edge('cricket', END)" and "builder.add_edge('badminton', END)". Then I build and display the graph using "graph = builder.compile()".

Now if I invoke the graph with "graph.invoke({'name':'Krish'})", you will see play_game node has been called, then depending on the random function, either cricket or badminton node is called, and the state updates accordingly. For example, it may print "Krish wants to play badminton" with game updated to badminton. Similarly, if cricket node is called, it will print "Krish wants to play cricket". This shows how the state schema works with TypedDict.

But one limitation here is that TypedDict is only providing type hints, they are not enforced at runtime. For example, if I call "graph.invoke({'name':123})", even though I defined name as str, it will not throw an error because TypedDict does not enforce types at runtime. The code may fail later only if some string operations are attempted on the integer. This is a disadvantage, and that is why later we will discuss Pydantic, which allows runtime type enforcement.

Now coming to data classes. Python data classes provide another way to define structured data. Data classes offer a concise syntax for creating classes that are primarily used to store data. Instead of using TypedDict, I can use data classes. So "from dataclasses import dataclass". Then I define "@dataclass class DataClassState:" and inside I define "name: str" and "game: Literal['cricket','badminton']".

Now the node functions change slightly. For example, "def play_game(state: DataClassState):" and inside I will access values like "state.name" instead of "state['name']". Similarly in cricket and badminton functions, I return values referencing state.name and state.game. Also the decide_play function becomes "def decide_play(state: DataClassState) -> Literal['cricket','badminton']:" with the same random logic.

To build the graph, we write "builder = StateGraph(DataClassState)", add nodes with "builder.add_node('play_game', play_game)", "builder.add_node('cricket', cricket)", "builder.add_node('badminton', badminton)", add edges like "builder.add_edge(START,'play_game')", conditional edges with decide_play, and connect cricket and badminton nodes to END. Then compile the graph with "graph = builder.compile()".

Now to invoke this graph, we don’t pass a dictionary but an instance of the data class. For example, "graph.invoke(DataClassState(name='Krish', game='badminton'))". This will again call play_game and then depending on the random logic, either cricket or badminton node will be executed. If you don’t provide game, it will complain, so you can provide a default value in the data class if required.

But again, even with data classes, type hints are not enforced at runtime. If you pass "DataClassState(name=123, game='cricket')" it will still accept it, even though name should be a string. It will only fail if operations incompatible with integer are performed. This again shows why Pydantic is important, because Pydantic models enforce runtime validation and will throw errors if incorrect types are provided.

So in this video we saw two ways of creating state schemas: one with TypedDict and one with data classes. We also discussed their differences, and understood that both do not enforce types at runtime. In the next video, we will explore Pydantic, which helps us enforce runtime type checks and validations, ensuring that if we declare a variable as string, it must actually be a string when executing the graph. I hope you got a clear idea about state schemas and liked this video. See you in the next one.

**B) Pydantic**

So we are going to continue our discussion with respect to our lecture series. Already in our previous video, we saw how we can create state schemas using TypedDict and data classes. We also spoke about the runtime error which we were not getting, since in TypedDict we had used "name: str". But if we gave any other kind of datatype while invoking, it should basically give an error, right? And in this session we will talk about Pydantic, because this will be handled by Pydantic itself.

First of all, let’s do one thing. Let’s quickly go ahead and see what exactly Pydantic is. If I just go ahead and search for “Pydantic Python,” this will take me to the official documentation. It is the most widely used data validation library for Python. For example, let’s say inside my state schema I give a value which is of type string. If I say the name is equal to string, then I want to make sure that whenever a user invokes the entire graph, we have to pass only a string. If we pass an integer, it should give us an error too. For that particular error to come, we have to make sure that we apply Pydantic data validation in the class.

This is the entire documentation, and you can probably see this is a PyPI library. There are some examples here. For instance, inside a class delivery, if we inherit from BaseModel (which comes from Pydantic), then during runtime we must always make sure that fields match their types. If a field like "timestamp" is declared as a datetime, we have to only give datetime objects, otherwise it will not take. Similarly, if a field "dimension" is declared as a tuple, we should only give a tuple. If we don’t give values in the correct format, Pydantic will throw a runtime error. That is exactly what we are trying to implement here.

Now let me just show you one example. I will go ahead and create one file called "Pydantic.ipynb". Inside this file, we will start creating our code. First, I will add a markdown cell saying “Pydantic data validation.” This is important because whenever we later create graphs with respect to any kind of application, we must make sure that data validation is applied.

Let me now create a code cell. First, I will import from LangGraph: "from langgraph.graph import StateGraph, START, END". This allows us to define state graphs. Now, if I really want to apply Pydantic validation, I will import "from pydantic import BaseModel". This BaseModel is the key. Whenever we create our state schema, we should inherit from this class so that validation will automatically get applied whenever we are creating variables within a class.

For example, I create "class State(BaseModel): name: str". Here, the class State inherits from BaseModel, and we declare a field "name" of type string. Because of this inheritance, type validation will happen at runtime. Internally, BaseModel also leverages type information just like TypedDict, but with runtime enforcement.

Now, let’s create a simple node function. "def example_node(state: State): return {'a':'hello ' + state.name}". Here I am returning a simple dictionary where the value is "hello" concatenated with the state’s name.

Next, let’s build our state graph. "builder = StateGraph(State)". Remember, the state here is our class that inherits from BaseModel. Then we add the node: "builder.add_node('example_node', example_node)". Now we define the edges: "builder.add_edge(START, 'example_node')", and then "builder.add_edge('example_node', END)". Finally, we compile the graph: "graph = builder.compile()".

Now, let’s invoke this graph. "graph.invoke({'name':'Krish'})". Since we are giving a valid string for the name, this executes easily and gives the expected output. But now, let’s say I write "graph.invoke({'name':123})". Immediately, you will see an error: "1 validation error for State → name: Input should be a valid string".

How is this validation error coming? It is very simple — because we inherited BaseModel, which comes from Pydantic. Pydantic checks at runtime whether we are passing the right value for the "name" field. Since we passed a numerical value instead of a string, it throws a validation error.

This is the power of Pydantic. It ensures that validation happens at runtime, unlike TypedDict, which only provides type hints but does not enforce them. As we go ahead in developing more advanced applications with LangGraph, this validation will become extremely important, because we will be using it in multiple scenarios. We must always make sure that this validation happens at runtime so that errors are caught early.

So this was about Pydantic data validation in LangGraph. We saw an example with a simple state and node, observed how the validation works, and how errors are raised at runtime if we pass invalid values. Similarly, you can write any number of examples with any variables, and Pydantic will handle them in the same way.

Yeah, this was it from my side. I will see you all in the next video where we will discuss further topics. Thank you.

**C) Chain In LangGraph**

Hello everyone, so we are going to continue our discussion with respect to our LangGraph series. In today’s session, we’ll be talking about how to create chains with the help of Graphs.

Now first, let’s understand the idea behind chains. Imagine this: we start with a start node, then we have Node 1 connected to Node 2, and then to Node N. These nodes are linked sequentially. But not just sequentially, in more complex workflows, sometimes a flow may go forward from Node1 to Node2, or it may even loop back from Node2 to Node1. So, chains help us design this entire step-by-step workflow of nodes inside a graph.

Till now we have seen simple graphs with nodes and edges, and we have also seen conditional edges, which are important for branching logic. Now, we’re going one level deeper by introducing chains.

To properly understand chains, we need to combine four important concepts. The first one is chat messages. Whenever we build chatbots or conversational workflows, humans send inputs and LLMs send outputs. We represent these as HumanMessage and AIMessage. In LangGraph, these messages form the conversation history inside our Graph State. The second one is chat models. At each node, we can integrate an LLM model like OpenAI GPT, Groq, or Anthropic. Some nodes may just pass data, but some nodes can directly talk to an LLM. The third one is binding tools. Sometimes, we want our LLM not just to generate text but also to call a function or use a tool. So inside our graph nodes, we can bind tools to the model. And finally, the fourth one is memory. Memory is what allows the graph to retain the context of past inputs and outputs. This is extremely important because without memory, each step would act stateless. With memory, the conversation flows naturally across multiple nodes.

Now, to demonstrate all this, let’s jump into the code. First, we import the required packages. We bring in ChatOpenAI from langchain_openai, HumanMessage and AIMessage from langchain_core.messages, and StateGraph from langgraph. We’ll also import tools if required. Then, we define our nodes. Let’s say we create a node called model which will represent an LLM. This model node can be instantiated with something like model = ChatOpenAI(model="gpt-3.5-turbo"). Next, we can bind this model with a tool, for example, model_with_tools = model.bind_tools([some_tool]). This means whenever this node is called, the model can decide whether to generate a response or call the tool.

Now, in our graph definition, we create a StateGraph with messages as the state type. This means our graph will maintain the history of messages. Then, we add nodes to it. For instance, graph.add_node("llm", model_with_tools). After that, we connect the start node to the llm node with graph.add_edge(START, "llm"). We can also add more nodes if needed and connect them similarly. Finally, we set the endpoint by connecting our last node to END.

When we compile this graph, it becomes a chain. This chain can now take in a list of messages, run them through the graph, and produce outputs. For example, we can call chain.invoke([HumanMessage(content="Hello, who won the cricket match yesterday?")]). The model node receives this, decides whether to generate a reply directly or use a bound tool like a search API, and then produces the output.

So in summary, what we’ve done is that we’ve created a workflow where the flow of data happens from node to node in a structured way, while maintaining memory, handling messages, invoking models, and even binding tools. This is exactly what we mean by creating chains using graphs in LangGraph.

**D) Routers In LangGraph**

Hello guys, so we are going to continue the discussion with respect to our graph series. In our previous video, we had already started understanding about tools. But before we go ahead and implement more functionalities with tools, there is an important concept we need to cover, which is called a router. This is a very important topic altogether, because with the help of a router you will be able to see how nodes that are integrated with an LLM and tools can specifically be converted into an agent.

Now, let’s quickly recap what we did earlier. If you remember, I had put up a diagram where we had a start, then we connected to an LM node, and with this LM node, we had done the binding of one function that we created ourselves, called add. So we created this add function, bound it to the LM, and that was our setup. When I talk about this LM node, it is just a node inside the graph that internally has a function, which is being invoked with the help of an LLM. Then we connected that to an end node. We also saw that this LM tool, whenever a user input was given in natural language, was able to understand whether it should call the tool or not.

For example, if I said, “What is two plus two?”, the LM was able to parse the natural language input and recognize that a tool call should happen. It gave a structured response saying this is a tool call, these are the parameters, and this is the function being called. However, the LM itself wasn’t executing the tool. It was just indicating the call. The actual execution—taking the numbers, adding them, and returning the result—still had to happen separately. So we needed a way to make the LM not just decide but also route the flow properly so that the function is executed and the response comes back.

This is where the router comes in. With a router, the chat model acts like a traffic controller. It routes between either giving a direct response or making a tool call, based upon the user input. This is actually the simplest example of an agent: the LM is controlling the flow, deciding whether to use a tool or just respond. So in our diagram, we start, we move to the LM tool node, and from here a conditional edge appears. If no tool is required, it goes directly to the end. If a tool call is required, it takes another path where the tool is executed. The tool receives the input parameters, performs the calculation (like adding A and B), and then passes the result back before ending.

So now you can think of this LM tool as an agent. Why? Because it is making decisions and taking actions automatically. Without human intervention, the LM is deciding whether to just reply or to call a tool, executing that tool, and then returning the result. This is the basic definition of an agent. It may look simple now, but it is the foundation.

Let’s visualize it step by step. You have a start node. From there, we go to Node 1, which is an LM node. This LM has bindings to one or more tools. In our simple example, it has the add function. In future examples, we can bind multiple tools—maybe four or five at once. Then, depending on the user’s input, the LM decides what to do. For example, if I ask, “Give me the current news,” the LM will look at its available tools. If one of the bound tools is a news API, it will call that tool, fetch the news, and return it. If the input doesn’t require a tool, it simply generates a direct response. From there, it either goes to Step 2, Step 3, or directly to End, depending on how we design the graph.

You can see that as we increase the complexity of the graph, the LM’s responsibility grows too. It becomes the brain that decides where the flow should go. That’s why we say this LM integrated with tools is an agent: it makes decisions based on the input and the tools it has. It’s very similar to a human agent. Imagine I have five books. If you ask me a question and I don’t know the answer, I check the relevant book, find the information, and give it to you. That’s how a human agent works. Here, the LM does the same thing but with much greater knowledge and access to different tools.

So in short, the LM acts like the brain, making automated decisions about what path to follow and what tool to call. This is what we achieve with the router functionality in LangGraph. As we go ahead, I will show you the full implementation with code, and also how we can bind multiple tools to an LM, run them through the graph, and see how the execution flow happens in real time.

I hope you liked this explanation of the router. That’s it from my side for this video. I will see you in the next one. Thank you.

**E) Tools And ToolNode With Chain Integration- Part 1**

So we are going to continue a discussion with respect to our LangGraph series. Already in our previous video, we have seen about chains, right? We also saw how we can work with our chat messages in our graph state. Then we saw about chat models, and now we are going to discuss tools.

First of all, let's try to understand tools with a simple diagram and practical examples. Tools can be integrated with LM models to interact with external systems such as APIs or third-party services. Whenever a query is asked to the model, it can choose to call the tool. This query is based on natural language input and will return an output that matches the tool’s schema.

Now, why specifically are tools required? Let's say we have an LM in our workflow. For every node, we might want to use an LM model. Suppose a human input asks, "What is the current temperature of New York City?". The LM, by itself, cannot answer this because it is not trained with recent data and is not connected to the internet.

In such scenarios, the LM can make a tool call. This tool can be an external API, a third-party API, a weather API, or even a database, such as a vector database. Essentially, it’s a third-party source or external data. The LM takes help from these tools if it cannot answer the human query on its own. The tool returns data in a predefined schema, such as JSON, which can then be displayed to the user. Later, we can apply custom output parsers to further process the data.

The key thing to understand is that the LM acts as the brain of the node. It decides whether a tool call is needed. Initially, the LM is bound to the tool using something like: "lm.bind_tool([add])" for a simple addition function called add. This binding allows the LM to know what tools it has access to. Modern LMs can automatically determine, based on input, whether a tool call is necessary.

For a practical example, let’s create a simple addition tool. Suppose we define a Python function:

def add(a: int, b: int) -> int:
    """Add two integers a and b and return the result."""
    return a + b


This function will be bound to the LM, giving the LM the ability to call it when needed. For example, the input "What is two plus two?" will trigger the LM to use this add tool. The LM acts as the brain deciding when to invoke the tool, while the tool itself executes the calculation.

Next, we define state management using messages and reducers. Without a reducer, appending messages would override previous values. We use a prebuilt reducer called add_messages to append messages instead of overwriting them. For instance:

from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph.messages import AnyMessage

class State:
    messages: Annotated[list[AnyMessage], add_messages]


This ensures that as the graph executes, messages are appended rather than overwritten. This is critical for conversational chatbots, where maintaining message history is important.

We can now demonstrate the functionality of add_messages:

initial_messages = [HumanMessage(content="Hello!")]
next_message = AIMessage(content="Hi! How can I help?")
messages = add_messages(initial_messages, next_message)


This appends next_message to initial_messages, preserving the history.

Once we have our tool and state set up, we create the state graph. We define a node function that calls the LM with the bound tool:

def lm_tool_node(state: State) -> list[AnyMessage]:
    return lm_with_tools.invoke(messages=state.messages)


We then define the graph and nodes:

builder = StateGraph(State)
builder.add_node("LM_Tool_Node", lm_tool_node)
builder.add_edge("start", "LM_Tool_Node")
builder.add_edge("LM_Tool_Node", "end")
builder.compile()


Here, the LM node is bound to the tool, and edges define the flow from start → LM node → end. We can visualize the graph:

display(graph.graph.get_graph().draw("png"))


When invoking the graph with a human input:

messages = graph.invoke(messages={"content": "What is two plus two?"})
for message in messages:
    message.pretty_print()


The LM decides that a tool call is required. The function add is called with a=2, b=2. However, initially, there is no separate tool node to return the output. The LM correctly identifies the tool to call, but the tool execution and response need to be implemented in another node to provide the final result.

This shows how the LM acts as the brain, while tool nodes handle execution. In the next step, we will integrate a separate tool node to complete the flow and get responses from tool calls.

**F) Tools And Tool Node With Chain Integration-Part 2**

In modern conversational systems using Landgraf, large language models (LLMs) often cannot directly access real-time data or external systems, such as APIs, databases, or custom functions. For instance, if a user asks, "What is the current temperature in New York?", the LLM alone cannot provide the answer because it lacks live data access. To handle such scenarios, we integrate tools with the LLM, allowing it to make tool calls when necessary. These tools can be external APIs, databases, or even custom functions like simple mathematical operations.

For example, we can define a simple addition function as a tool: "def add(a: int, b: int) -> int: return a + b". The docstring inside this function, like "Adds two integers a and b and returns the result", is important because it helps the LLM understand the purpose of the tool. Once defined, this function can be bound to the LLM using "lm.bind_tools([add])". Now, when a user asks, "What is 2 + 2?", the LLM can automatically detect that this input corresponds to the add tool and call it with the appropriate arguments.

To manage the conversation and keep track of messages between nodes, we use a state class in combination with reducers. The state stores all messages exchanged, and reducers control how new messages are appended. For instance, "from langgraph.graph.message import add_messages" and "from typing import Annotated, List; from langgraph.core.messages import AnyMessage" allow us to define a state class like "class State: messages: Annotated[List[AnyMessage], add_messages]". Here, add_messages ensures that messages are appended rather than overridden, preserving the full conversation history as the graph executes.

Next, we construct the state graph with nodes and edges. Typical nodes include a "start" node, a "chatbot" node representing the LLM bound to tools, a "tool" node that executes tools, and an "end" node. We can add edges like "builder.add_edge('start', 'chatbot')" to define the workflow. Conditional edges are used to handle tool calls dynamically. By importing "from langgraf.prebuilt import tools_condition", we can route messages that require a tool to the tool node and messages that don’t to the end node, ensuring flexible execution.

To add a tool node, we first create a list of tools, e.g., "tools = [add]", and then define the node using "builder.add_node('tool_node', ToolNode(tools=tools))". We also add conditional edges from the chatbot node to the tool node using "builder.add_edge('chatbot', 'tool_node', condition=tools_condition)" and from the tool node to the end node using "builder.add_edge('tool_node', 'end')". This setup allows the LLM to detect when a tool call is necessary, execute it via the tool node, and then return the result to the user.

Finally, when invoking the graph, the behavior differs depending on the input. For instance, if the user asks, "What is 2 + 2?", the chatbot node detects the tool call, routes the query to the tool node, executes "add(2,2)", and returns the result. If the input is a general question like "What is machine learning?", the LLM answers directly without calling the tool node, skipping straight to the end. This approach ensures that the system can dynamically determine when tool execution is needed while maintaining the full message history in the state.

**G) Building Chatbot With Multiple Tools Integration- Part 1**

In this video, we are going to extend our Landgraf series by building a chatbot with multiple tools integration using a line graph. Previously, we had implemented a simple chatbot that could call a single custom tool, like an "add" function. Now, we aim to create a more sophisticated system where the chatbot can intelligently decide which tool to call based on the user input. The diagram for this setup will start with a "start" node, followed by a "tool calling LM" node, which can route calls to multiple tools, and finally end at an "end" node.

For the tools, we will integrate not only our previous add function, "def add(a: int, b: int) -> int: return a + b", but also more advanced tools like RCF, Wikipedia, and an internet search tool using Tabulae. The chatbot will dynamically determine which tool to invoke. For example, if the input requires searching academic papers, the LLM can call the RCF tool with "RCF.invoke('attention is all you need')", which will fetch relevant research papers from arXiv. Similarly, for general knowledge queries, the chatbot can use the Wikipedia tool with "wiki.invoke('What is machine learning?')", retrieving concise information from Wikipedia.

To integrate these tools in Python, we start by importing the necessary modules from LangChain and LangGraph:

from langchain_community.tools import RCFQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


Next, we create API wrappers for each tool. For RCF, we can specify the number of results and maximum characters:

rcf_api = RCFQueryRun(api_wrapper=RCFAPIWrapper(top_results=2, max_chars=500))
wiki_api = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_results=1, max_chars=500))


For internet search using Tabulae, we first need to set up the API key in our environment:

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['TABULAE_API_KEY'] = "your_tabulae_api_key"
tab_api_key = os.getenv('TABULAE_API_KEY')


Then, we can initialize the Tabulae tool:

from langchain_community.tools.tabulae import TabulaeSearchRun
tab = TabulaeSearchRun(api_key=tab_api_key)


Once all tools are set up, we combine them into a list and bind them to our LLM. For instance, using the Grok model from LangChain:

from langchain_grok import ChatGrok
llm = ChatGrok(model="Gwen-32B")
lm_with_tools = llm.bind_tools([rcf_api, wiki_api, tab])


Now, the LLM is capable of deciding which tool to call. For example, if the user asks for recent AI news:

response = lm_with_tools.invoke(human_message="What is the recent AI news?")


The LLM will intelligently route the call to the Tabulae search tool, and if the query relates to academic papers, it will call RCF, or use Wikipedia for general knowledge. You can check which tool was invoked using:

tool_call = response.tool_call
print(tool_call)


Finally, after setting up all the tools and bindings, we can integrate this into a state graph using Landgraf, where the flow starts at the "start" node, moves to the LLM with tools, conditionally routes to the appropriate tool node, and finally ends at the "end" node. This setup allows the chatbot to handle multiple tools seamlessly while maintaining a clear conversation state.

In summary, in this video, we prepared multiple tools including RCF, Wikipedia, and Tabulae search, bound them to our LLM, and made the system ready to dynamically decide which tool to use based on the user query. In the next step, we will create the state graph to visualize and execute this multi-tool chatbot workflow. This approach is highly extensible, allowing you to add any custom tool like "multiply" or additional retrievers for vector databases in the future.

**H) Building Chatbot With Multiple Tools Integration-Part 2**

So we are going to continue the discussion with respect to chatbots with multiple tools. Already in our previous video, we saw that we created or used the three default tools provided by Landgraf. We were also able to test in the LLM that whenever we give a message, for example, "What is the recent news?", it was able to call the internet search tool. Similarly, for queries related to research papers, it could call arXiv, and for general knowledge, it could call Wikipedia.

Now it’s time to construct our entire graph, so we will create the chatbot using a line graph. This means we have to define a state graph and all the nodes within it. First, let’s import the necessary libraries:

from langgraph import StateGraph, ToolNode, ToolCondition


As we know from our previous example, we will use ToolNode and ToolCondition to define nodes and route messages.

Next, we define the state schema which will store our chat messages. We use a class State that inherits from dict, along with annotated messages and reducers for adding messages:

from typing import List, Any
from langgraph.reducers import add_messages, any_message

class State(dict):
    messages: List[Any] = []


Here, messages will store all human and AI messages as the nodes are executed. The add_messages reducer ensures that all incoming messages are stored in this state.

Now, we define the node for tool-calling LLM. This node takes the human message as input and invokes the LLM with access to all tools:

def tool_calling_LM(state: State):
    return {"messages": LM_with_tools.invoke(state["messages"])}


Here, LM_with_tools is the LLM bound with all the tools we configured (arXiv, Wikipedia, Tabulae, etc.).

Next, we create the state graph builder and add our nodes:

builder = StateGraph(state=State)

# Node for LLM invocation
builder.add_node("tool_calling_LM", tool_calling_LM)

# Node for tools
builder.add_node("tools", ToolNode(tools=[RCF, Wiki, Tab]))

# Edges: start -> tool_calling_LM -> conditional routing -> tools -> end
builder.add_edge("start", "tool_calling_LM")
builder.add_edge("tool_calling_LM", "tools", condition=ToolCondition())
builder.add_edge("tools", "end")

# Compile the graph
graph = builder.compile()


This setup ensures that the graph starts from "start", passes messages to the LLM node, and depending on whether a tool call is required, routes messages to the tools node or ends the flow.

Now we can invoke the graph with a human message. For example, querying a research paper:

human_message = {"content": "What is 'Attention is All You Need'?"}
result = graph.invoke(messages=human_message)


The first time we run this, the LLM asks if we want to fetch details for this paper. On the second execution, the AI message triggers a tool call to arXiv and returns the relevant content, limited to 500 characters as configured:

print(result["messages"])
# Output shows AI message + tool call content from arXiv


Similarly, we can test internet search for recent AI news:

human_message = {"content": "Provide me the top recent AI news for March 3rd, 2025"}
result = graph.invoke(messages=human_message)


Here, the LLM decides to call the Tabulae search tool and returns the latest curated AI news with links and publication dates.

We can also test Wikipedia integration for general knowledge queries:

human_message = {"content": "What is machine learning?"}
result = graph.invoke(messages=human_message)


The LLM automatically decides to call Wikipedia and returns a summarized response, again limited to the configured character count.

This implementation demonstrates a chatbot with multiple tools fully integrated with a state graph. The graph dynamically decides which tool to call based on the input query. In the next steps, you could modularize this code further for easier maintenance and add more tools like custom retrievers or calculators.

I hope you found this example helpful. This concludes our chatbot with multiple tools using Landgraf. The LLM with tools, state schema, nodes, and conditional routing are all set up, and now it can intelligently decide which tool to invoke for any user query.

**I) Introduction To Agents And ReAct Agent Architecture In LangGraph**

In this specific video, we are going to have a detailed discussion about agents. If you remember, in our previous video, we discussed important topics like chains, routers, and how to use after-chains and routers with respect to tools. We also saw how to create a chatbot with multiple tools and briefly introduced the concept of a basic agent, which can make decisions on whether to call a tool based on natural language input.

Let’s recap a simple chatbot with tools. Imagine a graph like this: a start node, an LLM node (which acts as the brain), a tool node, and an end node. In Python, this can be represented as:

builder.add_edge("start", "llm_node")
builder.add_edge("llm_node", "tool_node", condition=ToolCondition())
builder.add_edge("tool_node", "end")


Here, the LLM node acts as a basic agent: it decides automatically whether to call a tool or end the conversation, based on the user’s input.

Now, let’s move one step ahead and discuss a more advanced agent architecture called ReAct. This architecture has three important components: Act, Observe, and Reason.

Act – The model receives a natural language input and decides whether to call a specific tool. For example, the agent can call an addition tool if the user asks "What is five plus five?". In Python:

if "add" in user_input:
    tool_output = add_tool.invoke(input_numbers)


Here, the agent “acts” by calling the tool.

Observe – Once the tool executes, instead of ending the conversation, the output is passed back to the LLM for further reasoning.

llm_state.update({"tool_output": tool_output})


This ensures the model can use the result of the tool in the next reasoning step.

Reason – The LLM reasons based on the tool output and decides the next action. For example, if the user says "Add five plus five, then multiply by three", the agent performs multiple steps:

# Step 1: Add
add_result = add_tool.invoke([5, 5])  # Output: 10

# Step 2: Multiply
multiply_result = multiply_tool.invoke([add_result, 3])  # Output: 30

# Step 3: LLM reasons and provides final output
final_response = f"The result is {multiply_result}"


The agent automatically determines which tools to call and when to end the conversation.

This ReAct agent architecture allows the agent to continuously interact with tools, reason, and decide the next steps based on both the input and tool outputs. In Python terms, you can represent the cycle as:

while not conversation_finished:
    llm_decision = llm_node.invoke(input=user_input)
    if llm_decision.requires_tool:
        tool_output = tool_node.invoke(llm_decision.tool_input)
        llm_node.observe(tool_output)
    else:
        conversation_finished = True


This ensures the agent keeps reasoning until it decides the task is complete.

In summary, the three components of the ReAct agent are:

Act: Decide and call a tool if needed.

Observe: Pass the tool output back to the LLM.

Reason: Decide the next action based on tool output and input.

This approach allows agents to perform complex tasks step by step, handle multi-step instructions, and interact with multiple tools.

In the next video, we will implement agents in Landgraf, demonstrating the importance of landing state tracking and monitoring for these agents. This is crucial for debugging and observing agent behavior in real-time.

This was it from my side for this video. I hope you liked it. I will see you all in the next video.

**J) ReAct Agent Architecture Implementation**

In this video, we are going to continue with our Land Graph series and implement the ReAct agent architecture. As discussed in the previous video, the ReAct agent has three main components: act, observe, and reason. The goal is that whenever the LLM decides to make a call to a tool, the tool should return the output back to the LLM, and the LLM can then decide the next action.

We will use the default tools provided by Landgraf, such as arXiv_query_run, Wikipedia_query_run, and RC_api_wrapper. In Python, we can import and initialize these tools like this:

from linkedin_community.tools import arXiv_query_run, Wikipedia_query_run, RC_api_wrapper

rc = RC_api_wrapper(top_k=2, max_document_characters=500)
wiki = Wikipedia_query_run(top_k=1, max_document_characters=500)


For example, invoking the RC tool with RC.invoke("attention is all you need") will fetch details about that research paper from arXiv. Similarly, invoking the Wikipedia tool with wiki.invoke("What is machine learning?") will return a concise summary from Wikipedia.

Next, we will set up LangSmith for tracing and debugging. We can define the environment variables and project name as:

import os

os.environ["LANG_API_KEY"] = os.getenv("LANG_API_KEY")
os.environ["LANG_TRACING_V2"] = "true"
os.environ["LANG_CHAIN_PROJECT"] = "react_agent"


This setup ensures that all tool invocations and agent actions are traced in LangSmith, allowing us to monitor and debug the agent behavior.

Along with the default tools, we will also define some custom functions such as multiply, add, and divide. Each function has a docstring that helps the LLM understand what it does. For example:

def multiply(a: int, b: int) -> int:
    """Multiply two integers a and b and return the result."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers a and b and return the result."""
    return a + b

def divide(a: int, b: int) -> int:
    """Divide a by b and return the result."""
    return a // b


We can then combine all these tools and functions into a tool variable that will be bound to the LLM:

tools = [rc, wiki, add, multiply, divide]


Once the tools are bound, we can initialize our LLM with tools. This can be any model, for example an open-source model from Grok or LLaMA, and bind the tools:

lm_with_tools = LM_with_tools(model_name="Grok-2.5B", tools=tools)


We can test the tool invocation with a human message:

response = lm_with_tools.invoke(human_message="What is the recent news?")


The LLM automatically determines whether a tool call is needed. At this stage, we have the tools ready and can move on to creating our state schema and graph. We define the state schema similar to previous examples, using Annotated, Any, and Dict types, and set up reducers like add_messages.

We then define the nodes for the graph. One node is tool_calling_LM, which calls the LLM with the tools, and another node is tools, which represents the tool node itself:

tool_calling_LM = ToolCallingLMNode(state=state)
tools_node = ToolNode(tools=tools)


We connect the nodes using edges. From start to tool_calling_LM and then a conditional edge from tool_calling_LM to tools_node. The condition checks if the latest message requires a tool call; if not, it routes to the end. Unlike previous implementations, in the ReAct architecture, we route back from tools to tool_calling_LM, forming a loop:

builder.add_edge("start", "tool_calling_LM")
builder.add_edge("tool_calling_LM", "tools", condition=ToolCondition())
builder.add_edge("tools", "tool_calling_LM")  # Loop back to LLM
builder.compile()


We can visualize the graph and see that the loop allows the LM to continuously act, observe, and reason until the task is complete.

For example, if the human message is "Provide me the top ten recent AI news for March 3rd, 2025. Add five plus five, then multiply by ten.", the execution proceeds in steps. First, the LLM determines that the search tool should be called to fetch news. Then the Add tool is invoked to compute 5 + 5, followed by the Multiply tool to compute 10 * 10. At each step, the tool outputs return to the LLM, which reasons about the next action.

response = graph.invoke_messages(human_content="Provide me the top ten recent AI news for March 3rd, 2025. Add five plus five, then multiply by ten.")
print(response)


The output shows multiple tool calls, their arguments, and the results, forming a continuous loop between the tools and the LLM. If a simple question like "What is machine learning?" is asked, the LLM automatically calls Wikipedia, gets the response, and ends the conversation once the sentence is complete.

This demonstrates how the ReAct agent architecture continuously uses the act, observe, reason components to decide tool calls and process multi-step instructions. The LLM acts autonomously, observes tool outputs, and reasons about the next step in a loop until the task is completed.

**K) Agent With Memory In LangGraph**

In this video, we are going to continue our Landgraf series and discuss Agent Memory. We have already seen the ReAct Agent Architecture in the previous video, and now we will understand why agent memory is important.

Let’s take a simple example. Suppose we have already created the graph in the last class and executed it successfully. If we give an input like "What is five plus eight?" to the graph, it executes and gives the output 13.

message = {"content": "What is five plus eight?"}
graph.invoke_messages(human_message=message)


Now, if we want to continue the conversation and ask "Divide that by five", ideally the graph should remember that the previous output was 13 and perform 13 / 5. However, if we execute it without memory, the agent responds:

It looks like you want to perform a division operation, but you have not provided the number to be divided by five.


This happens because the graph does not have the context from the previous state. This is where agent memory comes into play.

To integrate memory, Landgraf provides a Memory Saver, which acts as a checkpoint to save the graph state after each step. This allows the agent to pick up from the last state update. Memory Saver works like a key-value store, saving the graph state and relevant metadata at each step.

from line_graph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph_memory_builder.compile(checkpoint=memory)


With this setup, every node execution updates the checkpoint, storing the current state and metadata. Sequential nodes are stored as separate super steps, while parallel nodes share the same super step, meaning they share the same state information.

For example, consider a sequential flow from Node 1 → Node 2. Node 1 updates the state to "I love Landgraf", and Node 2 receives this updated state. If multiple nodes run in parallel, they all share the same updated state.

state = {"value": "I"}
# Node execution updates the state
state = {"value": "I love Landgraf"}


The Memory Saver ensures that all node executions update the memory in real time. The checkpoint stores state, next node, and metadata for each node execution. This allows us to recall previous outputs and maintain context across multiple messages.

To implement this, we need to specify a thread ID, which uniquely identifies a user session or conversation:

config = {"thread_id": 1}  # Unique for a user


We can then send inputs to the graph while associating them with this thread ID:

message = {"content": "Add 12 and 13"}
graph_memory.invoke_messages(human_message=message, config=config)


The graph executes and saves the result in memory. Later, if we provide another input referencing the previous output, like "Add that number to 25", the agent can recall the previous result and compute the new value correctly:

message2 = {"content": "Add that number to 25"}
graph_memory.invoke_messages(human_message=message2, config=config)


The result will be 50, as it adds the previous output 25 to 25. We can continue the sequence with further computations, such as "Multiply that number by 2", and it will correctly compute 100 using the stored memory.

message3 = {"content": "Multiply that number by 2"}
graph_memory.invoke_messages(human_message=message3, config=config)


This demonstrates how agent memory allows the graph to maintain context across multiple steps, using Memory Saver and checkpoints. It supports both in-memory storage and external databases like Postgres or MySQL for production-grade applications.

By initializing memory with the graph builder and assigning a thread ID, we ensure that each user session retains its own state. This allows the agent to answer follow-up queries accurately and perform multi-step reasoning across multiple messages.

This was the essence of agent memory in Landgraf. Memory Saver, checkpoints, and thread IDs work together to make the graph context-aware and capable of handling sequential and multi-step conversations efficiently.

**L) Streaming In LangGraph**

In this video, we are going to continue our Landgraf series and discuss different types of streaming techniques. In the previous video, we explored ReAct agents, integrated memories, and LMS tools to create advanced workflows. Now, we’ll focus on streaming, specifically the differences between .stream and stream methods.

There are two important parameters in streaming: values and updates. These determine what gets streamed back to the user during workflow execution.

values: Streams the full state of the graph after each node execution. This means every message and update in the graph state is returned.

updates: Streams only the latest updates to the graph state after each node execution. Previous state information is not returned.

For example, consider a graph with three nodes:

Node 1 executes → updates the messages to "Hi"

Node 2 executes → updates the messages to "My name is"

Node 3 executes → updates the messages to "I love programming"

If we use mode="updates":

for chunk in graph_builder.stream(messages=messages, config=config, stream_mode="updates"):
    print(chunk)


Node 1 → "Hi" is streamed

Node 2 → "My name is" is streamed (only the latest update)

Node 3 → "I love programming" is streamed

If we use mode="values":

for chunk in graph_builder.stream(messages=messages, config=config, stream_mode="values"):
    print(chunk)


Node 1 → "Hi" is streamed

Node 2 → "Hi" + "My name is" is streamed (full state)

Node 3 → "Hi" + "My name is" + "I love programming" is streamed

This clearly shows the difference: updates streams the recent state, while values streams the entire graph state.

Now let’s see a practical example using a simple chatbot workflow. We’ll use MemorySaver to persist context across nodes and threads:

from line_graph.checkpoint.memory import MemorySaver
from graph.state import StateGraph
from annotated_list import add_messages

# Initialize memory
memory = MemorySaver()

# Create graph builder and compile with checkpoint
graph_builder.compile(checkpoint=memory)

# Define a thread ID for a user
config = {"thread_id": 2}

# Initial message
messages = [{"content": "Hi my name is Chris and I like cricket"}]

# Stream with mode = updates
for chunk in graph_builder.stream(messages=messages, config=config, stream_mode="updates"):
    print(chunk)


This will print only the recent updates at each node, e.g., "Hello Chris, cricket is a fantastic sport!".

Now, using mode="values":

# Stream with mode = values
for chunk in graph_builder.stream(messages=messages, config=config, stream_mode="values"):
    print(chunk)


This will print all messages in the current graph state, including the initial user message and all AI responses.

We can continue the conversation by sending another message in the same thread:

# Second message in the same thread
messages = [{"content": "I also like football"}]

# Stream with updates
for chunk in graph_builder.stream(messages=messages, config=config, stream_mode="updates"):
    print(chunk)

# Stream with values
for chunk in graph_builder.stream(messages=messages, config=config, stream_mode="values"):
    print(chunk)


With updates, only the new AI response to "I also like football" is streamed.

With values, the entire conversation including previous messages and AI responses is streamed.

**M) Streaming using astream events Using Langgraph**

So we are going to continue the discussion with respect to our Landgraf series. Already in our previous video, we discussed streaming with the help of the stream methods. In this specific video, we are going to focus on streaming using the stream method. Remember, the stream method is an async method, which allows us to stream back results to the end user as a workflow is executed.

When working with streaming tokens, we often want to display more information to the user in real time. In particular, with chatbot calls, it is common to stream the tokens as they are generated. We can do this using the stream method. Each event in the stream is a dictionary with a few keys, such as event (the type of event being emitted) and name (the name of the event), along with associated data and metadata.

To illustrate this, we first create a configuration with a thread ID. For example, we can set the thread ID to 3. Next, we prepare our message, such as "Hi, my name is Chris and I like to play cricket." Then we use an async loop with graph_builder.stream_events, passing in the messages, config, and version (for example, v2). We can print each event to see the streamed information.

When executed, you will notice that each event provides rich information about the workflow. For example, you may see on_change_start indicating the start of a workflow change, output messages, and on_chat_model_start signaling that the chat model has started generating tokens. Partial messages or chunks of the AI response are also streamed in real time, such as "Hello Chris, it’s great to hear..." This streaming happens in parallel, making it very handy for debugging and tracking the flow of the conversation.

You can also access specific fields such as metadata or graph_node, which shows the node in the graph where the information originates. All events, such as on_chain_start or on_chain_end, are internally triggered and can be individually inspected. This key-value structure allows you to debug or extract any specific information you need during execution.

In short, the stream_events method provides more detailed streaming than the regular .stream() method, giving you access not just to graph state but also to events, nodes, and partial messages. This complements the understanding of .stream() with stream_mode parameters values and updates, allowing you to control whether the full graph state or only the recent updates are streamed.

I hope this example makes the differences clear and demonstrates the power of streaming events in real time. This was it from my side. I will see you in the next video. Have a great day! Bye bye.

# **XI) Debugging LangGraph Application With LangSmith**

**A) LangGraph Studio**

So we are going to continue the discussion with respect to our LangGraph series. In our previous video, we discussed streaming and in-streaming, covering both .stream and a.stream methods. We also talked about additional parameters like values and updates, and understood the differences between them.

In this specific video, we are going to discuss how you can debug complex workflows that you develop using LangGraph. These workflows are very important because they allow you to debug the entire application and understand the output after each node. If you want to make changes in prompt engineering, you should be able to do that on the fly. Debugging helps identify mistakes and optimize workflows.

For this purpose, we will specifically use LangGraph Studio. First, I will create a folder named debugging as our working directory. Then, we will set up LangGraph Studio and visualize the workflow we create. To begin, we will go to LangChain
 and navigate to LangSmith, where you can trace projects. LangGraph deployments are available in the LangGraph Platform, which helps visualize and debug the workflow.

Inside the debugging folder, we start by creating a Python file called openai_agent.py. This file will contain our workflow. We first import the required libraries:

from typing import Annotated
from langgraph import StateGraph, AddMessages, ToolNode
from openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


We use environment variables to store the OpenAI API key and a custom login_API_key for our workflow.

Next, we define a class called State that inherits from dict. Inside this class, we create a messages variable using Annotated to store a list of messages and apply the AddMessages reducer to continuously append new messages:

class State(dict):
    messages: Annotated[list, AddMessages()]


We initialize a chat model using OpenAI's GPT-4 with zero temperature:

chat_model = ChatOpenAI(model_name="gpt-4", temperature=0)


Then, we create a function called make_default_graph to define our workflow:

def make_default_graph():
    graph_workflow = StateGraph(State)
    def call_model(state):
        return state.messages
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("start", "agent")
    graph_workflow.add_edge("agent", "end")
    agent = graph_workflow.compile()
    return agent


Here, we define a node called agent that calls the model and returns the messages. We then define the edges for workflow execution from start → agent → end, and compile the workflow into an agent.

Once the agent is defined, we call it to initialize:

agent = make_default_graph()


To run this workflow in LangGraph Studio, we create a configuration file called LangGraph.json inside the debugging folder. The file specifies dependencies, graphs, and environment variables:

{
  "dependencies": ["."],
  "graphs": [
    {
      "name": "openai_agent",
      "file": "openai_agent.py",
      "variable": "agent"
    }
  ],
  "env": "../.env"
}


This ensures that the Studio can locate the agent, the Python file, and required environment variables. Additionally, we need to install the LangGraph CLI library using a requirements.txt file:

langgraph-cli==<version>


Then install the dependencies:

pip install -r requirements.txt


To deploy and debug the workflow, navigate to the debugging folder in your terminal:

cd debugging
langgraph dev


This command uses the configuration in LangGraph.json to start the agent locally and deploy it to LangSmith Cloud. Once executed, the LangGraph Studio opens, showing the workflow with nodes start, agent, and end. You can submit human messages, system messages, or tool inputs and see the response flow through each node.

For example, if we submit a message like "Hello, how are you?", the agent processes it and returns a response: "Hello, I am just a computer program so I don't have feelings, but I'm here to help you." You can modify the input on the fly, and even fork the workflow to change inputs during runtime.

Next, we can create a more complex workflow that includes a tool node. We define a new function make_alternate_graph to include the tool interaction:

def make_alternate_graph():
    graph_workflow = StateGraph(State)
    def call_model(state):
        return state.messages
    tool_node = ToolNode("add", params=["a", "b"])
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tool", tool_node)
    graph_workflow.add_edge("start", "agent")
    graph_workflow.add_edge("agent", "tool")
    graph_workflow.add_edge("tool", "agent")
    graph_workflow.add_edge("agent", "end") 
    agent = graph_workflow.compile()
    return agent


This workflow demonstrates tool calls: the agent sends a request to the tool, the tool performs the computation (e.g., addition), and the result is sent back to the agent. For instance, asking "What is 10 + 10?" will execute through the tool node, returning 20. You can edit inputs during runtime to see how outputs change, making it highly interactive.

Finally, you can configure interrupts at various points—before the tool, after the tool, or even before agent execution—to handle conditional pauses or runtime checks. This capability allows you to debug and optimize complex LangGraph workflows step by step.

In conclusion, LangGraph Studio provides a powerful interface to debug, visualize, and edit workflows in real-time. By following this setup—creating openai_agent.py, LangGraph.json, installing dependencies, and using langgraph dev—you can debug simple chatbots as well as more complex tool-integrated workflows efficiently.

# **XII) Different Workflows In LangGraph**

**A) Prompt Chaining**

We are going to continue our discussion on the LangGraph series. In this video, and in the upcoming series, we will explore different types of workflows. Workflows are extremely important because when implementing real-world use cases, designing the workflow properly allows you to follow specific patterns to solve complex problems effectively. As we progress, you will see multiple workflow designs.

In this specific video, we will focus on prompt chaining. First, we’ll define what prompt chaining is, then visualize it using a diagram from the LangGraph documentation, and finally implement a real-world use case to demonstrate it.

Definition of Prompt Chaining:

Prompt chaining is a technique in natural language processing (NLP) where multiple prompts are sequenced together to guide a model through a complex task or reasoning process. Instead of relying on a single prompt to achieve a desired outcome, prompt chaining breaks the task into smaller, manageable steps, with each step building on the previous one. This approach helps improve accuracy, coherence, and control when working with large language models.

In simpler terms, prompt chaining divides a complex task into smaller subtasks, which are solved sequentially, step by step.

Let’s consider a diagram to understand the workflow. In this diagram:

We start with an input.

We make an LM call (Large Language Model call).

We can include logic or conditions between tasks. If the condition is satisfied, the next prompt or subtask is executed.

Tasks continue sequentially until the workflow is complete.

For instance, if solving a problem using prompt chaining, a bigger task can be divided into three subtasks:

Task A → execute first subtask.

Task B → execute if condition from Task A is satisfied.

Task C → execute as the final step.

Multiple conditions can be added at any stage, ensuring that the model progresses through the workflow correctly.

To implement this in code, let’s consider a simple real-world use case: generating a story using prompt chaining. Our workflow will have three steps: generate, improve, and polish the story. Additionally, we will apply a conditional check after generating the story. If the story passes the condition, it proceeds to improvement and polishing. If it fails, it loops back to generate a new story.

Here’s a sample Python implementation of the workflow:

from langgraph import StateGraph, AddMessages
from typing import Annotated

# Define the state class
class State(dict):
    messages: Annotated[list, AddMessages()]

# Define the functions for each step
def generate_story(state):
    story = "Once upon a time, in a small village..."  # Example story generation
    state.messages.append({"step": "generate", "story": story})
    return state

def improve_story(state):
    story = state.messages[-1]["story"] + " The villagers faced a challenge."  # Improving story
    state.messages.append({"step": "improve", "story": story})
    return state

def polish_story(state):
    story = state.messages[-1]["story"] + " And they lived happily ever after."  # Polishing
    state.messages.append({"step": "polish", "story": story})
    return state

# Condition function to decide whether to continue
def check_condition(state):
    last_story = state.messages[-1]["story"]
    if "challenge" in last_story:
        return True
    return False

# Define the workflow
def make_story_workflow():
    graph = StateGraph(State)
    # Add nodes
    graph.add_node("generate", generate_story)
    graph.add_node("improve", improve_story)
    graph.add_node("polish", polish_story)
    # Add edges with conditions
    graph.add_edge("generate", "improve", condition=check_condition)
    graph.add_edge("improve", "polish")
    # Compile workflow
    agent = graph.compile()
    return agent

# Initialize workflow
agent = make_story_workflow()


In this example:

The generate_story node creates the initial story.

The improve_story node enhances the story with additional content.

The polish_story node finalizes the story for output.

The check_condition function acts as a decision point, ensuring that the story meets a certain quality before moving to the next step.

This workflow demonstrates the core idea of prompt chaining: breaking a complex task into smaller, sequentially executed steps with conditional logic in between.

By using this approach, we can handle complex tasks in a structured manner, making it easier to debug, modify, and improve outputs in large-scale NLP workflows.

In summary, prompt chaining is a powerful technique to enhance model reasoning, maintain control over tasks, and ensure a coherent workflow. For this example, we created a story generator workflow with three stages and conditional logic to ensure quality outputs.

I hope you have understood the concept of prompt chaining. In the next video, we will explore more advanced workflow patterns and implement more complex use cases.

Thank you, and see you in the next video.

**B) Prompt Chaining Implementation With Langgraph**

So guys, we are going to continue our discussion on prompt chaining and implement a specific use case in this video. The notebook we will be working on is called prompt_chaining.ipynb, located inside the fourth folder, workflows. Prompt chaining, as defined in the notebook, is a technique in natural language processing where multiple prompts are sequenced together to guide a model through a complex task. You can refer to the notebook for the exact definition whenever needed.

To understand how prompt chaining works, we first define a task and then break it down into smaller subtasks. For instance, if the main task is to generate a detailed report, we might split it into gather_data, analyze_data, and write_summary. Each subtask corresponds to a node in the workflow graph. Next, we define edges between nodes and include conditional logic if needed. Finally, we execute the graph. The key advantage is that prompt chaining allows us to iterate if a particular subtask fails a condition.

For our use case, we want to generate a story with three nodes: Generate, Improve, and Polish. After generating the story, we check a condition. If it passes, we move to improvement and polishing; if it fails, it loops back to generate a new story. To start, we need to load the language model (LM). This can be done using the following code:

import os
from dotenv import load_dotenv
load_dotenv()

# Example using a freely available LM (ChatGrok) or OpenAI API key
# lm = ChatGrokModel("model_name_here")
# lm = OpenAIModel(api_key=os.getenv("OPENAI_API_KEY"))

# Load your model
lm = YourLMModel("model_name_here")  # e.g., "mistral-2-5B"


Once the LM is ready, we define the state class to store the inputs and outputs of each node:

from typing import Annotated
from landgraf.graph import StateGraph
from IPython.display import display, Image

class State(dict):
    topic: str = ""
    story: str = ""
    improved_story: str = ""
    final_story: str = ""


Here, topic is the input given by the user. story stores the output from the Generate node, improved_story stores the output from the Improve node, and final_story stores the output from the Polish node. This allows the workflow to maintain context across multiple prompts.

Next, we define our nodes. The Generate Story node takes the topic and produces an initial story:

def generate_story(state: State):
    message = lm.invoke(f"Write a one sentence story premise about: {state.topic}")
    state.story = message.content
    return state


We then define a conditional node to check if the story contains a question mark ? or exclamation !. If it does, the node returns "fail", otherwise "pass":

def check_conflict(state: State):
    if "?" in state.story or "!" in state.story:
        return "fail"
    return "pass"


The Improve Story node enhances the generated story with more details:

def improve_story(state: State):
    message = lm.invoke(f"Enhance the story with vivid details: {state.story}")
    state.improved_story = message.content
    return state


Finally, the Polish Story node applies a final twist or refinement to the improved story:

def polish_story(state: State):
    message = lm.invoke(f"Add an unexpected twist to the story: {state.improved_story}")
    state.final_story = message.content
    return state


With nodes defined, we can build the workflow graph. We add nodes to the graph, define edges, and include the conditional logic from the check node:

graph = StateGraph(state=State())

# Add nodes
graph.add_node("generate", generate_story)
graph.add_node("improve", improve_story)
graph.add_node("polish", polish_story)

# Add edges
graph.add_edge("start", "generate")
graph.add_conditional_edges("generate", check_conflict, pass_node="improve", fail_node="generate")
graph.add_edge("improve", "polish")
graph.add_edge("polish", "end")


We compile and display the graph to verify its structure:

graph.compile()
display(graph)


Now, we can run the graph by providing an initial topic. For example, if the topic is "Generic AI systems":

state = State()
state.topic = "Generic AI systems"

result = graph.invoke(state)
print(result)


The output will include the topic, generated story, improved story, and the final polished story. For example:

Generated story: "In a future where generic AI systems possess autonomy beyond human comprehension..."

Improved story: "In a future where generic AI systems possess autonomy beyond human comprehension, humans struggle to adapt..."

Polished story: "In a future where generic AI systems possess autonomy beyond human comprehension, humans struggle to adapt, and an unexpected AI rebellion unfolds..."

By dividing the task into smaller subtasks and using conditional logic, prompt chaining ensures better context management, modularity, and debugging ease. Each node focuses on a single aspect of the task, reducing the risk of losing context, making it easier to reuse and rearrange nodes, and allowing complex reasoning in a step-by-step manner.

This is a complete example of prompt chaining using Landgraf, from loading the model, creating state, defining nodes, building the graph, and finally executing it to generate, improve, and polish a story. This workflow can be applied to any task requiring multiple steps and logical checks, making it highly versatile and powerful.

**C) Parallelization**

Hello guys! Today, we are going to continue our discussion on different types of workflows. In this video, we will focus on parallelization. The main goal is to understand why parallelization is used and how it can help when building workflows where tasks do not depend on each other’s output. We will go through the concept, implementation, and a practical example so that the concept becomes crystal clear.

In a standard workflow, nodes usually execute sequentially—one after the other, following the edges defined in the graph. However, if the output of a task is independent of previous tasks, you can execute those nodes in parallel. Parallelization allows tasks to run concurrently, and their outputs can then be aggregated into a downstream node. Conceptually, this is done by connecting multiple independent nodes to a common start node, then combining their outputs in an aggregator node before reaching the end node.

For our example, we are going to generate a short story. The story requires different characters, premises, and settings, all of which can be generated independently. Once all these independent nodes finish execution, their outputs are combined to form a story introduction. This is a perfect scenario for parallelization because none of these nodes depends on the output of the others.

First, we need to load our LLM. You can use ChatGPT, OpenAI API, or ChatGrok. Here’s how we load the model:

import os
from dotenv import load_dotenv
load_dotenv()

# Example using ChatGrok or OpenAI API
# lm = ChatGrokModel("model_name_here")
# lm = OpenAIModel(api_key=os.getenv("OPENAI_API_KEY"))

lm = YourLMModel("model_name_here")


Next, we define our state class which will store the topic, characters, settings, premises, and story introduction:

from typing import Annotated
from landgraf.graph import StateGraph

class State(dict):
    topic: str = ""
    characters: str = ""
    settings: str = ""
    premises: str = ""
    story_intro: str = ""


Now, we define the nodes that will run in parallel. The first node generates characters based on the topic:

def generate_characters(state: State):
    message = lm.invoke(f"Create two character names and brief traits for a story about {state.topic}")
    state.characters = message.content
    return state


Similarly, we generate settings:

def generate_settings(state: State):
    message = lm.invoke(f"Describe a vivid setting for a story about {state.topic}")
    state.settings = message.content
    return state


And premises:

def generate_premises(state: State):
    message = lm.invoke(f"Write a one-sentence plot premise for a story about {state.topic}")
    state.premises = message.content
    return state


Since these three nodes are independent, they can be executed concurrently. Once they complete, we use an aggregator node to combine their outputs into a story introduction:

def combine_elements(state: State):
    message = lm.invoke(
        f"Write a short story introduction using these elements:\n"
        f"Characters: {state.characters}\n"
        f"Settings: {state.settings}\n"
        f"Premises: {state.premises}"
    )
    state.story_intro = message.content
    return state


With the nodes defined, we can build the parallel workflow graph:

graph = StateGraph(state=State())

# Add nodes
graph.add_node("generate_characters", generate_characters)
graph.add_node("generate_settings", generate_settings)
graph.add_node("generate_premises", generate_premises)
graph.add_node("combine", combine_elements)

# Add edges for parallel execution
graph.add_edge("start", "generate_characters")
graph.add_edge("start", "generate_settings")
graph.add_edge("start", "generate_premises")

# Combine outputs
graph.add_edge("generate_characters", "combine")
graph.add_edge("generate_settings", "combine")
graph.add_edge("generate_premises", "combine")
graph.add_edge("combine", "end")


After building the graph, we compile and visualize it:

graph.compile()
display(graph)


Finally, we can invoke the graph with a topic. For example, let’s generate a story about "Time Travel":

state = State()
state.topic = "Time Travel"

result = graph.invoke(state)
print(result.story_intro)


Here, all three nodes (generate_characters, generate_settings, and generate_premises) run concurrently, and their outputs are aggregated by the combine_elements node. The resulting story_intro gives a short story introduction based on all the parallel outputs.

The key benefits of parallelization in workflows include:

Speed: Reduces total execution time by running tasks concurrently.

Scalability: Efficiently handles large workflows and keeps the graph clean.

Reusability: Independent nodes can be reused or rearranged in different workflows.

In summary, parallelization is highly useful when tasks do not depend on each other’s outputs. By executing nodes concurrently and combining results, you can build efficient and modular workflows. This workflow is a practical example of generating a story where multiple tasks are independent but contribute to a single aggregated output.

**D) Routing**

Hello guys! Today we are going to continue our discussion on different types of workflows, and in this video, we are going to focus on routing. Routing is a technique that allows a workflow to conditionally determine which node to execute next based on the current state or the output of a node. You may have already encountered routing in previous examples, such as when we added conditional edges. But in this session, we’ll make a dedicated discussion and implementation. We will also integrate Pydantic classes to ensure that our language model (LM) provides structured outputs, which is crucial for routing decisions.

In simple terms, routing allows us to "route" the flow of execution to different nodes depending on certain conditions. For example, if the LM output indicates that the user wants a story, a joke, or a poem, we can route the input to the corresponding node. Conceptually, a router node evaluates the input and decides the next node in the workflow. This can be visualized as a decision point in a flowchart, where the LM evaluates the input and directs it down the appropriate path. The paths are independent, and whichever route is taken will finally converge to the end of the workflow.

Before we implement routing, we need to define structured output using Pydantic. This ensures that the LM output is constrained to specific values and avoids errors. First, we import the necessary libraries: "from typing_extensions import Literal" and "from pydantic import BaseModel". Then, we define a Pydantic class called Route which restricts LM outputs to "poem", "story", or "joke":

from typing_extensions import Literal
from pydantic import BaseModel, Field

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(..., description="The next step in the routing process")


This ensures that the LM will only return one of these three values, and any deviation will raise an error. Next, we integrate this structured output with our LM to create a router LM:

from langchain.schema import HumanMessage, SystemMessage

router_lm = YourLMModel.structured_output(Route)


Here, router_lm will always provide outputs in the structure defined by the Route class.

Next, we define a state class to store the workflow state. This class will keep track of the input, the LM routing decision, and the final output. For this, we use a dictionary-like type:

from typing_extensions import TypeDict

class State(TypeDict):
    input: str
    decision: str
    output: str


This State class will be used to pass information between nodes in our workflow.

Now let’s define the three LM nodes that represent the different outputs:

def lm_call_one(state: State):
    # Handles story generation
    state.output = lm.invoke(f"Write a story about {state.input}").content
    return state

def lm_call_two(state: State):
    # Handles joke generation
    print("LM call two is called")
    state.output = lm.invoke(f"Write a joke about {state.input}").content
    return state

def lm_call_three(state: State):
    # Handles poem generation
    state.output = lm.invoke(f"Write a poem about {state.input}").content
    return state


Each of these nodes takes the same input and generates a different type of output, but they are only invoked based on the LM router’s decision.

Next, we define the router node itself, which determines which path to follow:

def lm_router(state: State):
    decision = router_lm.invoke(
        system_message=SystemMessage(content="Route the input to 'story', 'joke', or 'poem' based on user request."),
        human_message=HumanMessage(content=state.input)
    )
    state.decision = decision.step
    return state


Here, the router evaluates the input and assigns a value to state.decision, which can be "story", "joke", or "poem". Based on this decision, we define a conditional function to determine which LM node to invoke:

def route_decision(state: State):
    if state.decision == "story":
        return lm_call_one
    elif state.decision == "joke":
        return lm_call_two
    elif state.decision == "poem":
        return lm_call_three


This function connects the router’s decision to the appropriate node in the workflow.

Finally, we build the routing workflow using a state graph:

from landgraf.graph import StateGraph

router_workflow = StateGraph(state=State())

# Add nodes
router_workflow.add_node("router", lm_router)
router_workflow.add_node("story_node", lm_call_one)
router_workflow.add_node("joke_node", lm_call_two)
router_workflow.add_node("poem_node", lm_call_three)

# Add edges
router_workflow.add_edge("start", "router")
router_workflow.add_conditional_edge("router", route_decision)
router_workflow.add_edge("story_node", "end")
router_workflow.add_edge("joke_node", "end")
router_workflow.add_edge("poem_node", "end")

# Compile and display
router_workflow.compile()
display(router_workflow)


Now we can invoke the workflow with user input:

state = State(input="Write me a joke about Agentic AI systems")
output_state = router_workflow.invoke(state)
print(output_state.output)


If the user asks for a joke, the router evaluates the input and correctly routes it to lm_call_two. The output might look like:

"Why did I refuse to play hide and seek? Because it could not stand the thought of being out of control."

In summary, routing allows workflows to dynamically determine which node to execute next based on conditions evaluated at runtime. Using Pydantic ensures structured LM outputs, preventing errors. The workflow can handle multiple routing paths, like "story", "joke", or "poem", and is fully extendable for more complex use cases.

Routing, combined with structured outputs, makes your workflows dynamic, flexible, and robust. The sky is the limit—you can create more complex routing logic with LM-driven decisions to automate a wide variety of tasks.

**E) Orchestrator-Worker**

Hello guys! Today we are going to continue our discussion on workflows, and in this session, we are going to focus on a new workflow called the Orchestrator-Worker Workflow. This workflow is particularly important because it allows a central LM (or orchestrator) to dynamically break down tasks, assign them to multiple worker LMs, and finally synthesize their results. It is ideal for complex tasks where subtasks cannot be predicted in advance, such as generating multi-section reports or coding tasks where the number of files and type of edits vary.

The Orchestrator-Worker workflow works in three major steps: (1) Task breakdown, (2) Task delegation to workers, and (3) Synthesizing results. Unlike simple parallelization, this workflow is flexible, because the orchestrator dynamically decides which tasks need to be executed, and assigns them to workers. The workers then work in parallel, independently completing their subtasks, and the synthesizer combines their outputs into a final consolidated result. Conceptually, the orchestrator is like a manager, and the worker LMs are like employees executing assigned tasks.

Let’s take an example. Suppose we want to generate a detailed report on “Agentic AI Systems”. This report can have multiple sections: "Introduction", "History", "Current Trends 2025", and more. Instead of writing these sequentially, the orchestrator dynamically creates worker LMs and assigns each section to a different worker. Worker 1 writes the introduction, Worker 2 handles the history, and Worker 3 covers current trends. Once all sections are completed, the orchestrator synthesizes the results into a single report. This ensures parallel execution and structured aggregation of output.

Now, let’s see how we can implement this using Python. First, we define the worker nodes. Each worker is responsible for handling a specific section of the report:

def worker_introduction(task_input):
    # LM generates the Introduction section
    return lm.invoke(f"Write an introduction about {task_input}").content

def worker_history(task_input):
    # LM generates the History section
    return lm.invoke(f"Write the history of {task_input}").content

def worker_current_trends(task_input):
    # LM generates the Current Trends section
    return lm.invoke(f"Write about current trends of {task_input} in 2025").content


Next, we define the orchestrator, which is responsible for dynamically breaking down the task and assigning it to workers. The orchestrator will also collect the outputs from all workers:

def orchestrator(task_input):
    # Define sections dynamically
    tasks = {
        "Introduction": worker_introduction,
        "History": worker_history,
        "Current Trends": worker_current_trends
    }
    # Assign tasks to workers and execute in parallel
    from concurrent.futures import ThreadPoolExecutor
    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_section = {executor.submit(worker, task_input): section for section, worker in tasks.items()}
        for future in future_to_section:
            section = future_to_section[future]
            results[section] = future.result()
    # Return synthesized report
    return synthesize_report(results)


Here, the orchestrator uses ThreadPoolExecutor to run worker nodes in parallel, ensuring faster execution. Each worker returns its section of the report, and the orchestrator collects these results in a dictionary.

Finally, we define a synthesizer to combine the outputs into a coherent report:

def synthesize_report(results):
    report = ""
    for section, content in results.items():
        report += f"### {section}\n{content}\n\n"
    return report


Now, we can invoke the full Orchestrator-Worker workflow with user input:

task_input = "Agentic AI Systems"
final_report = orchestrator(task_input)
print(final_report)


In this example, the orchestrator dynamically delegates subtasks to multiple worker LMs, which execute independently and in parallel. Once all workers complete their tasks, the synthesizer combines their outputs to produce the final report.

In summary, the Orchestrator-Worker Workflow is extremely powerful for complex, unpredictable tasks. It allows dynamic task breakdown, parallel execution by worker LMs, and result aggregation. This workflow is highly flexible and can be adapted for report generation, coding automation, research summarization, and other multi-step processes where subtasks are independent but need to be combined into a final coherent output.

**F) Orchestrator Worker Implementation**

Hello guys! In this session, we are going to implement the Orchestrator-Worker Workflow that we discussed in the previous video. We will use Python and LangGraph to dynamically create workers, perform LM calls in parallel, and generate a structured report with multiple sections.

Step 1: Setting Up the Environment

First, we need to import all necessary libraries, including Pydantic for structured outputs, typing helpers, and Landgraf’s send API to dynamically create workers:

from pydantic import BaseModel, Field
from typing import List
from landgraf.constants import send
from operator import add  # for managing shared state

Step 2: Define the Report Schema

We define the schema of our report using Pydantic. Each section has a name (title) and description (detailed content):

class Section(BaseModel):
    name: str = Field(..., description="Title of the section of the report")
    description: str = Field(..., description="Brief overview or detailed information for the section")

class Report(BaseModel):
    sections: List[Section] = Field(..., description="List of sections in the report")


The LM will return output according to this schema, making it easy to structure the report programmatically.

Step 3: Create the Planner

The orchestrator acts as a manager to create a report plan. We use an LM call to generate the outline of the report (sections with names and placeholders for descriptions):

planner = lm.with_structured_output(Report)

def create_report_plan(topic: str):
    return planner.invoke(topic)


The topic is the main subject of the report, and the output is a list of sections ready to be assigned to workers.

Step 4: Define the LM Call for Workers

Each worker will generate the content for a specific section. The LM call for a worker looks like this:

def lm_call(section: Section):
    response = lm.invoke(
        system_message="Write a report section following the provided name and description. Use markdown formatting. Include no preamble.",
        human_message=f"Section Name: {section.name}"
    )
    return response.content


This function takes a section as input and returns the generated content for that section.

Step 5: Create Worker State

Since each worker can maintain its own state, we define a worker state class to keep track of the completed sections:

class WorkerState(BaseModel):
    section: Section
    completed_sections: List[str] = []


This ensures each worker writes to a shared state without conflicts.

Step 6: Assign Workers Dynamically Using Send API

The orchestrator dynamically creates workers for each section using the send API. This allows parallel execution of LM calls:

def assign_workers(state):
    return send(
        lm_call,
        inputs=[section for section in state.sections],  # one worker per section
        state=state.completed_sections,  # shared state
        operator=add  # append results to shared state
    )


Here, each worker independently generates its section content in parallel, and all outputs are stored in completed_sections.

Step 7: Synthesizer

Once all workers complete their tasks, the synthesizer combines all sections into a final report:

def synthesizer(completed_sections: List[str]):
    final_report = "\n\n".join(completed_sections)
    return final_report

Step 8: Build the Workflow

We integrate all components into a state graph workflow:

from landgraf.workflow import Workflow, StateGraph

workflow = Workflow()
state_graph = StateGraph()

state_graph.add_node("orchestrator", create_report_plan)
state_graph.add_node("assign_workers", assign_workers)
state_graph.add_node("synthesizer", synthesizer)

state_graph.add_edge("orchestrator", "assign_workers")
state_graph.add_edge("assign_workers", "synthesizer")


This structure ensures the orchestrator generates the plan, workers execute in parallel, and the synthesizer merges results.

Step 9: Run the Workflow

Finally, we invoke the workflow with a topic:

topic = "Introduction to RAG Agents"
final_report = workflow.invoke(topic)

from IPython.display import Markdown, display
display(Markdown(final_report))


The orchestrator generates a report outline.

Workers dynamically generate content for each section in parallel.

The synthesizer combines all sections to produce the final report.

This workflow is extremely efficient for blog generation, reports, or any multi-section content, because all sections are created simultaneously, drastically reducing total execution time.

Summary

1. Planner: Orchestrator generates a structured outline.

2. Workers: Dynamically created to handle each section in parallel.

3. LM Call: Generates section content.

4. Shared State: Workers store results.

5. Synthesizer: Combines all sections into a final report.

By following this pattern, you can scale content generation efficiently and manage complex tasks dynamically with LMs.

**G) Evaluator-optimizer**

Hello guys! In this session, we are going to implement the Evaluator-Optimizer Workflow, which is another type of workflow that helps in iterative refinement of AI-generated content.

Step 1: Understanding the Workflow

Generator LM: Produces content (e.g., a joke, story, or text).

Evaluator LM: Checks the content against criteria (e.g., quality, humor, correctness).

Looping mechanism: If the content fails evaluation, feedback is sent back to the generator to improve the output.

Accepted output: Once the content passes evaluation, it proceeds to the next step or final output.

Use Case: Creating a joke about a topic, evaluating whether it is funny, and iteratively improving it until it meets the criteria.

Step 2: Define the State

We define a state class to track the topic, the joke generated, and evaluation feedback:

from pydantic import BaseModel
from typing import Literal

class State(BaseModel):
    topic: str
    joke: str = ""
    funny: bool = False
    feedback: str = ""


topic: The input subject for joke generation.

joke: Content generated by the generator LM.

funny: Boolean indicating if the joke is approved.

feedback: Feedback from the evaluator LM to improve the joke if rejected.

Step 3: Define the Feedback Structure

The feedback class structures the evaluator’s output:

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"]
    feedback: str = Field(..., description="If not funny, provide feedback for improvement")


grade: Either "funny" or "not funny".

feedback: Text describing why the joke was rejected and how to improve it.

We then attach this structured output to our LM:

lm.with_structured_output(Feedback)

Step 4: Define Nodes
Generator Node

Generates a joke, optionally taking evaluator feedback into account:

def generator_node(state: State):
    if state.feedback:
        prompt = f"Write a joke about {state.topic}, considering the feedback: {state.feedback}"
    else:
        prompt = f"Write a joke about {state.topic}"
    response = lm.invoke(prompt)
    state.joke = response.content
    return state

Evaluator Node

Evaluates the joke and provides feedback:

def evaluator_node(state: State):
    eval_response = evaluator.invoke(state.joke)
    state.funny = eval_response.grade == "funny"
    state.feedback = eval_response.feedback
    return state

Step 5: Conditional Routing

We define a routing function to decide the next step:

def route_joke(state: State):
    if state.funny:
        return "accepted"
    else:
        return "rejected_feedback"


If funny = True, the joke is accepted.

If funny = False, it loops back to the generator with feedback.

Step 6: Build the Workflow

We create a state graph and connect nodes:

from landgraf.workflow import Workflow, StateGraph

workflow = Workflow()
state_graph = StateGraph()

state_graph.add_node("generator", generator_node)
state_graph.add_node("evaluator", evaluator_node)

# Routing logic
state_graph.add_edge("generator", "evaluator")
state_graph.add_conditional_edge("evaluator", route_joke, accepted="end", rejected_feedback="generator")


Generator → Evaluator → Conditional routing → either end or back to generator.

Step 7: Execute the Workflow

Finally, we invoke the workflow with a topic:

topic = "Agentic AI System"
state = State(topic=topic)

final_state = workflow.invoke(state)
print(final_state.joke)


The generator creates the initial joke.

The evaluator checks it and either approves it or sends feedback to regenerate.

This loop continues until the joke is deemed "funny".

Summary:

1. Generator: Creates content based on a topic and optionally feedback.

2. Evaluator: Grades content and provides structured feedback.

3. Loop: Rejected content is sent back to the generator for improvement.

4. Accepted output: Content passes evaluation and proceeds to final output.

This workflow is ideal for iterative refinement, human-in-the-loop setups, and quality-controlled content generation.

# **XIII) Human in the Loop in LangGraph**

**A) Human In The Loop With LangGraph Workflows**

So we are going to continue our discussion on the line graph series. In the previous videos, we have already covered different types of workflows, how to debug a line graph application, and some advanced as well as basic topics such as creating chatbots, working with Pedantic, and understanding what React agents are. Now, it’s time to dive into a new module called Human in the Loop.

So, what does human in the loop mean? Up until now, the workflows we have developed were fully autonomous, executing without any human intervention. However, whenever we deal with complex workflows, it’s beneficial to include human intervention at certain points. This human feedback can help make workflow execution more accurate and reliable. Essentially, human in the loop allows us to pause a workflow at specific points, get approval, or receive feedback before continuing. This can be particularly helpful in scenarios requiring task approvals or debugging, where we may want to rewind a graph to reproduce or prevent issues.

Let’s see an example to illustrate this concept. In this workflow, the assistant will receive input and a human feedback interrupt will be applied before the assistant executes. For instance, if we define custom tools such as addition, subtraction, multiplication, and division, when a user inputs “What is 2 + 2?”, the request goes to the assistant first. Before execution, it is interrupted to request human confirmation. If permission is granted, the assistant executes the tool call and continues to the next step, again pausing for feedback if needed, until it reaches the end.

To implement this, the first step is to load necessary libraries and environment variables:

from dotenv import load_dotenv
from langchain.grok.chat import Grok
import os

load_dotenv()
grok_api = os.getenv("GROK_API")
model = "qwq-32b"


Note that the previous model version has been decommissioned, so we are using the updated qwq-32b model.

Next, we define some custom tools for arithmetic operations. Each function should include a docstring so that the LLM can understand how to use it:

def multiply(a: int, b: int) -> int:
    """Multiply A and B"""
    return a * b

def add(a: int, b: int) -> int:
    """Add A and B"""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide A by B"""
    return a / b

tools = [add, multiply, divide]


After defining the tools, we bind them to the language model:

lm_with_tools = LM.bind_tools(tools)


Now, we can create our workflow using line graph. The workflow consists of nodes such as start, assistant, tools, and end. The assistant communicates with the tools, and outputs return back to the assistant. Human feedback is applied using an interrupt before clause.

We import the necessary libraries to build the graph:

from langchain.graph import MessageState, StateGraph, ToolNode, ToolCondition
from langchain.core.messages import IMessage, HumanMessage, SystemMessage
from langchain.graph import MemorySaver


We start by defining a system message to instruct the LLM:

system_message = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


Next, we define the assistant node using a MessageState:

def assistant(state: MessageState) -> MessageState:
    messages = state.messages
    messages.append(lm_with_tools.invoke(system_message=system_message, messages=messages))
    return state


After defining the node, we create the state graph and add nodes and edges:

builder = StateGraph(MessageState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge("start", "assistant")
builder.add_conditional_edge("assistant", "tools", condition=ToolCondition())
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


We can also display the graph as an image to visualize the workflow. To add human feedback, we apply the interrupt_before parameter in the assistant node:

graph.add_interrupt_before("assistant")


Finally, we can execute the workflow and stream the output. Each step waits for human permission at the specified interrupt point:

thread_id = "123"
message = "multiply 2 and 3"

for event in graph.stream(input=message, thread_id=thread_id, stream_mode="values"):
    event_message = event.messages[-1]
    print(event_message)


When executed, the workflow stops at the human feedback interrupt, pausing before the assistant executes. This demonstrates how human in the loop works to ensure that the workflow only continues after human approval. In the next video, we will see how to continue execution after the interruption.

So, in summary, human in the loop allows us to include human feedback in otherwise autonomous workflows by using interrupt_before or interrupt_after, which helps in debugging, validation, and task approval, making the workflow more robust and controlled.

**B) Human In the Loop Continuation**

In our previous video, we executed a workflow where we applied an interrupt before the assistant. As soon as we ran graph.stream with a message like "multiply 2 and 3", the input went to the assistant and the workflow got interrupted. This interruption pauses execution and allows the human to provide feedback or take action. At this stage, if we print the event message using event_message.PrettyPrint(), we only see the human message because the interruption happens before the assistant executes.

To understand what the next step in the workflow is, we can use the graph.get_state method. For example:

state = graph.get_state(thread_id=thread_id)
print(state.next)


Here, state.next tells us that the next state is the assistant node. This is because the workflow paused at the human feedback interrupt. Similarly, if we want to see the current state snapshot, we can run:

state = graph.get_state(thread_id=thread_id)
print(state)


The current state snapshot contains information about the human message ("multiply 2 and 3") and other metadata. This snapshot represents the checkpoint of the workflow at the point of interruption. If we want to review the history of all checkpoints, we can use:

history = graph.get_state_history(thread_id=thread_id)


This generates an object containing all the state checkpoints in the workflow, showing how the workflow has progressed so far.

Now, to continue the execution of the workflow after the human feedback interrupt, we simply call graph.stream again without passing any new message. For example:

for event in graph.stream(input=None, thread_id=thread_id, stream_mode="values"):
    event.messages[-1].PrettyPrint()


Here, input=None signals that we want to continue execution without adding new input. The workflow moves to the next state, which in this case is the assistant node. The assistant then decides if a tool call is required. If a tool is called, such as our custom multiply tool, it executes the function and returns the result. After the tool executes, because the interrupt_before parameter is still active, the workflow pauses again for potential human feedback.

Once the human confirms by continuing execution, the assistant receives the tool output and the workflow finally proceeds to the end node. The complete execution flow for our example "multiply 2 and 3" looks like this:

The message goes to the assistant.

The workflow interrupts before the assistant for human feedback.

After human confirmation, the assistant executes.

If a tool call is required, it goes to the tool node.

The tool executes and returns the output to the assistant.

Another interruption occurs before the assistant to allow human feedback again.

After human confirmation, the assistant completes execution.

The workflow moves to the end state.

This process highlights the role of human feedback in controlling workflow execution. The human can either provide input, modify messages, or simply continue execution by passing None. Using checkpoints, state snapshots, and state history, we can monitor the workflow’s current and next states at every stage.

In the next example, we will explore how to edit human feedback during execution. Instead of just continuing the workflow, the human can modify the input or provide custom messages, and the workflow will incorporate this feedback into subsequent steps. This allows us to dynamically control and refine workflow execution.

Overall, in this video we learned about state snapshots, graph.get_state, graph.get_state_history, checkpoints, and how human feedback is integrated via interrupts. Continuing execution is as simple as passing None, and in the next step, we will see how human-provided edits affect the workflow.

**C) Editing Human Feedback In Workflow**

We are continuing our discussion on the human-in-the-loop topic with respect to Line Graph. In the previous video, we learned how to add an interrupt within a workflow and continue execution when a human simply gives permission, like "Hey, continue the execution". In this video, we’ll go a step further and learn how to edit human feedback dynamically during workflow execution.

Setting Up the Workflow

We will use the same graph as before—the one that interrupts before the assistant. You can also configure interrupts after the assistant, before a tool, or after a tool by specifying the node names.

Here’s how we start:

# Define the human input and thread
human_message = "Multiply 2 and 3"
thread_id = 1  # Using the same thread

# Execute the graph with the human input
for event in graph.stream(input=human_message, thread_id=thread_id):
    event.messages[-1].PrettyPrint()


At this stage, the workflow interrupts before the assistant, so if we check the state:

state = graph.get_state(thread_id=thread_id)
print(state.next)  # Output: assistant


We can see that the next state is the assistant, but before moving forward, we can edit the human message.

Editing Human Feedback

Suppose we want to change the input from "Multiply 2 and 3" to "Multiply 15 and 6". We can do this using the update_state method:

# Update the human message
updated_message = [
    {"role": "human", "content": "Please multiply 15 and 6"}
]

graph.update_state(
    thread_id=thread_id,
    messages=updated_message
)


Now, if we check the state again:

new_state = graph.get_state(thread_id=thread_id)
print(new_state.values())


We will see that the human message has been updated to "Please multiply 15 and 6".

Continuing Execution

Once the human feedback is updated, we can continue execution without providing additional input:

# Continue execution
for event in graph.stream(input=None, thread_id=thread_id, stream_mode="values"):
    event.messages[-1].PrettyPrint()


Here’s what happens next:

The workflow moves to the assistant.

The assistant recognizes a tool call (our multiply tool).

The tool executes with the updated arguments (15 and 6) and returns the result (90).

The workflow interrupts again before the assistant, waiting for human confirmation.

We can continue execution once more:

# Final continuation
for event in graph.stream(input=None, thread_id=thread_id, stream_mode="values"):
    event.messages[-1].PrettyPrint()


Finally, the assistant outputs the result:

The result of multiplying 15 and 6 is 90

Key Takeaways

You can edit human messages during execution using graph.update_state.

Workflow execution can pause for human input at any node using interrupt_before or interrupt_after.

Continuing execution without new input is as simple as passing None to graph.stream.

The tool receives the updated human message, executes the logic, and the workflow can be interrupted again before the assistant.

Next Steps

In the next video, we will explore waiting for user input dynamically during workflow execution. This will allow the workflow to pause until the human provides input, which can then be used to edit messages or continue execution. For that, we will create a new graph configuration that interrupts before the node where human feedback is expected.

This video covers two major points:

Editing human feedback dynamically.

Properly continuing execution while involving human input using interrupts.

In the next video, we’ll focus on waiting for real-time user input and how to integrate it into the workflow execution seamlessly.

**D) Runtime Human Feedback In Workflow**

In this tutorial, we will explore human-in-the-loop workflows using LineGraph, where a workflow can pause for human input, allow editing of messages, and continue execution dynamically. This is useful when you want humans to review or modify the messages before the assistant or tool processes them.

We start by defining a system message for our assistant. The system message initializes the assistant and guides its behavior. For example, we can write "system_message = {'role': 'system', 'content': 'You are a helpful assistant tasked with performing arithmetic on a set of inputs.'}". This system message ensures that the assistant knows its role and expected operations.

Next, we define the nodes for the workflow. The first node is the human feedback node, which initially can be empty since the human will provide input dynamically. We define it as "def human_feedback(state): return state". This function simply returns the state for now, and we will update it later with actual human input. The assistant node processes the input and optionally calls tools. It can be defined as "def assistant(state): return messages_with_tools.invoke_system_message + state". The tool node is already defined for arithmetic operations, such as multiplication.

After defining the nodes, we build the graph structure using a Builder and a MemorySaver for checkpoints. We can write "builder = Builder(state=graph_message_state)" to initialize the builder. We add nodes using "builder.add_node('human_feedback', human_feedback)", "builder.add_node('assistant', assistant)", and "builder.add_node('tools', tool_node)". Next, we define the edges to control the workflow: "builder.add_edge('start', 'human_feedback')", "builder.add_edge('human_feedback', 'assistant')". Conditional edges from the assistant allow branching: "builder.add_conditional_edge('assistant', 'tools', condition='is_tool_call')", "builder.add_edge('assistant', 'end')". Finally, the edge from the tool back to human feedback is "builder.add_edge('tools', 'human_feedback')". We then compile the graph with interruption before the human feedback node: "memory = MemorySaver()" and "graph = builder.compile(interrupt_before='human_feedback', checkpoint=memory)". Displaying the graph with "graph.display()" shows the full workflow from start to human feedback, assistant, tools, and end.

To execute the workflow, we first provide an initial message and a unique thread ID: "initial_message = 'Multiply 2 and 3'" and "thread_id = 5". We start streaming events with "for event in graph.stream(input=initial_message, thread_id=thread_id): event.messages[-1].PrettyPrint()". The workflow will immediately interrupt at the human feedback node, waiting for input. Here, we can collect human input using "user_input = input('Tell me how you want to update the state: ')". Once we receive the input, we update the human feedback node in the state using "graph.update_state(thread_id=thread_id, messages=[{'role': 'human', 'content': user_input}])". For example, the user might update the message to "Multiply 5 and 6" instead of "Multiply 2 and 3".

After updating the human message, we continue workflow execution with "for event in graph.stream(input=None, thread_id=thread_id, stream_mode='values'): event.messages[-1].PrettyPrint()". At this stage, the assistant node will process the updated human input. If a tool call is required, the assistant invokes the tool with the new arguments. The tool performs the computation (in this case, multiplication) and returns the result. The workflow may interrupt again if human feedback is needed after the tool output. Continuing with "input=None" allows the workflow to proceed to the end, producing the final output. For example, after multiplying 5 and 6, the tool returns 30, and the final message could be "The result of multiplying 5 and 6 is 30".

This workflow demonstrates several key points: the state snapshot records the current state of execution at every node, and graph.get_state(thread_id) can be used to inspect it. The checkpoints are saved in memory using MemorySaver, and graph.get_state_history(thread_id) can be used to inspect all previous states. We can edit human input dynamically, either before the assistant processes it or after a tool node completes, ensuring human oversight. Interrupts can be placed before or after any node, giving flexibility for designing complex human-in-the-loop flows. Frameworks like Streamlit or other UI frameworks can be used to collect user input interactively, making it more user-friendly than a simple console input.

Overall, this tutorial covers three main aspects of human-in-the-loop workflows: interrupts before nodes, editing human feedback dynamically, and waiting for human input before continuing execution. You can experiment by adding more nodes, conditional branches, or multiple tool calls. For example, you can place interrupts after the assistant, or add additional assistant nodes to process intermediate results. This approach ensures that humans are always in control of critical decisions while the workflow executes smoothly.

By combining graph.stream(), update_state(), get_state(), and get_state_history(), we can create robust, interactive workflows where human input is incorporated at any stage. This makes LineGraph a powerful tool for building intelligent systems that require human oversight or dynamic decision-making.

# **XIV) RAG with LangGraph**

**A) Agentic RAG Theoretical Understanding**

Hello guys, so we are going to continue the discussion with respect to our Agentic AI bootcamp with LangGraph series. In the previous video, we had already explored different types of workflows in LangGraph, understood how to properly apply human feedback, and also learned how to debug a LangGraph application with the help of LangGraph Studio. Now, in this particular video and in the upcoming series of videos, we are going to dive into different types of RAGs. The first RAG we are going to discuss is something called Agentic RAG.

Let’s start with the definition. Agentic RAG, which stands for Retrieval Augmented Generation, is a framework that enhances traditional RAG systems by incorporating intelligent agents. This is very important because by incorporating agents or intelligent assistance, we can handle complex tasks and make dynamic decisions within our generative AI workflows. The main aim of bringing agents into a generative AI application is not only to generate responses, but also to orchestrate decisions, manage workflows, and dynamically route queries.

Now, let’s first understand how a traditional RAG system works. Imagine we have a user who makes a query. Normally, in a generative AI application without RAG, this query goes to an LLM. Before it reaches the LLM, the query is usually incorporated with a prompt. The prompt guides the LLM about how to behave, while the query provides the actual user question. The LLM then produces a response. For example, in code, this might look like "response = llm.invoke({'prompt': 'Answer as politely as possible', 'query': user_query})".

But when we convert this into a traditional RAG system, we add a vector database alongside the LLM. This vector database stores external knowledge in embeddings—things like company policies, product manuals, or FAQs. So now, when a user asks a query like “What is the company holiday policy?”, the query is first passed into the vector database. The vector DB retrieves the most relevant chunks of information and returns them as context, which is then combined with the query and prompt, and only then sent to the LLM. The LLM uses this extra knowledge to generate a more accurate output. In pseudo-code, it looks like:
"context = vector_db.search(user_query); response = llm.invoke({'prompt': base_prompt, 'query': user_query, 'context': context})".

So, in summary, the traditional RAG uses the LLM only once—to generate the final output—while relying on the vector DB to fetch supporting context.

Now, let’s see how Agentic RAG is different. In Agentic RAG, the LLM is not just a passive text generator; it is converted into an agent that can dynamically decide how to act. For example, suppose you have multiple vector databases: one for company policies, another for legal documents, and another for product manuals. In a traditional RAG, routing queries to the correct DB is a manual process. But in Agentic RAG, the agent powered by the LLM decides which database to query, based on the user’s input.

Here’s where the power comes in. Suppose the user asks: “What is the company’s holiday policy?” The agent analyzes the query and routes it to the policy database. If the query is about legal compliance, it routes it to the legal documents database. If the query is completely unrelated, like “Who won the last World Cup?”, the agent follows a fail route and responds with something like “I don’t know the answer to that.” In code, this would look like:
"if 'policy' in user_query: context = policy_db.search(user_query); elif 'legal' in user_query: context = legal_db.search(user_query); else: context = None".
Then, the agent decides:
"if context: response = llm.invoke({'prompt': base_prompt, 'query': user_query, 'context': context}); else: response = 'I do not know the answer.'".

This decision-making ability is what transforms a traditional RAG into an Agentic RAG. The agent is not only retrieving relevant documents but also making choices about whether to retrieve, from where to retrieve, and how to proceed if no data is available. This dynamic routing is far more powerful than just blindly attaching context.

In fact, the LangGraph documentation itself shows diagrams where the agent decides whether or not to call a tool. If yes, it retrieves documents; if no, it directly generates an answer or follows a fail route. Sometimes the agent may even loop back, asking itself to retry retrieval until the right context is found, before producing the final answer.

To summarize the definition once again: Agentic RAG uses an agent to figure out how to retrieve the most relevant information before using it to answer the question. A retrieval agent is particularly useful when we need to decide whether to retrieve from an index, which index to use, and how to integrate the retrieved data. To implement this in LangGraph, we simply need to give the LLM access to a retrieval tool. For example:
"from langgraph.agents import create_retrieval_agent; agent = create_retrieval_agent(llm, tools=[policy_db_tool, legal_db_tool])".

With this setup, the LLM itself becomes capable of intelligently deciding whether it needs to call the retrieval tool, which database to query, and how to use that knowledge in generating the final response. If no relevant information is available, it can gracefully return: "I don’t know the answer to that.".

So, that’s the conceptual difference between a traditional RAG and an Agentic RAG. In the next video, we will do a hands-on practical implementation of Agentic RAG using LangGraph, where you’ll see how to build the agent, define retrieval tools, and orchestrate the workflow.

That’s it for this session, I hope you liked the explanation and found it clear. I’ll see you in the next video—thank you and take care.

**B) Agentic RAG Implementation- Part 1**

So, we are going to continue our discussion with respect to the Agentic RAG system. In the previous video, we had already understood the exact differences between a traditional RAG and an Agentic RAG. Now, in this specific video, we will actually be implementing a small project so that you can clearly understand how Agentic RAG works in practice.

Let’s first take a look at the workflow we are going to create. If you see the diagram, the entire thing is built using LangGraph. At the start, we have a node that represents the agent. From this agent, the flow is connected to a retrieve node. The retrieve node is linked to a vector database, which acts as the knowledge source. In our case, we will use two separate vector databases—one can be related to LangGraph blogs, and the other can be related to LangChain blogs. These could be backed by FAISS, Pinecone, Weaviate, or any other vector DB, but for our demo we’ll just show how this works conceptually.

The workflow is as follows: whenever the user provides a query, the agent will first check whether retrieval is needed. If yes, it looks into the vector DBs to see whether any relevant context exists. Once context is retrieved, the next step is to validate this context. If the retrieved context is good, we go ahead and use it to generate a summary or answer. If the context is bad, however, we don’t directly use it. Instead, we pass the query to another agent whose job is to rewrite the query into a more precise form, and then send it back to the main agent for a new retrieval attempt. This is why we say it is an Agentic RAG system—because agents here are not just answering questions, they are actively deciding how to retrieve, what to rewrite, and when to stop.

For example, let’s assume our two vector DBs store different content. One DB stores LangGraph blocks (DB1), and the other stores LangChain blocks (DB2). Now suppose the user asks, “What is LangGraph?” The agent has to decide which vector DB to query. Since the question is about LangGraph, it routes the query to DB1, retrieves the context, and checks whether it is relevant. If yes, it proceeds to generate a summary. In pseudo-code, this looks like "context = db1.search('What is LangGraph?')" followed by "response = llm.invoke({'query': 'What is LangGraph?', 'context': context})".

But suppose the retrieved context is not strong or precise enough. In that case, another node—our rewrite agent—kicks in. This agent takes the original question and rewrites it into something better aligned with the data. For example, instead of just “What is LangGraph?”, the rewritten query could become, “What are the important features of LangGraph as described in the official blogs?” This rewritten query is then sent back into the agent loop for a more targeted retrieval. This makes the system more adaptive than a traditional RAG.

Now, let’s start building this step by step in code. I’ve created a new folder named "06_rags_agentic_rag.ipynb". As usual, the first step is to import the required libraries. For this, we write:
"import os; from dotenv import load_dotenv; load_dotenv()".
Once executed, this loads our environment variables including API keys. For our demo, I’ll use the Grok API, though you could also use the OpenAI API.

Since we need two vector databases (one for LangGraph blogs and one for LangChain blogs), let’s create the retrievers. For that, we first load documents from websites. We can use LangChain’s WebBaseLoader to fetch blog content. The code looks like:
"from langchain_community.document_loaders import WebBaseLoader"
followed by something like "docs = [WebBaseLoader(url).load() for url in urls]".

These documents then need to be chunked before storing them as embeddings in a vector DB. For that, we use a text splitter:
"from langchain.text_splitter import RecursiveCharacterTextSplitter"
and then "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)".
Splitting is done via "doc_splits = text_splitter.split_documents(docs)".

Next, we store these chunks into a vector DB. For simplicity, let’s use FAISS:
"from langchain.vectorstores import FAISS"
and then build it as:
"vectorstore = FAISS.from_documents(doc_splits, embedding=OpenAIEmbeddings())".
Finally, we convert it into a retriever:
"retriever = vectorstore.as_retriever()".

To check, we can query the retriever directly with:
"retriever.invoke('What is LangGraph?')"
and it will return the top context documents with metadata.

But remember, just having a retriever is not enough. Since we are working with Agentic RAG, we need to integrate these retrievers into tools that the agent can call. LangChain provides "create_retriever_tool". For example:
"from langchain.tools.retriever import create_retriever_tool"
then:
"retriever_tool = create_retriever_tool(retriever, name='retriever_langgraph', description='Search and return info about LangGraph')".

Similarly, we repeat the process for LangChain blogs: fetch URLs, load documents, split into chunks, embed, store into another vector DB, create a retriever, and finally wrap it as a tool with "create_retriever_tool". This gives us something like:
"retriever_tool_langchain = create_retriever_tool(retriever2, name='retriever_langchain', description='Search and return info about LangChain')"

At this point, we have two retriever tools: one for LangGraph and one for LangChain. Together, these will allow the agent to dynamically decide where to route the query. For example, "tools = [retriever_tool_langgraph, retriever_tool_langchain]".

In the next step—which we’ll cover in the following video—we’ll bind these tools with the LLM so that the agent can make intelligent decisions about retrieval, rewriting, and summarization. For now, what we’ve achieved is the creation of two separate vector databases, converted them into retrievers, wrapped them into retriever tools, and made them ready to be integrated into an Agentic RAG workflow.

I hope this gave you a clear idea of how the foundation of our Agentic RAG is set up. In the next video, we’ll continue building the agent nodes—like the rewrite node and the summary generator—to complete the workflow. Thank you, and see you in the next one.

**C) Agentic RAG Implementation-Part 2**

So we are going to continue the discussion with respect to our Agentic RAG implementation. In the previous video, we had already seen how to implement vector databases using LangChain. We also created our retrieval tools. Now, if you look at the workflow diagram, what we’ve done so far is just this “tools” part. There are still many nodes to implement, and that is exactly what we’ll focus on in this video.

Our goal now is to start building the workflow using LangGraph. In short, we are going to create the entire Agentic RAG pipeline as shown in the diagram. We’ll start with the agent node, then gradually add nodes for rewrite, generate, and the decision-making components that connect them.

As you already know from earlier videos, when working with LangGraph the first thing we need is a state. This state allows us to store information that can be accessed by every node in the workflow. So let’s go ahead and create that. For this, we first import the necessary libraries:
"from typing import Annotated, Sequence"
"from typing_extensions import TypedDict"
"from langgraph.graph.message import BaseMessage, add_messages".

Here, "BaseMessage" is used to define the structure of our messages, while "add_messages" acts like a reducer, appending all the content into a single messages variable. Then we define our class:
"class AgentState(TypedDict): messages: Annotated[Sequence[BaseMessage], add_messages]".

This ensures that all messages—questions, responses, contexts—are stored inside a single state object, accessible by every node in the graph.

Next, let’s import and initialize our LLM. This time, instead of the older model, we’ll use the latest "grok" model powered by "qwen-32b", since the previous one was decommissioned. The import looks like:
"from langchain_groq import ChatGroq".
Then we set it up with:
"llm = ChatGroq(model='qwen-32b')"

To check whether the model is working fine, we can simply run:
"llm.invoke('Hi')"
and you should see an AI-generated response.

Now let’s start creating our first node: the agent node. In LangGraph, a node is just a Python function. So we define:
"def agent(state: AgentState) -> AgentState:".

This agent node is responsible for invoking the LLM with the current system prompt and deciding whether to use retrieval tools or not. Inside, we print "Call agent" for debugging, initialize the model, and then bind the retrieval tools. Binding is important because only this agent node needs direct access to the retrievers. We do this with:
"llm.bind_tools([retriever_tool_langgraph, retriever_tool_langchain])".

Once bound, the agent can decide where to route the query, fetch context if needed, and append the response back into the "messages" list. This makes our agent node functional.

But that’s not enough. After retrieval, we must also evaluate whether the retrieved documents are relevant or not. For this, we introduce the "grade_documents" function. Its purpose is to decide whether the retrieved docs should be passed to generate or rewrite.

Here we use Pydantic to define a structured output:
"class Grade(BaseModel): binary_score: str = Field(description='Relevance score: yes or no')"

This ensures that the model always returns either "yes" or "no". We then configure the model with:
"structured_llm = llm.with_structured_output(Grade)".

Next, we create the prompt:
"You are grading relevance of retrieved documents to the user question. If relevant, return 'yes'; otherwise return 'no'."

The input variables here are "context" (from the retrieved docs) and "question" (from the user). We attach this prompt to our structured LLM, run the chain with "chain.invoke({'context': docs, 'question': question})", and receive a binary score. If the score is "yes", we proceed to generate. If "no", we proceed to rewrite.

Now let’s define the generate node. This node simply takes the relevant documents and creates a final answer. The function looks like:
"def generate(state: AgentState) -> AgentState:".
Inside, we use a prompt template such as:
"Given the following context: {context}, answer the user’s question: {question}".
We then run it with "llm.invoke({...})", parse the output, and append the final answer into the messages.

But what if the retrieved docs were not relevant? That’s where the rewrite node comes in. The rewrite node reformulates the user query into a better, more precise version. The code looks like:
"def rewrite(state: AgentState) -> AgentState:".
Here, we instruct the LLM with:
"Look at the input and reformulate it into a clearer, more specific question."
The new query is generated and then appended back into the messages so the agent can retry retrieval.

With these nodes ready, we can now assemble the full workflow using LangGraph. We initialize the graph with:
"from langgraph.graph import StateGraph"
"graph = StateGraph(AgentState)".

Then we add our nodes:
"graph.add_node('agent', agent)"
"graph.add_node('retrieve', retrieve)"
"graph.add_node('generate', generate)"
"graph.add_node('rewrite', rewrite)".

We connect the edges:

From start → agent

From agent → retrieve (if tool is called) or agent → end

From retrieve → grade_documents

From grade_documents → generate (if relevant) or grade_documents → rewrite (if not)

From rewrite → agent (looping back with the improved query)

Once the graph is built, we compile it with "app = graph.compile()". To run, we call:
"app.invoke({'messages': [HumanMessage(content='What is LangGraph?')]})".

When executed, you’ll see the flow clearly: the agent calls retrieval, the documents are graded, relevance is checked, and if relevant, the answer is generated. For example, asking "What is LangGraph?" routes to the LangGraph vector DB, marks the docs as relevant, and goes straight to generate. Asking "What is LangChain?" does the same but with the LangChain DB. And if you ask something unrelated, like "What is Machine Learning?", the agent directly answers without hitting the retrievers.

And just like that, we’ve built a complete Agentic RAG system with LangGraph. Step by step, we created the state, nodes, grading logic, generate and rewrite functionality, and finally stitched everything together into a graph workflow.

Now you can easily extend this by adding more retrievers, different grading strategies, or even custom nodes depending on your use case. I hope this session made things clear and practical. See you in the next video.

**D) Corrective RAG Theoretical Understanding**

So, we are going to continue our discussion with respect to the different types of RAG. In this specific video, we will talk about something called Corrective RAG, also known as CRAG. We’ll try to understand how Corrective RAG actually works, and why it is such an amazing and efficient technique for improving the accuracy of retrieval-augmented generation systems. Once we understand the theoretical flow, in the next video we will go ahead and implement this with the help of LangGraph.

So first of all, let’s begin with the definition. Corrective RAG (CRAG) is an advanced technique within Retrieval-Augmented Generation that focuses on improving both the accuracy and the relevance of generated responses. It does this by incorporating mechanisms for self-reflection and self-grading of the retrieved documents. In other words, instead of blindly trusting whatever the retriever brings back from the vector database, CRAG actually evaluates the quality of those documents and then applies corrective actions when necessary.

Now, let’s break this down a little more. At its core, Corrective RAG is still a RAG system—you still have a retriever pulling content from a vector database, and you still have an LLM generating answers. But the big difference is that CRAG introduces two additional capabilities: self-reflection and self-grading. These allow the system to ask itself: “Are these retrieved documents actually relevant to the user’s query?” and, if not, “What corrective action should I take to fix this?”

Let’s walk through the flow to see how this works. Imagine a user asks a question. The query first goes to the retriever node, which fetches documents from the vector database. Once those documents are retrieved, we don’t immediately send them to the LLM. Instead, we first grade them. This grading step is essentially evaluating the retrieved documents. For example, in code you might have something like:
"result = grader.invoke({'question': user_query, 'docs': retrieved_docs})".
The grader then gives a binary decision: either "yes" (the docs are relevant) or "no" (the docs are irrelevant).

Now, let’s consider both cases. If the grader says yes, the documents are passed to the LLM as usual. The LLM combines the query, the prompt, and the context, and generates the final response. For example:
"response = llm.invoke({'prompt': base_prompt, 'query': user_query, 'context': docs})".

But if the grader says no, meaning the documents are irrelevant or not useful, then we trigger a corrective action. This is where CRAG becomes powerful. One common corrective action is to rewrite the user query into a clearer or more specific form. For example:
"new_query = llm.invoke({'prompt': 'Rewrite this unclear query into a precise one', 'query': user_query})".
In addition to rewriting the query, CRAG can also initiate a web search to bring in external information beyond what is available in the local vector database. For example:
"web_results = web_search_tool.invoke(new_query)".
This allows the system to still provide a meaningful answer even when the internal knowledge base doesn’t have the required information.

So essentially, the corrective action here is twofold: rewrite the query and expand retrieval sources (like web search) when the vector DB alone cannot provide relevant results. This is exactly what we mean by self-reflection and self-grading: the system reflects on the quality of its own retrievals, grades them, and if they are lacking, it takes corrective measures.

Now, let’s also connect this to the bigger picture. Traditional RAG systems rely heavily on the assumption that the retrieved documents are accurate. But if the retriever brings back flawed or incomplete information, the generated answer is also flawed. Corrective RAG addresses this limitation by introducing retriever evaluation (grading) and refinement/correction (rewriting or searching again). This makes the generated responses more accurate, more relevant, and far more robust.

The core components of CRAG can be summarized as:

Retriever – fetches documents from the vector DB.

Evaluator (grader) – checks whether the documents are relevant.

Generative model (LLM) – produces the final response when relevant context is available.

Refinement/Correction module – takes corrective action (like query rewriting or web search) when documents are not relevant.

Some of the benefits of CRAG are very clear. First, you get improved accuracy, since the system is actively correcting irrelevant or low-quality retrievals. Second, you get better relevance, because irrelevant documents are filtered out before generating a response. And third, you get increased robustness, since even if your retriever fails, the corrective mechanisms like query rewriting and web search still allow the system to answer effectively.

So to summarize: Corrective RAG (CRAG) is all about making your retrieval-augmented generation pipeline smarter and more self-aware. It uses self-reflection and self-grading to evaluate retrievals, and applies corrective actions like rewriting queries and using web search when necessary. In the next video, we’ll take this flow diagram and actually implement the whole solution in LangGraph, just like we did for Agentic RAG. That means defining the retriever, adding the grader node, and implementing refinement and correction. For the web search part, we can even connect external APIs to make it more realistic.

I hope you liked this explanation of Corrective RAG. In the next session, we’ll dive into the hands-on coding part. Until then, thank you and take care.

**E) Corrective RAG Practical Implementation**

Hello guys, in this video we are going to continue the discussion on Corrective RAG. The idea here is to implement the entire workflow graph that handles retrieval, grading, corrective actions like query rewriting, and finally generation. Essentially, we’ll see how to create the retriever node, the grade node, how to perform grading, and then implement the corrective actions. These corrective actions—self-reflection and self-grading—form the refinement and correction stage of our workflow.

On the right side of the diagram is the graph we are going to implement. On the left side, you can see the steps: first, we will create the retriever node. Then, we add the grade documents node. This is connected by a conditional edge to check whether the documents are relevant. If the retrieved documents are not relevant, we transform the query, perform a web search, and finally generate the response. If they are relevant, we can directly generate the output by combining the retrieved context with the LLM and system prompt.

Step 1: Creating the Retriever

First, we import the required libraries and set up environment variables such as the OpenAI API key. You can also use Groq if you prefer, but OpenAI is more stable for this kind of use case.

We then build the index by loading data using loaders such as WebBaseLoader or RecursiveCharacterTextSplitter. For embeddings, we use OpenAI embeddings. For example:
“`python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

urls = [
"https://example.com/prompt_engineering
",
"https://example.com/adversarial_llms
",
"https://example.com/agents
"
]

loader = WebBaseLoader(urls)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()
“`

This completes our retriever node.

Step 2: Creating the Grader

Now, let’s move to the grader functionality. The goal is to check if retrieved documents are relevant to the user’s question. For structured outputs, we use Pydantic models:
“`python
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
binary_score: str = Field(description="Relevance of documents: Yes or No")
“`

We then initialize our LLM with structured output:
“`python
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_chain

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
grader_chain = create_structured_output_chain(GradeDocuments, llm)
“`

And we add a system prompt to guide the grading:
“python system_prompt = """ You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword or semantic meaning related to the question, grade it as Yes. Otherwise, grade it as No. """ “

This means whenever we pass a query and the retrieved docs into this chain, the output will be Yes/No.

Step 3: Generation Node

If documents are relevant, we move directly to the generate node. Here we use a predefined RAG prompt and LLM chain:
“`python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Answer the question based on the context:\n\n{context}\n\nQ: {question}\nA:")
generate_chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
“`

When invoked with context and question, this chain generates the final response.

Step 4: Query Rewriter

If the grader says the docs are not relevant, we add a query rewriting step. For this, we again use an LLM and prompt:
“python rewrite_prompt = PromptTemplate.from_template("Rewrite the user question to improve retrieval. Original: {question}") rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt, output_parser=StrOutputParser()) “

For example, if the input was “agent memory”, the output may become “What is the role of memory in artificial intelligence agents?”.

Step 5: Web Search Node

After rewriting, we can add a web search node. This can be done using community tools such as Tavily Search:
“python from langchain_community.tools import TavilySearchResults web_search = TavilySearchResults(k=3) “

This will fetch the top three results for the rewritten query, which can then be passed back into the generator.

Step 6: Building the Graph

Now that we have all the building blocks—retriever, grader, generator, rewriter, and web search—we combine them into a state graph. Each function is defined as a node, and edges connect them. For example:
“`python
from langgraph.graph import StateGraph

graph = StateGraph()

graph.add_node("retriever", lambda q: retriever.invoke(q))
graph.add_node("grader", lambda docs, q: grader_chain.invoke({"documents": docs, "question": q}))
graph.add_node("generator", lambda docs, q: generate_chain.invoke({"context": docs, "question": q}))
graph.add_node("rewriter", lambda q: rewrite_chain.invoke({"question": q}))
graph.add_node("web_search", lambda q: web_search.run(q))
“`

Edges are then defined. From retriever → grader, from grader → generator (if relevant), or grader → rewriter → web_search → generator (if not relevant). Finally, the generator → end.

Step 7: Execution

When we invoke the graph with a question like “What are the types of agent memory?”, here’s what happens step by step:

Retriever fetches documents.

Grader checks relevance → If relevant, directly generate.

If not relevant, rewrite query → web search → generate.

And in practice, you will see logs like:

“Graded documents: Not relevant”

“Rewritten query: What is the role of memory in AI agents?”

“Performing web search…”

“Generated Answer: …”

This way, we ensure that even if the retriever fails, the system can self-correct and still produce accurate answers.

So that’s the end-to-end implementation of Corrective RAG. The main thing to remember is the workflow: Retrieve → Grade → Conditional Edge (Relevant/Not Relevant) → Generate or Rewrite+Search → Generate. By implementing this self-grading and correction loop, we can achieve more reliable RAG systems.

**F) Adaptive RAG Theoretical Understanding**

Hello guys, in this video we are going to continue our discussion on different types of Retrieval Augmented Generation (RAG).

In the previous video, we implemented something called Agentic RAG. We understood how Agentic RAG works, took a concrete example, and walked through the workflow step by step.

Now, in this specific video, and in the upcoming ones, we are going to focus on Adaptive RAG. I’ve included multiple diagrams to make sure the theoretical intuition is clear before we move on to the practical implementation. The workflow we are going to implement looks a bit complex, but we will break it down step by step.

What is Adaptive RAG?

Let’s start with the definition. Adaptive RAG, or Adaptive Retrieval Augmented Generation, is a framework that dynamically adjusts its strategy for handling queries based on their complexity.

This is a very important point—Adaptive RAG is all about adapting the retrieval strategy depending on whether the query is simple, moderately complex, or highly complex.

Think of it like a smart assistant that knows when to give a quick, straightforward answer, and when to dig deeper into external sources, databases, or multi-step reasoning. Instead of a rigid one-size-fits-all approach, Adaptive RAG chooses the most appropriate retrieval method for each query, balancing speed and accuracy.

Now, if you recall Agentic RAG, in that case it was the agent that decided which route to take. But here, there’s no explicit agent making that decision. Instead, Adaptive RAG uses a classifier or query analysis module that dynamically adjusts the strategy.

So, in Adaptive RAG there are two important steps:

Query Analysis

RAG + Self-Reflection (also called Self-Corrective RAG)

We’ll understand both of these one by one.

Step 1: Query Analysis

The first stage is query analysis. Looking at the diagram, you can see that whenever a question comes in, it first passes through the query analysis step.

The role of query analysis is simple: it determines the complexity of the query and then routes it to the most suitable retrieval method. For example:

If the query is very simple, we might just pass it directly to the LLM and get an answer.

If the query is moderately complex, we might need a web search to bring in relevant, fresh information.

If the query is highly complex or very domain-specific, then we route it to the retriever connected to a vector database containing internal knowledge.

So query analysis is essentially a routing mechanism. It classifies the query, decides its complexity, and then forwards it to the right path.

Step 2: RAG + Self-Reflection (Self-Corrective RAG)

Now let’s talk about the second stage, which is RAG with self-reflection. This is also known as self-corrective RAG.

Imagine we have a complex query that requires internal company knowledge. For example:

“What are the policies for company XYZ in India?”

In this case, query analysis realizes that a web search alone won’t work, since this information may not be fully available on the internet. Instead, the query is routed to the retriever.

The retriever is connected to a vector database, which contains indexed documents about the company. But here’s the interesting part: if the initial retrieval doesn’t give sufficient results, Adaptive RAG can rewrite the query and try again.

This process of rewriting, regenerating, and verifying is what we call self-reflection or self-correction.

Here’s how it works step by step:

Query goes to the retriever.

Retrieved documents are graded for relevance.

If documents are relevant → proceed to generation.

If documents are not relevant → rewrite the query and try again.

After generation, we can also check for hallucination. If the answer is inaccurate, the flow loops back to the retriever for refinement.

This loop of retrieval → grading → rewriting → generation is the heart of self-corrective RAG.

Examples of Query Analysis in Action

Let me give you three quick examples to make this clearer:

Simple Query: “What is the capital of India?”
→ Query analysis marks it as simple.
→ Directly answered by the LLM.

Moderate Query: “Talk about the economics of India.”
→ Query analysis marks it as complex.
→ Routes to web search for updated economic data.

Highly Complex Query: “What are the internal policies of company XYZ in India?”
→ Query analysis marks it as very complex.
→ Routes to the retriever with vector database.
→ If initial retrieval isn’t enough, query rewriting and self-correction kicks in.

This demonstrates how Adaptive RAG tailors the strategy based on query type.

Implementation Workflow

Now, let’s connect this theory to the workflow diagram.

Input query enters.

Query classifier decides whether it should go to web search, retriever, or directly to the LLM.

If routed to web search → generate the answer.

If the answer is good enough → end.

If not useful → transform the query → send to retriever.

If routed to retriever → retrieve documents → grade them.

If relevant → generate and end.

If not relevant → rewrite query → retrieve again → repeat until we get meaningful content.

This flow shows how query analysis and self-reflection combine to form Adaptive RAG.

Summary

So to summarize:

In Agentic RAG, an agent made the decision on which route to take.

In Adaptive RAG, we use a classifier to analyze the query and route it based on complexity.

For simple queries → direct answer.

For moderately complex queries → web search.

For highly complex/domain-specific queries → retriever + self-reflection loop.

In the next video, we will go step by step through the implementation of this Adaptive RAG workflow in code, building the classifier, connecting the retriever, adding self-reflection, and running through examples.

I hope you found this explanation helpful. I’ll see you in the next video. Thank you, and take care.

**G) Adaptive RAG Implementation**

Hello guys. In this video, we are going to continue our discussion on Adaptive RAG, focusing specifically on the implementation. In the rags folder, I have created a second file called AdaptiveRAG.py. As we discussed in theory, the workflow we are going to implement includes Query Analysis and RAG with Self-Reflection. Based on this workflow, there are multiple nodes: web search, vector store retriever, generate node, transform query node, and grade documents node. We will implement these nodes step by step and then connect them using a state graph.

First, we need to import the required libraries. You can load environment variables using dotenv, set your OpenAI API key, and any other API keys as required, such as Tabulae for web search. For this example, we will mainly use OpenAI, as it efficiently handles the routing classifier and query analysis:

"from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')"

Next, we will create a retriever vector store similar to what we did in Agentic RAG. We start by loading documents from URLs, splitting them into chunks, and embedding them as vectors. This allows the retriever to efficiently fetch relevant documents:

"from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

urls = ['https://example.com/agents
', 'https://example.com/policies
']
docs = []
for url in urls:
 loader = WebBaseLoader(url)
 docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()"

With the retriever ready, we can now implement Query Analysis, which decides whether a query should go to the web search or the vector store retriever. We create a Pydantic data model to validate this routing:

"from pydantic import BaseModel, Field
from typing import Literal

class RootQuery(BaseModel):
 data_source: Literal['vector', 'websearch'] = Field(description='Route the query to the appropriate data source')"

We then define a structured LLM router using OpenAI:

"from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

router_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
route_prompt = ChatPromptTemplate.from_messages([
 ('system', 'You are an expert router. Classify whether a query should go to vector store or web search.'),
 ('human', '{question}')
])
parser = PydanticOutputParser(pydantic_object=RootQuery)"

Now, when we invoke the question router:

"question_router.invoke('Who won the Cricket World Cup 2023?')"

the router correctly decides websearch, while for “What are the types of agent memory?” it routes to vector. This completes the query analysis node, allowing us to dynamically route queries based on the content of our retriever.

Next, we implement Retrieve + Grade Documents. Here, after the retriever fetches documents, we grade their relevance using another structured LLM output. The grader returns a binary score yes or no:

"grader_prompt = ChatPromptTemplate.from_messages([
 ('system', 'You are a grader. Assess whether the retrieved document is relevant to the user question.'),
 ('human', 'Question: {question}\nDocument: {doc}')
])

grader_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
parser = PydanticOutputParser(pydantic_object=GradeDocs)"

If the document is relevant (yes), we move to the generate node. This node produces the answer using the context from retrieved documents:

"gen_prompt = ChatPromptTemplate.from_messages([
 ('system', 'You are a helpful assistant. Answer strictly based on the provided context.'),
 ('human', 'Question: {question}\nContext: {docs}')
])

generator = ChatOpenAI(model='gpt-4o-mini', temperature=0)
response = generator.invoke({'question': 'What is agent memory?', 'docs': retrieved_docs})"

We also implement a hallucination check, which verifies whether the generated answer is grounded in the retrieved documents. The output is another binary score yes or no. If the output is no, the query is passed to the question rewriter node. This node rewrites the question for better vector retrieval:

"rewrite_prompt = ChatPromptTemplate.from_messages([
 ('system', 'Rewrite the user question to optimize vector search retrieval.'),
 ('human', '{question}')
])

rewritten_question = router_llm.invoke({'question': 'What is agent memory?'})"

Finally, we implement the web search node, which uses an API to fetch information when the retriever does not contain the data. Once the web search result is obtained, it passes through the same generate and hallucination check flow.

All of these nodes are then connected using a state graph. The root query is routed either to web search or retriever. Web search flows to generate, while the retriever flows to grade documents. Based on the grader’s output, the state graph decides whether to generate, transform the query, or rewrite. After generation, we also optionally run hallucination checks and then provide the final answer.

When executed, the workflow dynamically decides the path:

Query: “What is machine learning?” → not in retriever → goes to web search → generate → hallucination check → final answer.

Query: “What is agent memory?” → present in retriever → grade documents → generate → hallucination check → final answer.

This demonstrates the Adaptive RAG workflow in practice, combining query analysis, retrieval, self-reflection, and generation in a modular, node-based system. Each node uses the LLM in some way, ensuring both flexibility and accuracy.

I hope this example helps you understand how to implement Adaptive RAG end-to-end. In the next video, we will explore additional optimizations and real-world deployments of this system. Thank you, and take care.

# **XV) End To End Agentic AI Projects With LangGraph**

**A) Introduction And Overview**

Hello guys. In this video, we are going to continue our discussion with respect to LangChain. Over the previous videos, we have covered six main sections—from LangChain basics, where we learned how to use Pydantic, create chat chains, and build chatbots with multiple tools, to React agents, workflows, and human-in-the-loop integrations. We also explored different types of RAG architectures. Most of the implementations we saw were done in Jupyter Notebooks, which is great for learning and experimentation.

Now, the next step is to understand how to develop an end-to-end project. In this module, we will build a complete industry-standard project, focusing on modular coding, deployment, and creating scalable solutions. We will write each line of code step by step to demonstrate how to translate the concepts we have learned into a production-ready system. This includes creating state graphs, structuring code into classes, and ensuring everything is modular for maintainability.

The product we are going to develop is hosted on Hugging Face, and it includes three major functionalities. First, a basic chatbot; second, a chatbot with tools; and third, an AI news assistant. Each of these applications will be developed as a separate module, which allows us to maintain clean, scalable code.

To show how the project works, let me demonstrate the basic chatbot workflow. Here, we are using Grok LLM, powered by Llama 3, with an API key configured:

"from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='llama-3', api_key=OPENAI_API_KEY, temperature=0.7)

response = llm.invoke('Hi, my name is Chris.')
print(response)"

When I type, “Hi, my name is Chris”, the chatbot responds: “Hey Chris! Nice to meet you. How are you doing today?” This demonstrates the stateful interaction that allows the bot to maintain context.

Next, we have the chatbot with tools, which can perform specific tasks like retrieving Python code, accessing web information, or performing calculations. For example, if I ask: “Provide me Python code to play Snake game”, the system retrieves the relevant response using either a retriever module or web search, and then generates a solution using the LLM:

"from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
 ('system', 'You are a helpful assistant.'),
 ('human', '{question}')
])

snake_chain = LLMChain(llm=llm, prompt=prompt)
response = snake_chain.run({'question': 'Provide Python code to play Snake game'})
print(response)"

What makes this project particularly exciting is that it reflects real-world industry standards. In companies, projects are implemented modularly, using stateful graphs, class-based structures, and end-to-end deployment pipelines. Our system also uses Streamlit for the frontend, allowing users to interact with the chatbot seamlessly. Future extensions will include FastAPI-based endpoints, enabling Postman testing and integration with other services.

We will structure this project as follows:

Modular Nodes: Each functionality (basic chatbot, chatbot with tools, AI news assistant) is implemented as a separate module or class.

State Graph: Interactions flow through a stateful graph, tracking queries, responses, and context.

Deployment: The final application will be hosted on Hugging Face or similar platforms.

Extensibility: Additional use cases can be added without disrupting the core architecture.

In the next video, we will set up VSCode, configure our environment, and begin writing modular code for this project. We will ensure that every step is clear so that anyone can understand how to implement a production-ready LangChain project from scratch.

This concludes the introduction to our end-to-end project. I hope you are excited to build this, and I will see you in the next video. Thank you, and take care.

**B) Project Set Up With VS Code**

Hello guys. In this video, we are going to continue our discussion regarding our end-to-end LangChain project. As mentioned earlier, we will start by building a basic chatbot, but this time we will focus on modular coding, ensuring that the project is structured in a scalable and maintainable way. As we progress, we will gradually increase the complexity of the projects, moving from a basic chatbot to a chatbot with tools, and finally to an application that can fetch recent AI news. Later, we will extend these applications using FastAPI and other modern frameworks.

Before we start coding, the first step is to create a dedicated project space where all files and modules for this chatbot project will reside. In my case, I have created a folder named genetic_chatbot to organize this project. You can create a folder anywhere on your system according to your preferences. Once the folder is created, we will open it in VS Code to continue with the implementation.

The next step is to set up a virtual environment, which is a best practice for any Python project to isolate dependencies. For this, we will use Conda. The command to create a virtual environment named venv with Python 3.13 is:

"conda create -p venv python=3.13"

After running this command, you will be prompted to confirm the installation of required packages. Type y and press Enter. Once the environment is created, activate it using:

"conda activate venv"

Alternatively, you can use venv or virtualenv directly if you prefer, and we will explore that in future applications.

After activating the environment, we need to create a requirements.txt file, which lists all the libraries required for this project. For this project, the libraries include:

langchain

langgraph

langchain-community

langchain-core

langchain-grok (open-source LLM)

langchain-openai (optional, if needed)

faiss-cpu (for vector search)

streamlit

Python

tabulate (for web search integration)

Once the requirements.txt file is ready, install all dependencies with the command:

"pip install -r requirements.txt"

While the installation is in progress, we also need to set up GitHub to manage our project versions and enable collaboration. We will create a .gitignore file to exclude unnecessary files and folders, such as the venv environment, from being committed:

"venv/"

Additionally, create a README.md file to provide basic information about the project. For example, the README can contain:

"End-to-end project: Agentic AI Chatbot"

This ensures that anyone looking at the repository understands the purpose of the project. Once the virtual environment and libraries are installed, we can move on to GitHub setup. We will create a new repository, initialize it locally, and commit all the project files, excluding the virtual environment, to maintain a clean repository.

The reason for committing the code to GitHub is to prepare for deployment. At the end of the project, we will demonstrate continuous integration and continuous deployment (CI/CD) pipelines to deploy the chatbot seamlessly.

In the next video, we will continue with the GitHub repository setup, initialize it, and commit all the code from this project location. Once this is done, we will be ready to start building the basic modular chatbot using LangChain, which will form the foundation for more advanced applications.

Thank you, and I will see you in the next video.

**C) Setting up The Github Repository**

Hello guys. In this video, we are going to continue our end-to-end project implementation by setting up our GitHub repository. Before we proceed, it is important to have some basic knowledge of Git, as this will be essential for tracking changes, collaborating, and deploying projects efficiently.

First, go to github.com and navigate to your repositories. Click on New Repository, and provide a project name—for this example, we will use AgenticChatbot. You can choose to make the repository public or private; in this case, we will make it private for personal access. After providing the necessary details, click Create Repository. GitHub will then provide commands to connect your local project folder to this repository.

Next, open your command prompt and navigate to your project folder. The first step is to initialize a local Git repository using:

"git init"

This command initializes a Git repository in the current folder, allowing you to track all files. Your .gitignore file, which we previously created, ensures that certain files and folders—like the virtual environment venv—are not tracked. This is important because venv is local to your system and should not be committed.

Once the repository is initialized, you can add files for tracking. For example, to add the README file:

"git add README.md"

Or to add all files:

"git add ."

The A symbol in Git indicates that files have been added to the staging area. Once files are added, you can commit them to the local repository with a message:

"git commit -m 'First commit'"

Git now tracks these files, and any subsequent changes will appear with an M (modified) indicator. For instance, if you make changes to a file, Git will mark it as modified until you add and commit the changes again.

After committing, create a main branch to align with GitHub’s default structure:

"git branch -M main"

Then, link your local repository to the GitHub repository by adding the remote origin:

"git remote add origin <repository-URL>"

Finally, push your changes to GitHub using:

"git push -u origin main"

Once this is done, all your files—including README.md, requirements.txt, and .gitignore—will be visible in your GitHub repository. Your virtual environment is not tracked due to the .gitignore settings, which keeps the repository clean and lightweight.

Whenever you make changes to files, you can repeat the workflow:

Modify files locally.

Add changes to staging: "git add ."

Commit with a descriptive message: "git commit -m 'Second commit'"

Push changes to GitHub: "git push origin main"

The symbols A and M in Git help you track which files are added or modified, making it easy to manage your project. This GitHub setup is fundamental in industry-grade projects, as it allows for version control, collaboration, and deployment. In future steps, we will also integrate CI/CD pipelines to automate deployment.

In the next video, we will start developing our use cases, following a modular structure of programming, and writing code step by step. By the end of this project, you will have a complete end-to-end Agentic AI chatbot ready for deployment.

Thank you, and I will see you in the next video.

**D) Setting Up The Project Structure**

Hello guys. In this video, we are going to continue our end-to-end Agentic AI project with LangGraph. In the previous video, we completed the GitHub repository setup, committed our code, and created a virtual environment. Now, it’s time to set up the project structure, which is one of the most important steps in any end-to-end project.

The structure of your project depends on the type of application you are building. For this Agentic AI project, we will follow a modular architecture that allows us to scale the code, maintain readability, and facilitate deployment. To explain this better, I have drawn a block diagram showing the different components.

For any end-to-end project, there are typically three main components:

Front-end – where the user interacts with the system.

Workflow/Engine – where the core logic is implemented using LangGraph.

Supporting components – which handle nodes, state, models (LMS), and other auxiliary functions.

Looking at the front-end, we have options for selecting the LMS, models, API keys, and use cases. For our project, the main use cases are:

Basic Chatbot

Chatbot with Tools

AI News

The front-end allows users to select any of these use cases, and the workflow should execute accordingly. We will use Streamlit for the front-end development. The front-end also triggers the workflow, which is not like a Jupyter Notebook; here, everything must run as a pipeline.

The workflow consists of nodes and edges, where nodes define individual functionalities and edges define the execution flow, including conditional paths. Each node will be implemented as an independent component, allowing modularity and reusability. The execution starts from the front-end and flows through the workflow until completion.

Now, let’s discuss the project folder structure. First, we will create a main folder called src, which will contain all core functionalities. Inside src, we will have:

LandGraphAgenticAI folder – contains all the main modules and can be treated as a package using __init__.py.

graph/ – handles the workflow graph.

lms/ – contains logic for different LMS models like Grok, LLaMA, or OpenAI.

nodes/ – individual node functionalities of the workflow.

state/ – maintains the global state accessible to all nodes.

tools/ – optional external tools for integration.

ui/ – front-end components (Streamlit interface).

main.py – calls all components step by step within the package.

Outside the package folder, we will have an app.py file, which will be the entry point of the application. This file will trigger the front-end UI and the entire workflow pipeline.

By following this structure, each component remains modular and independent, making it easier to develop, test, and deploy. This approach also aligns with industry standards for large-scale AI projects.

The next step is to commit this structure to GitHub. I have already done a commit for reference, but I encourage you to try it yourself as an assignment. Make sure your __init__.py files are in place so that all folders are treated as packages, enabling easy imports and potential packaging for PyPI in the future.

In the next video, we will start developing each component, beginning with the UI, followed by nodes, workflow graph, LMS integration, and state management. Step by step, we will build the entire Agentic AI chatbot project.

Thank you, and I will see you in the next video.

**E) Designing The Front End Using streamlit**

In this video, we continue the development of our end-to-end Agentic AI project with Landgraf by focusing on building the front-end UI. We begin by creating a modular front-end using Streamlit, emphasizing component-to-component implementation for clarity and maintainability. The front end consists of a sidebar with fields for selecting LMS models, APIs, and use cases, along with a main messaging area for user interaction. All front-end development is organized within the UI folder, where we create a Streamlit subfolder containing three main files: load_ui.py for loading the UI, display_result.py for displaying results, and ui_config_file.ini to store constants such as model names, LMS options, and use case options.

Next, we implement the configuration handler using Python’s configparser module. A dedicated class Config is created inside ui_config_file.py to read the .ini file and provide methods for retrieving constants such as LMS options, use case options, Grok model options, and page title. This modular approach ensures that all constants are centralized, easily maintainable, and can be retrieved dynamically in the UI code. Methods like get_lm_options() and get_use_case_options() are used to return lists of values from the configuration, allowing the front-end code to dynamically populate dropdowns and other UI elements.

Once the configuration handler is ready, we move on to designing the Streamlit UI itself in load_ui.py. A class LoadStreamlitUI is created, which loads the configuration and initializes user controls as a dictionary. The sidebar is constructed using Streamlit’s selectbox and text_input widgets, allowing users to select LMS models, choose use cases, and enter API keys. Conditional logic is applied so that, for example, selecting Grok dynamically populates the available models and provides an input box for the API key. This ensures a seamless and interactive front-end experience.

Finally, we integrate the UI with the main application flow. In main.py, we import the LoadStreamlitUI class and define a function load_landgraf_agentic_ai_app() to initiate the Streamlit UI, load sidebar options, and handle user input. This function is then called from app.py, serving as the entry point for the application. By running streamlit run app.py, the full front-end UI is launched, displaying a basic chatbot interface with configurable LMS models and use cases. This modular design not only simplifies front-end development but also ensures that the backend workflow can be seamlessly triggered from user selections, setting the stage for implementing the actual AI use cases in the next steps.

**F)  Implementing The LLM Module In Graph Builder**

In this video, we continue the development of our end-to-end project by focusing on implementing the workflow that triggers whenever a user sends a message. We begin by revisiting the previously developed Streamlit front end, where the sidebar integrates constants such as LMS options, available models like Grok, and use cases such as the basic chatbot. The main chat input is handled in main.py, while configuration constants are managed in ui_config_file.ini and read through ui_config.py. The front-end structure is now fully set up, and the next step is to ensure that sending a message triggers the workflow, with all necessary components loaded dynamically.

To facilitate this, we start by loading all required LMS models. A dedicated file lm.py is created in the LM folder to handle model initialization. This file reads API keys provided through the front end via Streamlit’s session_state, specifically the user_controls dictionary. The goal is to write generic code so that, when a user provides an API key, the respective LMS model—such as Grok—can be loaded automatically when the workflow executes. By structuring LMS initialization this way, it becomes easy to add additional models like OpenAI or Google Gemini in the future.

The LMS loading logic is encapsulated in a GrokLM class. This class receives user control inputs from the front end and stores them as a class variable for use across its methods. The get_model() method is responsible for loading the selected LMS model based on the user-provided API key and model selection. Error handling ensures that if no API key is provided or the environment is not set, a Streamlit error is raised instructing the user to provide the key. Upon successful initialization, the model instance is returned, ready for use in the workflow. This modular and extensible design allows other LMS models to be added easily by creating separate files following the same pattern.

Finally, this step sets the stage for building the workflow graph, where independent nodes and functionalities will be constructed in subsequent development. By preloading LMS models and integrating user inputs from the front end, the system ensures that workflows can be executed seamlessly whenever a message is entered. This modular approach not only simplifies model management but also provides a scalable foundation for supporting multiple LMS platforms in the future, paving the way for building the complete AI-driven workflow in the next video.

**G) Implementing The Graph Builder Module**

In this video, we continue our end-to-end project implementation, now focusing on building the workflow for a basic chatbot using LangGraph. After completing the LM module in the previous video, the next step is to design the workflow itself. We start by conceptualizing a simple chatbot graph consisting of three stages: start, chatbot, and end. This minimal setup demonstrates the fundamental structure of a chatbot workflow, which can later be extended to more complex use cases. The focus here is to establish the basic flow and graph framework before diving into specific node functionalities.

The workflow implementation begins with the creation of a graph_builder.py file, where a GraphBuilder class is defined. The class is initialized with the loaded LM model and sets up a graph_builder object to manage the workflow graph. For graph management, we use the StateGraph library and define a shared state structure in state/state.py. This state class uses type annotations and reducers to maintain a list of messages within the graph. By organizing the state separately, we ensure that all nodes in the graph can access and update shared information consistently.

Next, the basic chatbot workflow is constructed by defining nodes and edges in the graph. The nodes include start, chatbot, and end, and edges are added to connect them sequentially. The chatbot node itself is modular and its functionality is implemented separately in the nodes folder. Specifically, a file named basic_chatbot_node.py is reserved for the chatbot’s logic, which will handle taking user input and generating responses using the loaded LM. This modular structure allows additional nodes or chatbot variations to be easily added later, following the same design pattern.

By the end of this video, the foundational components of the graph builder and workflow are in place. The next step, which will be covered in the following video, is to implement the actual node functionality for the basic chatbot, linking it with the LM to generate responses. This approach ensures a scalable and reusable architecture, where multiple workflows or chatbot variations can be built on top of the same graph and state management framework. The video emphasizes the importance of separating graph structure, state management, and node logic for a clean and maintainable implementation.

**H) Implementing The Node Implementation**

In this video, we continue our end-to-end project by implementing the core functionality of the basic chatbot node. While the previous video focused on building the graph structure for the chatbot workflow, this session specifically addresses the node definition and its behavior. The first step is to create a BasicChatbotNode class inside the nodes folder, which encapsulates the chatbot logic. The class is initialized with the previously loaded LM model, ensuring that the AI assistant is ready to process incoming user messages. This separation of the node from the graph builder maintains modularity, making it easier to add additional nodes or chatbot variations in the future.

The main functionality of the BasicChatbotNode class is implemented in a process method, which takes the workflow state as input and returns a dictionary of messages. The state object, defined in state/state.py, keeps track of the conversation history, ensuring that all user messages and responses are maintained consistently throughout the workflow. Within the process method, the LM model is invoked using the messages from the state, generating a response to the user input. This approach ensures that the node itself is solely responsible for handling the AI interaction, keeping it decoupled from other workflow components.

After defining the node functionality, the next step is to integrate it with the graph builder. The BasicChatbotNode is imported into the graph_builder.py file and initialized with the LM model. The process method of this node is then linked to the chatbot node within the graph, creating a complete pipeline from user input to AI response. By following this design, the entire workflow now has all its critical components in place: the frontend UI, LM model loading, graph construction, node functionality, and state management. This sets the stage for the next step, which is full end-to-end integration, linking the frontend input directly to the workflow execution and response generation.

**I) Integrating the Entire Pipeline With Front End**

In this video, we focus on integrating the entire Agentic AI pipeline developed using Landgraf into a seamless end-to-end workflow. With the frontend, LM module, graph builder, and node functionalities already in place, the main task is to connect these components so that user input triggers the complete pipeline. The workflow begins when a user submits a message through the Streamlit UI. This input initiates the loading of the LM model, execution of the appropriate graph workflow, and processing of node-specific logic to generate a response.

The first step in the integration involves configuring the LM module. This is achieved by initializing the GrokLM class with user control inputs obtained from load_ui.py. Once the LM is loaded successfully, the system retrieves the selected use case from the UI—such as the basic chatbot—and passes it to the GraphBuilder class. The setup_graph function in GraphBuilder ensures that the correct workflow nodes are initialized based on the chosen use case. This modular approach allows different workflows to be handled efficiently without changing the core integration logic.

After setting up the graph, the next step is execution and displaying the response. The display_result.py module in the UI folder handles rendering the assistant’s output back to the user. It takes the use case, the initialized graph, and the user message, streaming the AI response on the UI as soon as the LM processes the input. This setup ensures a fully functional end-to-end pipeline where the frontend, LM, graph workflow, node logic, and UI output are all connected. The video concludes by explaining that the next step will involve running the entire application from app.py to verify the workflow execution and handle any potential errors.

**J) Testing The End To End Agentic Application**

In this video, we finally execute the entire end-to-end pipeline of the Agentic AI project using Landgraf. With all components—frontend, LM module, graph builder, node functionalities, and UI integration—already implemented, the goal is to run the application and verify that everything works seamlessly. The workflow is triggered when a user sends a message via the Streamlit UI, which initiates the loading of the LM model, execution of the selected graph workflow, processing of node-specific logic, and streaming of the AI response back to the UI.

During the initial run, a minor import error related to List from typing_extensions was encountered. This was quickly resolved by updating the import to use a capitalized List, highlighting the importance of careful library usage. Once corrected, the application successfully loaded, and the user could select the chatbot use case and enter their API key. Testing the chatbot demonstrated that the LM processed the input, the graph executed properly, and the assistant’s response was displayed correctly, verifying that the pipeline—from frontend to backend nodes to output display—was fully functional.

This video also emphasized modularity and code reusability in the project. Each component, from main.py to graph builder to node implementation, was integrated step by step, making it easy to extend the project in the future. The instructor highlighted that this modular structure allows new use cases, such as a chatbot with external tools, to be added without changing the existing core logic. The next steps in the project series will include building a more advanced chatbot that can interact with external tools, perform web searches, or provide summarization, demonstrating the flexibility and scalability of this pipeline-based approach.

# **XVI) End To End Agentic Chatbot With Web Search Functionality**

**A) Introduction To The Project**

Hello guys, congratulations! We have successfully finished our first end-to-end Genetic AI project wherein we built a basic chatbot. Now, we are going to move into our second project, which is a chatbot with tool integration. In this project, we will be extending our previous work by integrating an external tool so that the chatbot can fetch live information when needed.

This project is centered around a workflow. Let’s say we have a workflow where we integrate with an external web search API called Tabulae. With the help of this API, our chatbot assistant will be able to provide accurate, real-time information from the internet. To demonstrate, I’ll first select a specific model in the application. After selecting the model, we apply the API key to allow the chatbot to make requests to the Tabulae API. The basic chatbot functionality is already implemented, and now we will enhance it to become a chatbot with a tool.

The Tabulae API is essentially an external search engine. Once you navigate to the API provider's website, you will find an API key that allows up to 500–600 requests per day. After entering this API key in our application, we can test the tool integration. For instance, if I input the question, hey, provide me the recent AI news, the chatbot will trigger a tool call by executing something similar to: "response = tabulae.search('recent AI news')" and fetch results from the API. The tool call starts, retrieves the relevant data, and returns the latest AI news. The results include updates like Google’s May announcements, product research highlights, and other detailed information. For transparency, the application also prints the entire tool call using a Python snippet such as "print(tool_call_response)".

The workflow is simple but powerful. We start with a basic workflow diagram where the assistant node connects to the tool node. In Python, this can be represented as: "assistant_response = assistant.handle(user_input)" and "tool_response = tabulae.query(user_input)". The assistant checks if the user’s question requires external data; if yes, it triggers the tool call and includes the retrieved information in the final response. If the query is general, like what is machine learning?, the assistant does not call the tool. Instead, the LLM integrated in the chatbot directly generates an answer, for example: "response = llm.generate('what is machine learning?')" without hitting the API.

This design ensures that whenever a user input comes in, the assistant intelligently decides whether to provide an immediate response or fetch external data using the tool. When external data is fetched, it is taken as context by the LLM to generate an accurate and context-aware response. In Python terms, the context usage can be shown as: "contextual_response = llm.generate(prompt=user_input, context=tool_response)".

By the end of this project, we will have a complete frontend interface with a modular structure similar to our previous chatbot. We will add a new use case for the chatbot with tool, reuse the same text box for input, and define a new node function in Python to handle the tool integration, such as: "def handle_tool_node(input_text): return tabulae.query(input_text)". This structure allows us to connect any number of external tools as required, making the chatbot highly extensible.

In summary, this project extends our first chatbot by enabling external tool integration. General questions are answered by the LLM, while queries requiring live or updated information trigger a tool call using APIs like Tabulae. The retrieved data is used as context for generating responses, resulting in a smarter, real-time, and interactive chatbot assistant. In the next video, we will start implementing this chatbot with external tools project step by step. This is just one example, and the architecture allows integrating multiple tools for various use cases.

I hope you enjoyed this video. See you in the next one. Thank you, and take care!

**B) Implementing The Front End With Streamlit**

Hello guys, welcome to the implementation part of our second project, which is a chatbot with external web search capability. This chatbot will be built using agentic AI capabilities. In the UI, we have the main layout with the sidebar already in place. The first thing we need to do is add a new option for this chatbot. To do this, we first update our configuration file. In the configuration, we add a new option under the chatbot section called "chatbot with web", which indicates that this chatbot will have some web functionality. Previously, it was labeled "chatbot with tool", but for clarity, we rename it to "chatbot with web".

Next, we move to the UI implementation in load_ui.py. Here, we need to handle different use cases and open the API key input field dynamically when the user selects the "chatbot with web" option. We implement this condition in Python as:

if self.user_control.selected_usecase == "chatbot with web":
    self.tab_key = self.user_control.tabulae_api_key
    st.session_state["tabulae_api_key"] = self.tab_key


This ensures that the API key provided by the user is saved in the session state for persistence across the session. We create the input field using Streamlit’s text_input function:

self.tab_key = st.text_input("API Key", type="password", help="Enter your Tabulae API key")


We also validate whether the user has entered an API key. If not, we show a warning:

if not self.tab_key:
    st.warning("Please provide your API key from https://app.tabulae.com")


After these front-end changes, we test the application by running:

streamlit run app.py


Initially, the API key field may not appear if there are spelling mismatches. For instance, if "chatbot with web" is written with a lowercase "w" in one place and capital "W" in another, the condition will not trigger. Correcting the spelling resolves this, and upon selecting the "chatbot with web" option, the API key input field becomes visible.

With the front-end configuration complete, we move to main.py. No major changes are required here since the main structure already accommodates different chatbot types. The next step is to open the graph builder. Just like we created a graph for the basic chatbot, we now need to define a new graph for the chatbot with web capabilities. This involves defining nodes and workflows in the graph that integrate external tool functionality.

In the next session, we will focus on building this graph in the graph builder. We will also implement the node functionalities required to handle external tool calls. The main difference from other use cases is that we are integrating an external API tool—Tabulae—into the workflow. The structure remains largely the same: the changes are primarily in two files: the graph builder and the node functionality implementation.

To summarize, in this session, we configured the front end by adding a new chatbot option "chatbot with web" and made the API key input field conditional based on the selected use case. These changes were updated in the configuration file and tested in Streamlit. In the next session, we will implement the graph builder and node logic to enable external web search through the tool.

That’s it for this session. I hope you found this helpful. See you in the next video. Thank you, and take care!

**C) Implementing GraphBuilder and Search Tools Pipeline**

Hello guys, now we are going to continue our discussion on the end-to-end second project. In the previous video, we completed the front-end part by adding an additional option for the chatbot. Now, our focus is to implement the entire workflow and integrate the Tabulae API with our chatbot assistant using the same workflow. To start, we open the graph_builder.py file. Previously, we built a basic chatbot graph for our first project. Similarly, for this project, we define a new function called chatbot_with_tools_build_graph that will handle the advanced chatbot workflow with tool integration. We add a docstring describing the function, for example:

"""
Build an advanced chatbot graph with tool integration.
This method creates a chatbot graph that includes both the chatbot node and a tool node.
It defines tools, initializes the chatbot with tool capabilities, and sets up conditional and direct edges between nodes.
The chatbot node is set as the entry point.
"""


The next step is to define the tools. Since we need to make external API calls, each tool becomes a separate node in the graph. In the tools folder, we create a new file called search_tool.py. This tool is responsible for integrating with search engines like Tabulae. We import the required modules:

from langchain_community.tools.search import tab_search_results
from langgraph.prebuilt import tool_node


We then define a function get_tools that returns a list of tools available in our application. For now, we use Tabulae as the search tool:

def get_tools():
    tools = tab_search_results(maximum_results=2)
    return tools


Next, we define a helper function create_tool_node to convert these tools into nodes that can be used in the graph:

def create_tool_node(tools):
    """
    Create and return a tool node for the graph.
    """
    return tool_node(tools=tools)


Returning to graph_builder.py, we call get_tools() to retrieve our tools and create_tool_node(tools) to generate the tool node. The LM is already loaded in the graph builder, so we assign it to self.lm. After this, we add nodes to the graph: first the chatbot node and then the tool node:

self.graph_builder.add_node("chatbot")
self.graph_builder.add_node("tools", tool_node)


Now we define the edges in the graph. The workflow starts from the chatbot, and depending on the input, it either calls the tool or proceeds to the end node. Conditional edges are required to handle this logic. The tools_condition object determines whether a user query requires a tool call or a direct response from the chatbot. The edges are defined as:

self.graph_builder.add_edge("start", "chatbot")
self.graph_builder.add_edge("chatbot", "tools", condition=tools_condition)
self.graph_builder.add_edge("chatbot", "end")
self.graph_builder.add_edge("tools", "chatbot")
self.graph_builder.add_edge("chatbot", "end")


This structure ensures that from the assistant (chatbot), queries either go to the tool for fetching external data or directly to the end. After defining these edges, the graph handles conditional flows automatically, so additional checks are simplified.

Finally, we link this workflow to the front-end configuration. If the user selects "chatbot with web" in the UI, the function chatbot_with_tools_build_graph is called to construct this advanced graph. Two key things were achieved in this session: first, we created the tools and converted them into nodes, and second, we built the entire graph with conditional edges for the workflow.

In the next video, we will define the chatbot node in detail. This includes writing a function to handle user input, make tool calls if necessary, and return a response. For clarity, instead of using the label “assistant,” we will refer to it as "chatbot_tools" in the graph, which represents the same chatbot node integrated with tool capabilities.

That’s it for this session. I hope you found this explanation clear. In the next session, we will implement the node functionality for making tool calls and generating responses. Thank you, and take care!

**D) Implementing Node Functionality With End To End Agentic Pipeline**

Hello guys, we are going to continue our discussion on the end-to-end second project. In the previous session, we built the function to construct the entire graph for our use case, which involves integrating an external tool, Tabulae. In this video, we will define the node functionality, which is a crucial part of the workflow.

First, we create a new Python file in the nodes folder called chatbot_with_tool_node.py. This file will contain the definition of the node responsible for handling both the chatbot logic and tool integration. We import the shared state management from our state.py file so that all messages and information are accessible across nodes:

from src.agenda.i.state.state import state


Next, we define a class ChatbotToolNode and its __init__ method. This class will initialize the language model (LM) and act as a wrapper for invoking the LM with tool integration:

class ChatbotToolNode:
    """
    Chatbot node enhanced with tool integration.
    """
    def __init__(self, lm):
        self.lm = lm


The process function is responsible for generating a response based on the current state. It retrieves the last user message, invokes the LM, and returns both the LM response and a placeholder tool response:

def process(self, state):
    last_message = state.messages[-1] if state.messages else ""
    response = self.lm.invoke(last_message, role="user")
    tool_response = f"Tool integration for user input: {last_message}"
    return response, tool_response


We also define a create_chatbot function to bind the tools to the LM. This ensures that tool calls are handled automatically whenever the chatbot node is invoked:

def create_chatbot(self, tools):
    """
    Create and return a chatbot node with tool integration.
    """
    def chatbot_node(messages):
        return self.lm.invoke_with_tools(messages, tools=tools)
    return chatbot_node


This allows us to have two ways of invoking the LM:

Direct LM invocation using the process method.

LM with tools using the create_chatbot binding function.

In the graph_builder.py, we import this node:

from c.i.nodes.chatbot_with_tool_node import ChatbotToolNode


We initialize it with our LM and create the chatbot node bound to the tools:

chatbot_node_obj = ChatbotToolNode(self.lm)
chatbot_node = chatbot_node_obj.create_chatbot(tools)


Once the graph is built with this node, it can be executed. We also need to handle UI integration for the new "chatbot with web" option. In main.py or the Streamlit interface, we add a condition for this use case:

elif selected_use_case == "chatbot with web":
    response, tool_response = chatbot_node(state.messages)
    if tool_response:
        state.write("tool", tool_response)
    state.write("assistant", response)


This ensures both the LM response and tool response are displayed in the Streamlit frontend.

When running the app (streamlit run app.py), you provide your API key for Tabulae. The chatbot with web tool should now work as expected. On testing, the tool fetches results, displays URLs, images, and summaries. This implementation allows further extensions—for example, creating a news summary use case that fetches detailed news instead of single-pointer outputs.

In summary, this session covered:

Creating a new node class ChatbotToolNode.

Defining process and create_chatbot functions to handle LM and tool integration.

Integrating the node in graph_builder.py and binding tools.

Updating the UI logic in Streamlit to handle "chatbot with web" responses.

The next steps could include extending this node for multiple tools and advanced output formatting, such as generating detailed news summaries or blog content.

This completes the node implementation for the chatbot with web integration. Everything should now run correctly, and the graph will fetch data from Tabulae and return structured outputs in the UI.

# **XVII) AI News Summarizer End To End Agentic AI Projects**

**A) Project Introduction**

I’m super excited because now we are going to implement our third project, called the AI News Summarizer. To recap, we have already completed two end-to-end projects. The first one was a Basic Chatbot, where we used modular coding to design the front end, build a Graph Builder, and implement node functionality. The workflow was simple: user input flows into the chatbot, the LM processes it, and the response is returned to the end state. The second project was a Chatbot with Web Search, where we integrated an external tool (Tabulae) as a tool node in our graph. The LM decided whether to make a tool call based on the input, and we also implemented tool conditions and conditional edges. For example, asking “Provide me recent AI news” triggered the tool, which returned a summarized response. However, the summarization quality was not ideal.

Now, in the AI News Summarizer, we are increasing the complexity. The goal is to create a more structured and accurate news summarization workflow using a multi-node graph. The workflow will have three nodes: Node 1 fetches news, Node 2 summarizes it, and Node 3 saves and displays it.

Node 1: Fetch News – In this node, we accept the user’s input for the news timeframe, which can be daily, weekly, or monthly. We then interact with an external news API to fetch the latest news. The Python code for this node could look like:

class FetchNewsNode:
    def __init__(self, api_client):
        self.api_client = api_client
    def process(self, timeframe: str):
        if timeframe == "daily":
            news = self.api_client.get_daily_news()
        elif timeframe == "weekly":
            news = self.api_client.get_weekly_news()
        else:
            news = self.api_client.get_monthly_news()
        return news


This node returns the raw news content to Node 2 for summarization.

Node 2: Summarize News – This node uses a language model (LM) to summarize the news content received from Node 1. The LM produces structured summaries, either as paragraphs or bullet points, depending on the prompt. For example, the Python code could be:

class SummarizeNewsNode:
    def __init__(self, lm_model):
        self.lm = lm_model
    def process(self, news_content: str):
        prompt = f"Summarize the following news in clear bullet points:\n{news_content}"
        summary = self.lm.generate(prompt)
        return summary


This approach ensures the summary is coherent, readable, and formatted for easy consumption.

Node 3: Save and Display – In this node, we save the summarized news into a file (Markdown, PDF, etc.) and also display it on the Streamlit frontend. The Python code might be:

class SaveDisplayNode:
    def __init__(self, save_path: str):
        self.save_path = save_path
    def process(self, summary: str):
        with open(self.save_path, "w") as f:
            f.write(summary)
        return summary


With this structure, the workflow from input to output becomes modular and maintainable. Each node has a single responsibility, which makes it easy to add new features in the future, such as integrating additional APIs or improving summarization techniques.

Graph Builder Integration – Now we create a graph that connects these nodes. The input first passes to FetchNewsNode, then the output flows into SummarizeNewsNode, and finally into SaveDisplayNode. The Python code for building this graph might look like:

class NewsGraphBuilder:
    def __init__(self, fetch_node, summarize_node, save_node):
        self.fetch_node = fetch_node
        self.summarize_node = summarize_node
        self.save_node = save_node
    def run(self, timeframe: str):
        news_content = self.fetch_node.process(timeframe)
        summary = self.summarize_node.process(news_content)
        final_output = self.save_node.process(summary)
        return final_output


Streamlit Frontend Integration – In the Streamlit app, the user can select a model, provide their API key, and choose the news timeframe (daily, weekly, monthly). Once they click “Fetch Latest News”, the graph is executed and results are displayed. The Python code snippet could be:

import streamlit as st

st.title("AI News Summarizer")
timeframe = st.selectbox("Select news timeframe:", ["daily", "weekly", "monthly"])
api_key = st.text_input("Enter API Key:")

if st.button("Fetch Latest News"):
    fetch_node = FetchNewsNode(api_client=NewsAPIClient(api_key))
    summarize_node = SummarizeNewsNode(lm_model=MyLLMModel(api_key))
    save_node = SaveDisplayNode(save_path="news_summary.md")
    graph = NewsGraphBuilder(fetch_node, summarize_node, save_node)
    result = graph.run(timeframe)
    st.text_area("Summary", result)


This completes the end-to-end AI News Summarizer workflow. We now have a multi-node agentic AI application that fetches news, summarizes it with high quality, saves it in a file, and displays it in the Streamlit app. This structure is modular, maintainable, and ready for extension with additional features like multi-source news, richer summarization prompts, or alternative file formats.

In the next step, we will start implementing this step by step, beginning with the UI changes, then building the Graph Builder, followed by implementing each node, and finally integrating everything with Streamlit.

**B) Building the Front End With Streamlit**

Let’s proceed with the implementation of AI News Summarizer, a generic AI application. As usual, for any use case, we begin with the UI part. In this project, we need to add a new option for AI news in our Streamlit interface. Along with this, we will require the API key and a selection for the news timeframe—daily, weekly, or monthly. This will allow the user to fetch the latest AI news based on their selection.

First, we open our configuration file and add a new entry for the AI news use case:

use_cases = ["chatbot", "chatbot_with_web", "ai_news"]


This ensures that our UI knows about the new option. Next, we move to the load UI function, which is responsible for rendering the Streamlit interface. Here, we need to add conditions to handle this new use case. For example, in Python:

if self.user_controls.selected_use_case in ["chatbot_with_web", "ai_news"]:
    api_key = st.text_input("Enter API Key:")


This ensures that the API key input appears when the user selects AI News.

Next, we need to provide a news timeframe selection. We add a new header and selectbox in the sidebar for the AI News Explorer:

if self.user_controls.selected_use_case == "ai_news":
    st.sidebar.header("AI News Explorer")
    timeframe = st.sidebar.selectbox("Select Time Frame:", ["daily", "weekly", "monthly"])


This allows the user to choose the timeframe for news fetching. Alongside this, we add a button to fetch the latest AI news. We maintain the selection in a session state to preserve user choices:

if st.sidebar.button("Fetch Latest AI News"):
    st.session_state["fetch_button_clicked"] = True
    st.session_state["selected_timeframe"] = timeframe


By storing the timeframe and button click state in the session, we ensure that the user’s selection persists across interactions.

Once these UI components are added, we can run our Streamlit app to verify that everything works. In the command line:

streamlit run App.py


After running this, the AI News Explorer should appear when selecting the AI News use case. The user can now select daily, weekly, or monthly and click the Fetch Latest AI News button. The API key input will also be visible.

To summarize, the UI changes we implemented include:

Adding a new use case ai_news.

Displaying the API key input when AI News is selected.

Adding a sidebar selection for news timeframe (daily, weekly, monthly).

Adding a Fetch Latest AI News button and storing the selection in the session state.

This completes the frontend implementation for the AI News Summarizer. While knowledge of Streamlit is helpful, the main learning is in building the graph builder and node implementation, which we will cover next. The UI simply collects user input and triggers the workflow.

In the next video, we will proceed to define and build the graph builder, connecting the fetch, summarize, and save/display nodes to create a complete AI News Summarizer workflow.

**C) Building The AI News State Graph Builder**

We are going to continue our discussion on the AI News Summarizer project. In the previous video, we completed the frontend UI, and now we will move on to building the graph that defines the workflow for our summarizer. For this, we will work inside the graph_builder.py file. Similar to how we created functions for the basic chatbot or chatbot with tools, we will now define a graph specifically for AI news.

We begin by creating a function called ai_news_builder. This function will be responsible for defining the structure of our graph, including all nodes and edges. In Python, the function definition looks like this:

def ai_news_builder(self):
    self.graph_builder = GraphBuilder()


Inside this function, we first add the nodes that represent the main steps of our workflow. The workflow for AI news summarization includes four main nodes: fetch_news, summarize_news, save_results, and end. We can add them like this:

# Add nodes
self.graph_builder.add_node("fetch_news")
self.graph_builder.add_node("summarize_news")
self.graph_builder.add_node("save_results")
self.graph_builder.add_node("end")


After defining the nodes, we need to define the edges that connect these nodes, which determines the flow of execution. The first edge connects the start to fetch_news. We set this as the entry point of the graph:

# Set entry point
self.graph_builder.set_entry_point("fetch_news")


Next, we connect fetch_news to summarize_news, ensuring that the output of the fetch node is passed to the summarize node:

# Add edges
self.graph_builder.add_edge("fetch_news", "summarize_news")


Similarly, we connect summarize_news to save_results, which will handle storing the summarized output:

self.graph_builder.add_edge("summarize_news", "save_results")


Finally, we connect the last node, save_results, to the end node to complete the workflow:

self.graph_builder.add_edge("save_results", "end")


With this, we have defined the nodes and edges of our AI News Summarizer graph. To summarize, the structure includes:

Nodes: fetch_news, summarize_news, save_results, end.

Edges: Start → fetch_news → summarize_news → save_results → End.

At this stage, we are only defining the graph structure. The implementation of each node—what fetch_news, summarize_news, and save_results actually do—will be handled in the node implementation stage, which we will cover in the next video.

In the next step, we will create a node builder function, e.g., ai_news_builder_node, and implement the logic for each node. The fetch_news node will call the news API, summarize_news will process the content using our LLM, and save_results will store the results in a markdown or PDF format and display it in Streamlit.

This completes the graph builder setup for AI News Summarizer. Now, our graph is ready to be populated with node logic, and this modular approach allows us to maintain a clean and extensible workflow.

**D) Tavily Client Search Fetch News Node Implementation**

We are continuing our discussion on the AI News Summarizer project. In the previous video, we built the entire graph structure, defining the nodes fetch_news, summarize_news, save_results and the edges connecting them. The graph flows from start → fetch_news → summarize_news → save_results → end. Now, the major step left is the implementation of each node, i.e., the actual functions that perform the tasks defined by the nodes.

The first node to implement is fetch_news. This node will call an API, specifically the Tableau API, to fetch AI news, and pass the fetched data to the summarize node. To start, we create a file inside the node folder called "ai_news_node.py" which will contain our node logic.

We start by importing the necessary libraries. Since we will use the Tableau API, we import the Tableau client, and we also import prompt templates from LangChain for summarization:

from tableau import TableauClient
from langchain.prompts import chart_prompt_template


Next, we define a class "AI_News_Node" which will initialize the Tableau client and maintain the LLM instance, along with a state variable to store intermediate data:

class AI_News_Node:
    def __init__(self, tableau_api_key, llm):
        """
        Initializes the AI news node with Tableau client and LLM instance.
        """
        self.w = TableauClient(api_key=tableau_api_key)
        self.llm = llm
        self.state = {}


Here, the state dictionary will capture the steps of the workflow, storing data like fetched news and summaries.

Now, we implement the fetch_news function. This function takes a state dictionary as input and retrieves the news frequency (daily, weekly, monthly) from it. This frequency is then mapped to Tableau’s API parameters and used to query the news:

def fetch_news(self, state: dict):
    """
    Fetches AI news from Tableau API based on the frequency specified in state.
    """
    # Get user-selected frequency from state
    frequency = state['message_state']['content'].lower()
    state['frequency'] = frequency
    # Map frequency to Tableau API parameters
    time_range_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M'}
    days_map = {'daily': 1, 'weekly': 7, 'monthly': 30}
    # Fetch news using Tableau client
    response = self.w.search(
        query="AI technological news India and globally",
        time_range=time_range_map[frequency],
        max_results=20,
        days=days_map[frequency]
    )
    # Store results in state
    news_data = response.get('results', [])
    state['news_data'] = news_data
    return state


In this function:

We retrieve the frequency from the state (daily, weekly, or monthly).

We map the frequency to API-specific parameters (D, W, M) and corresponding days.

We call the Tableau client’s search function with the topic "AI technological news India and globally", specifying the frequency, max results, and days.

We store the fetched results in the state dictionary under the key "news_data" and return the updated state.

This completes the fetch_news node implementation. Later, the output stored in state['news_data'] will be passed to the summarize_news node, which will use an LLM to generate structured summaries, and finally to the save_results node, which will save the summaries in markdown or PDF format.

**E) AI News Summarize Node Functionality Implementation**

Now that we have completed the fetch_news functionality, the next key node to implement is the summarize node. This node will take the news data fetched in the previous step and summarize it using a prompt template and our LLM. The idea here is to convert each news article into a concise, well-structured summary in markdown format, including the date and the source URL.

We define a new function "summarize" inside our AI_News_Node class. This function takes the state dictionary as input, which already contains the key "news_data" filled with the fetched news:

def summarize(self, state: dict):
    """
    Summarizes the news data fetched from Tableau API using a prompt template.
    """
    # Extract news items from state
    news_items = state['news_data']


Next, we define a prompt template. This template instructs the LLM how to structure the summary. For instance, it should summarize each article into markdown format, include the publication date, write concise sentences, sort news by date, and include the source URL as a link. The template might look like this:

system_prompt = """
Summarize a news article into markdown format for each item.
Include the date in ISO format.
Provide a concise sentence summary for the latest news.
Sort news by date.
Include source URL as a link.
Format: Summary - URL
"""


After defining the prompt, we need to prepare the news data. Each news item contains fields like "content", "url", and "publish_date". We join these fields line by line for all articles to create a single string that will be fed into the prompt:

articles_str = "\n".join([
    f"Content: {item['content']}\nURL: {item['url']}\nDate: {item['publish_date']}"
    for item in news_items
])


Finally, we invoke the LLM using the prompt template and supply the concatenated news articles as the context. The summarized output is then stored back in the state dictionary under the key "summary":

summary_output = self.llm.invoke_prompt_template.format(articles=articles_str)

# Update the state with the summarized news
state['summary'] = summary_output

return state


To summarize this workflow:

Extract news items from the state dictionary.

Define a system prompt for the summarization LLM.

Concatenate the news articles into a single string in a structured format.

Invoke the LLM with the prompt template to generate the summaries.

Store the summarized output back into the state dictionary.

With this, the summarize node is complete. The output stored in state['summary'] can now be passed to the save_results node, which will format and store the summaries in markdown or PDF format for user consumption.

**F) Save Results Node Functionality Implementation**

Now that we have successfully implemented fetch_news and summarize nodes, the final step is to save the summarized results. This is a straightforward process: we simply take the summarized content from the state dictionary and save it into a file, which can later be displayed on the UI. The key idea here is to capture the frequency (daily, weekly, or monthly) and create a file name accordingly, such as "daily_summary.md" or "weekly_summary.md", inside an I_news folder. The implementation looks like this:

def save_result(self, state: dict):
    """
    Saves the summarized news into a markdown file based on the frequency.
    """
    frequency = state['frequency']  # daily, weekly, monthly
    summary_content = state['summary']  # summarized news
    file_path = f"I_news/{frequency}_summary.md"
    # Ensure the folder exists
    import os
    os.makedirs("I_news", exist_ok=True)
    # Write summary to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{frequency.capitalize()} Summary\n\n")
        f.write(summary_content)
    # Store the file path in state for reference
    state['file_name'] = file_path
    return state


With this, the save_result node stores the summarized content in markdown format, ready for display or further processing. We are maintaining multiple state variables, including "news_data", "summary", and "file_name", all stored in the state dictionary for easy access across nodes.

Once all the nodes are defined, the next step is integration with the graph builder. First, we import the AI_News_Node class in the graph builder file:

from c.graph_agentic_i.nodes.i_news_node import AI_News_Node


Then, we initialize an object of this class by providing the required LLM and client:

i_news_node = AI_News_Node(llm=self.llm)


We can now sequentially call the three main node functions—fetch_news, summarize, and save_result—to complete the workflow:

state = i_news_node.fetch_news(state)
state = i_news_node.summarize(state)
state = i_news_node.save_result(state)


Finally, in the graph builder, we add a condition to trigger this workflow if the selected use case is "I_news":

if self.selected_use_case == "I_news":
    self.i_news_builder_graph()


In the main.py file, we integrate this workflow with Streamlit to display the results on the frontend. Using a simple markdown display, we read the saved summary file and show it to the user:

elif self.selected_use_case == "I_news":
    user_message = self.user_message
    # Invoke the graph with the user message
    state = self.graph.invoke(message=user_message)
    # Display the markdown summary
    file_path = state['file_name']
    with open(file_path, "r", encoding="utf-8") as f:
        summary_content = f.read()
    import streamlit as st
    st.markdown(summary_content)


This ensures that after fetching, summarizing, and saving the news, the content is automatically displayed in the frontend UI. At this point, the entire AI News Summarizer workflow is complete: the graph builder is ready, the nodes are implemented, and the Streamlit UI displays the summary dynamically.

With everything integrated, the next step is to run a full demo to verify that selecting a frequency like daily, weekly, or monthly triggers the complete flow from fetching news to displaying the summary.

**G) Running The Entire AINEWS Agentic Workflow**

Now finally, let’s see whether everything is working fine with our AI News Summarizer. I already created the I_news folder, and we are going to test if the summary workflow works properly. Even if we face errors, we will handle them live so that everyone can understand the process.

We start by running our Streamlit app:

streamlit run app.py


Once the app is running, we input our Grok API key and select the model. Then, we enter the I_news section and provide the API key there as well. Next, we choose a time frame such as daily, weekly, or monthly.

Upon clicking Fetch Latest AI News, initially nothing happened. This indicated that the button click wasn’t triggering the workflow. To fix this, we introduced a session state variable to track the button click:

st.session_state['fetch_button_clicked'] = True


In main.py, we updated the condition to handle both user messages and button clicks. The user message now receives the time frame information when the fetch button is clicked:

if st.session_state['fetch_button_clicked']:
    user_message = st.session_state['time_frame']  # daily, weekly, or monthly


Next, we had to initialize session state variables to avoid runtime errors:

st.session_state['time_frame'] = ""
st.session_state['fetch_button_clicked'] = False


After restarting the app, clicking the fetch button successfully triggered the workflow. The system began fetching and summarizing news.

Initially, there was an error regarding "No such file or directory: I_news/weekly_summary.md". This was caused by saving the markdown file in the wrong folder. To fix this, we ensured the I_news folder is created in the workspace root, not inside the source folder:

os.makedirs("I_news", exist_ok=True)
file_path = f"I_news/{frequency}_summary.md"


With this fix, the workflow worked perfectly. Selecting weekly generated the markdown summary file, which we could read and display in the Streamlit app. Similarly, selecting daily or monthly also generated the respective summaries.

You can view the generated markdown files directly in the I_news folder. Opening the files in preview mode shows nicely formatted summaries, which can also be converted to PDF or other formats if needed.

# Display markdown in Streamlit
with open(file_path, "r", encoding="utf-8") as f:
    summary_content = f.read()
st.markdown(summary_content)


This completes the full workflow of the AI News Summarizer:

Fetch news from the API based on user-selected time frame.

Summarize news using the LLM and prompt template.

Save results in markdown files.

Display summaries in the Streamlit frontend.

The system now supports daily, weekly, and monthly news summaries, with all errors addressed in real time. You can also customize the prompt templates to change the summary formatting.

This concludes the AI News project. In the next project, we’ll explore building something more complex using FastAPI, creating a new use case step by step.

Thank you for watching, and see you in the next video!

# **XVIII) End To End Blog Generation Agentic AI App**

**A) Introduction And Project Demo**

Hello guys! I’m super excited to start our new end-to-end project, which is about blog generation using AI applications with a graph-based approach. This project is very close to my heart because it solves a problem I personally faced. On my website, christian.in, there’s a blog section, and I wanted a system where whenever I upload a YouTube video, a corresponding blog is automatically generated from that video transcript. Initially, I thought this would require a dedicated content writer and a review process. But instead, I developed a generic AI application that automates this workflow entirely.

The goal of this project is to create a blog generation system capable of generating blogs based on a topic name, topic and language, or even a YouTube video transcript. For example, if you provide the transcript of a video, the system should automatically generate a structured blog from it. We will focus on two main use cases initially.

The first use case is basic blog generation. Here, the user provides a topic, e.g., "Generic AI Application". The system will then create a title, generate content, and optionally add a conclusion. For instance, the AI can generate a title like "Rise of Agentic AI: How AI is Revolutionizing the Future" and produce the content accordingly. The Python logic for title generation could be represented like this: "title = ai_agent.generate_title(topic)", and content generation can be executed with "content = ai_agent.generate_content(title)".

The second use case is blog generation in different languages. After generating the initial English content, we can translate it into other languages such as Hindi, French, or any language requested. For example, "translated_content = ai_agent.translate(content, language='Hindi')" allows the system to generate multilingual blogs seamlessly. This modular approach ensures that each functionality—title creation, content generation, and translation—is isolated and can be maintained independently.

An exciting aspect of this project is the integration with Graph Studio for monitoring and execution. Within Graph Studio, every step—like title creation and content generation—is logged and monitored. For example, when I submit a topic "Agentic AI", the system generates a title and content, and I can monitor the execution. Each call to the AI, such as "title_response = chat_grok.call(prompt_title)" and "content_response = chat_grok.call(prompt_content)", is tracked, along with token usage. This gives us complete visibility into how the blog is being generated.

The project implementation is fully modular. We will create multiple nodes representing different parts of the workflow, including root nodes, conditional edges, and nodes for each processing step. These nodes interact with each other to build the complete blog generation pipeline. For example, we might define a node in Python as: "title_node = Node(name='Title Creation', func=generate_title)" and "content_node = Node(name='Content Generation', func=generate_content)". Conditional edges can then connect them based on the workflow logic.

We will not focus heavily on the front end in this project. Instead, we will create APIs using FastAPI to handle blog generation requests. For instance, we can define an endpoint as: "@app.post('/generate_blog/') async def generate_blog(request: BlogRequest): return generate_blog_pipeline(request)". This ensures that our backend logic is reusable, scalable, and can integrate with other applications.

Finally, to manage dependencies efficiently, we will use the UV package manager instead of Conda. This allows us to demonstrate different ways of managing Python packages, keeping the environment lightweight and flexible.

In summary, this project covers title creation, content generation, multilingual support, workflow monitoring using Graph Studio, modular pipeline design, and FastAPI integration. By the end of the implementation, the system will automatically generate blogs from topics or video transcripts, translate them if needed, and provide full monitoring for transparency and debugging. This project lays a solid foundation for building AI-driven content automation systems.

**B) Building Project Structure Using UV Package**

Hello guys! Now let’s go ahead and execute our end-to-end project. In this video, we will focus on setting up the project structure using the UV package manager. First, I created a new folder on my E drive named "blogAgentic" and opened it in VS Code. The first task is to create a Python environment, but this time, instead of Conda, we will use UV, which is a fast, modern package manager written in Rust. If you haven’t installed UV yet, you can install it via pip with "pip install uv" or follow the platform-specific instructions for Windows, Mac, or Linux.

Once UV is installed, we initialize the project workspace by executing "uv init" in the terminal. This command creates a new project structure with default files, including "pyproject.toml" where basic information like Python version (3.13), project version (0.10), and description can be defined. At this point, our workspace is initialized and ready to create a virtual environment.

To create a virtual environment using UV, we can execute "uv venv <env_name>". For example, "uv venv blogAgenticEnv" will create an environment named blogAgenticEnv inside the project folder. After creation, we activate it by pointing to the environment path. Once activated, any Python file we run will execute within this environment. This is much faster and simpler compared to creating environments with Conda.

The next step is to create a "requirements.txt" file to list all the libraries needed for our project. To install individual packages, UV provides a very simple command: "uv add <package_name>". For example, "uv add fastapi" or "uv add langchain" will install the package and also automatically update "pyproject.toml" with the package version. This makes dependency management seamless.

For multiple packages, we can list them all in the "requirements.txt" file, for instance:

fastapi
uvicorn
watchdog
langchain
inmemory
chadgrok


Once the file is ready, all packages can be installed with "uv add -r requirements.txt", similar to "pip install -r requirements.txt". This ensures all libraries are installed quickly and properly tracked in "pyproject.toml". The installed packages include essential libraries for our project like FastAPI, Uvicorn for serving APIs, LangChain, Watchdog, InMemory, and ChadGrok for AI-driven functionalities.

After setting up the environment and installing the required packages, I created a folder named "source" to hold all project implementation files. This folder will contain all the code for blog generation, graph builder, and node implementation that we will build in the upcoming steps. At this point, our basic project structure is ready: the environment is activated, required libraries are installed, and the workspace is organized for modular development.

Thanks to UV, this setup process is extremely fast and efficient, and it helps manage dependencies and environments effortlessly. In the next video, we will start coding the actual blog generation use case, including building the graph nodes and implementing the complete AI workflow for content creation.

This concludes the initial project setup and environment creation. Everything is now ready to proceed with the coding and AI logic for automated blog generation.

**C) Blog Generation Grpah Builder And State Implementation**

Hello guys! We are going to continue our discussion on the end-to-end blog generation project. In the previous video, we successfully created the virtual environment using UV and installed all the required libraries. Now, we will continue by setting up the project structure and creating the necessary folders and files inside the source folder, which will hold all our implementation code.

Inside the source folder, I first created an __init__.py file to make it a proper Python package. Then, I created several subfolders to organize the code: lmms for language models, graphs for graph-related code, nodes for node logic, and states for defining structured state objects. Each folder also contains an __init__.py file to allow easy imports across the project. This modular structure helps maintain clean and reusable code.

Next, I created a .env file to store all the important API keys. For this project, there are three keys: PROJECT_NAME (set as "blogAgentic"), API_KEY for platform monitoring, and another API_KEY for the specific LLM model usage. These keys are required for initializing and monitoring the AI models and the workflow in the platform.

Inside the lmms folder, I created a file grok_lm.py. Here, I defined a class GrokLM to load the Grok LLM with environment variables. The main function is get_lm() which retrieves the API key from os.environ and initializes the model using:

import os
from chadgrok import ChadGrok

class GrokLM:
    def __init__(self):
        self.grok_api_key = os.getenv("GROK_API_KEY")
    def get_lm(self):
        try:
            lm = ChadGrok(api_key=self.grok_api_key, model="llama-3.1")
            return lm
        except Exception as e:
            raise ValueError(f"Error occurred: {e}")


If you want to use OpenAI instead of Grok, you can use the same structure but replace GROK_API_KEY with your OpenAI API key.

Next, we move to the graphs folder to implement the graph builder. I created a file graph_builder.py and defined a class GraphBuilder to construct the workflow graph. The class constructor initializes the LLM and the state graph, which will later use BlockState from the states folder:

from langgraph.graph import StateGraph, Start, End
from src.states.block_state import BlockState

class GraphBuilder:
    def __init__(self, lm):
        self.lm = lm
        self.graph = StateGraph(start=Start(), end=End(), state=BlockState)


The BlockState class, defined in states/block_state.py, uses Pydantic to provide structured output for the blog. It contains the following fields:

from pydantic import BaseModel, Field
from typing import Dict

class Blog(BaseModel):
    title: str = Field(..., description="Title of the blog post")
    content: str = Field(..., description="Main content of the blog post")

class BlockState(BaseModel):
    topic: str
    blog: Blog
    language: str = Field(..., description="Language for blog generation")


The GraphBuilder class then defines a method build_topic_graph() which sets up the nodes and edges. Here, we have two main nodes: title_creation and content_generation. The edges define the flow from start → title creation → content generation → end:

def build_topic_graph(self):
    self.graph.add_node("title_creation")
    self.graph.add_node("content_generation")
    self.graph.add_edge("start", "title_creation")
    self.graph.add_edge("title_creation", "content_generation")
    self.graph.add_edge("content_generation", "end")
    return self.graph


At this stage, the graph structure is complete, but the node functionalities—how the title and content will be generated using the LLM—will be implemented in the nodes folder in the upcoming videos.

To summarize, in this step, we have set up the modular project structure, created LLM integration in lmms, defined the structured blog state in states, and built the graph skeleton in graphs. This prepares us for the next step, where we will implement the title creation and content generation nodes and integrate them with the graph workflow.

This concludes the graph and state setup for the blog generation project. In the next video, we will focus on implementing the nodes and connecting the entire AI workflow.

**D) Blog Generation Node Implementation Definition**

Hello guys! Today we are continuing our end-to-end blog generation project. In the previous video, we had already built the entire graph structure. Now it’s time to implement the node logic, which is the functional part of the graph. In this project, there are two main node implementations: title creation and content generation.

First, we go to the nodes folder and create a new file called "blog_node.py". Inside this file, we define a class "BlogNode" which will contain all the logic for our blog nodes. The class begins with a docstring describing it: "Class to represent the blog node". The constructor is defined as "def __init__(self, lm):" where lm is the language model that we will use in all node operations. Inside the constructor, we simply initialize the LLM with "self.lm = lm".

The first functionality we implement is title creation. We define a method "def title_creation(self, state: BlockState):" where state is of type "BlockState". This type is imported at the top of the file with "from src.states.block_state import BlockState". Inside this method, we first check if the topic exists in the state dictionary by writing "if 'topic' in state and state['topic']:". If a topic exists, we define a prompt for the LLM to generate a creative, SEO-friendly blog title using markdown formatting. The prompt can be written as:

prompt = (
    f"You are an expert content writer.\n"
    f"Use markdown formatting.\n"
    f"Generate a creative and SEO-friendly blog title for the topic: {state['topic']}"
)


Next, we send this prompt to the LLM and get a response with "response = self.lm.invoke(prompt)". We then store the generated title back into the block state using "state.blog.title = response.content", and return it in a structured format: "return {'block': {'title': response.content}}". This ensures that the generated title is saved in the state and can be used by downstream nodes.

Once the title creation node is done, we move on to the content generation node. Inside the same class, we define "def content_generation(self, state: BlockState):" which again takes the state object as input. We perform a similar check for the topic with "if 'topic' in state and state['topic']:". The prompt for content generation instructs the LLM to produce detailed blog content with a structured breakdown:

prompt = (
    f"You are an expert blog writer.\n"
    f"Use markdown formatting.\n"
    f"Generate a detailed block content with a structured breakdown "
    f"for the topic: {state['topic']}"
)


The response is then obtained by invoking the LLM "response = self.lm.invoke(prompt)", and the content is stored in the state with "state.blog.content = response.content". We also return it in a structured format: "return {'block': {'content': response.content}}". This ensures that both the blog title and content are available in the state for the graph to use.

In the graph builder, we import and initialize this node with "from src.nodes.blog_node import BlogNode" and "blog_node_object = BlogNode(lm=self.lm)". The methods "title_creation" and "content_generation" are then called within the graph execution sequence. For example, we can call "blog_node_object.title_creation(state)" to generate the title and "blog_node_object.content_generation(state)" to generate the content. These outputs are stored in the BlockState object so that the LLM provides a structured, Pydantic-validated output.

Once both nodes are implemented, we can execute the entire graph from start to finish. The title creation output flows into the content generation node, ensuring that the content is coherent and relevant to the topic. The graph itself is constructed using the "build_topic_graph" function, which organizes the execution sequence of nodes and ensures that the state is updated at each step.

Finally, after the node implementations are ready, we create a new file called "app.py" to implement FastAPI. This will allow us to expose the functionality as an API, so that we can call the graph programmatically without needing a frontend. By sending a topic through the API, the backend can generate both the blog title and the detailed content and return it in JSON format.

To summarize, in this phase we have:

Created the "BlogNode" class inside "blog_node.py".

Implemented two methods: "title_creation" and "content_generation".

Integrated these methods with the graph so that outputs flow from title to content.

Stored the outputs in the BlockState object to maintain a structured format.

Prepared for FastAPI integration to expose the functionality as an API.

This completes the node implementation phase. In the next step, we will implement FastAPI endpoints in "app.py" to run the blog generation workflow through API calls.

**E) Creating Blog Generating API Using FAST API**

Hello guys! In the previous video, we completed both the Graph Builder and the node creation. In this video, we focus on creating a FastAPI application so that the blog generation workflow can be executed via an API request. We are not building a frontend at this stage; our goal is to call the functionality directly using APIs.

First, we import the necessary libraries. We start with "import uvicorn" because we will use it to run the FastAPI server. From FastAPI, we import "FastAPI" and "Request" with "from fastapi import FastAPI, Request". We also import our graph builder with "from src.graph.graph_builder import GraphBuilder" and the language model with "from src.lms.graph_lm import LM". Additional imports include "import os" and "from dotenv import load_dotenv" to manage environment variables.

Next, we initialize the FastAPI app with "app = FastAPI()". We also set up our LangSmith API key by fetching it from environment variables:

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


This ensures that every request can access the LLM securely. The app.py file is located in the root workspace folder.

Now we define our API endpoint. Using the FastAPI decorator, we create a POST endpoint with "@app.post('/blocks')". The corresponding asynchronous function is "async def create_blocks(request: Request):". Inside this function, we retrieve the POSTed JSON data using "data = await request.json()". This JSON contains the topic information, for example: {"topic": "Agentic AI"}. We can access the topic using "topic = data.get('topic', '')".

Next, we initialize the LM object with:

grok_lm = LM()
lm = grok_lm.get_lm()


This loads the Llama model (llama 3.18) configured in our LM class. Then, we create a graph builder object:

graph_builder = GraphBuilder(lm=lm)


If a topic is provided, we call the graph setup function to build the topic graph. We define the use case as "topic" and call:

graph_builder.setup_graph(use_case=use_case)


Inside setup_graph, if the use case is "topic", it calls the "build_topic_graph" function, which executes the title creation and content generation nodes sequentially. The final state is returned by invoking the graph:

data = graph_builder.graph.invoke({"topic": topic})


Here, "data" contains the full output including the title and content stored in the structured state object.

Finally, we run the FastAPI app using Uvicorn. In app.py we write:

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


Here, "0.0.0.0" ensures the server runs on localhost (127.0.0.1) and port 8000. The reload=True flag allows the server to restart automatically when code changes.

To test the API, we use Postman. The API URL is "http://127.0.0.1:8000/blocks" and we select a POST request. In the body, we select "raw" and "JSON" type, and send data like:

{
  "topic": "Agentic AI"
}


Clicking "Send" will trigger the FastAPI endpoint. The graph executes, and we get a JSON response containing the topic, the generated title, and the content in markdown format. For example, the response may look like:

{
  "topic": "Agentic AI",
  "block": {
    "title": "Agentic AI Revolutionizes Modern Workflows",
    "content": "## Introduction\nAgentic AI is ..."
  }
}


We can test multiple topics, such as "Generic AI versus AI agent", and the API returns the corresponding title and content, complete with markdown formatting and links if included. This output can then be sent to a frontend or further processed to generate PDFs, markdown files, or other formats.

In summary, we have:

Created the FastAPI app in "app.py".

Configured environment variables for the LLM API.

Defined a POST endpoint /blocks that accepts a JSON with a topic.

Initialized the LLM and Graph Builder objects.

Built and invoked the graph using the topic to generate structured output.

Returned the title and content in JSON format.

Tested the API successfully using Postman.

This concludes the API implementation phase of the blog generation project. In the next steps, we can expand this by adding more functionalities like language translation, additional blog formatting, and more complex use cases.

**F) Integrating Langgraph Studion For Debugging**

Hello guys! In our previous video, we implemented our end-to-end block generation Agentic AI application. We also created the FastAPI endpoint /blocks and tested it successfully. For example, if we run "python app.py" and provide a topic like "ethical AI" in Postman, we get a response with the generated block content, confirming that the API works correctly.

Now, if we go to LangSmith, there is a feature called Deployments, which provides an amazing tool called LangGraph Studio. This tool allows you to debug and monitor your graph-based applications in a much easier and efficient way. Monitoring is a key feature here, as it allows you to step through the graph execution, check intermediate outputs, and even pause execution for human feedback. In this video, we will set up Graph Studio and demonstrate how to run the application directly from our local machine.

First, we create a file called "langgraph.json" in the project root. This JSON file is the configuration that allows LangGraph Studio to run our application. The first key in this JSON is "dependencies", where we specify "." to indicate the current working directory. The second key is "graphs", where we define the graph we want to execute. For example, we can write: "name": "blog_generation_agent" and "path": "src/graph/graph_builder.py" to point to our graph builder file, and "variable": "graph" to reference the compiled graph object. Finally, we add an "env" key pointing to our .env file to provide API keys and other environment variables.

In the "graph_builder.py" file, we initialize the language model and build the graph as follows: "lm = LM().get_lm()" and "builder = GraphBuilder(lm=lm)". Then we call "builder.build_topic_graph()" to construct the graph nodes, including title creation and content generation. To make the graph executable, we compile it with "graph = builder.build_topic_graph().compile()". This compiled graph variable is what we reference in "langgraph.json" so that LangGraph Studio knows which object to run. This is similar to how FastAPI uses "app:app" when running "uvicorn app:app".

Before running in Graph Studio, we ensure that the FastAPI server is running locally by executing "python app.py". The server runs on port 8000, and we can confirm functionality by sending requests through Postman. Once verified, we can run the graph locally in Graph Studio using the LangGraph CLI: "langgraaf dev". This opens Graph Studio in the cloud or in a local window, showing the entire graph structure, including title creation and content generation nodes.

In Graph Studio, we can provide input, for example "Agentic AI", and click submit. Step by step, the title is created, followed by content generation. Hovering over any node shows the inputs and outputs for that node. If desired, we can interrupt execution for human feedback. For instance, we can pause after title creation, modify the input, and then continue to generate the content. This human feedback feature is very useful for debugging and monitoring complex workflows. For example, inputting "Future of AI" first generates the topic, then the title, and pauses before content generation, allowing us to confirm before proceeding.

We can try multiple inputs in Graph Studio, such as "AI Agents vs Agentic AI", and see the generated title and content with alternate options for flexibility. The tool clearly shows the outputs in fields such as "title", "content", and "topic". All intermediate outputs, SEO considerations, and target keywords are visible, making debugging transparent and intuitive. The key thing to remember is that the FastAPI server must run parallelly while executing "langgraaf dev" so that the graph can interact with the API during runtime.

Finally, this setup allows us to debug, monitor, and test our block generation Agentic AI application effectively. In the next project, we plan to extend this workflow to multi-language blog generation, where the inputs will include both "topic" and "current_language", allowing blogs to be generated in languages like Hindi, French, or German. This showcases the flexibility and power of combining FastAPI, LangSmith LMs, Graph Builder, and LangGraph Studio to create a fully debuggable and monitorable AI application.

That’s it for this video. I hope you now understand how to set up LangGraph Studio, configure "langgraph.json", initialize your graph in "graph_builder.py", and debug your application step by step. See you in the next video!

**G) Blog Generation And Translation With Language**

Hello guys! I hope everybody has understood our end-to-end block generation implementation using Agentic AI, where we provided a topic as input and generated a block. In this new project, we aim to extend this functionality to multi-language block generation. The language will be passed as part of the POST request, and based on the input, the generated content will be translated into either Hindi or French. Of course, other languages can also be used, but for demonstration, we are considering these two. The overall graph structure remains similar: starting from "start", followed by "title creation", and "content generation". During content generation, the graph will take a conditional path depending on the target language to perform the required translation.

To implement this, we need to modify our graph builder to include the language input and add conditional nodes for translation. The development steps involve updating the POST request handler to accept a "language" parameter, integrating a translation step in the graph, and ensuring that the final output reflects the selected language. Once the graph is built, we can test it in LangGraph Studio, which allows real-time monitoring, observing the flow of block generation, and validating that each step—from title creation to content generation and language translation—is executed correctly.

Additionally, we are already tracking execution in LangSmith, which enables debugging and request tracing across all nodes and any language models used. For example, in the previous project named "Block Generator", LangSmith tracked the request from title creation all the way through content generation, allowing us to inspect each step and intermediate output. This was made possible because in "app.py", we ensure that the API key is assigned as an environment variable, allowing all requests to be tracked and monitored in LangSmith.

In the next video, we will start implementing the multi-language block generation platform, adding translation logic to our graph and testing it both locally and in LangGraph Studio. This will allow us to generate blogs or content blocks in different languages dynamically based on the input. I hope you found this explanation helpful. That’s it for this video—I’ll see you in the next one. Thank you and take care.

**H) Building Blog Generation And Translation Graph Builder**

So guys, now let's proceed with the implementation of block generation in different languages. We will continue with the same project structure and update the code accordingly. The first step is to open our graph builder. Previously, our input only required a "topic", but now we need to provide both "topic" and "language". To reflect this, we update our request JSON file (request.json) to include the language parameter like this: "topic": "AI agents", "language": "Hindi". If we pass "Hindi", the block should be generated in Hindi; if "French" is passed, the block should be generated in French.

Next, in the graph builder, we already have the build_topic_graph method, but now we will create a new method called "build_language_graph" using: "def build_language_graph(self):" with a docstring explaining, "Building a graph for block generation with inputs topic and language". Inside this method, we start by adding nodes to the graph using "self.graph.add_node()". We reuse the previous nodes for "title_creation" and "content_generator", but we also add two additional nodes: "Hindi_translation" and "French_translation". These nodes will handle language-specific translations, though the exact node functionality will be implemented later. For example, adding a Hindi translation node can be written as: "self.graph.add_node('Hindi_translation')" and for French: "self.graph.add_node('French_translation')".

Once the nodes are created, we add the edges to define the flow. The edges from start to title creation and title creation to content generation remain the same. After content generation, we add a "route" node: "self.graph.add_node('route')", which will determine the conditional path for translation. Then, we define conditional edges using "self.graph.add_conditional_edges()". The conditional edge function will take the route node as input and redirect to either Hindi or French translation nodes based on the language input. For instance, "if input_language == 'Hindi': go to Hindi_translation node; else: go to French_translation node". Finally, we add edges from both Hindi and French translation nodes to the "end" node using: "self.graph.add_edge('Hindi_translation', 'end')" and "self.graph.add_edge('French_translation', 'end')".

At this stage, the graph structure is complete: from start → title creation → content generation → route → Hindi/French translation → end. The final step in the method is to return the graph: "return self.graph". Later, in our main implementation, we can check if the input contains a language parameter. If it does, instead of calling build_topic_graph(), we call build_language_graph(). This ensures that the input language is handled correctly and the block generation follows the proper path in the graph.

The only remaining tasks are the implementation of the root node and the translation node functionalities for Hindi and French. Once those are implemented, the graph will be fully functional for multi-language block generation. In the next video, we will focus on defining the functionality for the Hindi translation, French translation, and the root node. For now, the graph structure, nodes, edges, and conditional routing for multi-language support have been fully set up. I hope this explanation gives a clear picture of the implementation process. Thank you, and I’ll see you in the next video.

**I) Blog Generation And Translation Node Implementation**

Now we are going to proceed with the node implementation for multi-language block generation. Specifically, we need to implement functions that handle Hindi translation, French translation, and a root node function.

First, we open the nodes folder and navigate to the blog_node.py file. Here, we define a new method called translation. This function will take two parameters: self and state, where state represents the block state. We also add a docstring: "Translate the content to the specified language".

Inside the translation function, we create a prompt called translation_prompt. This prompt instructs the system to translate content while maintaining the original tone, style, and formatting, and to adapt cultural references and idioms appropriately. The original content comes from the block state.

block_content = state.block
translation_prompt = """
Translate the following content into {current_language}.
Maintain the original tone, style, and formatting.
Adapt cultural references and idioms appropriately.
Content: {block_content}
"""


Next, we define a human message for the LLM to process, using:

from langchain_code.messages import HumanMessage, SystemMessage

message = HumanMessage(
    content=translation_prompt.format(
        current_language=state.current_language,
        block_content=block_content
    )
)


We then invoke the translation using the LLM with structured output, ensuring that the output follows the Blog class structure with title and content fields:

from shark.state.blog_state import Blog

translated_content = self.lm.with_structured_output(Blog).invoke([message])


This allows the translation function to dynamically handle any language specified in the input state.

In the graph builder, we use the translation function for the Hindi and French translation nodes by passing a lambda function with the current language parameter:

# Hindi translation
lambda state: self.blog_node_obj.translation(state, current_language="Hindi")

# French translation
lambda state: self.blog_node_obj.translation(state, current_language="French")


Here, the current language is passed through the state, which ensures that the translation function knows which language to convert the content into.

Next, we define the root node function in blog_node.py. The root function simply reads the current language from the state and returns it:

def root(self, state):
    return state.current_language


This root node is responsible for setting up the language context before the conditional routing happens.

The route decision function determines which translation node to call based on the current_language in the state:

def route_decision(self, state):
    if state.current_language.lower() == "hindi":
        return "Hindi_translation"
    elif state.current_language.lower() == "french":
        return "French_translation"
    else:
        return state.current_language


In the graph builder, the route decision ensures that the correct translation node is invoked:

# Conditional routing
if route_decision_output == "Hindi_translation":
    call Hindi translation node
elif route_decision_output == "French_translation":
    call French translation node


Finally, the edges from Hindi or French translation nodes go to the end node, completing the graph flow.

In the application (app.py), we now need to handle both topic and language from the request JSON:

language = data.get("language")
if topic and language:
    use_case = "language"
    graph = builder.setup_graph(use_case)
elif topic:
    # fallback to topic-only graph
    graph = builder.setup_graph("topic")


We also ensure that the current_language field is set in the state:

current_language = language.lower()


This guarantees that the translation function and route decision can correctly pick the language for translation.

Recap of key functions:

Translation function: Translates content based on current_language and block content.

Root function: Reads the current language from the state.

Route decision: Determines which translation node to call (Hindi or French).

Graph builder: Passes the correct lambda function for translation nodes.

App.py changes: Handles both topic and language input, sets current_language in the state.

Once this is set up, you can run your app, send a POST request with topic and language (e.g., French), and see the content generated in the specified language.

**J) Testing In Postman And Langgraph Studio**

Hello guys! Now that we have integrated everything into the code, it’s time to test our multi-language block generation project. The goal is that when you provide a language like Hindi or French, the blog should be translated accordingly. First, we need to start our FastAPI application. Open the command prompt and run python app.py. This starts the server on port 8000, ready to receive requests.

Before testing in LangGraph Studio, we must ensure the Graph Builder is updated to support languages. Previously, we were using the function build_topic_graph, but now we should use build_language_graph so that the graph can handle both topic input and the requested language. Once this is updated, we can start LangGraph Studio by running langgraph dev in the command prompt. This opens the studio interface where we can visualize our graph, including nodes for title creation, content generation, root, and language-specific translation nodes like Hindi and French, along with the conditional routing edges that connect them.

Next, we can test our API in Postman. For example, to generate a French blog, we provide a JSON input like {"topic": "Ethical AI", "language": "French"} and send a POST request. The server will first generate the blog content in English using the content generator node and then translate it to French using the French translation node, which internally uses the translation function in our blog_node.py. The translation function is defined as def translation(self, state, **kwargs): and takes the block state along with a current_language parameter. Inside this function, we get the original blog content via block_content = state.blog.content and create a message prompt like translation_prompt = "Translate the following content into {current_language} maintaining tone, style, formatting, and culture: {block_content}". This prompt is sent to the LLM using self.lm.with_structured_output(blog).invoke([translation_prompt]) to produce the translated blog in the specified language. Similarly, for Hindi, we send {"topic": "Genetic AI", "language": "Hindi"} in Postman, and the content is translated to Hindi.

The routing in our graph is handled by a root node and a route decision function. The root node is implemented as def root(self, state): return state.current_language, which ensures that the current language is set in the block state. The route_decision function looks at state.current_language and returns either 'Hindi' or 'French', which determines which translation node to invoke. After the translation node completes, the flow moves to the end node. This setup is modular, so if you want to add more languages like German or Spanish, you simply create new translation nodes, reuse the generic translation function, and update the route decision logic.

Finally, you can also test the flow visually in LangGraph Studio to see each node execute step by step. This is particularly useful for debugging or verifying that the routing works correctly. The project is fully modular, uses FastAPI for API integration, supports multiple languages, and can easily be extended. With this setup, you can test topics in any language, debug in LangGraph Studio, and even add additional languages as needed.

# **XIX) Model Context Protocol**

**A) Demo of MCP with Claude Desktop**

Hello guys! Today, we are going to continue our discussion on Model Context Protocol (MCP). In our previous video, we discussed how communication happens between the MCP host, MCP server, and the language model (LM). We saw that as soon as input is received at the MCP host via the MCP client, it communicates with the MCP server, and then the context reaches the LM. In this video, I will show you some practical examples of MCP host, MCP client, and server interactions, along with step-by-step integration.

First, let’s talk about an MCP host. An MCP host can be any interface, including an IDE, a cursor, or an application. For this demonstration, we’ll use Cloudy Desktop, which is a host introduced by Anthropic. Cloudy Desktop internally creates an MCP client to communicate with multiple MCP servers. The architecture is simple: the host with MCP clients uses the MCP protocol to communicate with MCP servers, which can connect to remote APIs, databases, or external services.

To get started, you need to download Cloudy Desktop. For Windows, download the .exe file, and for Mac, use the respective installer. After downloading, install it by clicking Next until the installation completes. Once installed, Cloudy Desktop will look like a chatbot assistant where you can interact with multiple models like Cloudy Sonnet 3.7, Opus 2, Sonnet 4, and more. You can also access external integrations such as DuckDuckGo Search or Airbnb through MCP servers.

Now, let’s focus on configuring MCP servers. Cloudy Desktop has a JSON configuration file named cloudy_desktop_config.json, which specifies which MCP servers are connected. For example, to integrate Airbnb MCP server, the JSON might look like this:

{
    "mcp_servers": {
        "airbnb": {
            "command": "pnp",
            "args": ["airbnb-mcp-server"]
        }
    }
}


You can edit this file using Notepad or any text editor. After adding the configuration, you need to restart Cloudy Desktop to reload the MCP servers. It’s important to close the app fully from Task Manager to ensure the changes take effect. Once restarted, go to File > Settings > Developer to verify that the Airbnb MCP server is running.

After integration, you can test the Airbnb MCP server. For example, you can search for listings using:

Hey, list all the best hotels in San Jose, California.


Cloudy Desktop will prompt: "Cloudy would like to use an external integration." You can click Allow once, and it will fetch listings directly from Airbnb. The response includes room names, links, Wi-Fi availability, ratings, and more—all handled automatically by the MCP protocol.

Similarly, you can integrate DuckDuckGo Search MCP server by adding it to the JSON config:

{
    "mcp_servers": {
        "duckduckgo": {
            "command": "np",
            "args": ["-y", "duckduckgo-mcp-server"]
        }
    }
}


After saving the file and restarting Cloudy Desktop, you can perform web searches using:

Hey, provide me the latest news from DuckDuckGo web search.


Cloudy Desktop will fetch the information from DuckDuckGo, summarize it, and present it in the chat interface. You can also search for technical topics, like asking:

Tell me what is Model Context Protocol from DuckDuckGo web search.


The MCP server fetches the relevant information and sends it back through the MCP client.

The beauty of this setup is its modularity. To integrate any new MCP server, you just need to add its configuration in cloudy_desktop_config.json, restart the host, and it’s ready to use. There is also a website called Symmetry where you can find multiple MCP servers for different services, which will be discussed in upcoming videos.

In summary, MCP allows hosts like Cloudy Desktop to communicate seamlessly with external servers using a standard protocol. Integration is as simple as editing the JSON configuration file, restarting the host, and testing queries. We demonstrated Airbnb and DuckDuckGo MCP servers, and the process can be repeated for any other MCP server, providing a powerful and extensible architecture.

**B) Cursor IDE Installation**

Hello guys! Today, we are going to continue our discussion on Model Context Protocol (MCP). In our previous video, we installed Cloudy Desktop and integrated it with MCP servers. We also explored the cloudy_desktop_config.json file, which allows us to configure MCP servers like Airbnb and DuckDuckGo Search, providing features such as web search and external service integration.

In this video, we will focus on installing Cursor ID, one of the most popular IDEs that can serve as an MCP host. Cursor ID provides a rich UI and allows seamless integration with multiple MCP servers, similar to Cloudy Desktop. You can download it from www.cursor.com
. For Windows, simply click the download link, then double-click the .exe file and follow the installation wizard by clicking Next until the installation completes. Once installed, open Cursor ID by searching for “Cursor” on your system.

Once Cursor ID is open, you will notice a highly interactive interface. On the right-hand side, there’s a panel where you can ask questions, provide context, or even interact with AI agents for various tasks. This is where the MCP client communicates with MCP servers. You can also view your file repository and project files directly in Cursor ID, making it easier to manage code and configurations.

To view and manage MCP servers in Cursor ID, go to File > Preferences > Cursor Settings, then navigate to the MCP section. Here, you can see the MCP servers already configured. For example, I have integrated Playwright, Airbnb, DuckDuckGo Search, Heroku platform, and a custom Weather MCP server. Each MCP server has its own JSON configuration file, similar to how we configured Cloudy Desktop:

{
    "mcp_servers": {
        "airbnb": {
            "command": "pnp",
            "args": ["airbnb-mcp-server"]
        },
        "duckduckgo": {
            "command": "np",
            "args": ["-y", "duckduckgo-mcp-server"]
        },
        "weather": {
            "command": "python",
            "args": ["weather.py"]
        }
    }
}


For example, my custom Weather MCP server is implemented in weather.py, which allows you to fetch weather alerts. Here’s a snippet of how it works:

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/weather", methods=["GET"])
def get_weather():
    location = request.args.get("location", "California")
    # Here you can call an API or custom logic
    response = {
        "location": location,
        "alerts": ["Sunny", "High UV Index", "Windy conditions"]
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5000)


With this setup, when you ask Cursor ID:
Hey, provide me the weather alerts of California, the MCP client communicates with this custom server and retrieves responses like ["Sunny", "High UV Index", "Windy conditions"].

The advantage of Cursor ID is that it behaves like a wrapper over VSCode, so all VSCode functionalities remain intact. You can add or remove MCP servers at any time by updating the JSON configuration and restarting the IDE. Cursor ID also allows you to create, deploy, and interact with your own MCP servers while still supporting third-party services like Airbnb or DuckDuckGo Search.

In summary, this video focused on installing Cursor ID, exploring its interface, and understanding how to integrate both third-party MCP servers and custom MCP servers. This provides a modular approach where you can easily add new services, run them via Python or Node.js, and interact with them seamlessly through Cursor ID’s interface. In upcoming videos, we will build MCP servers from scratch and integrate them into both Cloudy Desktop and Cursor ID.

This concludes the video for today. The key takeaway is that Cursor ID simplifies MCP host management and makes integrating multiple MCP servers extremely straightforward.

**C) Getting Started With Smithery AI**

In this video, we are continuing our discussion on the Model Context Protocol (MCP). In our previous videos, we installed Cloudy Desktop and Cursor ID, and we also saw how to integrate MCP servers into both hosts. I showed you the entire mechanism step by step. Cursor ID acts as a wrapper on top of VSCode, which makes it easier to use if you are familiar with VSCode. The best part about Cursor ID is that it comes with tools that allow you to communicate with agents, provide information, and even integrate MCP servers directly within the IDE. We also explored how to see MCP settings in Cursor ID, and how to add a new MCP server by editing the MCP.json configuration file.

In this video, we will focus on a platform called Smithereen. This platform acts as an aggregator for various MCP servers provided by different service providers. Some of the popular MCP servers available on Smithereen include Memory Tool, Supabase, GitHub, Slack, Notion, and many more. There are also servers for specific purposes, such as DuckDuckGo Search for web search, Perplexity Search, and Playwright for browser automation. The key advantage of Smithereen is that it provides ready-to-use configurations for MCP servers, making it easier for developers to integrate these servers into their hosts or clients.

To demonstrate, let’s take an example of integrating the DuckDuckGo Search MCP server. When you navigate to this server on Smithereen, you will see several important pieces of information: the MCP server URL, a description stating that it enables web search capabilities and fetches web page content intelligently for enhanced LLM interaction, and the tools it provides, such as the search tool and fetch content tool. To integrate this MCP server into your host, go to the Install section on the right-hand side, select your client or host (for example, Cloudy Desktop), and choose the JSON option for configuration. This will give you a JSON snippet specifically for your platform, whether it is Windows, Mac, or Linux. Copy this configuration.

For example, the JSON snippet for Windows may look like this:

{
    "name": "duckduckgo_search",
    "command": "cmd",
    "args": ["--url", "https://example-mcp-server.com"],
    "key": "YOUR_API_KEY"
}


Next, open Cloudy Desktop and navigate to File > Settings > Edit Config. Paste the copied configuration into the cloud_desktop_config.json file. Make sure the formatting is correct and all commas are in place. Save the file, then force close Cloudy Desktop from the Task Manager and reopen it. Once reopened, you should see the new MCP server listed in the interface.

Now you can test the integration by performing searches directly from Cloudy Desktop. For example, you can do a Web Search by typing “Agentic AI” or a Research Paper Search by querying “Attention is All You Need.” The platform will fetch results, including research papers, web pages, and other relevant content. You can also approve responses, allowing the host to save them for later reference.

The process for integrating MCP servers into Cursor ID is very similar. First, copy the JSON configuration from Smithereen. Then, open Cursor ID and go to File > Preferences > Cursor Settings > MCP. Click on Add New Global MCP Server and paste the JSON configuration. Make sure that all commas and brackets are correctly placed. Save the configuration. Unlike Cloudy Desktop, Cursor ID does not require a restart, and the MCP server will appear immediately. You can now use this server to interact directly within Cursor ID, perform web searches, fetch research papers, and more.

Once your MCP servers are integrated, you can also manage them by enabling, disabling, or deleting any server configuration as needed. For example, if a particular MCP server is not working or you do not want to use it, you can simply remove it from the JSON configuration, save the file, and the server will no longer appear in the host interface. This flexibility allows you to customize your host setup according to your needs.

By using Smithereen and adding MCP servers to Cloudy Desktop or Cursor ID, you gain access to multiple services from different providers in one place. This approach ensures that whatever MCP server you need—whether for web search, research, or automation—you can integrate it with minimal effort. You also have the ability to explore each server in a playground environment, test its performance, and decide whether it fits your workflow before fully integrating it.

In summary, this video demonstrated how to integrate MCP servers from Smithereen into both Cloudy Desktop and Cursor ID. We saw how to copy JSON configurations, add them to the host settings, test web searches and research paper searches, and manage server configurations. This process allows you to extend the capabilities of your MCP client or host by connecting it to multiple external services, all while maintaining flexibility and control.

I hope you found this video helpful. In the next video, we will continue exploring more MCP servers and show additional advanced integrations. Thank you for watching, take care, and see you in the next video!

**D) Building MCP Servers With Tools And Client From Scratch Using Langchain**

(Transcript Not Available for now - will take notes from the Video later)

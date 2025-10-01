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

**XIX) Model Context Protocol**

**AA) Demo of MCP with Claude Desktop**

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



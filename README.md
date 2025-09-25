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



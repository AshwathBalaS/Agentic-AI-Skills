# Agentic-AI-Skills

**Table of Contents:**

**A) UV for creating Virtual Environment**

**B) Getting Started with LangGraph - Creating the Environment**

**C) Setting up OpenAI API Key**

**D) Setting up GROQ API Key**

**E) Getting up with LangSmith API Key**

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

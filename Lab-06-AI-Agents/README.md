# Lab Sheet 6 – Building a Tool-Using AI Agent with Ollama

**Module:** PUSL 3205 – Large Language Models and Generative AI  
**Practical:** Lab Sheet 6 – Building a Tool-Using AI Agent with Ollama  
**Domain:** AI Agents, Tool Calling, Memory, RAG and Workflow Automation  
**Technologies:** Python, Ollama, LangChain, SQLite, Google Calendar API, Docker and n8n

---

## 1. Objective

The objective of this practical session is to develop intelligent AI agents capable of interacting with external tools, maintaining conversational memory, retrieving information from knowledge bases, and performing automated tasks.

The practical explores how locally running Large Language Models (LLMs) can be integrated with tools and external services to create more capable AI applications. It also introduces workflow automation using n8n and demonstrates how AI agents can support real-world administrative tasks.

## 2. Tasks and Implementation

### Part A – Building a Calculator Agent

Developed a basic tool-using AI agent using Ollama and LangChain. The `llama3.2:3b` model was used as the local language model.

The implementation involved:

- Initializing the local LLM using LangChain and Ollama.
- Creating a calculator tool to perform mathematical operations.
- Registering the calculator tool with the language model.
- Implementing an agent execution loop to process user requests.
- Testing the agent with mathematical queries.

The agent was designed to recognize calculation requests, invoke the calculator tool, and automatically perform calculations using the Windows Calculator application.

### Exercise – Expanding the Agent with Multiple Tools

Extended the calculator agent by introducing additional tools to perform different system-level operations.

The following tools were implemented:

| Tool | Functionality |
|---|---|
| Calculator | Performs mathematical calculations. |
| Date and Time | Retrieves the current date and time. |
| Notepad | Opens Windows Notepad and writes user-provided text. |
| File Search | Searches for files by name or extension in a specified directory. |
| File Creation | Creates text files with user-provided content. |
| System Information | Retrieves operating system, processor and memory information. |

The tools were tested using different natural-language queries to evaluate the agent's ability to identify and execute the appropriate tool.

### Part B – Giving the AI Agent Memory

Extended the AI agent with conversational memory to maintain context across interactions.

SQLite was used to implement persistent memory storage. The agent was designed with the following components:

- **Short-Term Memory:** Maintains recent conversation context.
- **Long-Term Memory:** Stores useful user information using SQLite.
- **Working Memory:** Maintains information required during the current interaction.
- **Calculator Tool:** Performs mathematical calculations.
- **Memory Search Tool:** Retrieves previously stored user information.

The implementation demonstrated the difference between an agent without memory and an agent capable of remembering previously provided information.

### Part C – RAG-Powered AI Agent

Integrated Retrieval-Augmented Generation (RAG) capabilities into the AI agent using the RAG pipeline developed in Lab Session 5.

The agent was provided with three tools:

1. **Calculator Tool:** Performs mathematical calculations.
2. **Memory Search Tool:** Retrieves previously stored user information.
3. **RAG Search Tool:** Retrieves relevant information from a university knowledge base.

The RAG-powered agent can identify university-related questions and retrieve relevant information from the knowledge base instead of relying entirely on the LLM's internal knowledge.

This integration demonstrates how tool calling, conversational memory and document retrieval can work together within a single AI agent.

### Part D – Integrating Google Calendar Tools

Extended the AI agent by integrating Google Calendar to perform calendar-related operations.

Google Cloud was used to configure the Google Calendar API and OAuth 2.0 authentication.

The following tools were configured:

- **Get Calendar Events:** Retrieves existing events from the user's Google Calendar.
- **Create Calendar Event:** Creates new calendar events using information provided by the user.

The integration allows the AI agent to understand calendar-related requests, retrieve scheduled events and create new events through Google Calendar.

### Part E – Workflow Automation Using n8n

Introduced n8n as a workflow automation platform for developing AI-powered applications.

Docker was used to run n8n locally. An AI agent workflow was then created using the following components:

- **Chat Trigger:** Receives user messages through a chat interface.
- **AI Agent:** Processes user requests and determines the appropriate actions.
- **Ollama Chat Model:** Connects the workflow to a locally running LLM.
- **Simple Memory:** Maintains conversational context.
- **Google Calendar – Get Many Events:** Retrieves calendar events.
- **Google Calendar – Create Event:** Creates new calendar events.

The Ollama model was configured using the local Docker-accessible endpoint:

`http://host.docker.internal:11434`

The workflow demonstrates how locally hosted LLMs can be integrated with external tools and services through a visual automation platform.

## 3. Results and Evidence

The practical activities demonstrate the development of an AI agent capable of performing calculations, interacting with system tools, maintaining memory, retrieving university information and managing calendar events.

The following evidence should be included in the repository:

| Evidence | Description |
|---|---|
| Calculator Agent | Screenshot showing the agent performing calculations. |
| Multiple Tools | Outputs demonstrating the execution of different tools. |
| Agent Memory | Screenshot demonstrating information retrieval from SQLite memory. |
| RAG Integration | Output showing university information retrieved from the knowledge base. |
| Google Calendar | Screenshots showing calendar event retrieval and event creation. |
| n8n Workflow | Screenshot of the completed AI agent workflow. |
| Workflow Execution | Screenshot demonstrating successful interaction through the n8n chat interface. |

## 4. Observations

The practical demonstrated that LLMs become more capable when integrated with external tools and services.

The calculator agent showed how an LLM can delegate specific operations to external functions instead of generating every answer independently. Adding multiple tools expanded the agent's ability to interact with the operating system and perform practical tasks.

SQLite-based memory improved conversational continuity by allowing the agent to retrieve previously stored information. Integrating RAG further improved its ability to provide responses based on external university documents.

Google Calendar integration demonstrated how AI agents can interact with real-world applications through APIs and authentication mechanisms.

Finally, n8n provided a visual approach to connecting local language models, memory components and external services into automated workflows. The practical also highlighted the importance of configuring tool connections, credentials and system instructions correctly.

## 5. Learning Outcomes

Through this practical session, I gained practical experience in:

- Developing tool-using AI agents with Ollama and LangChain.
- Integrating multiple external tools with an LLM.
- Implementing conversational memory using SQLite.
- Understanding the role of short-term and long-term memory in AI agents.
- Integrating RAG capabilities into an AI agent.
- Connecting AI applications with Google Calendar through APIs and OAuth 2.0.
- Building AI-powered automation workflows using n8n.
- Connecting locally hosted LLMs with external services.
- Understanding how AI agents can automate practical tasks through tool calling and workflow orchestration.

## 6. Conclusion

This practical session provided hands-on experience in developing tool-using AI agents and integrating them with external applications and knowledge sources. Starting with a basic calculator agent, the implementation was progressively extended with multiple tools, persistent memory, RAG capabilities and Google Calendar integration.

The introduction of n8n demonstrated how these capabilities can be combined into automated workflows using locally hosted language models. Overall, this practical improved my understanding of AI agents, tool integration, memory management, information retrieval and intelligent workflow automation.


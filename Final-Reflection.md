# Final Reflection – Large Language Models and Generative AI


## 1. Introduction

Throughout the PUSL 3205 – Large Language Models and Generative AI module, I gained practical experience in understanding, developing, fine-tuning and integrating language models into intelligent applications. The laboratory sessions provided an opportunity to explore different approaches to artificial intelligence, starting with traditional machine learning and natural language processing and progressing towards Transformer architectures, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) and AI agents.

Rather than focusing only on theoretical concepts, the practical activities allowed me to implement different techniques using Python, machine learning libraries, locally hosted language models and external services. This portfolio documents my progress across six laboratory sessions, including implementations, experiments, results, observations and the challenges encountered throughout the development process.

## 2. Reflection on Laboratory Activities

### Lab 01 – Sentiment Analysis Using Traditional Machine Learning

The first laboratory session introduced the fundamental concepts of Natural Language Processing through a sentiment analysis task. I worked with a labelled review dataset and implemented text preprocessing techniques, including converting text to lowercase, removing unnecessary characters and punctuation, and cleaning whitespace.

I explored the dataset, examined positive and negative examples, and applied Bag-of-Words and TF-IDF techniques to represent textual information numerically. I then trained a traditional machine learning classification model using Scikit-learn and evaluated its performance using accuracy, precision, recall and F1-score.

This activity helped me understand how raw text can be transformed into meaningful numerical representations that machine learning algorithms can process. It also established an important foundation for understanding how traditional NLP approaches differ from modern language models.

### Lab 02 – AI in Everyday Challenges

The second laboratory session focused on applying LLMs to practical problems encountered in everyday university life. Through a group-based activity, I explored how an LLM could be used as a product-development partner to identify problems, generate ideas and develop potential solutions.

The activity involved identifying a real-world problem, developing a product concept, defining its target users and features, and preparing a visual prototype or mock-up. This demonstrated the importance of providing clear instructions and context when interacting with LLMs.

I learned that prompt engineering is not limited to generating text. It can also support brainstorming, product planning, creative thinking and problem-solving. The activity improved my understanding of how generative AI can be applied beyond traditional programming and data analysis tasks.

### Lab 03 – Developing a Tiny Transformer Language Model

The third laboratory session introduced the internal architecture and working principles of Transformer-based language models. Unlike using an already trained model, this practical involved implementing a small Transformer language model using Python and PyTorch.

I explored the TinyStories dataset, performed text preprocessing and tokenization, and prepared input and target sequences for next-token prediction. I then implemented the main Transformer components, including token embeddings, positional embeddings, masked multi-head self-attention, feed-forward networks, residual connections and layer normalization.

The model was trained using a cross-entropy loss function and an optimizer. I monitored training and validation loss, evaluated the model using perplexity and experimented with generating stories from starting prompts.

This was one of the most technically challenging activities because it required understanding how different neural network components work together. It provided valuable insight into how Transformers learn contextual relationships and generate text sequentially. I also gained a better understanding of how model architecture, training parameters and computational resources influence language generation performance.

### Lab 04 – Fine-Tuning an LLM Using LoRA

The fourth laboratory session introduced the process of adapting an existing pretrained language model to a specific domain using Parameter-Efficient Fine-Tuning (PEFT).

For this practical, I worked with the Qwen2.5-0.5B-Instruct model and prepared a Computer Shop Question Answering dataset containing 40 question-and-answer examples covering computer hardware, components, accessories, purchasing and troubleshooting. The dataset included 32 training examples and 8 unseen evaluation questions.

I implemented supervised fine-tuning using the Hugging Face Transformers ecosystem, PEFT, LoRA and TRL's SFTTrainer. LoRA was configured with a rank of 8, an alpha value of 16 and a dropout rate of 0.05. The model was trained for three epochs using a learning rate of 0.0002.

One of the challenges encountered during this activity was a BF16 compatibility error when training on the available Tesla T4 GPU. I addressed this by switching to FP16 precision and disabling BF16, allowing the training process to complete.

The recorded experiment completed in approximately 46.71 seconds with a training loss of 1.3604. Through qualitative comparisons, I observed that the fine-tuned model could produce shorter and more direct responses to domain-related questions. However, the experiment also highlighted that fine-tuning does not automatically guarantee factual accuracy, and that a small training dataset can limit generalization.

This activity helped me understand how pretrained models can be adapted efficiently without updating all their original parameters. It also improved my knowledge of dataset preparation, training configurations, GPU limitations and model evaluation.

### Lab 05 – Developing a Retrieval-Augmented Generation (RAG) Pipeline

The fifth laboratory session focused on Retrieval-Augmented Generation, an approach that combines information retrieval with language generation to provide responses based on external knowledge sources.

I explored the development of a RAG pipeline using Python, LangChain, Ollama, ChromaDB and embedding models. The implementation involved processing PDF documents, extracting their textual content, dividing the content into smaller chunks, generating vector embeddings and storing them in a vector database.

The pipeline was designed to retrieve relevant document sections based on user queries and provide the retrieved context to a locally hosted LLM for response generation. I used the Llama 3.2 3B model for generating responses and the Qwen3 embedding 0.6B model for producing embeddings.

I also worked on integrating the pipeline with a Streamlit interface to provide a more accessible way of interacting with the knowledge base. The implementation supported document ingestion and retrieval with source metadata.

This practical helped me understand the limitations of relying entirely on a language model's pretrained knowledge. I learned how external documents can provide additional context and how retrieval quality, document chunking, embeddings and similarity search influence the overall RAG process.

It also demonstrated the importance of providing source-related information so that generated answers can be connected to the original documents.

### Lab 06 – Building Tool-Using AI Agents with Ollama

The sixth laboratory session introduced AI agents and demonstrated how LLMs can interact with external tools, maintain memory and perform tasks beyond ordinary text generation.

I initially developed a calculator agent using Ollama and LangChain. I then extended its functionality by integrating additional tools for retrieving the current date and time, opening Notepad, searching and creating files, and retrieving system information.

The practical was further extended by introducing conversational memory using SQLite. This allowed the agent to retrieve previously stored information and maintain context across interactions.

I then integrated RAG functionality into the agent, combining calculator operations, memory retrieval and university knowledge retrieval within a single system. This demonstrated how different capabilities can be coordinated through an agent.

Another important part of this session involved integrating Google Calendar using authentication and API-based tools. I explored retrieving existing calendar events and creating new events through natural-language requests.

Finally, I explored n8n as a visual workflow automation platform. Using Docker, I configured a local n8n environment and developed an AI agent workflow integrating an Ollama chat model, conversational memory and Google Calendar tools.

This laboratory provided a broader understanding of how AI agents can interpret user requests, select appropriate tools and execute actions through external applications. It also demonstrated the potential of combining LLMs, RAG, memory, APIs and workflow automation to develop more practical intelligent systems.

## 3. Overall Challenges and Technical Experience

Throughout the laboratory sessions, I encountered several technical challenges related to model training, library compatibility, computational resources, data preparation and integration between different technologies.

Working with Transformer architectures and fine-tuning models required a better understanding of GPU memory, training precision, optimization and model configurations. In particular, the BF16 compatibility issue during LoRA fine-tuning demonstrated that the hardware environment and numerical precision settings can significantly affect the training process.

Developing the RAG pipeline introduced additional challenges involving document processing, text chunking, embeddings and retrieval configuration. Similarly, integrating AI agents with Google Calendar and n8n required an understanding of APIs, authentication, credentials and communication between locally hosted services.

These experiences taught me that developing an AI application involves more than selecting a model. Successful implementation also depends on appropriate data preparation, system configuration, integration, testing and evaluation.

I also learned the importance of examining actual model outputs rather than relying only on training loss or successful execution. A model may complete training successfully while still producing inaccurate or incomplete responses. Therefore, evaluating the quality and reliability of generated outputs is an essential part of AI development.

## 4. Overall Learning Outcomes and Skills Gained

By completing the six laboratory sessions, I developed a broader understanding of the technologies and techniques involved in modern AI application development.

The main knowledge and skills gained include:

- Applying natural language preprocessing and traditional machine learning techniques to text classification problems.
- Understanding prompt engineering and using LLMs for creative problem-solving and product development.
- Understanding Transformer architecture, attention mechanisms, tokenization and next-token prediction.
- Preparing datasets and fine-tuning pretrained language models using LoRA and PEFT.
- Developing RAG pipelines using document processing, embeddings and vector databases.
- Integrating locally hosted LLMs with external tools and applications.
- Implementing conversational memory and information retrieval using SQLite.
- Connecting AI applications with external services through APIs and authentication.
- Building automated AI workflows using Docker and n8n.
- Evaluating model behaviour, identifying implementation limitations and troubleshooting technical issues.

In addition to technical knowledge, these activities improved my problem-solving abilities, independent learning, experimentation and ability to document technical implementations in a structured manner.

## 5. Conclusion

Overall, the practical sessions provided a valuable opportunity to understand the progression of artificial intelligence technologies, from traditional machine learning and NLP to modern LLM-based systems and autonomous tool-using agents.

One of the most important lessons I gained from this module is that an effective AI application is not necessarily built around a single language model. Different techniques can be combined depending on the requirements of the application. Traditional machine learning can be suitable for classification tasks, fine-tuning can adapt pretrained models to specific domains, RAG can provide access to external knowledge, and AI agents can connect these capabilities with tools and real-world services.

The practical implementations also helped me recognize the limitations of current AI systems, particularly regarding factual accuracy, training data quality, computational requirements and reliable tool execution.

Maintaining this GitHub portfolio allowed me to organize my implementations, document experimental results and reflect on the knowledge gained throughout the module. It also provided an opportunity to develop a practical understanding of the technologies used in modern generative AI applications.

The knowledge and experience gained through these laboratory sessions will provide a useful foundation for developing more advanced AI and data science applications in future academic and professional projects.

# AI Engineering Bootcamp Cohort 4 Midterm

Richard Gower

## Links

[Chainlit Application (hosted on HuggingFace)](https://huggingface.co/spaces/rchrdgwr/AI4Midterm)

[Fine Tuning the Snowflake Model (hosted on Colab)](https://colab.research.google.com/drive/1xAxfWy_3kYg2Arem85Oz-VBb2QkQmU_f)

[Video of beta Chainlit App in action](https://www.loom.com/share/192df355aabc4bbc93912b6dee04ca57?sid=0f5dbdbb-26a1-41f2-8c2e-7a693c7165e0)

The two documents:
1. 2022: [Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) (PDF)
2. 2024: [National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (PDF)

[Snowflake base model](https://huggingface.co/Snowflake/snowflake-arctic-embed-m#:~:text=snowflake-arctic-embed%20is%20a%20suite%20of%20text%20embedding%20models%20that%20focuses)

[Snowflake fine tuned model](https://huggingface.co/rchrdgwr/finetuned-arctic-model)

[Snowflake fine tuned model - 2nd iteration](https://huggingface.co/rchrdgwr/finetuned-arctic-model-2)


## The Problem

`people are concerned about the implications of AI, and no one seems to understand the right way to think about building ethical and useful AI applications for enterprises.` 

This is a big problem and one that is rapidly changing.  Several people you interviewed said that *they could benefit from a chatbot that helped them understand how the AI industry is evolving, especially as it relates to politics.*  Many are interested due to the current election cycle, but others feel that some of the best guidance is likely to come from the government.

## Task 1: Dealing with the Data

#### Review of the two PDFs

The two pdf documents contain text and tables.

The AI Bill of Rights is 73 pages, mostly text with minimal tables, no images and a clear structural breakdown including sections etc. It outlines protections for individuals interacting with AI systems, emphasizing five key principles: 
- Safe and effective systems
- Algorithmic discrimination protections
- Data privacy
- Notice and explanation
- Human alternatives to AI decisions
 It aims to ensure fairness, transparency, and accountability in AI applications. The document serves as a guideline for organizations to manage AI responsibly and safeguard civil rights.

The AI Risk Management Framework is 64 pages, mostly text, but with a lot of tables listing action ID, suggested action and the Generative AI risks.

Possible questions about the AI Bill of Rights include:
- What are the core principles of the AI Bill of Rights?
- How does the document address issues of fairness and bias in AI systems?
- What are the key protections for data privacy and security outlined in the blueprint?
- How does the AI Bill of Rights aim to ensure transparency and explainability of AI systems?
- What measures are recommended for ensuring that AI systems are safe and effective?
- How does the blueprint propose to protect individuals from AI-driven surveillance?
- What are the enforcement mechanisms for the protections outlined in the document?
- How does the document address the rights of individuals when interacting with AI systems?
- How is accountability for AI failures or misuse handled?
- What role does the government play in implementing these AI protections?
- What are the key principles of data privacy protections in the document?
- What are the expected responsibilities for organizations using AI to prevent biased outcomes?
- How does the document propose handling violations of civil rights in AI-driven systems?

Possible questions about the AI Risk Management Framework include:
- What are the key risks associated with generative AI, according to the NIST framework?
- How does the framework suggest managing risks related to confabulations or AI "hallucinations"?
- What are the environmental impacts of training generative AI systems?
- How does the framework address data privacy concerns in AI?
- What guidelines does NIST provide for ensuring information security in generative AI systems?
- What strategies are recommended for mitigating harmful bias in AI models?
- How does the framework address the risks of AI-generated content related to copyright and intellectual property?
- What are the suggested actions for organizations to manage the risks posed by generative AI models?
- How does NIST propose addressing malicious use cases of generative AI, such as the creation of harmful content?
- What are the best practices for decommissioning AI systems, as per the framework?

### Chunking strategy

There are anumber of approaches to chunking the document:
- In all cases we need to consider that there are 2 documents - so when fine tuning we want to ensure that there are representative pages from each document for performing training, evaluation, and testing
- Since the documents are pdf, we need to ensure that the process to import them does not leave the documents split by pages (although that could be considered the simplest of chunking strategies if a really poor one since context typically is spread across pages)
- Since the documents are text rich, the most straight forward chunking strategy would be to break up the text based on characters or tokens and have a reasonable overlap. Chunk sizes of 500-1000 are typically recommended due to the size of the context windows. For this project I have chosen a chunk size of 1000 with an overlap of 100
- The project has been set up to easily test different chunk sizes. Initial tests that I run indicated a chunk size of 100 was poorly performing. A later phase would do a more detailed analysis of the chunk sizes to determine the optimum size for these documents

Alternative chunking strategies that are investigated within this project include:
- Table aware - use an extraction technique that takes into account text data stored in tables
    - Since the AI Risk Management Framework is almost entirely tables this could improve the retrieval process
- Section aware - use an extraction technique that takes into account the structure of the document based on the assumption that text within a section is more likely to be related than text between sections
    - Since the AI Bill of Rights is very structured, section aware chunking could greatly improve the retrieval of appropriate chunks 
- Semantic chunking - chunking based on semantic meaning of the chunks
    - In both documents, chunking based on the meaning of the data could improve retriebal of appropriate chunks

## Task 2: Building a Quick End-to-End Prototype

I created an end to end prototype allowing a user to ask questions about the two documents using the following components:
- [Chainlit](https://github.com/Chainlit/chainlit) - an open source application framework for rapid creation of user interfaces for LL<s
    - Free
    - Easy to use and light weight
    - Provides components for text-based conversations and feedback
    - Easy to deploy using git commands
    - Supports LandChain
    - Simple-to-use decoration wrappers to handle conversation init and message interaction
    - Can use our development stack 
    - Can develop and test locally, dockerize and run, push to host and run
- [LangChain](https://www.langchain.com/) - framework to develop applications that integrate with LLMs - 
    - Free
    - Currently one of the best frameworks for developing LLM applications
    - Great functionality supporting creation of the RAG pipelines
    - Tool I have been trained on
- [Qdrant](https://github.com/qdrant/qdrant) - Vector database for storing and managing embedings
    - Free
    - Performant
    - Scalable
    - Can be memory based, dockerized or accessed as SaaS
    - Fits within our toolset
    - Tool I have been trained on
- [RAGAS](https://docs.ragas.io/en/stable/) - evaluation framework for assessing RAG models
    - Free
    - One of the top tools in this field
    - Supports evaluation of RAG models based on different metrics
    - Supports creation of synthetic data
    - Fits in with the LangChain environment
    - Tool I have been trained on
- [OpenAI Models](https://openai.com/) - LLMs from OpenAI for different purposes
    - Readily accessible through APIs
    - Fairly cheap (sort of)
- [Snowflake Embedding Model](https://huggingface.co/Snowflake/snowflake-arctic-embed-m#:~:text=snowflake-arctic-embed%20is%20a%20suite%20of%20text%20embedding%20models%20that%20focuses) - Sentence transformer embedding model to be fine tuned 
    - Free model
    - Medium sized model can be easily fine tunes
    - Experience with fine tuning this model and achieving good success in achiving better metric scores
- [HuggingFace](https://huggingface.co/) hosting system - free hosting platform that supports multiple web-based applications
    - Free
    - Works as advertized
    - Used by many companies and individuals
    - Apps can be public and private
    - Works with our chosen stack
- [Colab](https://colab.research.google.com/) - cloud-based platform for developing code in Jupyter Notebook
    - Free or cheap
    - Great development environment
    - Provides access to GPU resources for model training
    - Can easily share code

Other components of the development framework:
- VS Code - development
- Git & Github - source code management
- Python and Interactive Python Notebook
- Numerous Python libraries

The code for the Chainlit app is in app.py

This code uses shared functions from utilities

The following video shows an early prototype of the application:
[Video of beta Chainlit App in action](https://www.loom.com/share/192df355aabc4bbc93912b6dee04ca57?sid=0f5dbdbb-26a1-41f2-8c2e-7a693c7165e0)

The application can be accessed here: 
https://huggingface.co/spaces/rchrdgwr/AI4Midterm

## Task 3: Creating a Golden Test Data Set

The software was developed with a modular architecture to allow easy testing and evaluating of different models, different parameters, and different chunking strategies. 

3 classes were developed to hold state to support this:
- AppState - this holds the source documents as well as information about the current run
- ModelRunState - this holds the models, the parameters, the retriever as well as the evaluation results
- RagasState - this holds the Golden Test data set to be used to evaluate the different models

rag_chain.ipynb was used to run the evaluations

Various functions within the utilities were used to support the document retrieval, chunking, and setting up the Qdrant retriever

The Golden Test dataset was created within the notebook and then saved (using Pickl). Subsequent runs retrieved this from disk so that a consistent question set was used.

Note 1: due to severe issues with OpenAI access and allocation limits, only 20 questions were created and subsequently only 5 were used for evaluations. Originally it was planned to create 80 questions.

Note 2: Due to allocation limit issues and timeout issues, some of the evaluation results using RAGAS timed out and resulted in not-a-number (nan) making this impossible to compare. What is interesting is this mostly happened for the RAGAS Context Precision metric with the Fine Tuned Snowflake model 

I used 2 embedding models for the initial analysis:
- text-embedding-3-small - since we have used this before I used it during the initial development and testing
- Snowflake/snowflake-arctic-embed-m - this was fine tuned later

The results from the RAGAS framework:
              Metric  Snowflake_Base/1000/100  TE3/1000/100  Difference
        Faithfulness                 0.400000      0.940000    0.540000
     AnswerRelevancy                 0.383450      0.977401    0.593951
       ContextRecall                 0.550000      1.000000    0.450000
    ContextPrecision                 0.203968      0.775980    0.572012
   AnswerCorrectness                 0.332609      0.750321    0.417711

Looking at the results of the TE3 embedding model, the faithfulness and the answer relevancy is high. The context precision and answer correctess are high, but could be improved

The base Snowflake model is challenged in all scores, but especially context precision and answer correctness.

The pipeline with the base Snowflake model is struggling to perform. 

Basic recommendations:
- Faithfulness - Perform fine tuning on the base Snowflake model using the two documents - this should improve faithfulness by ensuring the answers better match the source documents. We probably want to test and add human feedback into the release to see how well the answers are performing.
- Context Precision - modify the retrieval to ensure the model returns more relevant chunks of material. This could include increasing the chunk size or changing the search algorithms.
- Context Recall - improve the ability to retrieve appropriate information by fine tuning the model against the source data.

Looking at both models, it appears the base retrieval mechanism (chunking etc) could definitely be improved. Experimentation will be needed to identify the best parameters.

Since our goal is to use our own embedding model, we need to fine tune the Snowflake model against the documents and see how this modifies the metrics and also see how the fine tuned model performs against the TE3.

## Task 4: Fine-Tuning Open-Source Embeddings

The fine tuning software was developed on Colab

[Colab notebook - fine tuning the Snoflake embedding model](https://colab.research.google.com/drive/1xAxfWy_3kYg2Arem85Oz-VBb2QkQmU_f)

I ended up running it twice so presenting the 2 fine tuned models here. The performance is similar although the first model performs slightly better.

[Snowflake fine tuned model](https://huggingface.co/rchrdgwr/finetuned-arctic-model)

[Snowflake fine tuned model - 2nd iteration](https://huggingface.co/rchrdgwr/finetuned-arctic-model-2)

The Snowflake model Snowflake/snowflake-arctic-embed-m was selected for the following reasons:
- It is freely available
- The model was designed for natural language processing tasks and able to handle semantic search and text similarity
- Pretrained on wide range of datasets allowing its use across all domains
- It creates high quality embeddings to be used in RAG
- It can be customized through fine tuning to specific domain language
- It is a medium-sized model so can be more easily and quickly trained
- The model can easily be integrated in a variety of platforms and applications
- I have fine-tuned this model before with good success

## Task 5: Assessing Performance

Task 5: Assess the performance of 1) the fine-tuned model, and 2) the two proposed chunking strategies

First a look at the fine tuned model relative to TE3 and the base Snowflake model

| Metric            |   TE3/1000/100 |   Snowflake_Base/1000/100 |   Snowflake_Fine/1000/100 |
|:------------------|---------------:|--------------------------:|--------------------------:|
| Faithfulness      |       0.94     |                  0.4      |                  1        |
| AnswerRelevancy   |       0.977401 |                  0.38345  |                  0.973854 |
| ContextRecall     |       1        |                  0.55     |                  1        |
| ContextPrecision  |       0.77598  |                  0.203968 |                nan        |
| AnswerCorrectness |       0.750321 |                  0.332609 |                  0.632057 |


The fine tuned model has shown incredible improvement in all metrics compared with the base Snowflake model

It is now performing on par with the TE3 model (better in some cases). 

It is still a little lower than I would like on Answer Correctness although it is an improvement on the base model.

It would be interesting to perform mor fine tuning on the model, increasing the epochs and possivly increasing the number of questions it was trained on. It may also be interesting to investigating processing the 2 documents differently - not sure if that is a possible strategy. I would also like to experiment on different chunking sizes and overlaps to determine if there is a better chunking size.

The following table is a comparison of the different chunking strategies tested:
| Metric            |   Snowflake_Fine/1000/100 |   Snowflake_FineSection/1000/100 |   Snowflake_FineTable/1000/100 |   Snowflake_FineSemantic/1000/100 |
|:------------------|--------------------------:|---------------------------------:|-------------------------------:|----------------------------------:|
| Faithfulness      |                  1        |                         0.928342 |                       0.890942 |                          0.798095 |
| AnswerRelevancy   |                  0.973854 |                         0.963411 |                       0.96981  |                          0.967873 |
| ContextRecall     |                  1        |                         0.9      |                       1        |                          0.5      |
| ContextPrecision  |                nan        |                       nan        |                     nan        |                        nan        |
| AnswerCorrectness |                  0.632057 |                         0.466659 |                       0.380591 |                          0.604548 |

Summary of comparisons:
- The fine tuned model that used the recursive text strategy is the best performant
- Changing the chunking strategy adversly affected the answer correctness for all 3 strategies
- The section aware chunking is least impacted
- I am surprised by these differences

### Model Selected
At this point I would recommend the TE3 embedding model although the fine tuned Snowflake model is very close. However it is critical that the first impression that the internal stakedolders get is that the AI tool provides accurate and correct answers.

With some more experimentation and fine tuning I believe that the Snowflake model will become the better choice.

## Task 6: Managing Your Boss and User Expectations

### Story for CEO
This application will allow our employees to interact with two important documents to understand the principles of the AI Bill of Rights and the NIST AI Risk Management Framework. The chatbot is designed to ensure transparency, fairness, and privacy when handling employee queries. 

It will provide answers based on the content of the documents. However sometimes it may provide incorrect information. It has been trained to respond that it cannot answer a question if it has not found the answer within the documents. 

We are working on improving the accuracy of the answers and encourage everyone to use the chatbot and report any incorrect responses or any irregularities.

By providing the documents asa Chatbot people will more easily find the answers to their questions.

We are hoping to add additional documents.


### Incorporating future White-House briefin information
There are a number of things we want to consider when adding additional information to the application:
- Receive notification that changes or new documents exist
- If adding new documents to the embedding model we need to ensure that the document aligns with the tone, structure, and focus of the original documents
- Changes or additions should complement the existing content without causing conflicting responses
- If there are changes in policies, then the previous documents will need to be removed from the vector store
- If necessary we may need to fine tune the model again based on the new documents - this will require a review process
- We may want document versioning or year of publishing so that we can identify and manage changes
- Following addition of new documents there will need to be an evaluation of the chat bots performance both automated and by a human


## Your Final Submission

Please include the following in your final submission:

1. A public link to a **written report** addressing each deliverable and answering each question.
2. A public link to any relevant **GitHub repo**
3. A public link to the **final version of your application** on Hugging Face
4. A public link to your **fine-tuned embedding model** on Hugging Face

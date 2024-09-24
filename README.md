# AI Engineering Bootcamp Cohort 4 Midterm

Richard Gower

## Links

[Chainlit Application (hosted on HuggingFace)](https://huggingface.co/spaces/rchrdgwr/AI4Midterm)

[Fine Tuning the Snowflake Model (hosted on Colab)](https://colab.research.google.com/drive/1xAxfWy_3kYg2Arem85Oz-VBb2QkQmU_f)

[Video of beta Chainlit App in action](https://www.loom.com/share/192df355aabc4bbc93912b6dee04ca57?sid=0f5dbdbb-26a1-41f2-8c2e-7a693c7165e0)

The two documents:
1. 2022: [Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf) (PDF)
2. 2024: [National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) (PDF)


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

Alternative chunking strategies to be investigated include:
- Table aware - use an extraction technique that takes into account text data stored in tables
- Section aware - use an extraction technique that takes into account the structure of the document based on the assumption that text within a section is more likely to be related than text between sections
- Semantic 

 and decide how best to chunk up the data with a single strategy to optimally answer the variety of questions you expect to receive from people.

*Hint: Create a list of potential questions that people are likely to ask!*

</aside>

✅ Deliverables:

1. Describe the default chunking strategy that you will use.
2. Articulate a chunking strategy that you would also like to test out.
3. Describe how and why you made these decisions

## Task 2: Building a Quick End-to-End Prototype

**You are an AI Systems Engineer**.  The SVP of Technology has tasked you with spinning up a quick RAG prototype for answering questions that internal stakeholders have about AI, using the data provided in Task 1.

<aside>
📝

Task 2: Build an end-to-end RAG application using an industry-standard open-source stack and your choice of commercial off-the-shelf models

</aside>

✅ Deliverables:

1. Build a prototype and deploy to a Hugging Face Space, and ~~include the public URL link to your space~~  create a short (< 2 min) loom video demonstrating some initial testing inputs and outputs.
2. How did you choose your stack, and why did you select each tool the way you did?

## Task 3: Creating a Golden Test Data Set

**You are an AI Evaluation & Performance Engineer.**  The AI Systems Engineer who built the initial RAG system has asked for your help and expertise in creating a "Golden Data Set."

<aside>
📝

Task 3: Generate a synthetic test data set and baseline an initial evaluation

</aside>

✅ Deliverables:

1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, answer relevancy, context precision, and context recall.  Provide a table of your output results.
2. What conclusions can you draw about performance and effectiveness of your pipeline with this information?

## Task 4: Fine-Tuning Open-Source Embeddings

**You are a Machine Learning Engineer.**  The AI Evaluation and Performance Engineer has asked for your help in fine-tuning the embedding model used in their recent RAG application build. 

<aside>
📝

Task 4: Generate synthetic fine-tuning data and complete fine-tuning of the open-source embedding model

</aside>

✅ Deliverables

1. Swap out your existing embedding model for the new fine-tuned version.  Provide a link to your fine-tuned embedding model on the Hugging Face Hub.
2. How did you choose the embedding model for this application?

## Task 5: Assessing Performance

**You are the AI Evaluation & Performance Engineer**.  It's time to assess all options for this product.

<aside>
📝

Task 5: Assess the performance of 1) the fine-tuned model, and 2) the two proposed chunking strategies

</aside>

✅ Deliverables

1. Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Test the two chunking strategies using the RAGAS frameworks to quantify any improvements. Provide results in a table. 
3. The AI Solutions Engineer asks you “Which one is the best to test with internal stakeholders next week, and why?”

## Task 6: Managing Your Boss and User Expectations

**You are the SVP of Technology**.  Given the work done by your team so far, you're now sitting down with the AI Solutions Engineer.  You have tasked the solutions engineer to test out the new application with at least 50 different internal stakeholders over the next month.

1. What is the story that you will give to the CEO to tell the whole company at the launch next month?
2. There appears to be important information not included in our build, for instance, the [270-day update](https://www.whitehouse.gov/briefing-room/statements-releases/2024/07/26/fact-sheet-biden-harris-administration-announces-new-ai-actions-and-receives-additional-major-voluntary-commitment-on-ai/) on the 2023 executive order on [Safe, Secure, and Trustworthy AI](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/).  How might you incorporate relevant white-house briefing information into future versions? 

## Your Final Submission

Please include the following in your final submission:

1. A public link to a **written report** addressing each deliverable and answering each question.
2. A public link to any relevant **GitHub repo**
3. A public link to the **final version of your application** on Hugging Face
4. A public link to your **fine-tuned embedding model** on Hugging Face

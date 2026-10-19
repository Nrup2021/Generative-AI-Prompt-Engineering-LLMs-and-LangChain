# Generative AI, Prompt Engineering, LLMs, and LangChain

A beginner-friendly guide to understanding Artificial Intelligence, Generative AI, NLP, NLU, LLMs, Prompt Engineering, and LangChain using real-world Indian recipe examples.

## Table of Contents

1. [Introduction to AI](#1-introduction-to-artificial-intelligence-ai)
2. [Generative AI vs. Discriminative AI](#2-generative-ai-vs-discriminative-ai)
3. [NLP, NLU, and NLG](#3-nlp-nlu-and-nlg)
4. [What Is an LLM?](#4-what-is-a-large-language-model-llm)
5. [How an LLM Works](#5-how-an-llm-works)
6. [Important LLM Terminology](#6-important-llm-terminology)
7. [What Is Prompt Engineering?](#7-what-is-prompt-engineering)
8. [Types of Prompting](#8-types-of-prompting)
9. [What Makes a Good Prompt?](#9-what-makes-a-good-prompt)
10. [What Is LangChain?](#10-what-is-langchain)
11. [Main Components of LangChain](#11-main-components-of-langchain)
12. [Chains and Sequential Chains](#12-chains-and-sequential-chains)
13. [Memory in LangChain](#13-memory-in-langchain)
14. [Agents and Tools](#14-agents-and-tools)
15. [Embeddings, Vector Databases, and RAG](#15-embeddings-vector-databases-and-rag)
16. [Pandas DataFrame Agent](#16-pandas-dataframe-agent)
17. [Complete AI Application Workflow](#17-complete-ai-application-workflow)
18. [Interview Revision Table](#18-interview-revision-table)
19. [Interview Questions and Answers](#19-interview-questions-and-answers)
20. [Final Summary](#20-final-summary)

---

## 1. Introduction to Artificial Intelligence (AI)

**Artificial Intelligence (AI)** is a field of computer science that focuses on building systems capable of performing tasks that normally require human intelligence.

Examples include understanding language, recognizing images, making predictions, solving problems, and making decisions.

### Real-world example

Imagine an Indian Recipe AI Assistant.

A user asks:

> Suggest a healthy Gujarati dish for dinner.

An AI-powered application can interpret the request and recommend a suitable dish, such as vegetable handvo, with a recipe and nutritional information if reliable data is available.

### Types of AI applications

| Application | Example |
|---|---|
| Recommendation systems | Suggesting recipes based on preferences |
| Computer vision | Recognizing a dish from a photograph |
| Predictive AI | Estimating food demand for a restaurant |
| Conversational AI | Answering cooking questions |
| Generative AI | Creating a new recipe description |

**Interview takeaway:** AI is the broad field. Machine Learning, Deep Learning, and Generative AI are related areas within the broader AI ecosystem.

---

## 2. Generative AI vs. Discriminative AI

These approaches solve different types of problems.

### Generative AI

Generative AI creates new content, such as text, images, audio, video, or code.

**Example:**

Prompt:

```text
Create a simple Gujarati Dhokla recipe for beginners.
```

Generated response:

```text
Ingredients:
- 1 cup gram flour (besan)
- Yogurt and water
- Salt and spices
- Fruit salt

Method:
Prepare the batter, add fruit salt, and steam until cooked.
```

The output is newly generated content. Its correctness still needs to be checked.

### Discriminative AI

Discriminative AI learns to distinguish between classes or predict an outcome based on input data.

**Example:**

Input: A food photograph.

Output:

```text
Predicted dish: Dhokla
Confidence: 94%
```

The confidence value here is illustrative, not a real model prediction.

### Comparison

| Feature | Generative AI | Discriminative AI |
|---|---|---|
| Main purpose | Generate new content | Classify or predict |
| Typical output | Text, images, audio, code | Class label, score, probability |
| Example | Generate a Dhokla recipe | Classify a photograph as Dhokla |
| Common models | Generative language and image models | Classifiers and regression models |

**Interview takeaway:** Generative models learn to produce data, while discriminative models learn to distinguish classes or predict targets. The distinction is conceptual; some modern models can perform both kinds of tasks.

---

## 3. NLP, NLU, and NLG

These three terms are related to how computers process human language.

### NLP — Natural Language Processing

NLP is the broad field of enabling computers to process, analyze, and work with human language.

Example:

```text
User: How do I make Gujarati Dhokla?
```

An NLP application processes the user's text.

### NLU — Natural Language Understanding

NLU focuses on interpreting the meaning, intent, and relevant information in language.

The assistant might identify:

```text
Intent: Get a recipe
Dish: Gujarati Dhokla
Difficulty: Not specified
```

### NLG — Natural Language Generation

NLG focuses on generating natural-language responses.

Example:

```text
To make Dhokla, prepare a gram-flour batter,
add fruit salt, and steam it until cooked.
```

### Comparison

| Term | Purpose | Recipe example |
|---|---|---|
| NLP | Process human language | Process the user's question |
| NLU | Interpret meaning and intent | Understand that the user wants a recipe |
| NLG | Generate language | Explain the recipe in natural language |

**Remember:** NLP is the broader field; NLU focuses on understanding, and NLG focuses on generating language. These functions can overlap in modern AI systems.

---

## 4. What Is a Large Language Model (LLM)?

A **Large Language Model (LLM)** is a model trained on large amounts of text and other language data to learn patterns in language and generate contextually relevant output.

Examples of LLM families include GPT, Gemini, Claude, and Llama.

LLMs are used for:

- Answering questions.
- Summarizing documents.
- Generating text and code.
- Extracting information.
- Translating languages.
- Supporting conversational applications.

### Real-world example

User:

```text
Explain how to make Dhokla in five simple steps.
```

The LLM generates an answer based on the prompt and its learned patterns.

An LLM does not automatically know every current fact or verify every response. External data retrieval, tools, or other validation may be needed for reliable results.

### LLM vs. Chatbot

| LLM | Chatbot |
|---|---|
| A language model that processes and generates language | An application that interacts with users conversationally |
| Can power many types of applications | May use an LLM, rules, retrieval, or a combination |
| Does not necessarily provide a user interface | Usually provides a conversation interface |

**Interview takeaway:** An LLM is a model; a chatbot is an application or interface that can use that model.

---

## 5. How an LLM Works

A simplified view of text generation:

```text
User Prompt
    |
    v
Tokenization
    |
    v
Token IDs / Embeddings
    |
    v
Neural Network Processing
    |
    v
Predict Next Token
    |
    v
Generate Additional Tokens
    |
    v
Decode Tokens into Text
    |
    v
Final Response
```

### Step-by-step explanation

1. **Tokenization:** The input text is divided into tokens, which can be words, word fragments, or other text units.
2. **Embeddings:** Tokens are represented numerically so the model can process them.
3. **Model processing:** A neural network, commonly based on the Transformer architecture, processes the input and its context.
4. **Next-token prediction:** The model estimates probabilities for possible next tokens.
5. **Text generation:** A decoding strategy selects tokens repeatedly until generation stops.
6. **Final output:** The resulting tokens are converted back into readable text.

### Example

Input:

```text
Gujarati Dhokla is usually made from
```

The model may predict a continuation such as:

```text
gram flour...
```

This is a simplified illustration, not a guarantee of a particular output.

**Interview takeaway:** Many LLMs generate text autoregressively, predicting one token at a time based on the preceding context.

---

## 6. Important LLM Terminology

| Term | Meaning | Simple example |
|---|---|---|
| Token | A unit of text processed by a model | A word or word fragment |
| Tokenization | Splitting text into tokens | Converting a question into token IDs |
| Embedding | A numerical representation of content | Representing a recipe for similarity search |
| Transformer | A neural-network architecture using attention mechanisms | A common architecture for modern LLMs |
| Attention | Helps the model weigh relationships between tokens | Relating "it" to a previously mentioned dish |
| Context window | The amount of input and conversation context a model can process in one request | Recipe instructions plus conversation history |
| Parameters | Learned numerical values within a model | Values adjusted during model training |
| Temperature | A generation setting affecting output variability | Lower settings often produce more predictable output |
| Max output tokens | A limit on generated output length | Restricting a recipe to a short answer |
| Hallucination | An unsupported or incorrect generated statement | Inventing an ingredient or cooking time |
| Inference | Using a trained model to produce a result | Generating a recipe |
| Fine-tuning | Further training a model on a targeted dataset | Adapting a model for a specialized task |
| Foundation model | A broadly trained model that can support multiple downstream tasks | A model adapted for recipe assistance |

**Important distinction:** Embeddings used for semantic search and token embeddings used internally by language models are related ideas, but they serve different purposes and are not interchangeable.

---

## 7. What Is Prompt Engineering?

**Prompt Engineering** is the process of designing, structuring, and refining instructions given to an AI model to improve the relevance, accuracy, and format of its responses.

A prompt may contain:

- Instructions.
- Context.
- Input data.
- Examples.
- Constraints.
- An output format.

### Poor prompt

```text
Tell me about food.
```

This prompt is vague. The model does not know which food, what details, or how much information to provide.

### Improved prompt

```text
Act as an experienced Indian chef.

Explain how to prepare Gujarati Dhokla
for a beginner.

Include:
1. Ingredients with quantities
2. Preparation steps
3. Cooking time
4. One common mistake to avoid

Use simple English and keep the answer concise.
```

The second prompt provides a role, a specific task, context, constraints, and an expected structure.

### Main goal

The objective is not simply to write a longer prompt. It is to give the model enough relevant information to produce the desired result.

**Interview takeaway:** Prompt engineering improves how a model responds by making the task, context, constraints, and expected output clearer.

---

## 8. Types of Prompting

The most common beginner-level prompting approaches include zero-shot, one-shot, and few-shot prompting.

### 8.1 Zero-shot prompting

**Definition:** Ask the model to perform a task without providing examples of the expected answer.

Prompt:

```text
Give me a recipe for Gujarati Dhokla.
```

The model uses its existing capabilities and the instructions in the prompt.

**When to use:** Straightforward, familiar tasks.

### 8.2 One-shot prompting

**Definition:** Provide one example to demonstrate the desired response pattern.

Prompt:

```text
Example:

Dish: Poha
Ingredients: Flattened rice, onion, mustard seeds.
Steps: Wash the poha, prepare the seasoning, and mix.

Now provide a recipe for Gujarati Dhokla
using the same format.
```

The model can use the example as a guide for formatting and level of detail.

**When to use:** When one example clarifies the desired structure or style.

### 8.3 Few-shot prompting

**Definition:** Provide multiple examples before asking the model to perform the task.

Prompt:

```text
Example 1:
Dish: Poha
Category: Breakfast

Example 2:
Dish: Dhokla
Category: Snack

Example 3:
Dish: Thepla
Category: Breakfast

Now classify Chole using the same format.
```

Possible output:

```text
Dish: Chole
Category: Main Course
```

The output is illustrative; the classification depends on the model and the requested categories.

**When to use:** Classification, extraction, formatting, and tasks where examples demonstrate a pattern.

### 8.4 Role-based prompting

**Definition:** Give the model a role or perspective.

Prompt:

```text
Act as an experienced Gujarati chef.
Explain how to prepare Dhokla for a beginner.
```

**Purpose:** Establishes the perspective, style, or expertise expected in the response.

### 8.5 Contextual prompting

**Definition:** Provide background information and constraints relevant to the task.

Prompt:

```text
I am a beginner and have only 30 minutes.
I have gram flour, yogurt, and basic spices.

Suggest a practical Gujarati snack recipe.
```

**Purpose:** Helps the model tailor its response to the situation.

### 8.6 Instruction-based prompting

**Definition:** State clearly what the model must do.

Prompt:

```text
Explain the Dhokla recipe in five numbered steps.
Mention the ingredients before the instructions.
```

**Purpose:** Makes the task and requirements explicit.

### 8.7 Structured-output prompting

**Definition:** Request the response in a defined format, such as JSON, a table, or a list.

Prompt:

```text
Return a Dhokla recipe as JSON with these fields:
- dish_name
- ingredients
- steps
- estimated_cooking_time
```

Possible output:

```json
{
  "dish_name": "Gujarati Dhokla",
  "ingredients": ["Gram flour", "Yogurt", "Spices"],
  "steps": [
    "Prepare the batter",
    "Add fruit salt",
    "Steam until cooked"
  ],
  "estimated_cooking_time": "25-35 minutes"
}
```

The example is illustrative. In production, use schema validation or a supported structured-output feature when machine-readable output must be reliable.

### 8.8 Chain-of-thought-related prompting

**Definition:** Ask the model to solve a problem carefully or provide a concise explanation of its approach.

Prompt:

```text
Calculate the ingredient quantities needed
to prepare Dhokla for eight people instead of four.

State the scaling factor and provide
the final quantities in a table.
```

The model can provide a concise calculation or explanation. Requesting a detailed internal chain of thought is not necessary; the important part is obtaining a verifiable answer.

### Prompting comparison

| Type | Number of examples | Main purpose |
|---|---:|---|
| Zero-shot | 0 | Perform a task from instructions alone |
| One-shot | 1 | Demonstrate the desired pattern |
| Few-shot | 2 or more | Teach a pattern through multiple examples |
| Role-based | Not defined by example count | Establish a role or perspective |
| Contextual | Not defined by example count | Supply relevant background |
| Instruction-based | Not defined by example count | Specify the task and constraints |
| Structured-output | Not defined by example count | Request a particular response format |
| Chain-of-thought-related | Not defined by example count | Encourage careful problem-solving |

**Remember:** Zero-shot, one-shot, and few-shot are distinguished by the number of examples provided. The other techniques describe different prompt features and can be combined with them.

---

## 9. What Makes a Good Prompt?

A useful prompt often contains five elements.

| Element | Purpose | Dhokla example |
|---|---|---|
| Role | Defines a perspective | Act as an Indian chef |
| Task | States what to do | Explain a Dhokla recipe |
| Context | Provides background | The reader is a beginner |
| Constraints | Sets limits | Keep it under 200 words |
| Output format | Defines the structure | Ingredients followed by numbered steps |

### Reusable prompt template

```text
Role:
[Who should the AI act as?]

Task:
[What should the AI do?]

Context:
[What background information matters?]

Constraints:
[What limitations or requirements apply?]

Output Format:
[How should the answer be presented?]
```

### Common prompt-engineering mistakes

- Asking vague questions.
- Providing conflicting instructions.
- Omitting important context.
- Assuming the model always knows current information.
- Failing to specify the output format.
- Trusting generated facts without validation.
- Providing examples that do not match the expected task.

**Interview takeaway:** A good prompt makes the task clear, provides relevant context, and defines what a useful answer should look like.

---

## 10. What Is LangChain?

**LangChain** is an open-source framework for building applications powered by Large Language Models.

It helps developers connect language models with prompts, application logic, data sources, tools, and retrieval systems.

Without an orchestration framework, developers may need to manage these integrations and workflows manually. LangChain provides reusable abstractions for building them.

### Real-world example: Indian Recipe AI Assistant

Imagine building an application that can:

1. Recommend a popular dish from an Indian state.
2. Retrieve a recipe from a trusted source.
3. Estimate cooking time.
4. Remember the user's dietary preferences.
5. Answer follow-up questions.
6. Search recipe documents when required.

LangChain can help connect the model, prompts, retrieval system, tools, and workflow into a single application.

<img width="1536" height="1024" alt="LangChain Recipe Infographic" src="https://github.com/user-attachments/assets/82e0e492-af4d-4cec-be50-f4c0366a58e8" />


### Basic architecture

```text
User
 |
 v
Prompt / Application Input
 |
 v
LangChain Workflow
 |
 +------> Language Model
 |
 +------> Retrieval / Database
 |
 +------> Tools / External APIs
 |
 v
Process and Combine Results
 |
 v
Response to User
```

LangChain does not itself provide all models, databases, or external tools. These are integrated through the application and its configured providers.

**Interview takeaway:** LangChain is an orchestration framework, not an LLM itself.

---

## 11. Main Components of LangChain

| Component | Meaning | Recipe example |
|---|---|---|
| Models | Interfaces to language models or chat models | Generate a recipe explanation |
| Prompt Templates | Reusable prompts with variables | Insert a state or dish name |
| Chains / Runnables | Connect operations into workflows | Generate a recipe and then summarize it |
| Messages | Represent system, user, and assistant conversation messages | Store recipe-related messages |
| Tools | Functions or services that perform actions | Search a recipe database |
| Agents | Select tools and actions based on a task | Search, retrieve, and calculate quantities |
| Message History | Stores and retrieves conversational messages | Recall earlier conversation messages |
| Documents | Text and associated metadata | Store recipe instructions |
| Output Parsers | Format or validate model output | Parse a response into JSON |
| Retrievers | Find relevant documents | Retrieve Dhokla recipes |
| Vector Stores | Store and search vector embeddings | Find semantically similar recipes |

The exact classes and APIs available depend on the LangChain version and integrations used.

### Prompt Template example

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Suggest a famous dish from {location}."
)

formatted_prompt = prompt.format(
    location="Gujarat"
)

print(formatted_prompt)
```

Output:

```text
Suggest a famous dish from Gujarat.
```

Here, `{location}` is a placeholder that is replaced when the prompt is formatted.

---

## 12. Chains and Sequential Chains

A **chain** connects operations so that the output of one operation can become the input to another.

### Example: Find a Dish, Generate a Recipe, Estimate Cooking Time

Suppose the user provides:

```text
Location: Gujarat
```

The application runs three steps.

#### Chain 1: Dish selection

Input:

```text
Gujarat
```

Output:

```text
Gujarati Dhokla
```

#### Chain 2: Recipe generation

Input:

```text
Gujarati Dhokla
```

Output:

```text
Ingredients and preparation instructions
```

#### Chain 3: Cooking-time estimation

Input:

```text
The generated recipe
```

Output:

```text
Estimated cooking time: 25-35 minutes
```

### Sequential workflow

```text
Location
   |
   v
Chain 1: Identify Dish
   |
   v
Dish Name
   |
   v
Chain 2: Generate Recipe
   |
   v
Recipe
   |
   v
Chain 3: Estimate Cooking Time
   |
   v
Final Response
```

**Key concept:** The output from one step becomes the input for the next step.

### What is the pipe operator (`|`)?

In LangChain Expression Language (LCEL), the pipe operator connects runnable components into a sequence.

Example:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "Suggest a famous dish from {location}."
)

# Assume `model` is an initialized chat model.
chain = prompt | model | StrOutputParser()

result = chain.invoke({"location": "Gujarat"})

print(result)
```

### How the sequence works

```text
Input Dictionary
      |
      v
Prompt Template
      |
      v
Chat Model
      |
      v
String Output Parser
      |
      v
Final Text
```

- `prompt` formats the input.
- `model` generates a response.
- `StrOutputParser()` converts the model response into a string.
- `invoke()` executes the sequence with the supplied input.

**Interview takeaway:** LCEL uses `|` to compose runnable components into a workflow. Older tutorials may use classes such as `LLMChain` or `SequentialChain`; check the API version when reading examples.

---

## 13. Memory in LangChain

Memory-related mechanisms help an application maintain context across multiple interactions.

In modern LangChain applications, conversation history and agent state are often managed using message histories or LangGraph persistence, depending on the architecture.

### Example conversation

```text
User: I prefer vegetarian food.

Assistant: I can suggest vegetarian dishes.

User: Suggest something for dinner.

Assistant: You could try vegetable handvo.
```

The assistant can use the earlier preference when generating a later answer if that information is available in the conversation history or application state.

### Simplified workflow

```text
New User Message
       |
       v
Retrieve Relevant History
       |
       v
Combine Context and Prompt
       |
       v
Generate Response
       |
       v
Save Updated Conversation State
```

**Important:** A model does not automatically remember every previous conversation. The application must supply the relevant history or use a configured persistence mechanism.

**Interview takeaway:** Memory and message history provide continuity; they do not automatically give an LLM unlimited or permanent memory.

---

## 14. Agents and Tools

An **agent** uses a language model to decide which actions or tools are appropriate for a task.

A tool is a callable capability, such as a search function, database query, calculator, or external API.

### Real-world example

User asks:

```text
Find a popular Punjabi dish,
get its recipe, and calculate the ingredients
needed for four people.
```

The agent might choose to:

1. Search for a popular Punjabi dish.
2. Retrieve its recipe.
3. Use a calculator or Python function to scale ingredients.
4. Combine the results into a final response.

### Agent workflow

```text
User Request
     |
     v
Agent / LLM
     |
     v
Choose an Action
     |
     +------> Search Tool
     |
     +------> Recipe API
     |
     +------> Calculator
     |
     v
Combine Tool Results
     |
     v
Final Answer
```

### Chain vs. Agent

| Chain | Agent |
|---|---|
| Follows a predefined workflow | Can select actions based on the task |
| Steps are generally specified in advance | Tool selection may vary |
| Useful for predictable tasks | Useful for tasks requiring flexible decisions |
| Example: recipe, then cooking time | Example: search, retrieve, and calculate |

**Interview takeaway:** A chain defines the workflow; an agent can choose the workflow or tools needed to complete a request.

---

## 15. Embeddings, Vector Databases, and RAG

### What is an embedding?

An embedding is a numerical representation of content that helps systems compare semantic similarity.

For example, a recipe search system may represent these queries similarly:

```text
Find a chickpea curry.
Find an Indian curry made with chickpeas.
```

The wording differs, but the meanings are related.

### What is a vector database?

A vector database stores and retrieves vector representations so an application can find semantically similar content.

Examples include vector search capabilities offered by dedicated vector databases and other database systems.

### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with language generation.

Instead of relying only on the model's learned patterns, the application retrieves relevant information and supplies it as context to the model.

### Recipe example

User asks:

```text
How long should I steam Gujarati Dhokla?
```

A RAG application can:

1. Search a collection of trusted recipe documents.
2. Retrieve relevant cooking instructions.
3. Pass the retrieved text to the LLM.
4. Generate an answer grounded in the available information.

### RAG workflow

```text
User Question
      |
      v
Create Query Embedding
      |
      v
Search Vector Store
      |
      v
Retrieve Relevant Documents
      |
      v
Add Retrieved Context to Prompt
      |
      v
LLM Generates Response
      |
      v
Answer Based on Retrieved Context
```

### Why use RAG?

- Provides access to external or private documents.
- Makes it easier to update knowledge without retraining the model.
- Can improve factual grounding when relevant, reliable information is retrieved.
- Can support citations or source references when the application implements them.

RAG does not guarantee correctness. Retrieval quality, source quality, and model behavior still matter.

### RAG vs. Fine-tuning

| RAG | Fine-tuning |
|---|---|
| Retrieves relevant external information at runtime | Further trains a model on a targeted dataset |
| Useful for changing documents and knowledge sources | Useful for adapting behavior, style, or task performance |
| Does not inherently change model weights | Updates model weights during training |
| Example: retrieve the latest recipe document | Adapt a model to follow a specialized recipe format |

**Interview takeaway:** RAG provides relevant external context; fine-tuning changes the model through additional training.

---

## 16. Pandas DataFrame Agent

A Pandas DataFrame is a tabular data structure in Python, similar to a spreadsheet.

A DataFrame agent lets a user ask natural-language questions about that data.

### Example dataset

Imagine a dataset containing 139 Indian recipes.

| Dish | Region | Cooking Time |
|---|---|---:|
| Dhokla | Gujarat | 30 minutes |
| Chole | Punjab | 45 minutes |
| Dal Baati | Rajasthan | 60 minutes |

The values are illustrative.

### Example questions

```text
How many rows are in the dataset?

Which dish has the longest cooking time?

How many recipes are from Gujarat?
```

If the DataFrame contains 139 rows, the answer to the first question is 139.

A configured DataFrame agent can translate a natural-language question into Python operations, execute them, and return the result.

### Workflow

```text
Natural-Language Question
          |
          v
LLM Interprets the Request
          |
          v
Generate Python Operations
          |
          v
Execute Against DataFrame
          |
          v
Return the Result
```

**Security note:** Executing model-generated Python code can be dangerous. Use restricted execution environments, appropriate permissions, and safeguards. Do not run untrusted generated code with unrestricted access to files, credentials, or networks.

**Interview takeaway:** A DataFrame agent allows natural-language interaction with structured data by using programmatic operations.

---

## 17. Complete AI Application Workflow

Consider the complete Indian Recipe AI Assistant.

User asks:

```text
Suggest a vegetarian dish from Gujarat.
Give me the recipe and estimated cooking time.
Remember that I prefer simple recipes.
```

### Application architecture

```text
                USER
                  |
                  v
           User's Question
                  |
                  v
        Prompt Engineering
      Instructions + Context
                  |
                  v
           LANGCHAIN APP
                  |
          +-------+-------+
          |       |       |
          v       v       v
         LLM    Memory   Retriever
          |       |       |
          |       |       v
          |       |   Recipe Documents
          |       |
          +-------+
                  |
                  v
       Chain or Agent Workflow
                  |
                  v
        Validate / Format Output
                  |
                  v
           Final Response
```

### How the concepts fit together

1. **Generative AI:** Generates the recipe explanation.
2. **NLP / NLU:** Helps process the user's natural-language request and identify the intended task.
3. **LLM:** Processes the prompt and generates text.
4. **Prompt engineering:** Defines the task, constraints, context, and expected format.
5. **LangChain:** Connects model calls, prompts, retrieval, and tools.
6. **Chains:** Execute predefined steps in sequence.
7. **Memory:** Supplies relevant conversation history when configured.
8. **Agents:** Select tools when a task needs flexible action selection.
9. **RAG:** Retrieves relevant recipe documents to ground the answer.
10. **Output parsing:** Formats the result for display or downstream processing.

Not every application needs every component. A simple chatbot may need only a prompt and a model; a more advanced assistant may need retrieval, tools, memory, and validation.

---

## 18. Interview Revision Table

| Question | Short answer |
|---|---|
| What is AI? | The broad field of building systems that perform tasks associated with intelligence. |
| What is Generative AI? | AI that creates new content such as text, images, audio, or code. |
| What is Discriminative AI? | AI that distinguishes classes or predicts targets from input data. |
| What is NLP? | The field of processing and working with human language. |
| What is NLU? | Interpreting the meaning and intent of language. |
| What is NLG? | Generating natural-language output. |
| What is an LLM? | A large language model trained to process and generate language. |
| What is a token? | A unit of text processed by a model. |
| What is prompt engineering? | Designing instructions and context to guide a model's response. |
| What is zero-shot prompting? | Performing a task without providing examples in the prompt. |
| What is one-shot prompting? | Providing one example in the prompt. |
| What is few-shot prompting? | Providing multiple examples in the prompt. |
| What is LangChain? | A framework for connecting LLMs with prompts, tools, data, and workflows. |
| What is a prompt template? | A reusable prompt with variables. |
| What is a chain? | A sequence of connected operations. |
| What does `|` mean in LCEL? | It composes runnable components into a sequence. |
| What is memory? | A mechanism for retrieving and maintaining conversation state or history. |
| What is an agent? | A system that uses an LLM to select actions or tools. |
| What is a tool? | A callable function or external capability available to the application. |
| What is an embedding? | A numerical representation of content. |
| What is a vector database? | A system for storing and searching vector representations. |
| What is RAG? | Retrieving external information and supplying it to a model as context. |
| What is fine-tuning? | Further training a model on a targeted dataset. |
| What is hallucination? | Generated content that is incorrect or unsupported. |
| What is a DataFrame agent? | An agent that answers questions about tabular data using programmatic operations. |

---

## 19. Interview Questions and Answers

### Q1. What is the difference between AI, Generative AI, and an LLM?

**Answer:**

AI is the broad field of building intelligent systems. Generative AI focuses on creating new content. An LLM is a type of model used for language tasks and is commonly used in Generative AI applications.

### Q2. What is the difference between NLP, NLU, and NLG?

**Answer:**

NLP is the broad field of language processing. NLU focuses on understanding meaning and intent, while NLG focuses on generating natural-language output.

### Q3. What is prompt engineering, and why is it important?

**Answer:**

Prompt engineering is the process of designing and refining model instructions. It is important because clear tasks, relevant context, constraints, and output formats can improve the usefulness of responses.

### Q4. Explain zero-shot, one-shot, and few-shot prompting.

**Answer:**

Zero-shot prompting provides no examples, one-shot provides one example, and few-shot provides multiple examples. Examples help demonstrate the expected pattern or format.

### Q5. What is LangChain, and why would you use it?

**Answer:**

LangChain is an open-source framework for building LLM applications. It helps connect prompts, models, tools, retrieval systems, and application workflows using reusable components.

### Q6. What is the difference between a chain and an agent?

**Answer:**

A chain executes a predefined sequence of operations. An agent uses an LLM to select actions or tools based on the current task.

### Q7. What is the pipe operator in LangChain?

**Answer:**

In LangChain Expression Language, the pipe operator (`|`) connects runnable components. For example, a prompt can be connected to a model and then an output parser.

### Q8. How does memory work in an LLM application?

**Answer:**

The application retrieves relevant conversation history or stored state and supplies it to the model. After processing, the application may save new messages or updated state for later requests.

### Q9. What is the difference between RAG and fine-tuning?

**Answer:**

RAG retrieves external information and adds it to the prompt at runtime. Fine-tuning involves additional model training to adapt its behavior or performance for a particular task.

### Q10. Can an LLM independently access a database or execute Python?

**Answer:**

Not inherently. The application must provide suitable tools or execution capabilities. An agent may choose to use those tools, while the application or runtime performs the actual operations.

### Q11. What is a hallucination, and how can it be reduced?

**Answer:**

A hallucination is an incorrect or unsupported generated response. Risk can be reduced through reliable retrieval, clear instructions, constrained output formats, validation, and source verification, although it cannot be eliminated completely.

### Q12. Does every AI application need LangChain?

**Answer:**

No. A simple application can call a model provider directly. LangChain is useful when an application benefits from reusable orchestration, retrieval, tools, or more complex workflows.

---

## 20. Final Summary

The easiest way to remember these concepts is to follow the journey of an Indian Recipe AI Assistant:

```text
AI
 |
 v
Generative AI
 |
 v
NLP / NLU / NLG
 |
 v
LLM
 |
 v
Prompt Engineering
 |
 v
LangChain
 |
 +----> Chains
 |
 +----> Memory
 |
 +----> Agents and Tools
 |
 +----> Embeddings and RAG
 |
 v
Complete AI Application
```

- **AI** is the broad field.
- **Generative AI** creates new content.
- **NLP, NLU, and NLG** describe language-related processing, understanding, and generation.
- **LLMs** process and generate language.
- **Prompt engineering** guides the model using instructions and context.
- **LangChain** connects models with application logic, data, and tools.
- **Chains** execute connected steps.
- **Memory** maintains access to relevant conversation history or state.
- **Agents** choose actions and tools.
- **RAG** retrieves external information to support generated responses.

The central idea is that an LLM provides language capabilities, while prompt engineering and application frameworks such as LangChain help developers turn those capabilities into useful software.

---

## Suggested Learning Order

1. Understand AI, Generative AI, Discriminative AI, NLP, NLU, and NLG.
2. Learn LLMs, tokens, embeddings, inference, and hallucinations.
3. Practice prompt engineering with zero-shot, one-shot, and few-shot examples.
4. Learn LangChain prompts, models, runnables, and the pipe operator.
5. Build sequential workflows, message history, and tool integrations.
6. Learn embeddings, vector stores, and RAG.
7. Build a small AI assistant and practice explaining its architecture in an interview.

**Best interview strategy:** For each concept, prepare three things: a one-sentence definition, one real-world example, and one comparison with a related concept.

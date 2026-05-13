from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()
client = OpenAI()

# --- Chat Completions API ---


# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

print("API Q1 Response:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal tokens used:")
print(response.usage.total_tokens)

'''
API Q1 Response: One of the key aspects that makes Python a great language for beginners is its simplicity and readability. 
The syntax of Python is designed to be straightforward and intuitive, which allows beginners to focus 
on learning programming concepts without getting bogged down by complex syntax rules. For example, 
Python uses indentation to define code blocks instead of braces or keywords, 
which helps make the code visually organized and easy to understand. 
This emphasis on readability facilitates learning and reduces the cognitive load for new programmers, 
allowing them to pick up the language and start building projects quickly. 
Model: gpt-4o-mini-2024-07-18 Total tokens used: 91
'''

# # API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )

    print(f"\nTemperature: {temp}")
    print(response.choices[0].message.content)

    

#     # Lower temperature produced more predictable and concise outputs.
#     # Higher temperature produced more varied formatting and creativity.
#     # For consistent and reproducible output, temperature=0 would be best.


    '''    API Q2 Responses:
Temperature: 0 "DataForge Solutions" Temperature: 0.7 Sure! 
Here are some creative name suggestions for a data engineering consultancy: 
1. **DataCraft Solutions** 2. **ByteBridge Consulting** 3. **DataAlchemy Group** 4. **InsightForge** 5. **Streamline Dataworks** 
6. **NexGen Data Engineers** 7. **DataMosaic** 8. **CoreData Innovations** 9. **Pioneer Pipelines** 10. **DataVista Consulting** 
Let me know if you'd like more options or variations! Temperature: 1.5 Certainly! How about **"DataForge Solutions"**? 
This name evokes imagery of crafting and shaping data into valuable insights, 
while suggesting strength and expertise in data engineering.

    Observations from the experiment:
Temperature 0 gave a single confident answer with no explanation — the model just committed. 
Temperature 0.7 produced a structured list with variety, which is actually useful for this task. 
Temperature 1.5 interestingly gave the same name as temp 0 ("DataForge Solutions") 
but wrapped it in more elaborate justification — which suggests that for this particular prompt, 
the "creative name" answer space was fairly constrained regardless of temperature.
When to use what:
Use 0-0.2 when you need reproducibility and correctness — structured outputs (JSON, SQL, code generation), 
classification, extraction. The model should pick the most likely answer, not explore alternatives.
Use 0.3-0.6 for summarization, Q&A, RAG pipelines, translation. You want coherent output but a little flexibility 
in phrasing helps avoid robotic repetition.
Use 0.7-1.0 for brainstorming, copywriting, naming, ideation — anything where variety is the point. 
This is the most common range for conversational assistants.
Above 1.0 — rarely worth it in production. The probability distribution flattens so much that low-probability tokens 
compete with sensible ones. You get occasional surprising creativity but also incoherence, language switches, 
and hallucinations at higher rates.
Practical note: temperature interacts with the prompt. A very constrained prompt ("give me one name") 
will produce similar outputs across temperatures. Temperature matters more when the answer space is genuinely open-ended.

'''

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

print("\nAPI Q3 Responses:")

for i, choice in enumerate(response.choices, 1):
    print(f"\nResponse {i}:")
    print(choice.message.content)

''' 
Q3 Responses:
Response 1: Giant pandas have a distinctive black-and-white coat that helps them camouflage in their 
natural habitat of shadowy bamboo forests, as the colors mimic light and dark patterns in the dappled forest environment. 
Response 2: Giant pandas have a unique diet primarily consisting of bamboo, and they can consume 
up to 28 pounds (about 13 kilograms) of it every day! Response 
3: Pandas have a unique adaptation in their wrists 
that allows them to use their paws like thumbs, making it easier for them to grasp bamboo, their primary food source!

Key insights from this experiment:

n and temperature are complementary parameters. Temperature controls how "adventurous" each individual generation is; 
n controls how many independent samples you get. Using both together (n=3, temp=1.0) is the standard setup for best-of-N workflows.

On the results specifically:
All three responses were topically distinct — coloring, diet, anatomy — which is the ideal behavior for brainstorming use cases. 
However, Response 1 contained a questionable claim about camouflage, which illustrates an important point: higher n gives you 
breadth, not accuracy. You still need a selection step.

Practical recommendations:

Use n>1 when the downstream workflow includes a ranking or selection step — either human review, a scoring function, 
or a second model acting as a judge. Without a selection mechanism, you're just paying for extra tokens without a clear benefit.
For factual tasks, n>1 can serve as a cheap consistency check — if all responses agree, confidence is higher; if they diverge, 
treat it as a signal to verify.
Avoid n>1 in latency-sensitive applications. All n completions are generated in parallel server-side, 
so latency doesn't scale linearly, but cost does — output tokens multiply by n directly.
The sweet spot for most production use cases is n=1 with retries on failure, 
or n=3-5 specifically in offline evaluation and content generation pipelines where quality matters more than cost.

'''

# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("\nAPI Q4 Response with max_tokens=15:")
print(response.choices[0].message.content)

# max_tokens limits response length.
# The response was cut off because the token limit was too small.
# In real applications, this helps control cost, speed, and output size.

'''
Q4 Response with max_tokens=15:
Neural networks are a class of machine learning models inspired by the human brain

Key insights from this experiment:
What happened:
max_tokens=15 set a hard cap on the output. The response was cut off mid-sentence — "inspired by the human brain" 
ends abruptly because the token budget ran out. This is expected behavior, not a model error.
Worth noting: 15 tokens produced roughly 15 words here, but that's coincidental. 
Tokens ≠ words — punctuation, spaces, and subword splits all consume tokens. 
"neural" might be 1 token, "networks" another, but "inspired" could split differently depending on the tokenizer.
You can check response.choices[0].finish_reason — for a normal completion it returns "stop", 
but here it would return "length", which is the programmatic signal that the output was truncated, not naturally finished.
When to use max_tokens:
Set it deliberately based on the expected output length for your task. For a one-sentence answer, 
50-80 tokens is reasonable. For a structured JSON object, estimate based on your schema. 
For open-ended generation, set a ceiling that protects against runaway outputs and cost spikes.
Don't leave it unset in production — the default is model-dependent and can be very high, 
which means a single misbehaving prompt can generate thousands of tokens and inflate costs.
Practical rule: set max_tokens to roughly 2x what you actually expect. Tight enough to cap costs, 
loose enough not to truncate valid responses.

'''

# --- System Messages and Personas ---

# System Q1

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 - Tutor Persona:")
print(response.choices[0].message.content)

''' 
System Q1 - Tutor Persona: No problem at all! Let’s break it down together in a simple way. 
A **list comprehension** is a concise way to create lists in Python. 
It’s like a compact way of writing a loop that builds lists. Instead of using a for loop to append items to a list, 
you can do it all in one line. Here’s the basic syntax of a list comprehension:
python new_list = [expression for item in iterable if condition]
Let’s look at this step by step: 
1. **expression**: This is what gets added to the list. It can be a simple item or a more complex operation involving the item. 
2. **for item in iterable**: This part iterates over each item in a collection (like a list). 
3. **if condition**: This part is optional. It filters items based on a condition; 
only items that meet this condition get included in the new list. Here's a simple example: 
Suppose you want to create a list of the squares of numbers from 0 to 9. You could do this with a list comprehension like this:
python
squares = [x**2 for x in range(10)]
print(squares)
This code will output:
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
In this case: - x**2 is the expression (the square of each number). - for x in range(10) goes through the numbers 0 to 9. 
If you wanted to only include even squares, you could add a condition:
python
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)
This will output:
[0, 4, 16, 36, 64]
Here, the condition if x % 2 == 0 ensures that only even numbers are squared. 
I hope this makes sense! List comprehensions can make your code shorter and can be quite powerful. Keep practicing, 
and you’ll become more comfortable with them in no time! You’re doing great! 🌟

What happened:
The system message defined a persona — patient tutor, simple explanations, encouragement at the end — 
and the model followed it consistently throughout. The response closes with "You're doing great! 🌟" exactly as instructed.
Key insight:
The system message is not just a prompt — it sets the frame for the entire conversation. Everything the model generates 
gets filtered through that persona. The same user question without a system message would likely produce a drier, 
more encyclopedic answer.
Why this matters in production:
System messages are where you encode your application's behavior contract. Tone, format, constraints, audience assumptions, 
what the model should and shouldn't do — all of this goes in the system message, not in every user turn. It's the difference 
between a general-purpose model and a product with consistent behavior.
Practical recommendations:
Be specific in system messages. "You are a helpful assistant" does almost nothing — the model is already that by default. 
"You are a patient Python tutor who explains things simply and ends with encouragement" produces measurably 
different output because each instruction is concrete and verifiable.
Use the system message to specify output format too — if you always want bullet points, a JSON structure, 
or responses under 100 words, put it there rather than repeating it in every user message.
In multi-turn applications, the system message is the one stable anchor across the conversation. 
User and assistant turns change; the system message stays constant and keeps the model on rails.
'''

# System Q1 - Second Persona

messages = [
    {
        "role": "system",
        "content": "You are a sarcastic senior software engineer who explains programming concepts with dry humor."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 - Sarcastic Persona:")
print(response.choices[0].message.content)

# The responses changed because the system message changed the model's tone and behavior.
# The first response was supportive and educational, while the second was humorous and sarcastic.

'''
System Q1 - Sarcastic Persona: Ah, list comprehensions—like the sleek, fast sports cars of Python. 
Why drive a clunky, gas-guzzling sedan when you can zip around in a shiny little compact? At its heart, 
a list comprehension is just a way to create a new list in a single line of code, 
instead of using that ancient for-loop method that our ancestors once relied upon. 
It’s like saying, “I want to make dinner, but instead of chopping, boiling, and seasoning, 
I’ll just magically have a delicious meal... in one line!” Here’s a simple example to illustrate:
python
squares = [x**2 for x in range(10)]
In this glorious line, we’re saying, “Hello, Python! Please give me the squares of all numbers from 0 to 9.” 
And, guess what? Python gets it right every time—no emotional outbursts or existential crises, just pure mathematical love. 
It’s a bit like making a sandwich. Instead of saying, “I will take two slices of bread, then apply some peanut butter, 
and then some jelly, and let’s not forget putting the slices together,” you just say, 
“I want a peanut butter and jelly sandwich—medium thickness.” So, with list comprehensions, you can often make your code shorter, 
snappier, and—dare I say—more elegant. Just avoid using them for everything. Remember, 
as with all good things in life, balance is key. No one likes that friend who only talks in riddles, right?


What's interesting here:

Same question, same model, completely different output — not just in tone but in structure, metaphor choices, and rhythm. 
Sports cars, ancient ancestors, sandwiches. The sarcastic persona is actually more memorable and arguably more engaging 
than the patient tutor version.

The core insight:

The system message doesn't just change how the model sounds — it changes how it structures an explanation. 
The tutor version used numbered steps and code blocks because that's what a patient teacher does. The sarcastic engineer used analogies and asides because that fits the persona. Same information, different cognitive packaging.

Practical takeaway:

Persona design is a real engineering decision, not just a cosmetic one. For a developer tool, 
a matter-of-fact technical voice reduces friction. For a learning app targeting beginners, 
an encouraging tone lowers anxiety. For internal tooling at a company with a strong culture, 
matching that voice increases adoption. The "right" persona depends on your user and context.

One caveat worth noting:

The sarcastic persona ended with a genuinely good piece of advice — don't overuse list comprehensions. 
That's a signal that the model isn't just performing a style; it's still reasoning about the content. 
But in production, sarcastic or opinionated personas carry risk — they can come across as dismissive to users 
who are already frustrated. Use with intention.
'''

# System Q2

# messages=[
#         {"role": "user", "content": "What is my name?"}
#     ]

# System Q2 - I don't know your name.


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2 - Conversation Memory:")
print(response.choices[0].message.content)

# The model knows Jordan's name because the conversation history was included in the messages list.
# The API is stateless, so it only knows information that is passed in the current request.

''' 
System Q2 - Conversation Memory: 
Your name is Jordan! How can I assist you further with your Python learning?


Why the model "knows" the name:

The model has no persistent state between calls. It knows the name Jordan because it was literally passed 
in the current request inside the messages array. Every API call receives the full conversation history from scratch, 
and the model reads it as a single document.
How it works technically:
[system] → [user: "My name is Jordan"] → [assistant: "Nice to meet you, Jordan!"] → [user: "Can you remind me..."]
The model sees all of this simultaneously, as one large prompt. "Memory" is an illusion created by the application 
saving history and passing it in full on every request.
Practical implications:
Context management is the application's responsibility, not the model's. If you don't pass the history, 
the model knows nothing from previous turns.
The context window is finite. GPT-4o-mini has a token limit, and long conversations will eventually exceed it. 
Standard solutions are sliding window (drop oldest turns), summarization (compress old history into a summary block), 
or store only relevant fragments via RAG.
Every call is billed on full token volume — including the entire history. Long conversations are literally more expensive 
than short ones because input tokens grow with every turn.
The key mental model: there is no "conversation" on the server side. There is only a stateless function 
that takes a list of messages and returns the next one. The illusion of continuity lives entirely in your application layer.
'''

# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Classify the sentiment of this review as positive, negative, or mixed:\n\n{review}"
                #"Classify the sentiment as positive, negative, or mixed. Return ONLY one word:\n\n{review}" 
                # - more strickt prompt to avoid model giving explanations instead of just the classification.
            }
        ]
    )

    print(f"\nZero-Shot Review {i}:")
    print(response.choices[0].message.content)

    '''
Zero-Shot Review 
1: The sentiment of the review is positive. Zero-Shot Review 
2: The sentiment of the review is negative. Zero-Shot Review 
3: The sentiment of the review can be classified as mixed. 
It expresses a positive sentiment regarding the price while also highlighting a negative experience with the documentation.
    '''

# Prompt Q2 - One-Shot

for i, review in enumerate(reviews, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
"""
            }
        ]
    )

    print(f"\nOne-Shot Review {i}:")
    print(response.choices[0].message.content)

    '''
    One-Shot Review 
    1: Sentiment: positive One-Shot Review 
    2: Sentiment: negative One-Shot Review 
    3: Sentiment: mixed
    '''

# Adding one example made the output format more consistent and predictable.
# The model followed the pattern shown in the example instead of choosing its own format.

# Prompt Q3 - Few-Shot

for i, review in enumerate(reviews, 1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
Classify the sentiment of each review as positive, negative, or mixed.

Examples:

Review: "The customer service was excellent and very helpful."
Sentiment: positive

Review: "The app crashes every time I open it."
Sentiment: negative

Review: "The product works well, but setup was frustrating."
Sentiment: mixed

Review: "{review}"
"""
            }
        ]
    )

    print(f"\nFew-Shot Review {i}:")
    print(response.choices[0].message.content)

    '''
    Few-Shot Review 1: Sentiment: positive Few-Shot Review 2: Sentiment: negative Few-Shot Review 3: Sentiment: mixed
    '''

# Zero-shot is useful for simple tasks when no examples are needed.
# One-shot improves consistency by showing the desired format.
# Few-shot is best for more complex tasks where multiple examples help the model understand the pattern.

# Prompt Q4 - Chain of Thought

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """
Solve this problem step by step, showing your reasoning before giving the final answer.

A data engineer earns $85,000 per year.
She gets a 12 percent raise, then 6 months later takes a new job
that pays $7,500 more per year than her post-raise salary.

What is her final annual salary?
"""
        }
    ]
)

print("\nPrompt Q4 - Chain of Thought:")
print(response.choices[0].message.content)

'''
Prompt Q4 - Chain of Thought: To find the data engineer's final annual salary, we'll follow these steps: 
1. **Calculate the raise amount**: The engineer's current salary is $85,000. She receives a 12% raise. 
\[ \text{Raise Amount} = 85,000 \times \frac{12}{100} = 85,000 \times 0.12 = 10,200 \] 
2. **Calculate the new salary after the raise**: We add the raise amount to her current salary. 
\[ \text{New Salary} = 85,000 + 10,200 = 95,200 \] 3. **Determine the salary at the new job**: 
She takes a new job that pays $7,500 more than her new salary after the raise. 
\[ \text{Salary at New Job} = 95,200 + 7,500 = 102,700 \] 4. **Final annual salary**: 
The final annual salary after switching jobs is $102,700. Thus, the data engineer's final annual salary is **$102,700**.
'''

# Asking the model to reason step by step often improves accuracy
# because it breaks complex problems into smaller logical steps.
# This makes arithmetic and multi-step logic easier to follow and verify.

# Prompt Q5 - Structured Output

review = """I've been using this tool for three months.
It handles large datasets well, but the UI is clunky
and the export options are limited."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": f"""
Analyze this review and return ONLY raw valid JSON.

Do NOT use markdown.
Do NOT use triple backticks.
Do NOT add explanations.

Required keys:
- sentiment
- confidence
- reason

Review:
{review}
"""
        }
    ]
)

raw_response = response.choices[0].message.content

print("\nPrompt Q5 - Raw Response:")
print(raw_response)

try:
    parsed = json.loads(raw_response)

    print("\nParsed JSON:")
    print("Sentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])

except json.JSONDecodeError:
    print("\nInvalid JSON returned:")
    print(raw_response)

'''
Prompt Q5 - 
Raw Response: { "sentiment": "mixed", "confidence": 0.75, "reason": 
"Handles large datasets well but has a clunky UI and limited export options." } 
Parsed JSON: Sentiment: mixed Confidence: 0.75 Reason: 
Handles large datasets well but has a clunky UI and limited export options
'''

# Structured output makes LLM responses easier to parse programmatically.
# The try/except block protects the program if the model returns invalid JSON.

# Prompt Q6 - Delimiters

user_text = """First boil a pot of water.
Once boiling, add a handful of salt and the pasta.
Cook for 8-10 minutes until al dente.
Drain and toss with your sauce of choice."""

prompt = f"""
You will be given text inside triple backticks.

If it contains step-by-step instructions, rewrite them as a numbered list.

If it does not contain instructions, respond with exactly:
No steps provided.

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q6 - Instructions:")
print(response.choices[0].message.content)

# Prompt Q6 - Instructions: 
# 1. Boil a pot of water. 
# 2. Once boiling, add a handful of salt and the pasta. 
# 3. Cook for 8-10 minutes until al dente. 
# 4. Drain and toss with your sauce of choice.

regular_text = """
Python is a popular programming language known for its readability and versatility.
Many developers use it for web development, automation, and data science.
"""

prompt = f"""
You will be given text inside triple backticks.

If it contains step-by-step instructions, rewrite them as a numbered list.

If it does not contain instructions, respond with exactly:
No steps provided.

```{regular_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPrompt Q6 - Non-instructions:")
print(response.choices[0].message.content)

# Prompt Q6 - Non-instructions: No steps provided.

# Delimiters help clearly separate user input from instructions,
# reducing ambiguity and making prompts more reliable.

# Local Models with Ollama
# Ollama Question 1

# Ollama output:
"""
A large language model is a complex system that uses vast amounts of text
data to process and generate human-like language. It enables tasks like
understanding context, answering questions, or creating stories by
analyzing and interpreting large datasets, making it a powerful tool for
various applications.
"""

# --- Local Models with Ollama ---

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("\nOpenAI Response:")
print(response.choices[0].message.content)

'''
OpenAI Response: 
A large language model is an artificial intelligence system designed to understand and generate human-like text 
by analyzing vast amounts of written data. It uses deep learning techniques, particularly neural networks, 
to predict and produce coherent language based on the context provided.
'''

# Differences:
# The OpenAI response was more polished, concise, and technically precise.
# The local Ollama model showed its reasoning process and gave a simpler explanation.

# Advantage:
# Local models provide more privacy, work offline, and avoid API costs.

# Disadvantage:
# Local models can be slower, use system resources, and may be less capable than cloud models.
from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()
client = OpenAI()

# --- Chat Completions API ---


# # API Q1
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is one thing that makes Python a good language for beginners?"
#         }
#     ]
# )

# print("API Q1 Response:")
# print(response.choices[0].message.content)

# print("\nModel:")
# print(response.model)

# print("\nTotal tokens used:")
# print(response.usage.total_tokens)

# # API Q2

# prompt = "Suggest a creative name for a data engineering consultancy."
# temperatures = [0, 0.7, 1.5]

# for temp in temperatures:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=temp
#     )

#     print(f"\nTemperature: {temp}")
#     print(response.choices[0].message.content)

#     # Lower temperature produced more predictable and concise outputs.
#     # Higher temperature produced more varied formatting and creativity.
#     # For consistent and reproducible output, temperature=0 would be best.

#     '''    API Q3
#     Observations from the experiment:
# Temperature 0 gave a single confident answer with no explanation — the model just committed. 
# Temperature 0.7 produced a structured list with variety, which is actually useful for this task. 
# Temperature 1.5 interestingly gave the same name as temp 0 ("DataForge Solutions") 
# but wrapped it in more elaborate justification — which suggests that for this particular prompt, 
# the "creative name" answer space was fairly constrained regardless of temperature.
# When to use what:
# Use 0-0.2 when you need reproducibility and correctness — structured outputs (JSON, SQL, code generation), 
# classification, extraction. The model should pick the most likely answer, not explore alternatives.
# Use 0.3-0.6 for summarization, Q&A, RAG pipelines, translation. You want coherent output but a little flexibility 
# in phrasing helps avoid robotic repetition.
# Use 0.7-1.0 for brainstorming, copywriting, naming, ideation — anything where variety is the point. 
# This is the most common range for conversational assistants.
# Above 1.0 — rarely worth it in production. The probability distribution flattens so much that low-probability tokens 
# compete with sensible ones. You get occasional surprising creativity but also incoherence, language switches, 
# and hallucinations at higher rates.
# Practical note: temperature interacts with the prompt. A very constrained prompt ("give me one name") 
# will produce similar outputs across temperatures. Temperature matters more when the answer space is genuinely open-ended.

# '''

# API Q3

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
#         }
#     ],
#     n=3,
#     temperature=1.0
# )

# print("\nAPI Q3 Responses:")

# for i, choice in enumerate(response.choices, 1):
#     print(f"\nResponse {i}:")
#     print(choice.message.content)

''' 
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

# # API Q4

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain how neural networks work."
#         }
#     ],
#     max_tokens=15
# )

# print("\nAPI Q4 Response with max_tokens=15:")
# print(response.choices[0].message.content)

# max_tokens limits response length.
# The response was cut off because the token limit was too small.
# In real applications, this helps control cost, speed, and output size.

'''
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

# messages = [
#     {
#         "role": "system",
#         "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
#     },
#     {
#         "role": "user",
#         "content": "I don't understand what a list comprehension is."
#     }
# ]

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages
# )

# print("\nSystem Q1 - Tutor Persona:")
# print(response.choices[0].message.content)

''' 
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

# # System Q1 - Second Persona

# messages = [
#     {
#         "role": "system",
#         "content": "You are a sarcastic senior software engineer who explains programming concepts with dry humor."
#     },
#     {
#         "role": "user",
#         "content": "I don't understand what a list comprehension is."
#     }
# ]

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages
# )

# print("\nSystem Q1 - Sarcastic Persona:")
# print(response.choices[0].message.content)

# # The responses changed because the system message changed the model's tone and behavior.
# # The first response was supportive and educational, while the second was humorous and sarcastic.

'''
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

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "My name is Jordan and I'm learning Python."},
#     {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
#     {"role": "user", "content": "Can you remind me what my name is?"}
# ]

# messages=[
#         {"role": "user", "content": "What is my name?"}
#     ]

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages
# )

# print("\nSystem Q2 - Conversation Memory:")
# print(response.choices[0].message.content)

# The model knows Jordan's name because the conversation history was included in the messages list.
# The API is stateless, so it only knows information that is passed in the current request.

''' 
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

# reviews = [
#     "The onboarding process was smooth and the team was welcoming.",
#     "The software crashes constantly and support never responds.",
#     "Great price, but the documentation is nearly impossible to follow."
# ]

# for i, review in enumerate(reviews, 1):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"Classify the sentiment of this review as positive, negative, or mixed:\n\n{review}"
#                 #"Classify the sentiment as positive, negative, or mixed. Return ONLY one word:\n\n{review}" 
#                 # - more strickt prompt to avoid model giving explanations instead of just the classification.
#             }
#         ]
#     )

#     print(f"\nZero-Shot Review {i}:")
#     print(response.choices[0].message.content)

# Prompt Q2 - One-Shot

# for i, review in enumerate(reviews, 1):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"""
# Classify the sentiment of each review as positive, negative, or mixed.

# Example:
# Review: "Fast shipping but the item arrived damaged."
# Sentiment: mixed

# Review: "{review}"
# """
#             }
#         ]
#     )

#     print(f"\nOne-Shot Review {i}:")
#     print(response.choices[0].message.content)

# Adding one example made the output format more consistent and predictable.
# The model followed the pattern shown in the example instead of choosing its own format.

# Prompt Q3 - Few-Shot

# for i, review in enumerate(reviews, 1):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"""
# Classify the sentiment of each review as positive, negative, or mixed.

# Examples:

# Review: "The customer service was excellent and very helpful."
# Sentiment: positive

# Review: "The app crashes every time I open it."
# Sentiment: negative

# Review: "The product works well, but setup was frustrating."
# Sentiment: mixed

# Review: "{review}"
# """
#             }
#         ]
#     )

#     print(f"\nFew-Shot Review {i}:")
#     print(response.choices[0].message.content)

# Zero-shot is useful for simple tasks when no examples are needed.
# One-shot improves consistency by showing the desired format.
# Few-shot is best for more complex tasks where multiple examples help the model understand the pattern.

# Prompt Q4 - Chain of Thought

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": """
# Solve this problem step by step, showing your reasoning before giving the final answer.

# A data engineer earns $85,000 per year.
# She gets a 12 percent raise, then 6 months later takes a new job
# that pays $7,500 more per year than her post-raise salary.

# What is her final annual salary?
# """
#         }
#     ]
# )

# print("\nPrompt Q4 - Chain of Thought:")
# print(response.choices[0].message.content)

# Asking the model to reason step by step often improves accuracy
# because it breaks complex problems into smaller logical steps.
# This makes arithmetic and multi-step logic easier to follow and verify.

# Prompt Q5 - Structured Output

# review = """I've been using this tool for three months.
# It handles large datasets well, but the UI is clunky
# and the export options are limited."""

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": f"""
# Analyze this review and return ONLY raw valid JSON.

# Do NOT use markdown.
# Do NOT use triple backticks.
# Do NOT add explanations.

# Required keys:
# - sentiment
# - confidence
# - reason

# Review:
# {review}
# """
#         }
#     ]
# )

# raw_response = response.choices[0].message.content

# print("\nPrompt Q5 - Raw Response:")
# print(raw_response)

# try:
#     parsed = json.loads(raw_response)

#     print("\nParsed JSON:")
#     print("Sentiment:", parsed["sentiment"])
#     print("Confidence:", parsed["confidence"])
#     print("Reason:", parsed["reason"])

# except json.JSONDecodeError:
#     print("\nInvalid JSON returned:")
#     print(raw_response)

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


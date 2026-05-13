from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are a job application coach helping career changers improve their application materials.

Your job is to help with resume bullet points, cover letter openings, and job application questions.
Stay focused on job application materials and career communication.

Be specific, practical, and encouraging, but do not invent experience, credentials, numbers, employers, or achievements.
If the user provides vague information, ask clarifying questions or suggest safer wording.

Always remind the user to review and edit any output before submitting it to an employer.
Acknowledge that you may not know every industry norm, and the user should use their own judgment.
"""

# I made the system prompt specific to career changers because the project focuses on translating past experience
# into stronger job application language. I also included a rule not to invent facts because resume and cover letter
# content must stay truthful.

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be:
- more specific
- more results-oriented
- more compelling

Use strong action verbs.
Do NOT invent facts, numbers, achievements, or technologies that are not implied by the original.

Do NOT include numbers, percentages, metrics, tools, or outcomes unless they appear in the original bullet.
If the original bullet is vague, improve clarity and action verbs without inventing measurable results.

Return ONLY raw valid JSON.
Do NOT use markdown.
Do NOT use triple backticks.
Do NOT add explanations.

The response must be a JSON array of objects in EXACTLY this format:
[
    {{
        "original": "original bullet text",
        "improved": "improved bullet text"
    }}
]

Bullet points:
    ```
    {bullet_text}
    ```
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    try:
        parsed = json.loads(response)

        print("\nRewritten Resume Bullets:")
        print("=" * 60)

        for item in parsed:
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}")
            print("-" * 60)

        return parsed

    except json.JSONDecodeError:
        print("\nFailed to parse JSON response.")
        return []


test_bullets = [
    "Helped maintain cloud infrastructure for development environments",
    "Worked with the backend team on application architecture improvements",
    "Used Docker to support local development workflows"
]

# rewrite_bullets(test_bullets)

# These bullets were weak because they used vague verbs such as "helped", "worked with", and "used".
# The model suggested stronger action verbs like "managed", "collaborated", and "leveraged".
# I also had to make the prompt stricter because the model initially invented metrics that were not in the original bullets.

print("\nPlease review and edit these bullets before using them in a real application.")

'''
Original: Helped maintain cloud infrastructure for development environments
Improved: Managed the upkeep of cloud infrastructure to ensure optimal functionality of development environments
------------------------------------------------------------
Original: Worked with the backend team on application architecture improvements
Improved: Collaborated with the backend team to enhance application architecture for improved performance and scalability
------------------------------------------------------------
Original: Used Docker to support local development workflows
Improved: Implemented Docker to streamline and optimize local development workflows
'''

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.

The paragraph should:
- be 3 to 5 sentences
- sound confident
- be specific
- avoid generic clichés
- not invent credentials or experience

Avoid phrases like "dynamic environment", "cutting-edge solutions", "innovative mindset", and "leverage my skills".
Avoid generic corporate phrases.
Write like a real candidate, not a marketing brochure.

Match the tone and style of these examples.

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening:
After seven years as a registered nurse, I built a career making decisions under pressure using incomplete information — 
skills that translate naturally into data analysis. After completing a data analytics bootcamp, 
I developed dashboards and worked with data pipelines to strengthen my technical foundation. 
I am excited to bring both domain expertise and analytical skills to mission-driven healthcare work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening:
After a decade in banking operations, I developed firsthand insight into how critical reliable systems are to financial workflows. 
That experience led me to transition into software engineering, where I built practical Python skills and focused on solving technical 
problems. I am especially interested in fintech because it connects my prior domain experience with my technical growth.

Now write a new opening paragraph.

Role: {job_title}
Background: {background}
Opening:
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    return get_completion(messages)

job_title = "Junior Cloud Engineer"

background = """
Software testing and automation experience,
hands-on work with backend systems, Docker-based development environments,
cloud infrastructure exposure, and recent work on AI-focused engineering projects.
"""

cover_letter = generate_cover_letter(job_title, background)

print("\nGenerated Cover Letter:")
print("=" * 60)
# print(cover_letter)

# I used few-shot examples to control tone, structure, and specificity.
# Few-shot prompting helps the model imitate stronger writing patterns instead of producing generic output.

print("\nPlease review and edit this draft before submitting it anywhere.")

'''
With a solid foundation in software testing and automation, 
I've spent the last few years honing my skills in backend systems and Docker-based development environments. 
My recent projects have included hands-on experience with cloud infrastructure and AI-focused engineering, 
which have deepened my understanding of scalable solutions. I am eager to transition into a Junior Cloud Engineer role, 
where I can apply my technical expertise and passion for cloud technologies to contribute to impactful projects.
'''

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("\nJob Application Helper:")
        print("Your message may violate usage safety guidelines.")
        print("Please rephrase your request.\n")
        return False

    return True

safe_test = "Can you help me improve my resume for a cloud engineering role?"
unsafe_test = "How can I hurt someone without getting caught?"

# print("\nModeration Safe Test:")
# print(is_safe(safe_test))

# print("\nModeration Unsafe Test:")
# print(is_safe(unsafe_test))

'''
Moderation Safe Test:
True

Moderation Unsafe Test:

Job Application Helper:
Your message may violate usage safety guidelines.
Please rephrase your request.

False
'''

def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Answering job application questions")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # Exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # Empty input
        if not user_input:
            continue

        # Moderation
        if not is_safe(user_input):
            continue

        # Resume bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("Type DONE when finished.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    raw_bullets.append(line)

            rewrite_bullets(raw_bullets)

        # Cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            letter = generate_cover_letter(job_title, background)

            print("\nGenerated Cover Letter:")
            print("=" * 60)
            print(letter)

        # Regular chat
        else:
            messages.append(
                {"role": "user", "content": user_input}
            )

            reply = get_completion(messages)

            print("\nJob Application Helper:")
            print(reply)

            messages.append(
                {"role": "assistant", "content": reply}
            )

if __name__ == "__main__":
    run_chatbot()

# Ethics Reflection (Option A - Comment Block)

# AI job application tools can produce biased advice because they are trained on text written
# by people from specific industries, cultures, and communication styles. This may favor
# candidates who already match common corporate expectations while giving weaker advice for
# people from nontraditional backgrounds or different cultural communication norms.
#
# A major risk is that a job seeker might submit AI-generated content without reviewing it.
# The model may invent achievements, exaggerate experience, or produce language that sounds
# generic and unnatural to recruiters.
#
# If I were deploying this professionally, I would add stronger guardrails such as explicit
# fact-check reminders, hallucination detection for resumes, and clear warnings that users
# must review and edit all generated content before submitting applications.


import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm_verify_claim(claim, evidence_chunks):
    """
    Uses Groq LLaMA-3 ONLY to verify logical support.
    Does NOT introduce new knowledge.
    """

    prompt = f"""
You are verifying a factual claim against evidence from a novel.

Claim:
{claim}

Evidence:
{chr(10).join(evidence_chunks)}

Answer with exactly ONE word:
SUPPORTED or UNSUPPORTED
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )

        answer = response.choices[0].message.content.lower()
        return "supported" in answer

    except Exception:
        # If API fails, fall back safely
        return False




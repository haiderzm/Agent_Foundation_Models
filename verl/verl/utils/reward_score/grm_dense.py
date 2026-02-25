import asyncio
import json
import logging
import os
import random
import re
import traceback
from typing import Dict, List

import openai
from verl.utils.reward_score.utils import verify_format_simple, verify_format_repetition
import time

# Setup logging
logger = logging.getLogger(__name__)

# LLM评估模板
LLM_EVALUATION_PROMPT_TEMPLATE = """
Please determine if the predicted answer is equivalent to the labeled answer. 
Question:  {question} 
Labeled Answer:  {gt_answer} 
Predicted Answer: {pred_answer}  

**Rules**:
If the prediction and answer are semantically equivalent despite the expression order, the description format, and the use of measurement units and the order, then your judgement will be correct.
{{  
"rationale": "your rationale for the judgement, as a text", 
"judgement": "your judgement result, can only be 'correct' or 'incorrect' 
}}
"""

OBS_CONSISTENCY_PROMPT_TEMPLATE = """
You are checking whether a final answer is SUPPORTED by the provided evidence.
You MUST NOT use outside knowledge.
If the evidence does not explicitly contain the answer or does not allow it to be directly derived, mark it as not supported.

Return ONLY valid JSON:
{{
  "supported": 0 or 1,
  "rationale": "short reason"
}}

Question:
{question}

Final answer:
{final_answer}

Evidence (from tool observations):
{evidence}
"""

# Environment variables for LLM judge
LLM_JUDGE_API_KEY = os.getenv("GRM_API_KEY")
LLM_JUDGE_BASE_URL = os.getenv("GRM_BASE_URL", "https://api.openai.com/v1")
LLM_JUDGE_MODEL_NAME = os.getenv("GRM_MODEL_NAME", "gpt-4.1-mini")

# OBS_JUDGE_API_KEY = os.getenv("OBS_JUDGE_API_KEY", LLM_JUDGE_API_KEY)
# OBS_JUDGE_BASE_URL = os.getenv("OBS_JUDGE_BASE_URL", LLM_JUDGE_BASE_URL)
# OBS_JUDGE_MODEL_NAME = os.getenv("OBS_JUDGE_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507")

OBS_JUDGE_API_KEY = os.getenv("GRM_API_KEY")
OBS_JUDGE_BASE_URL = os.getenv("GRM_BASE_URL", "https://api.openai.com/v1")
OBS_JUDGE_MODEL_NAME = os.getenv("GRM_MODEL_NAME", "gpt-4.1-mini")

obs_client = openai.AsyncOpenAI(
    api_key=OBS_JUDGE_API_KEY,
    base_url=OBS_JUDGE_BASE_URL,
)


client = openai.AsyncOpenAI(
    api_key=LLM_JUDGE_API_KEY,
    base_url=LLM_JUDGE_BASE_URL,
)

client_sync = openai.OpenAI(
    api_key=LLM_JUDGE_API_KEY,
    base_url=LLM_JUDGE_BASE_URL,
)

def extract_answer(response: str) -> str:
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, response, re.DOTALL))
    
    if not matches:
        return ""
    
    # Return the last answer if multiple exist
    return matches[-1].group(1).strip()

def extract_crawl_observations_from_response(response: str) -> str:
    """
    Extract ONLY <observation> blocks that correspond to <crawl_page>.
    Ignores web_search observations.
    """
    if not response:
        return ""

    # Match: <crawl_page>...</crawl_page> ... <observation>...</observation>
    pairs = re.findall(
        r"<crawl_page>.*?</crawl_page>\s*<observation>(.*?)</observation>",
        response,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Clean + dedup
    cleaned = []
    seen = set()
    for p in pairs:
        txt = p.strip()
        if not txt:
            continue
        key = " ".join(txt.split())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(txt)

    return "\n\n---\n\n".join(cleaned)

async def llm_judge_single(question: str, pred_answer: str, gt_answer: str, max_retries: int = 3) -> float:
    """
    Judge a single prediction using LLM.
    Returns:
        Score: 1.0 if correct, 0.0 if incorrect
    """
    do_print = random.randint(1, 32) == 1
    
    formatted_prompt = LLM_EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        pred_answer=pred_answer,
        gt_answer=gt_answer
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=LLM_JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": formatted_prompt},
                ],
            )
            response_text = response.choices[0].message.content

            try:
                response_json = json.loads(response_text)
                judgement = response_json.get("judgement", "").lower()
                
                if judgement == "correct":
                    score = 1.0
                elif judgement == "incorrect":
                    score = 0.0
                else:
                    raise ValueError(f"Invalid judgement: {judgement}")

                if do_print:
                    print("--- LLM Judge Evaluation ---")
                    print(f"Question: {question}")
                    print(f"Predicted: {pred_answer}")
                    print(f"Ground Truth: {gt_answer}")
                    print(f"Score: {score}")
                    print(f"Rationale: {response_json.get('rationale', '')}")
                    print("---------------------------\n")

                return score

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[WARNING] Could not parse LLM judge response on attempt {attempt + 1}: {response_text}")
                print(f"[WARNING] Got {e}")

        except Exception as e:
            print(f"[WARNING] An error occurred during LLM judge API call on attempt {attempt + 1}: {e}")
            traceback.print_exc()

        # If not the last attempt, wait before retrying
        if attempt < max_retries:
            print(f"[INFO] LLM Judge fail, Retrying LLM judge in 1 second...")
            await asyncio.sleep(1)
    
    print(f"[ERROR] All {max_retries + 1} LLM judge attempts failed. Returning default score of 0.")
    return 0.0

async def obs_consistency_single(question: str, final_answer: str, evidence: str, max_retries: int = 2) -> float:
    if not final_answer.strip() or not evidence.strip():
        return 0.0

    prompt = OBS_CONSISTENCY_PROMPT_TEMPLATE.format(
        question=question,
        final_answer=final_answer,
        evidence=evidence,
    )

    for attempt in range(max_retries + 1):
        try:
            resp = await obs_client.chat.completions.create(
                model=OBS_JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a strict evidence-grounding checker."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()
            data = json.loads(text)
            supported = data.get("supported", 0)
            
            print("$"*30)
            print("Evidences : ", evidence)
            print("Final Ans : ", final_answer)
            print("Supported : ", supported)
            print("$"*30)

            return 1.0 if supported in (1, "1", True) else 0.0
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                return 0.0
            
def compute_observation_consistency_batch(questions: List[str], responses: List[str]) -> List[float]:
    final_answers = [extract_answer(r) for r in responses]
    evidences = [extract_crawl_observations_from_response(r) for r in responses]

    async def _run():
        tasks = [
            obs_consistency_single(q, a, e)
            for q, a, e in zip(questions, final_answers, evidences)
        ]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())


def compute_score_grm_batch_with_obs(
    questions: List[str],
    ground_truths: List[Dict],
    responses: List[str],
    prompts: List[str],
    data_sources: List[str],
    extra_infos: List[Dict],
    wf: float = 0.7,
    wo: float = 0.3,
    lambda_rep: float = 0.05,
    **kwargs
) -> List[Dict]:
    # 1) final correctness (GPT)
    pred_answers = [extract_answer(r) for r in responses]
    gt_answers = [gt["target"] for gt in ground_truths]

    async def _run_all():
        final_tasks = [
            llm_judge_single(q, pa, ga)
            for q, pa, ga in zip(questions, pred_answers, gt_answers)
        ]
        evidences = [extract_crawl_observations_from_response(r) for r in responses]
        obs_tasks = [
            obs_consistency_single(q, pa, ev)
            for q, pa, ev in zip(questions, pred_answers, evidences)
        ]
        final_scores = await asyncio.gather(*final_tasks)
        obs_scores = await asyncio.gather(*obs_tasks)
        

        return final_scores, obs_scores
        # return await asyncio.gather(*tasks)

    # final_correct_scores = asyncio.run(_async_final())
    final_correct_scores, obs_scores = asyncio.run(_run_all())

    # 2) observation consistency (Qwen)
    # obs_scores = compute_observation_consistency_batch(questions, responses)

    # 3) repeated_steps (rule-based) — easiest from steps
    def repeated_steps_from_response(response: str) -> int:
        def norm(x: str) -> str:
            return " ".join(x.lower().split())

        web = [norm(q) for q in re.findall(r"<web_search>(.*?)</web_search>", response, re.DOTALL) if q.strip()]
        crawl = [norm(u) for u in re.findall(r"<crawl_page>(.*?)</crawl_page>", response, re.DOTALL) if u.strip()]

        return (len(web) - len(set(web))) + (len(crawl) - len(set(crawl)))

    results = []
    for fc, oc, resp in zip(final_correct_scores, obs_scores, responses):
        rep = float(repeated_steps_from_response(resp))

        score = wf * float(fc) + wo * float(oc) - lambda_rep * rep
        score = max(-1.0, min(1.0, score))

        results.append({
            "score": score,
            "llm_judge": float(fc),
        })

    return results
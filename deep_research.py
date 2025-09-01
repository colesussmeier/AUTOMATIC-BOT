import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from forecasting_tools import clean_indents
from datetime import datetime

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = AsyncOpenAI(api_key=api_key, timeout=3600)

async def call_deep_research(question: str, type: str, lower_bound: str = None, upper_bound: str = None):

    prompt = format_prompt(question, type, lower_bound, upper_bound)

    response = await client.responses.create(
        model="o3-deep-research",
        input=prompt,
        tools=[
            {"type": "web_search_preview"},
            {
                "type": "code_interpreter",
                "container": {"type": "auto"}
            },
        ],
    )

    response = f"""
    <<<DEEP RESEARCH>>>

    {response.output_text}
    """
    
    return response


def format_prompt(question: str, type: str, lower_bound: str = None, upper_bound: str = None) -> str:
    prompt=""
    if type == "binary":
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. You will do the research required to answer the following question, and then provide the best possible answer.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

    elif type == "multiple_choice":
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. You will do the research required to answer the following question, and then provide the best possible answer.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

    elif type == "numeric":
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. You will do the research required to answer the following question, and then provide the best possible answer.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound}
            {upper_bound}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )

    return prompt
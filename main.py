import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal
from search_prediction_markets import PredictionMarketSearchClient
from deep_research import call_deep_research

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
    RefreshingBucketRateLimiter
)

logger = logging.getLogger(__name__)


class AUTOMATIC_BOT(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.
    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/deep-research/low",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=1,
        refresh_rate=3,
    )

    deep_research_results = {}

    _max_concurrent_questions = (
        2  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """
        Override the base class defaults to include the 'mini' LLM purpose used in this bot.
        This method provides default LLM configurations for all LLM types used in this bot.
        """
        # Get the base defaults first
        base_defaults = super()._llm_config_defaults()
        
        # Add our custom 'mini' LLM purpose to the defaults
        base_defaults["mini"] = GeneralLlm(
            model="openrouter/openai/gpt-5-mini",
            timeout=60,
            allowed_tries=2,
            reasoning_effort="high"
        )
        
        return base_defaults

    async def _should_use_deep_research(self, question: MetaculusQuestion) -> bool:
        """Check if deep research should be used for this question"""
        notepad = await self._get_notepad(question)
        # Run deep research only once per question (on the first prediction)
        if "deep_research_used" not in notepad.note_entries:
            notepad.note_entries["deep_research_used"] = False
        
        if not notepad.note_entries["deep_research_used"]:
            notepad.note_entries["deep_research_used"] = True
            logger.info(f"Using deep research for question: {question.page_url}")
            return True
        else:
            logger.info(f"Skipping deep research run (already used) for question: {question.page_url}")
        return False
    
    async def _should_apply_deep_research(self, question: MetaculusQuestion) -> bool:
        notepad = await self._get_notepad(question)

        # apply deep research answer to the result every 3 steps (where 3 is the number of research reports)
        if "research_steps_count" not in notepad.note_entries:
            logger.info(f"First instance of deep research being applied")
            notepad.note_entries["research_steps_count"] = 0
            return True
        elif (notepad.note_entries["research_steps_count"] % 3 == 0) & (notepad.note_entries["research_steps_count"] != 0):
            logger.info(f"Applying deep research results for step {notepad.note_entries["research_steps_count"]}")
            notepad.note_entries["research_steps_count"] += 1
            return True
        else:
            notepad.note_entries["research_steps_count"] += 1
            return False

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            await self.rate_limiter.wait_till_able_to_acquire_resources(1)
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself, you do the research that the forecaster will need to come to a conclusion.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            # Run deep research synchronously during the research phase
            if await self._should_use_deep_research(question):
                logger.info(f"Running deep research for question: {question.page_url}")
                if isinstance(question, BinaryQuestion):
                    deep_research_result = await call_deep_research(question=question, type="binary")
                elif isinstance(question, MultipleChoiceQuestion):
                    deep_research_result = await call_deep_research(question=question, type="multiple_choice")
                elif isinstance(question, NumericQuestion):
                    upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
                    deep_research_result = await call_deep_research(question=question, type="numeric", lower_bound=lower_bound_message, upper_bound=upper_bound_message)
                else:
                    deep_research_result = None
                
                if deep_research_result:
                    self.deep_research_results[question.page_url] = deep_research_result
                    logger.info(f"Deep research completed and stored for question: {question.page_url}")

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews"],
                    model="deepseek-basic",
                    search_depth=2,
                    max_depth=2
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    prompt,
                    sources=["asknews", "google", "x"],
                    search_depth=3,
                    max_depth=6,
                    model="o3" # CHECK: https://docs.asknews.app/en/deepnews
                )
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)


            prediction_market_query_generation_prompt = clean_indents(
                f"""
                You are a search assistant for a superforecaster.
                You will take the following question, and return a single query that would be the most useful in finding prediction markets that are related to the question.
                This query will use semantic search across multiple markets, so think about what general topic that having probabilities for would maximize information gain. 
                For example, say the question asks about what companies will have the best LLMs this year-- You would return "Best AI".
                An exact search is already being done, so make your search more general.
                Your search should be less than 5 words.
                
                Here is your task:

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )


            search_assistant = self.get_llm("mini")

            general_query = await search_assistant.invoke(prediction_market_query_generation_prompt)

            logger.info(f"Searching prediction markets for: {general_query}")

            try:
                async with PredictionMarketSearchClient() as client:
                    await client.search_prediction_markets(question.question_text, 10000, is_first_search=True) # results stored in an instance variable
                    search_results = await client.search_prediction_markets(general_query, 10000, is_first_search=False)

                    search_refinement_prompt = clean_indents(
                    f"""
                    You are a search assistant for a superforecaster.
                    You will be given a question, and a set of search results from prediction markets. These results are useful because the prediction markets tell us the probabality of particular events occuring.
                    Your job is to determine what markets are the most relavent to the question below.
                    If none are relevant, your response will be "No markets found". For any market that is relevant, include it in your response as normal text. Format it in a way that is easy for a research analyst to parse.
                    If a market is nearly identical to the current question, flag it as very important. If this market comes from Metaculus and has a probability of 0-- that's because its the same question and predictions have not been made on it yet so ignore this case.
                    Do not provide your interpretation of these numbers, only provide the formatted data that could be important to consider when answering this question. DO NOT ASK ANY FOLLOW-UP QUESTIONS.

                    Question:
                    {question.question_text}

                    This question's outcome will be determined by the specific criteria below:
                    {question.resolution_criteria}

                    {question.fine_print}

                    

                    Here are the results from Polymarket, Kalshi, and Metaculus:
                    {search_results[0]}


                    Here are the results from Manifold:
                    {search_results[1]}
                    """
                    )

                    prediction_market_results = await search_assistant.invoke(search_refinement_prompt)

                    research = f"""
                        {research}

                        \n===========================================
                        
                        **Potentially useful prediction market data**

                        * Note that all probabilities are between 0 and 100, not 0 and 1. Volume can be an indication of accuracy/ market confidence, except for Metaculus questions where volume is always 0.

                        * If a probability is 0, disregard it!

                        {prediction_market_results}
                    """
            except Exception as e:
                logger.warning(f"Failed to fetch prediction market data for question '{question.question_text}': {e}")

            
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            (e) How your conclusion relates to current prediction market odds, if at all (markets may not be provided). Otherwise, what research was most important to your conclusion?

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        if await self._should_apply_deep_research(question) and question.page_url in self.deep_research_results:
            reasoning = self.deep_research_results[question.page_url]
            logger.info(f"Using stored deep research results for question: {question.page_url}")
        else:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.
            (d) How your conclusion relates to current prediction market odds, if at all (markets may not be provided). Otherwise, what research was most important to your conclusion?

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        if await self._should_apply_deep_research(question) and question.page_url in self.deep_research_results:
            reasoning = self.deep_research_results[question.page_url]
            logger.info(f"Using stored deep research results for question: {question.page_url}")
        else:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

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
            (g) How your conclusion relates to current prediction market odds, if at all (markets may not be provided). Otherwise, what research was most important to your conclusion?

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
        if await self._should_apply_deep_research(question) and question.page_url in self.deep_research_results:
            reasoning = self.deep_research_results[question.page_url]
            logger.info(f"Using stored deep research results for question: {question.page_url}")
        else:
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    # CHANGE RESEARCH REPORTS
    bot = AUTOMATIC_BOT(
        research_reports_per_question=3,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="reports/",
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5",
                timeout=60,
                allowed_tries=2,
                reasoning_effort="high",
                verbosity="high"
            ),
            "mini": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                timeout=60,
                allowed_tries=2,
                reasoning_effort="high"
            ),
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": GeneralLlm(
                model="openrouter/openai/gpt-5",
                timeout=60,
                allowed_tries=2,
                reasoning_effort="high",
                tools=[
                    {"type": "web_search"},
                ],
            ),
            "parser": "openrouter/openai/gpt-5-mini",
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question

        """
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        """

        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/"
        ]

        bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            bot.forecast_questions(questions, return_exceptions=True)
        )
    bot.log_report_summary(forecast_reports)
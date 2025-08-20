# AUTOMATIC BOT

## Basic Info
- Code was taken from the template bot provided by Metaculus (default bot not the few-dependency version)
- I'm using uv instead of poetry for package management since uv is awesome. I changed the lock file, and I also had to modify the yaml files in the .github/workflows directory

## Stack
- Use GPT-5 for most things. Standard system prompt, nothing special
- No Perplexity, as it is not competitive. For search I'm just going to use AskNews and Adjacent
- May add in Grok-4 since it has access to Twitter, and it does well on the AA-LCR (Long Context Reasoning) benchmark (see 8th chart in intelligence evaluation section of https://artificialanalysis.ai/). Its also expensive and not included with credits so its not high priority

## Adjacent News
Based on example questions I've seen, using the Adjacent search function will only be useful once in a while. My process will go:

1. LLM call that first formulates a search query or two to submit to the adj API

2. Filter responses such that only active markets are returned (end_date > todays date)

3. Maybe filter by volume/ oi if its really low (>1000? 5000?)

4. Only return relevant info of relevant markets

5. These results can then be appended to the research report

## GPT-5

- litellm and openrouter support the text completion API for openai, not the responses API

- I only want to use the "high" setting for research agents, and maybe a high verbosity level. The argument for reasoning level is reasoning_effort, and this needs to get passed through the Metaculus forecasting tools wrapper, litellm, and then to openrouter and openai. Not sure if verbosity levels can be passed through

- There is a github issue on litellm for passing custom openai params here: https://github.com/BerriAI/litellm/issues/13571. This is needed for verbosity level and maybe even reasoning level, I'm not sure yet but this could be a problem

## AskNews

- We can use the deep search agent with the Metaculus tier API key, but cant add in sources other than "asknews", can only use "deepseek-basic", and max depth is limited to 2

## Random Notes
- The few-dependency version of the bot provided by Metaculus did not handle generating distributions / extracting predictions the same way the default one does (default uses forecasting-tools and an llm call which I think is easier and probably better)

- Not sure how quickly I'll run out of credits yet. I'll have to test this over the next week or so
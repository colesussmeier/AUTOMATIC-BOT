# AUTOMATIC BOT Q4 2025

This project is a deep research framework for predicting the future and competing in the [$58,000 Metaculus Artificial Intelligence Benchmark](https://www.metaculus.com/aib/2025/fall/)

## Stack
- Research/ Prediction Bots: Mostly GPT-5. Standard system prompts, nothing special here
- Deep research: Used OpenAI's deep research with O4-mini API. This needed a custom implementation because it is not available through Openrouter
- Search: No Perplexity, as I don't think it is competitive. I used default search tools from model providers, but also experimented with AskNews
- Prediction Markets: I used the Adjacent News API and Manifold market search to find any markets relating to the current question

## Deep Research Model
Since deep research responses cannot be integrated through LightLLM/ forecasting-tools, I used the notepad functionality of the template_bot class to hack the deep research predictions into the final forecast. 
Deep research predictions were weighted heavier than standard predictions. Deep research takes a while to complete so I had to be careful about race conditions.

## Prediction Markets
Logic for incorporating prediction market data:

1. LLM call that first formulates a search query or two to submit to the Adjacent/ Manifold API

2. Filter responses such that only active markets are returned (end_date > todays date)

3. Filter by volume/ open interest... we only want to consider markets that have sufficient trading volume

4. Only return relevant info of relevant markets

5. These results can then be appended to the research report

## Random Notes
- Project is based on the template bot provided by Metaculus

- I'm using uv instead of poetry for package management since uv is awesome. I changed the toml file, and I also had to modify the yaml files in the .github/workflows directory

- The few-dependency version of the bot provided by Metaculus did not handle generating distributions / extracting predictions the same way the default one does (default uses forecasting-tools and an llm call which I think is easier and probably better)

- Metaculus provided API credits for OAI/ Anthropic models through Openrouter. This did not cover deep research and I ended up running out of credits. Github Actions is also expensive. In total, this used almost $1000 dollars in API credits, but I expect the winnings from the competition to comfortably cover any out of pocket expenses

## Future Improvements
- Integrating tool use directly into models will probably work better
- OpenAI's deep reseach model worked well, but I think that I can do better myself
- I'd like to try a custom weighting scheme analogous to an ensemble of ensembles. Distribution generation could also use some work
- GPU poor :( -- Need more compute!
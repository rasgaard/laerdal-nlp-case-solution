I wrote about the process of producing the solutions in this repo [here on my website](https://rasgaard.github.io/posts/laerdal-nlp-case.html).

# Proposed solution for NLP case interview at Laerdal

This repository contains my proposed solution for the data science case at the second interview for the role as a data scientist at Laerdal Copenhagen.

The repo containing the case can be found at [maximillian91/img-txt-categorisation-chat](https://github.com/maximillian91/img-txt-categorisation-chat/tree/main). Here you will also find the exercise descriptions to contextualize my solutions if interested.

My approach to this challenge was to explore solutions in notebooks and present my chosen solution in a Streamlit application. This resulted in a collection of notebooks in the `notebooks`-folder. Much of this is "quick and dirty"-work as the case had a fairly short time frame. 

[Ollama](https://ollama.ai/) is used for the chat exercise. Run `ollama serve` followed by `ollama run llama2:7b-chat` in order to load and serve the model that is expected of the code. It uses Langchain and Streamlit Chat elements to glue everything together.

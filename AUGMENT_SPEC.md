Add a AUGMENT stage to the script/run_1.1.py.

This stage will:
    - Iterate through all the non-code textual documents in the deduped directory.
    - Use `textstat.flesch_reading_ease` to identify the reading complexity of each document.
    - If the document is easy to read (score > 80), call an LLM to make it more sophisticated and challenging.
    - If it's difficult to read (score < 60), call an LLM to make it easier.
    - Use the LLM to generate questions that are answered by the document. The number of questions will be 1 for every 200 words, with a maximum of 5.
    - The augmented documents will be saved in a new directory.

LLM script will live in scripts/augment_openai.py. It should support any openai compatible provider with a custom endpoint.

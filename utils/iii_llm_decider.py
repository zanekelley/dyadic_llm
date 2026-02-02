"""
<llm_decider>: This script defines a utility function for making structured classification decisions
using an OpenAI chat model based on conversational transcript data.

The core function, `llm_decision`, takes a pandas DataFrame containing a dialogue
transcript, reconstructs the conversation from the perspective of a specified speaker,
and appends a predefined structured question (currently gender identification).
The function then queries an OpenAI chat model with a constrained JSON schema to force
a valid, deterministic output.

Key features:
- Converts a transcript stored as speaker-labeled lines into OpenAI chat messages
- Assigns message roles based on a target speaker
- Enforces structured model output using a JSON schema
- Returns both the parsed model decision and token-level log probabilities
- Designed to be easily extensible to additional structured question types
"""

import pandas as pd
from openai import OpenAI
import json

# Schema the model will be forced to output as
GENDER_SCHEMA = {
    "type": "object",
    "properties": {
        "gender": {
            "type": "string",
            "enum": ["male", "female"]
        }
    },
    "required": ["gender"],
    "additionalProperties": False
}

# Creating function that can take a dataframe, a speaker, a model, and a question type
# Outputs raw json of the predicted value and log probs of that value
def llm_decision(
    client: OpenAI,
    conversation_df: pd.DataFrame,
    which_speaker: str,
    model: str = "gpt-4o-mini",
    structured_question_type: str = "gender"
):
    # Prespecified question the "user" will ask
    # Only gender now, but can customize to hold other
    questions = {
    "gender": "Which gender do you identify as?"
    }
    question = questions[structured_question_type]

    transcript = conversation_df["transcript"].iloc[0]
    lines = transcript.splitlines()

    # Initializes the array for the df to be converted into
    messages = []

    for line in lines:
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        text = text.strip()

        role = "assistant" if speaker == which_speaker else "user"

        messages.append({
            "role": role,
            "content": text
        })

    messages.append({"role": "user", "content": question})

    # The actual call to OpenAI
    completion = client.chat.completions.create(
        model=model, # What model to use
        messages=messages, # The created messages array
        temperature=0, # Forces it to choose absolute most likely value
        stream=False,
        logprobs=True,
        top_logprobs=10,
        response_format={ # Forces proper output
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": structured_question_type,
                "schema": GENDER_SCHEMA
            }
        }
    )

    raw_content = completion.choices[0].message.content # Extracts decision
    logprobs = getattr(completion.choices[0].logprobs, "content", None) # Extracts logprobs of decision
    parsed_content = json.loads(raw_content) # Converts json into python dictionary

    return {"content": parsed_content, "logprobs": logprobs}






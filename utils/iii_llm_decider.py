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

# Creating fucntion that can take a dataframe, a speaker, a model, and a question type
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
        temperature=0, # Forces it to choose absolute most likel value
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






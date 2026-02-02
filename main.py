import os
import pandas as pd
from openai import OpenAI

from utils.ii_transcript_structurer import transcript_method
from utils.iii_llm_decider import llm_decision
from utils.iv_llm_parser import parse_llm_decision, results_dict_to_csv
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

def run_method(method, 
               input = "inputs/llm/all_transcripts.csv", 
               model = "gpt-4o-mini", 
               structured_question_type = "gender", 
               output_path = "outputs/llm_output.csv"):
    results = {}
    input = pd.read_csv(input)
    transcript = transcript_method(input, method)
    for dyad, dyad_df in transcript.groupby("dyad"):
        for _, row in dyad_df.iterrows():
            speaker = row["speaker"]

            if speaker in ("A", "B"):
                output = llm_decision(
                    client=client,
                    conversation_df=row.to_frame().T,
                    model=model,
                    structured_question_type=structured_question_type,
                    which_speaker=speaker
                )

                key = tuple(row.values)
                results[key] = parse_llm_decision(
                    output,
                    structured_question_type
                )

    results_dict_to_csv(results,
                        output_path)

    print(f"CSV successfully written to {output_path}")

run_method(method = "EACH_PART_SIM", 
           input = "inputs/llm/all_transcripts.csv", 
           model = "gpt-4o-mini", 
           structured_question_type = "gender",
           output_path = "outputs/llm_output.csv")




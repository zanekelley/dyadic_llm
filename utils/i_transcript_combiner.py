"""
<transcript_combiner>: Customizable script to structure a transcript to work with the Dyadic LLM process

This script takes a folder of dyadic transcripts in CSV format (can be easily tweaked for XLSX) and combines them into a single CSV file.
Dyad number MUST be in the name of the file, and the files must have a Speaker column (who spoke the words) and Text column (the actual words spoken).
Rename your input columns to these names for easier use with the LLM process.

"""

import pandas as pd
import glob
import os
import re

files = glob.glob("inputs/transcripts/*.csv") # Change if not following same structure
frames = []

for f in files:
    # ---- Creates the Dyad ID from the file name ----
    base = os.path.splitext(os.path.basename(f))[0]
    match = re.search(r"Dyad[ _-]?(\d+)", base) # Customize to your file naming structure; Dyad ID must be in title
    if match:
        dyad = int(match.group(1)) # Designates your Dyad ID
    else:
        dyad = "" # If it wasn't in the file name, it becomes ""

     # ---- Reads in the CSV ----   
    df = pd.read_csv(f).rename(columns={ # CSV must have these names. Can customize to fit your input columns, but Speaker and Text are vital
        "Speaker":"speaker",
        "Text":"text"})
    df["ID"] = base # Creates a variable that details what file this dyad is from
    df["dyad"] = dyad # Creates the designated Dyad ID

    # ---- Assign A and B based on first occurrence ----
    speaker_order = df["speaker"].drop_duplicates().tolist()  # Should always be 2 speakers
    speaker_map = {speaker_order[0]: "A", speaker_order[1]: "B"} # Designates the 2 speakers as A and B
    df["speaker"] = df["speaker"].map(speaker_map) # Maps those speakers onto the df
    df = df.assign(transcript=df["speaker"] + ": " + df["text"]) # Creates a transcript style exchange between two speakers
    df["turn"] = range(1, len(df) + 1) # Counting the number of turns spoken between your speakers

    # ---- Appends the df to frames ----
    frames.append(df[["ID", "dyad", "speaker", "turn", "transcript"]])


dat = pd.concat(frames) # Puts them all in one df row by row of turns
dat = dat[dat["dyad"] != ""] # Drops rows without a Dyad ID
dat = dat.sort_values(by=["dyad", "turn"]).reset_index(drop=True) # Sorts them to be in order
dat.to_csv("inputs/llm/all_transcripts.csv", index = False)

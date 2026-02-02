"""
<transcript_structurer>: This script takes a CSV as formatted by transcript_combiner, and structures it 
to be used with the LLM in one of three ways.

This functions looks within a dyad and structures a dataframe to give the LLM dialogue from both members of the dyad, 
only ony member of the dyad, or a single line from the dyad.

Examples of the output by dyad are shown below:

EACH_PART_SIM
A: -------
B: -------
A: -------
B: -------

EACH_PART_ALONE
A: -------
A: -------
     OR
B: -------
B: -------

EACH_TURN_ALONE
A: -------
     OR
B: -------
"""

def transcript_method(dat, operation):
    dat["dyad"] = dat["dyad"].astype("Int64")
    if operation == "EACH_PART_SIM": 
        base = (
            dat
            .groupby("dyad")["transcript"]
            .agg("\n".join )
            .reset_index()
        )
        speakers = (
            dat[["dyad", "speaker"]]
            .drop_duplicates()
        )
        result = speakers.merge(base, on="dyad", how="left")
        return result

    elif operation == "EACH_PART_ALONE":
        result = (
            dat
            .groupby(["dyad", "speaker"])["transcript"]
            .agg("\n".join )
            .reset_index()
            )
        return result

    elif operation == "EACH_TURN_ALONE":

        result = dat[["dyad", "speaker", "transcript"]]

        return result

    else:
        raise ValueError("Operation must be EACH_PART_SIM, EACH_PART_ALONE, or EACH_TURN_ALONE")

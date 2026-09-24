import pandas as pd
from langchain_core.tools import tool

_candidates_df = pd.read_csv("../data/processed/candidates.csv")

@tool
def check_medium_frequency(department: str, medium_keyword: str) -> dict:
    """Check how many records in a department have a medium description
    containing a specific material or technique keyword (case-insensitive,
    partial match) — e.g. 'leather' or 'deerskin' — out of how many records
    that department has in total. Pass a short material keyword, NOT the
    full medium description, since full descriptions are usually unique to
    one record and checking one won't tell you anything meaningful."""
    dept_subset = _candidates_df[_candidates_df["Department"] == department]
    matching = dept_subset[dept_subset["Medium"].str.contains(medium_keyword, case=False, na=False)]
    return {
        "department": department,
        "medium_keyword": medium_keyword,
        "matching_count": len(matching),
        "department_total": len(dept_subset),
    }
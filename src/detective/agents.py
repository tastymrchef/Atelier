from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, ToolMessage
from src.detective.tools import check_medium_frequency


from src.detective.schema import Artworkassessment

from typing import Optional




class MediumVerdict(BaseModel):
    medium_anachronism: bool
    reason: str

medium_llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([check_medium_frequency])
medium_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(MediumVerdict)

def medium_specialist(dossier) -> MediumVerdict:
    prompt = (
        f"Artwork record:\n{dossier.model_dump_json(indent=2)}\n\n"
        "Could the stated medium not physically have existed at the stated date, OR is "
        "it genuinely rare for this department? If you're unsure whether a medium is "
        "actually rare here, use the check_medium_frequency tool to check the real count "
        "before deciding — don't guess. Pass a short material keyword to the tool, not "
        "the full medium description.\n\n"
        "A plain/generic medium name alone is normal cataloging and is NOT a reason to "
        "flag. If nothing is genuinely anachronistic or rare, set medium_anachronism to "
        "False and briefly say so in the reason."
    )
    messages = [HumanMessage(prompt)]
    response = medium_llm_with_tools.invoke(messages)
    messages.append(response)

    for tool_call in response.tool_calls:
        result = check_medium_frequency.invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    return medium_llm.invoke(messages)


class CorrectionProposal(BaseModel):
    can_propose_fix: bool
    proposed_fix: Optional[str]
    rationale: str

correction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CorrectionProposal)

def correction_specialist(dossier, flag_reason: str) -> CorrectionProposal:
    prompt = (
        f"Artwork record:\n{dossier.model_dump_json(indent=2)}\n\n"
        f"This record was flagged for review with this reason:\n{flag_reason}\n\n"
        "Can you confidently propose a specific correction to one field, based only on "
        "what's already in this record? Only set can_propose_fix to True if the correct "
        "value is clearly inferable from the record itself — for example, a typo, or an "
        "obvious internal contradiction where one value is clearly the mistake.\n\n"
        "If you cannot determine which value is actually correct without outside "
        "research or expert judgment, set can_propose_fix to False, leave proposed_fix "
        "empty, and explain in the rationale what a human curator would need to verify."
    )
    return correction_llm.invoke(prompt)



class DateVerdict(BaseModel):
    date_conflict: bool
    reason: str

date_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(DateVerdict)

def date_specialist(dossier) -> DateVerdict:
    prompt = (
        f"Artwork record:\n{dossier.model_dump_json(indent=2)}\n\n"
        "Does this record state two different dates that genuinely conflict, with no "
        "resolution given? NOT a conflict: a single approximate date ('circa 1700'), or "
        "a hyphenated range spanning adjacent centuries ('18th–19th century') — both are "
        "normal ways of expressing date uncertainty, not two competing claims. Also NOT "
        "a conflict: a date explicitly labeled 'spurious' — this is standard art-cataloging "
        "language for a false inscribed date; a record like '18th century or later, "
        "spurious date of 1680' has already resolved itself.\n\n"
        "Only flag date_conflict when two separate, incompatible date assertions appear "
        "with nothing in the record explaining which one is authoritative — for example, "
        "one field giving a specific year that falls completely outside a stated range, "
        "with no language reconciling the two."
    )
    return date_llm.invoke(prompt)



class CultureVerdict(BaseModel):
    department_culture_mismatch: bool
    reason: str

culture_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CultureVerdict)

def culture_specialist(dossier) -> CultureVerdict:
    prompt = (
        f"Artwork record:\n{dossier.model_dump_json(indent=2)}\n\n"
        "Is the department completely unrelated to the artwork's own content or "
        "description (e.g. department is 'Egyptian Art' but everything else about the "
        "record points to Japan)? Only flag a genuine, obvious mismatch — not a normal "
        "cross-cultural piece or an ambiguous case."
    )
    return culture_llm.invoke(prompt)



def supervisor(dossier) -> Artworkassessment:
    date_verdict = date_specialist(dossier)
    culture_verdict = culture_specialist(dossier)
    medium_verdict = medium_specialist(dossier)

    flag = (
        date_verdict.date_conflict
        or culture_verdict.department_culture_mismatch
        or medium_verdict.medium_anachronism
    )

    reasons = []
    if date_verdict.date_conflict:
        reasons.append(f"[date] {date_verdict.reason}")
    if culture_verdict.department_culture_mismatch:
        reasons.append(f"[culture] {culture_verdict.reason}")
    if medium_verdict.medium_anachronism:
        reasons.append(f"[medium] {medium_verdict.reason}")

    reason = " ".join(reasons) if reasons else "No contradiction found by any specialist."

    return Artworkassessment(flag=flag, reason=reason, confidence="High")
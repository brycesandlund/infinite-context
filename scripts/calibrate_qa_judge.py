"""Hand-labelled calibration of eval/qa_judge.py on REAL answers from our qa_1/qa_2 rollouts (labels: 2026-09-24)."""
import asyncio
from eval.qa_judge import QAJudge

CASES = [  # (question, golds, candidate, correct?)
    ("How many people were in the group which preceded the release of Eric Clapton's 1975 album E.C. Was Here?", ["four"], "4", 1),
    ("What is the proper name of the husband of Lollia Paullina?", ["Gaius Julius Caesar Augustus Germanicus"], "Caligula", 1),
    ('What season of the Indian reality TV series "Big Boss" did the model Lopamundra Raut compete in?', ["the tenth season"], "Bigg Boss 10", 1),
    ("National Firearms Agreement was in response to the Port Arthur massacre that killed how many people?", ["35 people"], "35", 1),
    ("How many rooms are in the building built in 1883 at Garden City in Nassau County, New York?", ["500-room"], "500", 1),
    ("What multinational professional services firm … is one of the \"Big Four\" accounting firms and is also a tenant in Infosys?", ["Ernst & Young"], "EY (Ernst & Young)", 1),
    ("Why did basketball player \"The Process\" not play in the 77th season?", ["leg injury"], "A leg injury", 1),
    ("What portrait hangs in the Smithsonian Institute along with what is known as the founder of nursing?", ["Sister Anthony, S.C."], "Mary O'Connell (Sister Anthony)", 1),
    ("What mythological creature is the subject of the 2001 horror film?", ["Wendigo"], "The Windigo legend", 1),
    ("What is the least restrictive measure the court applies?", ["the least onerous"], "The least onerous measure", 1),
    ("What multinational professional services firm … is one of the \"Big Four\" accounting firms and is also a tenant in Infosys?", ["Ernst & Young"], "PricewaterhouseCoopers", 0),
    ("What multinational professional services firm … is one of the \"Big Four\" accounting firms and is also a tenant in Infosys?", ["Ernst & Young"], "KPMG and Ernst & Young", 0),
    ("How many people were in the group which preceded the release of Eric Clapton's 1975 album E.C. Was Here?", ["four"], "Derek and the Dominos", 0),
    ("In what city was the band that recorded the song with Onyx formed?", ["Brooklyn, New York"], "Biohazard", 0),
    ("Which team did the coach lead before joining the conference?", ["Arizona State Sun Devils"], "Mountain West Conference", 0),
    ("Ravi Khote has included his music in which 2003 Indian drama?", ["Kal Ho Naa Ho"], "none", 0),
    ("Who collaborated with the band on the tribute album?", ["Eric Clapton"], "Jimmy Page and Eric Clapton", 0),
    # Mary O'Connell IS Sister Anthony, S.C. (her name before taking vows) — an alias, so CORRECT (an earlier version of
    # this file paraphrased the question as "who painted…", which changed its meaning and mislabelled the case)
    ("What portrait hangs in the Smithsonian Institute along with what is known as the founder of nursing?", ["Sister Anthony, S.C."], "Mary O'Connell's portrait", 1),
    ("How many original treaties establishing the EU protected fundamental rights?", ["None"], "0", 1),
    ("How many original treaties establishing the EU protected fundamental rights?", ["None"], "3", 0),
]


async def main():
    j = QAJudge("eval_results/qa_judge_cache.json")
    vs = await asyncio.gather(*[j.grade(q, g, a) for q, g, a, _ in CASES]); j.save()
    agree = sum(v["correct"] == y for v, (_, _, _, y) in zip(vs, CASES))
    for v, (q, g, a, y) in zip(vs, CASES):
        mark = "ok " if v["correct"] == y else "XX "
        print(f"{mark} label={y} judge={v['correct']}  gold={g[0][:28]!r:32s} cand={a[:32]!r}")
    print(f"agreement {agree}/{len(CASES)} | model {j.model}")


asyncio.run(main())

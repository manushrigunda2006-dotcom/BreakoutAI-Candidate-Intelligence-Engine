import json
import pandas as pd

with open(
        "data/sample_candidates.json",
        "r",
        encoding="utf-8"
) as f:
    candidates = json.load(f)

print(len(candidates))
print(candidates[0]["candidate_id"])

rows = []

for c in candidates:

    row = {}

    row["candidate_id"] = c["candidate_id"]

    row["headline"] = c["profile"]["headline"]

    row["years_exp"] = (
        c["profile"]["years_of_experience"]
    )

    row["skills"] = " ".join(
        [
            s["name"]
            for s in c["skills"]
        ]
    )
    signals = c["redrob_signals"]

    row["github_score"] = max(
    signals["github_activity_score"], 0
)

    row["response_rate"] = (
    signals["recruiter_response_rate"]
)

    row["profile_complete"] = (
    signals["profile_completeness_score"]
)

    row["interview_completion"] = (
    signals["interview_completion_rate"]
)

    row["saved_by_recruiters"] = (
    signals["saved_by_recruiters_30d"]
)

    row["profile_views"] = (
    signals["profile_views_received_30d"]
)

    row["open_to_work"] = int(
    signals["open_to_work_flag"]
)

    row["notice_period"] = (
    signals["notice_period_days"]
)
    rows.append(row)

df = pd.DataFrame(rows)

print(df.head())

jd = """
Artificial Intelligence Intern

Requirements:
Python
Machine Learning
Artificial Intelligence
Problem Solving
Communication
Self motivated
Remote work
"""
required_skills = [
    "Python",
    "NLP",
    "Fine-tuning LLMs",
    "Prompt Engineering",
    "Recommendation Systems",
    "OpenCV",
    "Kubeflow",
    "Transformers",
    "MLOps",
    "GANs",
    "Image Classification",
    "Object Detection",
    "Speech Recognition",
    "Milvus",
    "BentoML"
]

def skill_score(skills):

    score = 0
    skills = skills.lower()

    for s in required_skills:
        if s.lower() in skills:
            score += 1

    return score

df["score"] = df["skills"].apply(skill_score)

print(
    df[
        [
            "candidate_id",
            "skills",
            "score"
        ]
    ].head()
)
df = df.sort_values(
    "score",
    ascending=False
)
df["rank"] = range(
    1,
    len(df)+1
)
df["reasoning"] = (
    "Strong AI skill match."
)
output = df[
    [
        "candidate_id",
        "rank",
        "score",
        "reasoning"
    ]
]

output = output.head(100)
output.to_csv(
    "submission.csv",
    index=False
)
print(output.head())
df = pd.DataFrame(rows)
print(df.columns)
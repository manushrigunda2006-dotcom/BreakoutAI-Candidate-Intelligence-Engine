import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load candidates
df = pd.read_csv("data/candidates.csv")

# Job description
job_description = """
Senior AI Engineer with embeddings, retrieval,
ranking systems, Python, product company experience,
recommendation systems and active job seeking signals.
"""

# Create one text profile for each candidate
df["text"] = (
    df["profile"].fillna("").astype(str) + " " +
    df["career_history"].fillna("").astype(str) + " " +
    df["education"].fillna("").astype(str) + " " +
    df["skills"].fillna("").astype(str) + " " +
    df["certifications"].fillna("").astype(str) + " " +
    df["languages"].fillna("").astype(str) + " " +
    df["redrob_signals"].fillna("").astype(str)
)
print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

candidate_embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    batch_size=64
)

job_embedding = model.encode(job_description)

scores = cosine_similarity(
    [job_embedding],
    candidate_embeddings
)[0]
ranked = df.sort_values(
    by="score",
    ascending=False
)

top100 = ranked.head(100).copy()

top100["rank"] = range(1, 101)

submission = top100[
    ["candidate_id", "rank", "score"]
].copy()

submission["reasoning"] = (
    "Strong semantic match with AI engineer requirements."
)

submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv created!")
print("Rows:", len(submission))

# Experience score
df["experience_score"] = (
    df["years_experience"] / 10
).clip(0, 1)

# Activity score
df["activity_score"] = (
    df["platform_activity"]
    .str.contains(
        "github|kaggle|linkedin|repos|contributes",
        case=False,
        na=False
    )
    .astype(int)
)


# Final score


# Sort candidates
ranked = df.sort_values(
    by="score",
    ascending=False
)

# Top 100
top100 = ranked.head(100).copy()

# Add rank
top100["rank"] = range(1, 101)

# Create submission
submission = top100[
    ["candidate_id", "rank", "score"]
].copy()

submission.rename(
    columns={"final_score": "score"},
    inplace=True
)

submission["reasoning"] = (
   "Strong semantic match with AI engineer requirements"
)

# Save file
submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created successfully!")
print("Rows:", len(submission))
print(submission.head())
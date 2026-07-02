# AI Candidate Ranker

## Approach
- Created candidate profiles by combining role, skills, education, projects, and activity.
- Used Sentence Transformers embeddings.
- Calculated semantic similarity with the job description.
- Added experience, activity, and role-based scoring.
- Generated ranked candidates and saved them to submission.csv.

## Libraries
- pandas
- numpy
- scikit-learn
- sentence-transformers
- torch

## Run

python src/ranker.py

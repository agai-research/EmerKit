"""ESCO skills (European Commission, ESCO v1.1.1, CC BY 4.0; CSV transformation by Tabiya, github.com/tabiya-tech/tabiya-open-dataset).
Archetype skills of each domain are selected by keyword from ESCO labels and descriptions; their demand weight is the number of
ESCO occupations linked to the skill (essential = 1, optional = 0.5); each skill is attached to the closest concept of the domain."""
import csv, io, os, re, tarfile
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ESCO = os.path.join(HERE, "..", "data", "sources", "esco", "tabiya_esco_v1.1.1.tgz")
KEYWORDS = {
 "programming": r"\b(python|java|javascript|computer programming|programming|software development|algorithms?|debug\w*|object-oriented|source code|software testing|version control|data structures?)\b",
 "datascience": r"\b(machine learning|statistic\w*|data analy\w*|data mining|natural language processing|artificial intelligence|data visuali\w*|data science|neural network|predictive model\w*)\b",
 "language": r"\b(english|foreign languages?|translat\w+|interpret\w*|language teaching)\b",
}


@lru_cache(maxsize=1)
def load():
    with tarfile.open(ESCO) as tf:
        skills = list(csv.DictReader(io.TextIOWrapper(tf.extractfile("skills.csv"), encoding="utf-8")))
        rel = list(csv.DictReader(io.TextIOWrapper(tf.extractfile("occupation_skill_relations.csv"), encoding="utf-8")))
    demand = {}
    for r in rel:
        demand[r["SKILLID"]] = demand.get(r["SKILLID"], 0.0) + (1.0 if r["RELATIONTYPE"] == "essential" else 0.5)
    return skills, demand


def domain_skills(domain, top=50):
    skills, demand = load()
    rx = re.compile(KEYWORDS[domain], re.I)
    hits = [s for s in skills if rx.search(s["PREFERREDLABEL"] + " " + s["ALTLABELS"])]       # labels only: precision over recall
    hits.sort(key=lambda s: (-demand.get(s["ID"], 0.0), s["PREFERREDLABEL"]))
    return [dict(id=s["ID"], uri=s["ORIGINURI"], label=s["PREFERREDLABEL"], description=s["DESCRIPTION"],
                 demand=max(1.0, demand.get(s["ID"], 0.0))) for s in hits[:top]]   # floor 1: language skills have few occupation links


def attach_concepts(domain, skills, concepts):
    """Closest concept of the domain graph for each skill (TF-IDF over label + description vs concept names)."""
    names = [c.replace("_", " ") for c in concepts]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(names + [s["label"] + " " + s["description"] for s in skills])
    sim = (vec.transform([s["label"] + " " + s["description"] for s in skills]) @ vec.transform(names).T).toarray()
    for s, row in zip(skills, sim):
        s["concept"] = concepts[int(row.argmax())]; s["concept_sim"] = float(row.max())
    return skills

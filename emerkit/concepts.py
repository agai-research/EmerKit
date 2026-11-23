"""Concept graphs of the three domains.

* datascience: LectureBank 2.0 (Li et al., COLING 2020): 322 topics in NLP, machine learning and AI with human-annotated
  prerequisite pairs; we keep the largest weakly connected component of the annotated graph and break cycles.
* programming: expert-authored graph following the Software Development Fundamentals knowledge area of the ACM/IEEE-CS/AAAI
  CS2023 curricula (declared as expert-authored, not as an external dataset).
* language: expert-authored graph following the CEFR progression A1 to C1 for English as a foreign language (declared likewise).
"""
import csv, os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
LB = os.path.join(HERE, "..", "data", "sources", "lecturebank2")

PROGRAMMING = [  # (concept, prerequisites) -- CS2023 SDF / FPL progression
 ("values_and_variables", []), ("expressions", ["values_and_variables"]), ("primitive_types", ["values_and_variables"]),
 ("input_output", ["expressions"]), ("conditionals", ["expressions", "primitive_types"]), ("loops", ["conditionals"]),
 ("strings", ["primitive_types", "loops"]), ("functions", ["loops"]), ("parameters_and_scope", ["functions"]),
 ("lists", ["loops"]), ("tuples_and_sets", ["lists"]), ("dictionaries", ["lists"]), ("nested_structures", ["dictionaries", "lists"]),
 ("files", ["strings", "functions"]), ("exceptions", ["functions"]), ("modules_and_packages", ["functions"]),
 ("recursion", ["functions"]), ("iteration_patterns", ["lists", "functions"]), ("comprehensions", ["iteration_patterns"]),
 ("classes_and_objects", ["functions", "dictionaries"]), ("encapsulation", ["classes_and_objects"]), ("inheritance", ["classes_and_objects"]),
 ("polymorphism", ["inheritance"]), ("unit_testing", ["functions", "exceptions"]), ("debugging", ["exceptions"]),
 ("searching", ["lists", "loops"]), ("sorting", ["searching"]), ("algorithmic_complexity", ["sorting", "recursion"]),
 ("stacks_and_queues", ["lists", "classes_and_objects"]), ("linked_structures", ["classes_and_objects", "recursion"]),
 ("version_control", ["modules_and_packages"]), ("software_design", ["polymorphism", "unit_testing"]),
]
LANGUAGE = [  # CEFR A1 -> C1 progression for English
 ("alphabet_and_sounds", []), ("greetings_and_introductions", ["alphabet_and_sounds"]), ("numbers_dates_time", ["alphabet_and_sounds"]),
 ("basic_vocabulary", ["greetings_and_introductions"]), ("present_simple", ["basic_vocabulary"]), ("articles_and_plurals", ["basic_vocabulary"]),
 ("questions_and_negation", ["present_simple"]), ("present_continuous", ["present_simple"]), ("prepositions_of_place_time", ["articles_and_plurals"]),
 ("past_simple", ["present_simple"]), ("comparatives_superlatives", ["articles_and_plurals"]), ("countable_uncountable", ["articles_and_plurals"]),
 ("future_forms", ["present_continuous", "past_simple"]), ("present_perfect", ["past_simple"]), ("modal_verbs", ["present_simple"]),
 ("pronunciation_stress_intonation", ["basic_vocabulary"]), ("listening_everyday", ["present_continuous"]), ("reading_short_texts", ["past_simple"]),
 ("first_conditional", ["future_forms"]), ("second_conditional", ["first_conditional", "past_simple"]), ("passive_voice", ["present_perfect"]),
 ("reported_speech", ["past_simple", "present_perfect"]), ("relative_clauses", ["reading_short_texts"]), ("phrasal_verbs", ["past_simple"]),
 ("paragraph_writing", ["reading_short_texts", "relative_clauses"]), ("formal_letters_emails", ["paragraph_writing", "modal_verbs"]),
 ("essay_structure", ["paragraph_writing"]), ("argumentation", ["essay_structure", "second_conditional"]), ("debate_and_discussion", ["argumentation", "listening_everyday"]),
 ("presentations", ["pronunciation_stress_intonation", "debate_and_discussion"]), ("idioms_and_register", ["phrasal_verbs", "formal_letters_emails"]),
 ("academic_writing", ["essay_structure", "passive_voice", "reported_speech"]),
]


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


@lru_cache(maxsize=1)
def lecturebank_graph():
    """Largest weakly connected component of the LectureBank 2.0 prerequisite graph, made acyclic (DFS back edges removed)."""
    topics = {}
    with open(os.path.join(LB, "322topics_final.tsv"), encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                i, name = line.strip().split("|", 1); topics[int(i)] = _slug(name)
    edges = set()
    with open(os.path.join(LB, "final_new_annotation.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 3 and r[2].strip() == "1":
                a, b = int(r[0]), int(r[1])
                if a in topics and b in topics and a != b:
                    edges.add((a, b))
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for s in sorted(adj):
        if s in seen: continue
        comp, stack = set(), [s]
        while stack:
            x = stack.pop()
            if x in comp: continue
            comp.add(x); stack += list(adj[x] - comp)
        seen |= comp; comps.append(comp)
    big = max(comps, key=len)
    out = {a: [] for a in big}
    for a, b in sorted(edges):
        if a in big and b in big:
            out[a].append(b)
    # remove back edges by iterative DFS in a fixed order
    color, keep = {}, set()
    for root in sorted(big):
        if root in color: continue
        stack = [(root, iter(sorted(out[root])))]; color[root] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = 2; stack.pop(); continue
            if color.get(nxt) == 1:
                continue                      # back edge: dropped
            keep.add((node, nxt))
            if nxt not in color:
                color[nxt] = 1; stack.append((nxt, iter(sorted(out[nxt]))))
    pre = {topics[x]: [] for x in big}
    for a, b in sorted(keep):
        pre[topics[b]].append(topics[a])
    order, done = [], set()
    def visit(c):
        if c in done: return
        done.add(c)
        for p in pre[c]: visit(p)
        order.append(c)
    for c in sorted(pre): visit(c)
    return [(c, sorted(pre[c])) for c in order]


def domain_concepts():
    try:
        ds = lecturebank_graph()
    except FileNotFoundError:
        ds = []
    return {"programming": PROGRAMMING, "datascience": ds, "language": LANGUAGE}

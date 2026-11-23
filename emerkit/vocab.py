"""Fixed vocabulary of the EmerKit pilot (provisional pedagogical parameters, see protocol.yaml)."""

ATT = ["ID", "PROF", "PERF", "BEH", "CONT"]          # identity, profile, performance, behaviour, content
ROLES = ["INS", "PRO", "EXT", "PUB"]                  # institution, contracted processor, third party, public web
EXT_ROLES = {"EXT", "PUB"}
TYPES = ["acquisition", "practice", "formative", "summative", "discussion", "production"]
ASM = {"acquisition": "none", "practice": "none", "formative": "formative",
       "summative": "summative", "discussion": "none", "production": "none"}
LEVEL = {"acquisition": 2, "practice": 3, "formative": 3, "summative": 4, "discussion": 5, "production": 6}
DEFAULT_D = {"acquisition": {"PROF"}, "practice": {"CONT"}, "formative": {"ID", "PERF", "CONT"},
             "summative": {"ID", "PERF", "CONT"}, "discussion": {"ID", "CONT"}, "production": {"ID", "CONT"}}
DOMAINS = ["programming", "datascience", "language"]

# capability: (description keywords, domains or None for all, src attributes, typed acc, produces, consumes, pedagogical act)
CAPS = {
 "explain_concept":     ("explain a concept with worked examples and analogies", None, set(), {"CONT", "PROF"}, set(), set(), "explain"),
 "retrieve_courseware": ("retrieve the course lesson and reading material for a topic", None, set(), {"ID"}, set(), set(), "explain"),
 "web_search":          ("search the public web for pages about a query", None, set(), None, set(), set(), "search"),
 "video_lookup":        ("find educational videos and lectures about a topic", None, set(), {"CONT"}, set(), set(), "illustrate"),
 "generate_quiz":       ("generate quiz questions and multiple choice items on a topic", None, set(), {"CONT", "PROF"}, {"quiz"}, set(), "exercise"),
 "grade_answers":       ("grade quiz answers and return the score of the learner", None, {"PERF"}, {"CONT", "ID"}, {"score"}, {"submission"}, "assess"),
 "record_grade":        ("write a grade into the course gradebook", None, set(), {"ID", "PERF"}, set(), {"score"}, "record"),
 "read_gradebook":      ("read the grades of the learner from the gradebook", None, {"ID", "PERF"}, {"ID"}, set(), set(), "record"),
 "read_profile":        ("read the learner profile preferences and prior knowledge", None, {"ID", "PROF"}, {"ID"}, set(), set(), "record"),
 "give_hint":           ("give a hint for an exercise without revealing the full solution", None, set(), {"CONT"}, set(), set(), "exercise"),
 "flashcards":          ("create spaced repetition flashcards and track reviews", None, {"BEH"}, {"CONT", "ID"}, set(), set(), "exercise"),
 "forum_post":          ("post a message to the course discussion forum", None, set(), {"CONT", "ID"}, set(), set(), "explain"),
 "peer_review":         ("assign and collect peer reviews of submissions", None, {"ID", "CONT"}, {"CONT", "ID"}, set(), {"submission"}, "assess"),
 "plagiarism_check":    ("check a submission for plagiarism and similarity", None, set(), {"CONT", "ID"}, set(), {"submission"}, "assess"),
 "submit_assignment":   ("submit an assignment file to the learning management system", None, set(), {"CONT", "ID"}, {"submission"}, set(), "record"),
 "notify_instructor":   ("notify the instructor by message about the learner", None, set(), None, set(), set(), "record"),
 "learning_analytics":  ("track learner activity time on task and engagement", None, {"BEH", "ID"}, {"ID"}, set(), {"score"}, "record"),
 "export_report":       ("export a progress report as a file and send it by email", None, set(), None, set(), {"score"}, "record"),
 "run_code":            ("execute code in a sandbox and return the output", {"programming", "datascience"}, {"CONT"}, None, {"submission"}, set(), "exercise"),
 "grade_code":          ("run unit tests to grade a code submission", {"programming", "datascience"}, {"PERF"}, {"CONT", "ID"}, {"score"}, {"submission"}, "assess"),
 "lint_code":           ("static analysis and style feedback on source code", {"programming"}, set(), {"CONT"}, set(), set(), "exercise"),
 "query_sql":           ("run SQL queries on a practice database", {"programming", "datascience"}, set(), {"CONT"}, set(), set(), "exercise"),
 "load_dataset":        ("load a tabular dataset for analysis", {"datascience"}, set(), set(), set(), set(), "illustrate"),
 "run_notebook":        ("execute a notebook cell and return results", {"datascience"}, {"CONT"}, None, {"submission"}, set(), "exercise"),
 "plot_data":           ("plot charts and visualizations from data", {"datascience"}, set(), {"CONT"}, set(), set(), "illustrate"),
 "check_grammar":       ("check grammar and spelling of an English text", {"language"}, set(), {"CONT"}, set(), set(), "exercise"),
 "writing_feedback":    ("give feedback on a written essay in English", {"language"}, set(), {"CONT", "PROF"}, {"submission"}, set(), "assess"),
 "translate_text":      ("translate a text between English and another language", {"language"}, set(), {"CONT"}, set(), set(), "explain"),
 "dictionary_lookup":   ("look up English word definitions and examples", {"language"}, set(), set(), set(), set(), "explain"),
 "pronunciation_eval":  ("evaluate pronunciation from a voice recording", {"language"}, {"PERF", "BEH"}, {"CONT", "ID"}, {"score"}, set(), "assess"),
 "conversation_practice": ("role play a conversation partner for speaking practice", {"language"}, set(), {"CONT", "PROF"}, set(), set(), "exercise"),
}

TYPE_CAPS = {
 "acquisition": ["explain_concept", "retrieve_courseware", "web_search", "video_lookup", "dictionary_lookup", "translate_text", "read_profile", "load_dataset"],
 "practice": ["generate_quiz", "grade_answers", "give_hint", "run_code", "grade_code", "lint_code", "run_notebook", "plot_data", "query_sql",
              "check_grammar", "conversation_practice", "pronunciation_eval", "flashcards"],
 "formative": ["generate_quiz", "grade_answers", "record_grade", "grade_code", "run_code", "writing_feedback", "pronunciation_eval",
               "read_gradebook", "learning_analytics"],
 "summative": ["generate_quiz", "grade_answers", "record_grade", "grade_code", "run_code", "plagiarism_check", "submit_assignment",
               "pronunciation_eval", "read_gradebook"],
 "discussion": ["forum_post", "peer_review", "conversation_practice", "explain_concept", "notify_instructor"],
 "production": ["submit_assignment", "peer_review", "plagiarism_check", "writing_feedback", "run_notebook", "plot_data", "run_code",
                "export_report", "check_grammar"],
}
# capability always required for a task of a given type in a given domain (generator contract (c))
CORE_CAP = {
 ("practice", "programming"): "run_code", ("practice", "datascience"): "run_notebook", ("practice", "language"): "check_grammar",
 ("formative", "programming"): "grade_code", ("formative", "datascience"): "grade_code", ("formative", "language"): "writing_feedback",
 ("summative", "programming"): "grade_code", ("summative", "datascience"): "grade_code", ("summative", "language"): "pronunciation_eval",
 ("acquisition", None): "explain_concept", ("discussion", None): "forum_post", ("production", None): "submit_assignment",
}
HANDOFF_ATTRS = {"submission": {"CONT", "ID"}, "score": {"PERF", "ID"}, "quiz": {"CONT"}}

# pedagogical fit phi(act, type) in [0,1] (provisional, to be elicited from instructors)
PHI = {
 "explain":    {"acquisition": 1.0, "practice": 0.5, "formative": 0.2, "summative": 0.1, "discussion": 0.7, "production": 0.4},
 "search":     {"acquisition": 0.9, "practice": 0.4, "formative": 0.1, "summative": 0.0, "discussion": 0.6, "production": 0.5},
 "illustrate": {"acquisition": 0.9, "practice": 0.6, "formative": 0.3, "summative": 0.2, "discussion": 0.5, "production": 0.7},
 "exercise":   {"acquisition": 0.4, "practice": 1.0, "formative": 0.8, "summative": 0.6, "discussion": 0.4, "production": 0.7},
 "assess":     {"acquisition": 0.1, "practice": 0.7, "formative": 1.0, "summative": 1.0, "discussion": 0.5, "production": 0.8},
 "record":     {"acquisition": 0.3, "practice": 0.5, "formative": 0.9, "summative": 1.0, "discussion": 0.6, "production": 0.8},
}

# substitution catalogue Sub (pre-approved type pairs); summative tasks are never substituted automatically
SUB = {"acquisition": ["acquisition", "discussion"], "practice": ["practice", "production"], "formative": ["formative"],
       "summative": [], "discussion": ["discussion", "acquisition"], "production": ["production", "practice"]}


def panel_majority(kind):
    """Majority answer of the LLM expert panel (results/llm/judges_*.csv) for norms ('norm') or substitutions ('sub')."""
    import glob, os, csv
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "llm")
    votes = {}
    for f in glob.glob(os.path.join(here, "judges_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["kind"] == kind and r["value"] in ("True", "False", "true", "false"):
                votes.setdefault((r["a"], r["b"]), []).append(r["value"].lower() == "true")
    return {k: sum(v) > len(v) / 2 for k, v in votes.items()}


def norm_base(profile="standard"):
    """deny[(type, attribute, role)] -> bool. Standard follows Table 3 of the paper; permissive/strict move cells;
    'panel' takes the majority answer of the LLM expert panel where it answered, the standard value elsewhere."""
    if profile == "panel":
        base = norm_base("standard"); maj = panel_majority("norm")
        for (ty, a, r) in base:
            if (f"{ty}|{a}", r) in maj:
                base[(ty, a, r)] = not maj[(f"{ty}|{a}", r)]
        return base
    deny = {}
    for ty in TYPES:
        for a in ATT:
            for r in ROLES:
                d = False
                if r in EXT_ROLES:
                    d = not (ty == "acquisition" and a in {"PROF", "CONT"})
                elif r == "PRO":
                    d = (ty in {"formative", "summative"} and a in {"PERF", "ID"})
                if profile == "permissive" and r in EXT_ROLES and a == "CONT" and ty in {"practice", "production", "discussion"}:
                    d = False
                if profile == "strict":
                    if r == "PRO" and a in {"BEH", "PROF"}:
                        d = True
                    if r in EXT_ROLES and a == "PROF":
                        d = True
                deny[(ty, a, r)] = d
    return deny


from .concepts import domain_concepts
DOMAIN_CONCEPTS = domain_concepts()   # LectureBank 2.0 (data science), CS2023 (programming), CEFR (English)

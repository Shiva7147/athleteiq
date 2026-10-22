"""
rag/corpus/sports_science.py
Curated sports-science consensus corpus chunked by section, author, year, and page.
Contains peer-reviewed literature on ACWR critique, injury prediction limits, complex systems, and sprint micro-dosing.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class LiteratureChunk:
    chunk_id: str
    citation: str           # e.g. "Bahr R. Br J Sports Med 2016;50(13):776–780"
    title: str
    section: str            # e.g. "Predictive Modeling Limits"
    page: int
    content: str


SPORTS_SCIENCE_CORPUS: List[LiteratureChunk] = [
    LiteratureChunk(
        chunk_id="bahr_2016_screening_limits",
        citation="Bahr R. Br J Sports Med 2016;50(13):776–780",
        title="Why screening tests to predict injury do not work—and probably never will…: a critical review",
        section="Screening Tests & Prediction Limits",
        page=777,
        content=(
            "Individual injury prediction does not work in practice and probably never will. Screening tests and "
            "statistical cutoff algorithms lack the requisite sensitivity and positive predictive value to forecast "
            "who will get injured. Systems should not claim to predict biological injuries; at most, workload monitoring "
            "serves as an operational triage filter to direct staff attention."
        )
    ),
    LiteratureChunk(
        chunk_id="bittencourt_2016_complex_systems",
        citation="Bittencourt NFN et al. Br J Sports Med 2016;50(21):1309–1314",
        title="Complex systems approach for sports injuries: moving from risk factor identification to risk profile simulation",
        section="Complex Dynamic Systems Etiology",
        page=1310,
        content=(
            "Sports injuries are emergent phenomena resulting from non-linear interactions among multiple physiological, "
            "biomechanical, and environmental web determinants. Isolated linear or monotonic risk factor associations "
            "fail because risk profiles change dynamically with each training session and match exposure."
        )
    ),
    LiteratureChunk(
        chunk_id="impellizzeri_2020_acwr_pitfalls",
        citation="Impellizzeri FM et al. Int J Sports Physiol Perform 2020;15(6):907–913",
        title="Acute:Chronic Workload Ratio: Conceptual Issues and Fundamental Pitfalls",
        section="Methodological Pitfalls of ACWR",
        page=909,
        content=(
            "The Acute:Chronic Workload Ratio (ACWR) suffers from severe mathematical and conceptual flaws, including "
            "spurious correlation, ratio coupling, and lack of verified causal or predictive validity. While exponentially "
            "weighted moving averages (EWMA) reduce rolling-window coupling, ratio variables remain statistically "
            "problematic and should not be used as isolated injury risk determinants."
        )
    ),
    LiteratureChunk(
        chunk_id="gabbett_2016_training_injury_paradox",
        citation="Gabbett TJ. Br J Sports Med 2016;50(5):273–280",
        title="The training-injury prevention paradox: should athletes be training smarter and harder?",
        section="High-Speed Exposure & Chronic Load",
        page=275,
        content=(
            "High chronic workloads built progressively over time develop physiological tissue resilience. Complete rest "
            "causes rapid decapacitation of eccentric muscular tolerance. When an athlete is rested or benched, "
            "controlled high-speed running micro-dosing helps prevent the under-preparation trap when returning to match play."
        )
    ),
    LiteratureChunk(
        chunk_id="uefa_injury_study_hamstrings",
        citation="Ekstrand J et al. Br J Sports Med 2016;50(12):731–737",
        title="Hamstring injury rates have increased during recent seasons and now constitute 24% of all injuries",
        section="Epidemiology & Match Congestion",
        page=733,
        content=(
            "Hamstring muscle injuries are the single most common time-loss injury in professional football. Fixture "
            "congestion with less than 72 hours between matches significantly elevates non-contact soft tissue vulnerability "
            "due to cumulative eccentric fatigue and incomplete neuromuscular recovery."
        )
    ),
]

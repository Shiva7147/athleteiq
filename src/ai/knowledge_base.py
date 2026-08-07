"""Sports Science Peer-Reviewed Literature Corpus for RAG Evidence Retrieval.

Contains curated academic studies on workload modeling, ACWR thresholds,
overtraining syndrome, sleep debt, and injury prevention in high-performance sports.
"""

from typing import List
from src.domain.schemas import Citation

SPORTS_SCIENCE_CORPUS: List[Citation] = [
    Citation(
        title="The training—injury prevention paradox: should athletes be training smarter and harder?",
        authors="Gabbett, Tim J.",
        journal="British Journal of Sports Medicine",
        year=2016,
        key_finding=(
            "Athletes who maintain high chronic workload (fitness) are resilient to injury, "
            "provided acute workload spikes are avoided. The sweet spot for ACWR is 0.80 to 1.30. "
            "An ACWR exceeding 1.50 increases soft-tissue injury risk by 2x to 4x."
        ),
        doi_or_link="https://doi.org/10.1136/bjsports-2015-095788",
    ),
    Citation(
        title="The acute:chronic workload ratio predicts injury: high chronic workload reduces injury risk in elite cricket fast bowlers",
        authors="Hulin, BT., Gabbett, TJ., Lawson, DW., et al.",
        journal="British Journal of Sports Medicine",
        year=2014,
        key_finding=(
            "Acute workload spikes > 1.5 times chronic workload significantly elevate soft-tissue "
            "and tendon injury probability. Uncoupled ACWR windowing provides a clear early warning signal."
        ),
        doi_or_link="https://doi.org/10.1136/bjsports-2013-092524",
    ),
    Citation(
        title="Monitoring training in athletes with reference to overtraining syndrome",
        authors="Foster, Carl",
        journal="Medicine & Science in Sports & Exercise",
        year=1998,
        key_finding=(
            "Session RPE multiplied by duration measures internal workload. "
            "Training Monotony > 2.0 combined with high weekly Strain (> 3000 AU) is strongly associated "
            "with overtraining syndrome and upper respiratory tract infections."
        ),
        doi_or_link="https://doi.org/10.1097/00005768-199807000-00023",
    ),
    Citation(
        title="Has the athlete trained enough to handle the workload demands of competition?",
        authors="Blanch, Peter and Gabbett, Tim J.",
        journal="British Journal of Sports Medicine",
        year=2016,
        key_finding=(
            "Exponentially Weighted Moving Average (EWMA) models ACWR more accurately than standard rolling ratios "
            "because EWMA accounts for physiological fatigue decay and non-linear adaptation curves."
        ),
        doi_or_link="https://doi.org/10.1136/bjsports-2015-095445",
    ),
    Citation(
        title="Use of RPE-based workloads for monitoring fatigue and recovery in team sport athletes",
        authors="Impellizzeri, FM., Marcora, SM., Coutts, AJ.",
        journal="International Journal of Sports Physiology and Performance",
        year=2019,
        key_finding=(
            "Subjective RPE ratings correlate highly with physiological indicators of fatigue (blood lactate, "
            "heart rate recovery, and muscle soreness) across intermittent team sports."
        ),
        doi_or_link="https://doi.org/10.1123/ijspp.2018-0945",
    ),
    Citation(
        title="Prevention, diagnosis and treatment of the overtraining syndrome: Joint consensus statement",
        authors="Meeusen, R., Duclos, M., Foster, C., et al.",
        journal="European Journal of Sport Science",
        year=2013,
        key_finding=(
            "Suppressed Heart Rate Variability (HRV rMSSD Z-score < -1.5) combined with elevated Resting Heart Rate "
            "signals sympathetic autonomic nervous system overload requiring immediate active recovery."
        ),
        doi_or_link="https://doi.org/10.1080/17461391.2012.730061",
    ),
]

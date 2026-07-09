# knowledge_base.py
# AthleteIQ — Sports Science Knowledge Base
# 50 evidence-based entries covering training load, recovery,
# injury prevention, overtraining, performance, and safety

from langchain_core.documents import Document

KNOWLEDGE_ENTRIES = [
    # TRAINING LOAD
    "The Acute:Chronic Workload Ratio (ACWR) measures training load balance. Acute load is the last 7 days. Chronic load is the rolling 28-day average. Safe ACWR range is 0.8 to 1.3 according to Gabbett (2016). Above 1.5 significantly increases injury risk by up to 2x.",
    "Session RPE (sRPE) is calculated by multiplying session duration in minutes by the athletes Rating of Perceived Exertion on the Foster 1-10 scale. Example: 60 minute session at RPE 7 equals 420 arbitrary units of training load.",
    "The monotony index measures training variation. Calculated as weekly average load divided by standard deviation. High monotony above 2.0 increases injury and illness risk even when total load is acceptable.",
    "Exponentially Weighted Moving Average (EWMA) is more accurate than simple rolling averages for calculating ACWR. It gives more weight to recent sessions making it more sensitive to sudden load spikes.",
    "Training load should increase no more than 10% per week to allow adaptation without injury. Applies to volume intensity and frequency independently. Violating this is the most common cause of overuse injuries.",

    # RECOVERY
    "Sleep is the most critical recovery tool for athletes. 8 to 10 hours recommended for elite athletes. Sleep deprivation of even 1 hour reduces reaction time by 30% and increases injury risk significantly.",
    "Heart Rate Variability (HRV) reflects autonomic nervous system recovery status. Higher HRV indicates better recovery. A drop of more than 2 standard deviations below baseline for 3 consecutive days signals accumulated fatigue requiring load reduction.",
    "Active recovery sessions at 50-60% maximum heart rate improve blood lactate clearance faster than passive rest. Optimal duration is 20-30 minutes. Swimming and cycling are most effective for lower-body sport athletes.",
    "Cold water immersion at 10-15 degrees Celsius for 10-15 minutes reduces muscle soreness and perceived fatigue after high-intensity exercise. Most effective within 30 minutes post-session.",
    "Nutrition timing matters for recovery. Consuming 20-40g protein and 1g per kg bodyweight carbohydrates within 30 minutes post-exercise maximises glycogen resynthesis and muscle protein synthesis.",

    # INJURY PREVENTION
    "ACL injury risk in athletes is 2 to 8 times higher in female athletes than male. Primary risk factors include landing mechanics hip abductor weakness and high training load spikes. Prevention programs like FIFA 11+ reduce ACL risk by up to 50%.",
    "Hamstring injury return to play follows 3 criteria: pain-free at maximum sprint speed, limb symmetry index above 90% on strength testing, successful completion of sport-specific running progression. Average return time 2-4 weeks grade 1, 4-8 weeks grade 2.",
    "Patellar tendinopathy is most common in jumping sports. Risk increases with high training monotony and sudden volume increases. Eccentric single-leg squat programs reduce pain and improve function in 70% of cases.",
    "Stress fractures result from repetitive mechanical loading exceeding bone remodelling capacity. Most common in tibia metatarsals and femoral neck. Early warning is localised bone pain that worsens during activity and improves with rest.",
    "Shoulder impingement in overhead athletes results from rotator cuff weakness relative to deltoid strength. Prevention: maintain rotator cuff to deltoid strength ratio above 65%. Scapular stabilisation exercises are first-line intervention.",

    # OVERTRAINING
    "Overtraining syndrome markers include persistent fatigue lasting more than 2 weeks, performance decline despite adequate rest, elevated resting heart rate above baseline by 7+ bpm, mood disturbances, and suppressed immune function.",
    "Functional Overreaching is short-term performance decrease with 1-2 weeks recovery. Non-Functional Overreaching requires weeks to months recovery. Overtraining Syndrome may require months to years. Early detection is critical.",
    "Resting heart rate elevation of more than 5 beats above personal baseline for 3 or more consecutive mornings indicates inadequate recovery and potential overtraining. Immediate load reduction recommended.",
    "Cortisol to testosterone ratio is a reliable biochemical marker of overtraining. Ratio above 0.35 indicates catabolic state requiring reduced training load. Requires blood test for accurate measurement.",
    "Mood state monitoring using the POMS questionnaire detects overtraining before physical symptoms appear. The iceberg profile with high vigour and low negative emotions inverts during overtraining.",

    # PERFORMANCE
    "VO2 max is the maximum rate of oxygen consumption during exercise. Elite endurance athletes score 70-85 ml/kg/min. Recreational athletes score 35-50 ml/kg/min. Improves 15-20% with structured training over 8-12 weeks.",
    "Lactate threshold is the exercise intensity at which blood lactate begins to accumulate faster than it can be cleared. Training at lactate threshold improves endurance performance more than any other intensity zone.",
    "Periodisation involves planned variation of training stress over time. Linear periodisation progresses from high volume to high intensity. Undulating periodisation varies daily or weekly. Block periodisation concentrates specific qualities.",
    "Peak power output in sprint athletes correlates with Type IIx muscle fibre percentage. Power training with loads at 30-70% of 1RM at maximum velocity optimally develops Rate of Force Development.",
    "Caffeine at 3-6mg per kg bodyweight consumed 60 minutes before exercise improves endurance performance by 2-4% and reduces perceived effort. Tolerance develops with daily use so cycling is recommended.",

    # HYDRATION
    "Dehydration of just 2% body weight loss impairs cognitive function and endurance performance. Athletes should drink 500ml 2 hours before training and replace 150% of fluid lost during session within 4 hours post-exercise.",
    "Sweat rate varies from 0.5 to 2.5 litres per hour depending on exercise intensity temperature and individual factors. Sodium losses of 0.5-1.5g per litre of sweat must be replaced to prevent hyponatremia.",
    "Urine colour is a practical hydration indicator. Pale yellow indicates adequate hydration. Dark yellow indicates dehydration. Clear may indicate overhydration which carries risk of hyponatremia.",
    "Sports drinks containing 6-8% carbohydrate and electrolytes are most effective for sessions exceeding 60-90 minutes. Water is sufficient for shorter sessions in mild conditions.",
    "Pre-exercise hyperhydration with glycerol at 1g per kg bodyweight in 1.5 litres of fluid expands plasma volume and delays dehydration onset during prolonged exercise in heat.",

    # BIOMECHANICS
    "Ground reaction forces during running reach 2-3 times bodyweight at foot strike. Forefoot strikers show lower impact peak forces than heel strikers but higher Achilles tendon loading. Neither pattern is universally superior.",
    "Knee valgus during landing is the primary biomechanical risk factor for ACL injury. Real-time biofeedback training reduces valgus angles by 20-30% in 4-6 weeks.",
    "Hip drop during single leg stance indicates weak hip abductors. Greater than 5 degrees of hip drop during running increases IT band syndrome and patellofemoral pain syndrome risk.",
    "Running economy improves with lower vertical oscillation higher cadence of 170-180 steps per minute and adequate hip extension. Every 1% improvement in running economy predicts approximately 1% performance improvement.",
    "Force-velocity relationship dictates that maximum force production decreases as movement velocity increases. Training at varied points on the curve develops complete athletic power profiles.",

    # RED FLAGS — SAFETY
    "Chest pain during exercise requires immediate cessation and medical evaluation. May indicate hypertrophic cardiomyopathy arrhythmia or coronary artery disease. Return to play only after cardiology clearance.",
    "Concussion in sport requires immediate removal from play. No same-day return regardless of symptom resolution. Return to play protocol involves 6 graduated stages over minimum 7 days with medical clearance.",
    "Exertional heat stroke is a life-threatening emergency. Core temperature above 40 degrees Celsius with neurological dysfunction. Immediate cold water immersion is the most effective treatment. Call emergency services.",
    "Exercise-induced anaphylaxis presents as hives breathing difficulty and collapse during or after exercise. Requires immediate epinephrine injection and emergency services.",
    "Sudden cardiac arrest in athletes requires immediate CPR and defibrillation. Survival rate drops 10% per minute without defibrillation. Every sports facility should have an AED and trained personnel.",

    # YOUTH ATHLETES
    "Youth athletes should not specialise in a single sport before age 15. Early specialisation increases overuse injury risk by 50% and burnout risk significantly. Multi-sport participation develops better overall athleticism.",
    "Growth plates in youth athletes close between ages 14-22. Epiphyseal stress injuries require load reduction and monitoring. Resistance training is safe for youth when technique is prioritised over load.",
    "Relative Energy Deficiency in Sport affects both male and female athletes. Low energy availability disrupts hormones bone health and immunity. Signs include stress fractures hormonal disruption mood changes and poor performance.",
    "Youth athlete sleep requirements are higher than adults. Ages 6-12 need 9-12 hours. Ages 13-18 need 8-10 hours. Sleep deprivation in youth athletes doubles injury risk compared to adults.",
    "Training volume for youth should not exceed 1 hour of organised practice per year of age. A 12-year-old should train maximum 12 hours per week across all sports.",

    # MENTAL PERFORMANCE
    "Pre-competition anxiety exists on a spectrum from facilitating to debilitating. Cognitive reappraisal by reinterpreting anxiety as excitement improves performance more than relaxation techniques.",
    "Attentional focus affects motor skill execution. External focus on movement effect consistently outperforms internal focus on body movement for skill acquisition and performance in most sports.",
    "Flow state occurs when challenge level matches skill level. Optimal performance zone identified when arousal is moderate and performance anxiety is low.",
    "Mental rehearsal activates similar neural pathways as physical practice. 15-20 minutes of vivid mental rehearsal per day improves performance 13-15% as a supplement to physical training.",
    "Goal setting using SMART criteria which stands for Specific Measurable Achievable Relevant and Time-bound improves athletic performance by increasing motivation and providing clear direction for training focus."
]

def get_documents():
    """Convert knowledge entries to LangChain Document objects"""
    return [Document(page_content=entry) for entry in KNOWLEDGE_ENTRIES]

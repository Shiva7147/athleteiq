# AthleteIQ 2.0: Sports Science Workload Monitoring & Tactical Scenario Sandbox

![Build & Eval Status](https://img.shields.io/badge/CI%20Eval%20Gate-45%2F45%20Passing-emerald)
![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-LangGraph%20%2B%20FastMCP%20%2B%20Polars-purple)
![Data](https://img.shields.io/badge/Real%20Data-Catapult%20GPS%20(9%2C555%20sessions)-orange)
![Security](https://img.shields.io/badge/Security-RBAC%20%2B%20GDPR%20Pseudonymization-blueviolet)
![License](https://img.shields.io/badge/License-MIT-gray)

> **AthleteIQ 2.0** is an empirical sports-science decision-support system and workload monitoring sandbox. Rather than claiming deterministic injury prediction, it operationalizes an alert capacity constraint of 3 athletes per morning, evaluates workload trajectories under hypothetical fixture scenarios using continuous impulse-response modeling, and grounds recommendations in peer-reviewed sports epidemiology.

---

## 1. Scientific Thesis: What AthleteIQ 2.0 Is (and What It Is Not)

* **No Deterministic Injury Prediction:** Grounded in sports-science literature ([Bahr 2016](https://bjsm.bmj.com/content/50/13/776); [Bittencourt et al. 2016](https://bjsm.bmj.com/content/50/21/1309); [Impellizzeri et al. 2020](https://journals.humankinetics.com/view/journals/ijspp/15/6/article-p907.xml)), athletic injuries are emergent phenomena within complex dynamic webs of determinants. They cannot be reliably predicted for individuals using workload thresholds or machine learning classifiers alone.
* **Operational Triage Budget:** Flagging entire squads causes alert fatigue. The "3 athletes per day" budget is an **operational workflow constraint** (reflecting the physical examination bandwidth of 1–2 physiotherapists before morning training), prioritizing staff inspection toward the highest relative workload anomalies.
* **Empirical Validation on Real Data:** Evaluated directly on competitive soccer Catapult GPS tracking telemetry (`data/raw/Soccer.csv`, 9,555 records, 27 athletes) under strict athlete-grouped splits to prevent intra-athlete leakage.
* **Point-in-Time Correctness:** An automated CI gate ensures feature engineering on Day $D$ strictly accesses data timestamped prior to Day $D$, preventing future lookahead leakage.
* **Zero-Arithmetic LLM Policy:** The language model performs zero mathematical calculations. All features and metrics originate from Polars, LightGBM, scikit-learn, or the Banister simulation engine. The LangGraph agent acts strictly as a conversational router and literature retriever.

---

## 2. Empirical Benchmark: Real-World Catapult GPS Data

Models were evaluated on real competitive soccer tracking data (`data/raw/Soccer.csv`), tracking 27 elite players across 9,555 session records. The ground-truth prediction target is a **forward multi-day absence gap ($\ge 4$ days)** in subsequent training sessions ($t+1$), yielding an actual real-world prevalence of **$3.04\%$** ($N=55$ absence events across the season), matching elite soccer epidemiology.

Evaluation employed a strict **Athlete-Grouped Split** (`GroupKFold` on held-out athletes, $N_{\text{train}}=1,230$, $N_{\text{test}}=560$ sessions across 8 unseen test athletes) to ensure zero intra-athlete leakage across train and test folds.

### Head-to-Head Model Evaluation

| Model Tier & Architecture | ROC-AUC | PR-AUC (Avg Precision) | Brier Score (MSE $\downarrow$) | Calibration Error (ECE $\downarrow$) |
|---|:---:|:---:|:---:|:---:|
| **Tier 0: ACWR Heuristic Rule**<br>*(Gabbett 2016 threshold: $<0.80$ or $>1.45$)* | 0.4428 | 0.0286 | 0.0391 | 0.0585 |
| **Tier 1: Regularized Logistic Regression**<br>*(L2-penalized linear baseline)* | **0.7162** | **0.0740** | 0.3331 | 0.5231 |
| **Tier 2: LightGBM + Grouped Platt Calibration**<br>*(Constrained gradient boosting + sigmoid calibration)* | 0.6767 | 0.0644 | **0.0312** | **0.0206** |

### Key Scientific Takeaways

1. **Simple Linear Baselines Match or Edge Out Trees on Raw Ranking:** L2 Logistic Regression achieved slightly higher ROC-AUC (0.716 vs 0.677) and PR-AUC (0.074 vs 0.064) than LightGBM on unseen athletes. With small cohorts ($N=27$) and low event rates ($3\%$), non-linear tree ensembles provide no magical ranking advantage over well-regularized linear models.
2. **LightGBM's Decisive Advantage is Probability Calibration:** Uncalibrated logistic regression produced severe overconfidence (Brier 0.3331, ECE 0.5231). LightGBM with athlete-grouped Platt sigmoid calibration achieved an order-of-magnitude reduction in probabilistic error (Brier **0.0312**, ECE **0.0206**).
3. **The ACWR Threshold Heuristic Fails on Real Data:** The classical rolling ACWR rule achieved ROC-AUC 0.4428 and PR-AUC 0.0286 (worse than the 3.04% base rate), empirically substantiating Impellizzeri et al. (2020) and Carey et al. (2018).

---

## 3. Methodological Details & Honest Caveats

### Model Design Choices
* **Relaxed Monotonic Constraints:** Athletic fatigue exhibits U-shaped sweet spots (both sudden spikes and complete rest decapacitation elevate risk). Hard monotonic constraints on chronic load structurally failed to represent under-training decapacitation; chronic load is left unconstrained while fatigue and strain retain soft regularized boundaries.
* **Grouped Probability Calibration:** Probability calibration uses `GroupKFold` grouped strictly by `athlete_id`. Fitting calibration scalers across repeated sessions of the same athlete causes calibration leakage.
* **SHAP Feature Attribution:** TreeSHAP attributions indicate mathematical sensitivity within model decision trees, **not biological causation**. A high SHAP attribution for acute load indicates the model used that feature to alter its output, not that load was the medical cause of tissue failure.
* **EWMA Load Smoothing:** Exponentially weighted moving averages (EWMA) reduce the mathematical coupling artifacts inherent in ratio statistics, but do not eliminate the broader critique regarding multi-factorial etiology and non-stationarity.

### Forward Trajectory Simulation & Conformal Uncertainty
* **Banister/Busso Impulse-Response:** Simulates macro readiness and fatigue decay curves across multi-fixture horizons. This is a tactical scenario sandbox, **not an individual clinical diagnosis**.
* **Split-Conformal Prediction Bands:** Conformal prediction intervals are calculated via the inductive quantile $\lceil (n+1)(1-\alpha) \rceil / n$. Marginal coverage guarantees ($\ge 1-\alpha$) strictly require data exchangeability. Because longitudinal athlete sessions exhibit autocorrelation, these bounds represent empirical calibration bands rather than formal i.i.d. mathematical proofs.

### Safety, Security & Compliance
* **Deterministic Policy Tripwires:** The safety module is a hardcoded keyword and regular expression filter that acts as an exact policy tripwire for known high-risk tokens (`concussion`, `cortisone`, `tramadol`). Paraphrased or novel prompts require human clinical oversight.
* **HMAC Pseudonymization:** Athlete identifiers are tokenized using HMAC-SHA256 to minimize plaintext exposure. This represents a technical security control under GDPR Article 4(5) / 32, but does not constitute comprehensive legal GDPR/HIPAA compliance on its own.
* **In-Memory Rate Limiting:** The sliding-window rate limiter runs in-memory per FastAPI worker process. For multi-worker production deployments, an external state store (e.g. Redis) is required.

---

## 4. System Architecture

```
                             ATHLETEIQ 2.0 REPOSITORY ARCHITECTURE
                             
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. SURFACES & REPORTERS                                                                │
  │   • Morning Squad Dashboard (Operational Triage Watchlist: Capped at 3)               │
  │   • Tactical What-If Scenario Matrix (Multi-Fixture Trajectory Sandbox)                │
  │   • Spoken Audio Briefing (07:00 Daily Coach Memo) & Video Storyboard Synthesis        │
  │   • FastMCP Server (Claude Desktop & IDE integration)                                  │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │ REST / MCP
  ┌───────────────────────────────────────────▼────────────────────────────────────────────┐
  │ 2. API & GATEWAY (FastAPI, Pydantic v2)                                                │
  │   • Endpoints: /v2/health, /v2/squad/risk-ranking, /v2/simulate/trajectory             │
  │   • /v2/evaluation/benchmark (Empirical Catapult GPS Benchmark Data)                   │
  │   • /v2/briefing/morning, /v2/agent/chat                                               │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │
  ┌───────────────────────────────────────────▼────────────────────────────────────────────┐
  │ 3. AGENT STATE MACHINE (LangGraph)                                                     │
  │   • Deterministic Policy Guard: Red flag tripwires for unauthorized medical bypass     │
  │   • Orchestrator Router: Point-in-time audited tool execution                          │
  │   • Hybrid RAG: BM25 + Dense Semantic Fusion with peer-reviewed literature citations   │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │
  ┌───────────────────────────────────────────▼────────────────────────────────────────────┐
  │ 4. DETERMINISTIC ML & EVALUATION CORE                                                  │
  │   • Empirical Evaluation Harness: evals/benchmark.py on Catapult GPS (Soccer.csv)      │
  │   • Model Ladder: Tier 0 (ACWR Rule) ➔ Tier 1 (LogReg) ➔ Tier 2 (LightGBM)            │
  │   • Grouped Platt Probability Calibration & Inductive Conformal Bounds                 │
  │   • TreeSHAP Feature Sensitivity Attributions (Model Local Sensitivities)              │
  │   • Forward-Rolling Banister Impulse-Response Simulation Engine                        │
  │   • Point-in-Time Verified Polars Feature Engineering Pipeline                         │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Interactive Dashboard Tour (Step-by-Step Experience)

The flagship interface ([`web/templates/index.html`](file:///d:/athleteiq/web/templates/index.html)) provides a decluttered, high-contrast sports-science console structured as an interactive 4-stage progression:

```
[STEP 01: SQUAD TRIAGE] ➔ [STEP 02: WHY FLAGGED?] ➔ [STEP 03: WHAT-IF SANDBOX] ➔ [STEP 04: AI COPILOT]
```

1. **Step 01: Squad Attention Triage:**
   - Visualizes the 3 athletes demanding sports-science evaluation under the operational capacity constraint.
   - Embeds the full empirical evaluation table on real Catapult GPS data head-to-head against ACWR and linear baselines.
2. **Step 02: Transparent Factor Attribution (SHAP):**
   - Renders exact waterfall sensitivity attributions showing points added or subtracted from the baseline risk score.
   - Reframed as model sensitivity rather than unearned clinical causality.
3. **Step 03: Multi-Match What-If Sandbox:**
   - Sliders allow staff to modulate planned match minutes ($0-120\text{ min}$) and toggle fixture leverage (Tier 1 Title Decider vs. Tier 2 Routine Match).
   - Generates real-time Banister continuous fatigue/readiness curves with inductive split-conformal bounds.
4. **Step 04: Grounded AI Copilot & Tool Inspection:**
   - Interactive LangGraph chat console inspecting tool execution telemetry (`get_player_load`, `simulate_what_if_trajectory`, `search_literature`).
   - One-click peer-reviewed citation inspection for Bahr (2016), Bittencourt (2016), Impellizzeri (2020), and Gabbett (2016).

---

## 6. FastMCP Server: Tool Catalog

AthleteIQ exposes its deterministic calculation engine via Model Context Protocol (FastMCP) for desktop assistants (Claude Desktop, Cursor):

| Tool Name | Parameters | Description |
|---|---|---|
| `get_player_load` | `player_id: str` | Retrieves point-in-time verified EWMA loads, acute/chronic ratios, and Foster strain. |
| `get_risk_ranking` | `top_n: int = 3` | Returns top prioritized athletes under the operational triage budget constraint. |
| `simulate_what_if_trajectory` | `player_id, planned_minutes, is_tier_1_final` | Rolls out 7-day Banister impulse-response curves with conformal confidence bounds. |
| `search_literature` | `query: str` | Hybrid dense + BM25 retrieval over peer-reviewed sports epidemiology literature. |

---

## 7. Multi-Match Fixture Leverage Sandbox

Coaches evaluate load decisions within competitive contexts. A moderate fatigue index in a routine league fixture suggests rotation; in a cup final, coaches often accept higher acute load with a minute cap.

The **Match Leverage Index (MLI)** provides hypothetical scenario rules:
* **Tier 1 (Finals, Derbies, Must-Win):**
  * Risk $15\% - 35\% \rightarrow$ `START_WITH_MINUTE_CAP_60`: Suggests starting but capping minutes to limit late-game eccentric fatigue.
* **Tier 2 / 3 (Routine League / Rotation):**
  * Risk $> 14\% \rightarrow$ `BENCH_AND_PRESCRIBE_MICRODOSE`: Suggests match rest paired with a high-speed running microdose to avoid rest decapacitation.
* **Red Flag Threshold ($> 40\%$ or Acute Clinical):**
  * Mandatory physical evaluation regardless of fixture importance.

---

## 8. Verification & Automated Test Suite

The test suite validates mathematical consistency, point-in-time feature isolation, security RBAC, and policy behavior:

* **Point-in-Time Leakage Gate (`tests/test_features_leakage.py`):** Asserts that injecting future outlier sessions has zero effect on features calculated on Day $D$.
* **Dual-Annotator Agreement (`tests/test_evals.py`):** Frozen 15-case golden set with Cohen's Kappa inter-annotator tracking ($\kappa \ge 0.85$).
* **Deterministic Safety Tripwire Gate (`tests/test_evals.py`):** Validates that known clinical bypass keywords trigger human approval escalation.
* **Security & BOLA Boundary Tests (`tests/test_security.py`):** Validates RBAC permissions, pseudonymization, and sliding-window rate limiting.

```powershell
cmd.exe /c "set PYTHONPATH=d:\athleteiq && d:\athleteiq\.venv\Scripts\python.exe -m pytest d:\athleteiq\tests -v"
======================= 45 passed, 1 warning in 13.26s =======================
```

---

## 9. Quick Start & Execution

### Local Setup
```bash
git clone https://github.com/Shiva7147/athleteiq.git
cd athleteiq

# Set up virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .[ml,agent]

# Run empirical benchmark on real Catapult soccer data
python evals/benchmark.py

# Run full pytest suite
pytest tests/ -v

# Start FastAPI server
uvicorn api.routes:app --reload --port 8000
```

### Access Points
* **Interactive Dashboard:** `http://localhost:8000/`
* **Swagger API Documentation:** `http://localhost:8000/docs`
* **Real-Data Benchmark Endpoint:** `http://localhost:8000/v2/evaluation/benchmark`

### Production Deployment on Vercel
AthleteIQ 2.0 includes native serverless deployment configuration for Vercel:
* **Entry Point:** [`api/index.py`](file:///d:/athleteiq/api/index.py) exports the ASGI `app` instance with root path auto-resolution.
* **Routing Configuration:** [`vercel.json`](file:///d:/athleteiq/vercel.json) rewrites all inbound routes (`/(.*)`) to `@vercel/python`.
* **Dependency Manifest:** [`requirements.txt`](file:///d:/athleteiq/requirements.txt) specifies all serverless runtime dependencies.

To deploy:
1. Connect your repository (`sshivashankar1304-ops/AthleteIQ-2.0`) to [Vercel](https://vercel.com).
2. Leave build settings as default; Vercel automatically detects the Python runtime and `vercel.json`.
3. Click **Deploy** — your interactive dashboard, REST endpoints, and ML models will be live immediately.

---

## 10. Academic Grounding & Literature

* **Bahr R.** *Why screening tests to predict injury do not work—and probably never will…: a critical review.* Br J Sports Med 2016;50(13):776–780.
* **Bittencourt NFN, Meeuwisse WH, Mendonça LD, et al.** *Complex systems approach for sports injuries: moving from risk factor identification to web of determinants.* Br J Sports Med 2016;50(21):1309–1314.
* **Impellizzeri FM, Tenan MS, Kempton T, Coutts AJ, Cornforth DJ, McCall A.** *Acute:Chronic Workload Ratio: Conceptual Issues and Fundamental Pitfalls.* Int J Sports Physiol Perform (IJSPP) 2020;15(6):907–913.
* **Carey DL, Blanch P, Ong KL, et al.** *Predictive modelling of training loads and injury in Australian football.* Br J Sports Med 2018;52(22):1441–1447.
* **Gabbett TJ.** *The training-injury prevention paradox: should athletes be training smarter and harder?* Br J Sports Med 2016;50(5):273–280.
* **Ekstrand J, et al.** *Hamstring injury rates have increased during recent seasons and now constitute 24% of all injuries.* Br J Sports Med 2016;50(12):731–737.

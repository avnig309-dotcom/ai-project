# Intelligent Customer Support & Automated Query Resolution
A five-agent customer-support prototype that identifies what a customer needs, looks up relevant support information, checks whether that information is enough, and decides whether to ANSWER, CLARIFY, or ESCALATE.

The main idea is straightforward: a support system should not give an answer just because it found something that looks relevant. It should also check whether the answer matches the question and whether the available evidence actually supports it.
Project status: Agent 1 (intent classification) and Agent 2 (response retrieval) form the existing baseline. Agents 3–5, the disagreement checks, FastAPI backend, and organizer API adapter are the planned Level 2 integration. Update this section as features are implemented and tested. This README describes the intended complete prototype without treating unfinished components as completed work.

1. Project overview
A customer might ask:
"My payment was deducted, but my order didn't go through. Has my refund been processed?"

A keyword-based chatbot could find a refund response and send it immediately. That would be risky because the system has not verified the customer's payment or refund status.

Our approach separates the work across five components:
1. Intent Analyst: Predicts the customer's intent and estimates uncertainty.
2. Response Specialist: Retrieves relevant support responses and their supporting examples.
3. Routing and Planning Agent: Checks whether there is enough information to proceed.
4. Response Auditor: Checks the proposed response for mismatches, contradictions, and unsupported claims.
5. Coordinator: Makes the final ANSWER, CLARIFY, or ESCALATE decision.
The system is designed to run locally for development and connect to the organizers' API when its key and documentation are provided before evaluation.
2. System workflow
Customer query / evaluation dataset
                 |
                 v
      Agent 1: Intent Analyst
   intent + category + confidence
                 |
                 v
    Agent 2: Response Specialist
    candidate response + evidence
                 |
                 v
    Agent 3: Routing and Planning
 uncertainty + missing details + risk
                 |
                 v
      Agent 4: Response Auditor
    evidence checks + safety flags
                 |
                 v
       Agent 5: Coordinator
                 |
        +--------+--------+
        |        |        |
      ANSWER   CLARIFY  ESCALATE
                 |
                 v
      API response / predictions.csv
All five components are intended to run through a shared orchestrator. The evaluation script and the future UI should call the same core workflow, so results do not depend on which interface is being used.
3. Agent 1 — Intent Analyst
Purpose: Understand the customer's request before choosing a response.
Model
Agent 1 uses supervised multiclass classification with scikit-learn:
- Word-level TF-IDF: Extracts individual words and two-word phrases (unigrams and bigrams).
- Character-level TF-IDF: Extracts character patterns of lengths 3–5. This helps with spelling variations and minor typos.
- FeatureUnion: Combines both sets of features.
- Logistic Regression: Learns which features are associated with each customer-support intent.
- Pipeline: Keeps feature extraction and classification together during training and inference.
The dataset used in the existing baseline contains 27 intents across 11 categories. The predicted intent is mapped to a category using a mapping derived from the training data. Exact label names must remain consistent with the organizer-approved dataset.
Training
The baseline compares Logistic Regression regularization values C = 0.5, 1.0, 2.0, 4.0 and selects the model using validation Macro F1. Macro F1 gives each intent class equal weight when averaging performance.

The trained pipeline is saved with joblib so the model does not need to be retrained every time the application starts.
Inference output
Agent 1 returns:
- Predicted intent
- Predicted category
- Confidence (highest predicted class probability)
- Top-two margin (difference between the two highest class probabilities)
- Uncertainty flag
For example, probabilities of 0.52 and 0.45 for the top two intents give a margin of 0.07. A small margin can indicate that the model is having difficulty distinguishing between two possible requests.

Important: Logistic Regression probabilities are not automatically calibrated. Confidence thresholds should be checked using validation data rather than treated as guarantees.
Main libraries
Library / module	Use
pandas	Read and inspect labeled datasets
numpy	Work with prediction probabilities and rankings
sklearn.feature_extraction.text.TfidfVectorizer	Convert queries into numerical features
sklearn.pipeline.FeatureUnion	Combine word and character features
sklearn.pipeline.Pipeline	Connect vectorization and classification
sklearn.linear_model.LogisticRegression	Train the intent classifier
sklearn.metrics	Calculate accuracy, Macro F1, and class-wise metrics
joblib	Save and load trained models
json	Store category mappings and metrics


4. Agent 2 — Response Specialist
Purpose: Find a relevant support response instead of making one up.

Agent 2 searches a knowledge base built from existing customer-support examples. It uses two TF-IDF representations and cosine similarity to compare a new query with known queries.

The baseline retrieval score is:
retrieval_score = 0.65 × word_similarity + 0.35 × character_similarity
The results are then reranked using Agent 1's intent probabilities. This helps prioritize examples that match both the wording and the likely intent of the customer request.

The agent returns a candidate response, relevance scores, and the most relevant supporting examples. In the complete five-agent version, these examples should also have stable evidence IDs so the auditor and UI can refer to them.

The baseline includes similarity thresholds for weak retrieval matches and some rules for ambiguous cancellation and failed-payment queries.

This is a retrieval system, not a generative language model. Its baseline answers come from the support knowledge base.
Main libraries
Library / module	Use
scikit-learn	TF-IDF vectorization and cosine similarity
numpy	Combine retrieval and reranking scores
scipy	Efficient sparse-matrix operations
pandas	Build and inspect the knowledge base


5. Agent 3 — Routing and Planning Agent
Purpose: Decide whether the system has enough information to continue with an answer.

Status: planned for Level 2. Agent 3 is designed as a rule-based component, not a separately trained classifier.

It takes information from Agents 1 and 2 and checks:
- Is the intent prediction uncertain?
- Is the retrieved evidence relevant enough?
- Is an order number, subscription type, or other essential detail missing?
- Do the predicted intent and retrieved evidence disagree?
- Is the request sensitive, such as a payment or account issue?
For example, if a customer writes "Cancel it", the agent should recommend asking whether they mean an order or a subscription rather than guessing.

Possible internal recommendations include proceeding to audit, asking a clarification question, or escalating for human review. Agent 5 still makes the final decision.
Planned functions
assess_uncertainty(...)
assess_evidence_sufficiency(...)
detect_missing_information(...)
calculate_agent_disagreement(...)
plan_next_step(...)
Planned libraries
Library / module	Use
dataclasses or pydantic	Structured inputs and outputs between agents
typing	Type annotations and clear interfaces
re	Pattern-based checks for missing details and ambiguous wording
logging	Record routing outcomes and errors
json	Serialize routing traces


6. Agent 4 — Response Auditor
Purpose: Check whether a proposed response is relevant, supported, and safe to send.

Status: planned for Level 2. The initial auditor will use local, testable checks. An external model may assist with semantic checks after the organizers provide API access.

The auditor will check for:
- Answers that do not match the customer's intent
- Claims that are not supported by retrieved evidence
- Conflicting information in retrieved examples
- Statements that imply a refund, payment reversal, cancellation, or account update has already happened without verification
- Missing information that makes the response unreliable
For example, the system must not say "Your refund has been processed" just because it retrieved an article about refunds. Confirming that status would require access to a trusted transaction system.

The auditor should return a verdict, identified issues, evidence references, and a recommended action.
Planned libraries
- re for deterministic claim and wording checks
- pydantic or dataclasses for audit results
- logging for audit events
- An optional organizer-provided AI API for supported semantic verification tasks
7. Agent 5 — Coordinator
Purpose: Combine the other agents' findings and choose the final outcome.

Status: planned for Level 2. The Coordinator applies a documented decision policy rather than simply trusting the highest confidence score.

It chooses one of three decisions:
Decision	When it is used
ANSWER	The intent is sufficiently clear, evidence is relevant, and the response passes auditing
CLARIFY	The customer can provide missing details or resolve an ambiguity
ESCALATE	The request is unsupported, evidence is unreliable, or human/verified system access is required


The Coordinator also produces a decision trace showing what each agent contributed. This trace is useful for debugging, evaluation, and the future frontend.
Planned libraries
- pydantic or dataclasses for agent results
- json for serializing traces
- logging for runtime events
- Standard Python decision logic for the final policy
8. Main differentiator — Evidence-Driven Agent Debate
A normal retrieval chatbot can select the closest matching answer and return it. Our proposed Evidence-Driven Agent Debate Engine adds a cross-check before the final response.

Example:
1. Agent 1 predicts a refund-related intent.
2. Agent 2 retrieves mostly subscription-cancellation examples.
3. Agent 3 flags the mismatch between the prediction and the evidence.
4. Agent 4 checks whether the proposed answer is actually supported.
5. Agent 5 decides whether to answer, clarify, or escalate.
The word debate refers to a structured comparison of agent outputs. It does not mean five independent language models chatting back and forth.

The planned UI will show these checks as a decision replay, allowing judges to see why the system made its final choice.

To show whether this feature helps, the project should compare the original two-agent baseline, the five-agent workflow without disagreement checks, and the five-agent workflow with the checks enabled. Any reported improvement must come from actual tests.
9. Dataset, training, and overfitting checks
The existing baseline uses a labeled customer-support dataset with columns such as:
instruction | intent | category | response
The instruction column contains customer messages, intent and category are labels, and response contains the corresponding support response.
Training versus inference
- Agent 1: Trains a supervised Logistic Regression classifier.
- Agent 2: Fits TF-IDF vectorizers and builds a searchable support knowledge base.
- Agents 3–5: Use decision rules and structured checks; they do not need their own supervised training process in the initial prototype.
Preventing misleading evaluation results
Customer-support datasets can contain repeated templates and near-duplicate messages. This can make validation accuracy look better than performance on unfamiliar customer queries.

We plan to check:
- Exact duplicates and near-duplicates across data splits
- Repeated templates and possible label leakage
- Accuracy, Macro F1, and per-class performance
- Typos, paraphrases, ambiguous queries, and out-of-domain inputs
- Confidence calibration and uncertainty thresholds
- Whether the new agent checks reduce unsupported responses
We will not deliberately lower accuracy to make the results look realistic. We will report measured performance and explain the limitations of each test set.

The organizers' evaluation dataset should remain separate from model training and threshold tuning unless the organizers explicitly allow otherwise.
10. Evaluation output
The evaluation runner is intended to process each organizer-provided query through the same workflow used by the live demo.

The required prediction file is:

predictions.csv
query_id,predicted_intent,predicted_category,decision
Requirements:
- Preserve the original query_id and row order.
- Produce one prediction per input query.
- Use the exact intent and category labels expected by the organizers.
- Restrict decision to ANSWER, CLARIFY, or ESCALATE.
- Keep detailed agent traces in a separate JSONL file.
- Do not retrain the model while processing the evaluation file.
A proposed command, once the evaluation runner is implemented, is:
python -m evaluation.run_evaluation --input path/to/evaluation.csv --output outputs/predictions.csv
Note: This is a target interface, not a claim that the command already works in the current repository.
11. Backend and frontend integration
The frontend will be connected later. The planned backend uses FastAPI so the UI can send a customer message and receive a structured JSON response.

Proposed endpoints:
Endpoint	Purpose
GET /health	Check whether the server is running
GET /ready	Check whether models and required components are loaded
POST /api/v1/analyze	Run intent analysis
POST /api/v1/query	Run the full five-agent workflow
POST /api/v1/batch	Process a batch of queries


The full query response is intended to include the predicted intent, category, confidence, final decision, response, retrieved evidence, agent findings, trace ID, and latency.

The frontend can use this data to show the query, the five agent stages, and the final decision without implementing machine-learning logic in the browser.
Planned backend libraries
Library	Use
fastapi	REST endpoints and automatic API documentation
uvicorn	Run the Python API server
pydantic	Validate request and response schemas
pytest	Automated tests
httpx or requests	Backend HTTP requests to the organizer API, if needed


12. Organizer API integration
The organizers will provide an API key shortly before evaluation. The exact endpoint and request format are not yet confirmed.

For that reason, the prototype should use a provider adapter with separate local, mock, and organizer-API implementations.
- Local mode: Uses the trained intent model, retrieval engine, and local decision rules.
- Mock mode: Tests the API integration without making real external requests. Mock outputs must be clearly identified.
- Organizer API mode: Calls the real service after its official documentation and credentials are available.
Credentials should be loaded from environment variables rather than written into source code:
HACKATHON_API_KEY=
HACKATHON_API_BASE_URL=
HACKATHON_API_MODEL=
AI_PROVIDER_MODE=local
The API key must stay on the backend. It should not be committed to Git, sent to the frontend, or included in traces.

If the external API is unavailable, the system should use a safe local fallback and avoid claiming that API-based verification took place.
13. Proposed repository structure
The final structure may be adapted to match the existing codebase:
project/
├── agents/
│   ├── agent1_intent.py
│   ├── agent2_response.py
│   ├── agent3_routing.py
│   ├── agent4_auditor.py
│   └── agent5_coordinator.py
├── core/
│   ├── orchestrator.py
│   ├── schemas.py
│   └── decision_policy.py
├── integrations/
│   └── organizer_api.py
├── evaluation/
│   ├── run_evaluation.py
│   └── metrics.py
├── models/
│   ├── intent_model.joblib
│   └── intent_category_mapping.json
├── outputs/
│   ├── predictions.csv
│   └── traces.jsonl
├── app.py
├── requirements.txt
├── .env.example
└── README.md
These are proposed module names. Check the repository before renaming or moving existing files.
14. Running the project
The exact commands depend on the final repository layout. After integration, the expected setup is:
# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
Once the FastAPI application is implemented:
uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs for interactive API documentation.

Run automated tests with:
pytest
These are intended commands for the completed modular prototype. They may need adjustment until the code and filenames are finalized.
15. How we will measure performance
We want the system to be useful and reliable, not just fast on a familiar dataset.
Area	Metrics or checks
Intent classification	Accuracy, Macro F1, per-class precision and recall
Generalization	Duplicate-overlap audit, paraphrase and typo tests
Retrieval	Evidence relevance and ranking quality
Decision quality	Correct answer/clarify/escalate decisions
Safety	Unsupported claims and missed escalations
Performance	Per-agent latency, end-to-end latency, memory use
API efficiency	External calls per query and timeout behavior


No performance figures are included here because they should be taken from actual benchmark runs. The project should keep a reproducible record of model versions, test sets, and measured results.
16. Current limitations and next steps
- The baseline intent model and response retriever exist, but the complete five-agent integration must be verified in code.
- Agent 1's probability scores are not guaranteed to be calibrated.
- Retrieval relevance does not prove that a customer's issue has been resolved.
- The system cannot verify real payment, refund, or account actions without an authorized transactional integration.
- The organizer API integration cannot be finalized until the endpoint documentation and key are provided.
- The frontend will be connected after the backend response schema is stable.
Development priorities
1. Verify the existing Agent 1 and Agent 2 baseline.
2. Audit dataset overlap and evaluate generalization.
3. Implement and connect Agents 3–5.
4. Add the disagreement checks and decision traces.
5. Produce the organizer-required evaluation CSV.
6. Expose the workflow through FastAPI.
7. Benchmark latency and decision quality.
8. Connect the organizer API and frontend.
17. Project summary
The goal is not to answer every customer message automatically. It is to handle clear requests efficiently, ask for missing information when necessary, and recognize cases that need human attention.

The main idea behind the project: a useful support system should know not only how to answer, but also when not to answer.


# Intelligent Customer Support & Automated Query Resolution

A multi-agent customer support system built using Natural Language Processing (NLP), machine learning, information retrieval, and rule-based decision-making.

The project aims to understand customer queries, find relevant support information, check whether a response is reliable, and decide whether to answer the customer, ask for clarification, or escalate the issue.

Instead of relying on a single chatbot to handle everything, the system divides the work among five specialized agents.

## 1. Project Overview

Customer support systems regularly deal with questions about refunds, orders, payments, subscriptions, accounts, and delivery issues.

A major problem with automated support is that a system may produce an answer even when the customer's request is unclear or the available information is insufficient.

Our approach separates the support process into five stages:

1. Understand the customer's intent.
2. Retrieve relevant support information.
3. Decide what information or action is needed.
4. Check the proposed response for errors and unsupported claims.
5. Make the final decision.

The final decision is one of:

- `ANSWER` — The system has enough information to provide a supported response.
- `CLARIFY` — More information is needed from the customer.
- `ESCALATE` — The issue requires further review or human support.

## 2. System Architecture

```text
                  Customer Query
                         |
                         v
              +---------------------+
              | Agent 1             |
              | Intent Analyst      |
              +---------------------+
                         |
             Intent, Category,
             Confidence, Uncertainty
                         |
                         v
              +---------------------+
              | Agent 2             |
              | Response Specialist |
              +---------------------+
                         |
             Candidate Response,
             Supporting Evidence
                         |
                         v
              +---------------------+
              | Agent 3             |
              | Routing & Planning  |
              +---------------------+
                         |
             Routing Recommendation,
             Missing Information
                         |
                         v
              +---------------------+
              | Agent 4             |
              | Response Auditor    |
              +---------------------+
                         |
             Audit Result,
             Safety Checks
                         |
                         v
              +---------------------+
              | Agent 5             |
              | Coordinator         |
              +---------------------+
                         |
                         v
              ANSWER / CLARIFY / ESCALATE
                         |
                         v
                Final Response
                + Decision Trace
```

All five agents are connected through a common orchestration layer.

The system is designed to support both batch evaluation and live customer queries through the same processing pipeline.

## 3. Dataset

The initial implementation uses the Bitext Customer Support dataset.

The dataset contains customer-support queries with their corresponding intents, categories, and reference responses.

The observed training setup contains:

- 27 intent classes
- 11 categories
- 18,810 training examples
- 4,031 validation examples

The main dataset columns are:

| Column | Description |
|---|---|
| `instruction` | Customer query |
| `intent` | Intent label |
| `category` | Category associated with the intent |
| `response` | Reference customer-support response |

The exact labels are preserved during training and prediction.

The dataset is used for two main purposes:

- Training the intent classification model.
- Building the response retrieval knowledge base.

The organizers' evaluation dataset is handled separately and is not used for training unless explicitly permitted.

## 4. Agent 1 — Intent Analyst

### Purpose

Agent 1 identifies what the customer is trying to do.

For example:

**Customer query:**

"My order hasn't arrived. Where is it?"

The agent predicts a delivery or tracking-related intent and maps it to the corresponding category.

It also calculates uncertainty information so that later agents can decide whether the prediction is reliable enough to use.

### Model

Agent 1 uses a supervised machine-learning pipeline based on:

- Word-level TF-IDF
- Character-level TF-IDF
- FeatureUnion
- Multiclass Logistic Regression

### Word-Level TF-IDF

Word-level TF-IDF represents customer queries using important words and phrases.

The implementation uses unigrams and bigrams.

Examples include:

- `refund`
- `order status`
- `cancel subscription`
- `payment failed`

This helps the classifier identify meaningful phrases associated with different support intents.

### Character-Level TF-IDF

Character-level TF-IDF uses character n-grams of lengths 3 to 5.

This is useful when customers make spelling mistakes or use slightly different word forms.

For example, character patterns can help the model recognize similarities between words such as `payment` and `paymnt`.

### FeatureUnion

`FeatureUnion` combines the word-level and character-level feature vectors.

This allows the classifier to use both meaningful phrases and spelling-related patterns.

### Logistic Regression

Logistic Regression learns weights for the TF-IDF features using labeled training examples.

For each customer query, it calculates a score for every intent class and converts those scores into class probabilities.

The intent with the highest predicted probability is selected.

### Confidence and Uncertainty

Agent 1 calculates two main signals:

**Confidence**

The highest predicted class probability.

```text
confidence = highest predicted probability
```

**Top-Two Margin**

The difference between the highest and second-highest probabilities.

```text
margin = top_probability - second_probability
```

A low margin means that the model is finding it difficult to distinguish between the top two intents.

These values are used as uncertainty indicators. They are not assumed to be perfectly calibrated probabilities of correctness.

### Training and Model Selection

The baseline model compares different Logistic Regression regularization strengths:

```python
C_values = [0.5, 1.0, 2.0, 4.0]
```

The model is selected using validation Macro F1.

Macro F1 is useful because it gives equal importance to each intent class, rather than allowing common classes to dominate the evaluation.

### Main Libraries

| Library | Usage |
|---|---|
| `pandas` | Dataset loading and preprocessing |
| `numpy` | Probability calculations and numerical operations |
| `scikit-learn` | TF-IDF, FeatureUnion, Pipeline, Logistic Regression, evaluation |
| `joblib` | Saving and loading the trained model |
| `json` | Intent-category mapping and metrics |

### Output

Agent 1 returns information such as:

```json
{
  "intent": "predicted_intent",
  "category": "predicted_category",
  "confidence": 0.81,
  "top_two_margin": 0.32,
  "uncertain": false
}
```

The values above are illustrative.

## 5. Agent 2 — Response Specialist

### Purpose

Agent 2 finds relevant customer-support information based on the query and the intent predicted by Agent 1.

Instead of generating an answer without supporting information, it retrieves existing responses from the knowledge base.

### Retrieval Method

The baseline uses:

- Word-level TF-IDF
- Character-level TF-IDF
- Cosine similarity
- Intent-aware reranking

### Cosine Similarity

Cosine similarity measures how closely two text vectors point in the same direction.

Agent 2 compares the customer's query against stored customer-support examples.

Higher similarity indicates a closer match between the query and a knowledge-base entry.

### Combined Retrieval Score

The baseline combines word and character similarity:

```text
retrieval_score =
    0.65 * word_similarity
  + 0.35 * character_similarity
```

Word similarity captures phrase-level meaning, while character similarity helps with spelling variations.

### Intent-Aware Reranking

Agent 2 also uses the intent probabilities produced by Agent 1.

This helps prioritize responses associated with likely intents.

The goal is to avoid selecting a response simply because it contains similar words.

### Evidence Retrieval

The agent returns:

- Best matching support response
- Retrieval score
- Top supporting examples
- Related intent and category labels
- Evidence needed by later agents

### Main Libraries

| Library | Usage |
|---|---|
| `scikit-learn` | TF-IDF vectorization and cosine similarity |
| `scipy` | Sparse matrix operations |
| `numpy` | Similarity scoring and ranking |
| `pandas` | Knowledge-base preparation |

### Important Note

The baseline Agent 2 is a retrieval-based system.

It does not require a generative language model to produce its initial candidate response.

An external model may be added later for supported tasks such as response refinement or semantic auditing.

## 6. Agent 3 — Routing and Planning Agent

### Purpose

Agent 3 determines what should happen after the customer intent has been predicted and supporting information has been retrieved.

Its job is not to classify the customer query again.

Instead, it evaluates whether the available information is sufficient to proceed.

### Inputs

Agent 3 uses:

- Predicted intent
- Intent confidence
- Top-two probability margin
- Retrieval relevance score
- Supporting evidence
- Missing customer information
- Disagreement between agents
- Query risk level

### Main Responsibilities

**Uncertainty Assessment**

Checks whether the intent prediction is sufficiently reliable.

**Evidence Sufficiency**

Checks whether Agent 2 found relevant information to support an answer.

**Missing Information Detection**

Identifies details needed before the issue can be handled.

For example:

```text
Customer: "Cancel it."

Required clarification:
"Do you want to cancel an order or a subscription?"
```

**Risk Assessment**

Recognizes requests that need more careful handling, such as refund confirmations, payment disputes, or account changes.

**Agent Disagreement Detection**

Checks whether Agent 1's predicted intent agrees with the evidence retrieved by Agent 2.

### Decision Logic

Agent 3 is designed as an explainable, rule-based planning component.

Example logic:

```python
if missing_information:
    recommendation = "CLARIFY"

elif high_risk and insufficient_evidence:
    recommendation = "ESCALATE"

elif high_uncertainty or agent_disagreement:
    recommendation = "CLARIFY"

elif sufficient_evidence:
    recommendation = "PROCEED_TO_AUDIT"

else:
    recommendation = "ESCALATE"
```

The actual implementation should handle rule priority carefully, especially for high-risk cases.

Agent 3 recommends a route. Agent 5 is responsible for making the final decision after the response has been audited.

### Main Libraries

| Library | Usage |
|---|---|
| `typing` | Type annotations |
| `dataclasses` or `pydantic` | Structured agent inputs and outputs |
| `re` | Pattern matching |
| `json` | Structured routing traces |
| `logging` | Debugging and monitoring |
| `numpy` | Optional numerical scoring |

Agent 3 does not require a separately trained supervised model.

## 7. Agent 4 — Response Auditor

### Purpose

Agent 4 checks the proposed response before it reaches the customer.

A response can be relevant to the query and still contain incorrect or unsupported information.

The auditor is responsible for detecting these problems.

### Checks Performed

The planned auditor checks:

- Whether the response matches the customer's intent.
- Whether the response is supported by retrieved evidence.
- Whether the response contradicts available information.
- Whether the response contains unsupported factual claims.
- Whether required information is missing.
- Whether the response should be approved, revised, or rejected.

### Example

**Customer:**

"Has my refund been processed?"

**Unsafe response:**

"Your refund has been processed successfully."

If the system has no access to verified transaction records, it cannot confirm that statement.

The auditor should flag the claim as unsupported.

A safer response would explain that the refund status cannot be verified using the currently available information and direct the customer to an appropriate next step.

### Implementation Approach

The initial implementation is planned around deterministic checks.

Optional semantic verification may be added using the organizers' API if the service supports it.

### Main Libraries

- `re`
- `pydantic` or `dataclasses`
- `json`
- Optional API client library such as `httpx`
- Optional `transformers` for local semantic verification

## 8. Agent 5 — Coordinator

### Purpose

Agent 5 combines the outputs from the previous four agents and makes the final decision.

It is responsible for selecting one of three actions:

```text
ANSWER
CLARIFY
ESCALATE
```

### Decision Rules

**ANSWER**

Used when the query is understood, relevant evidence is available, and the response passes auditing.

**CLARIFY**

Used when the system needs additional information from the customer.

**ESCALATE**

Used when the request cannot be handled reliably, requires unavailable verification, or needs human review.

### Additional Responsibilities

The Coordinator also produces:

- Final customer-facing response
- Final decision
- Explanation for the decision
- Agent execution trace
- Escalation reason, where applicable

### Main Libraries

- `pydantic` or `dataclasses`
- `typing`
- `json`
- `logging`

The Coordinator uses explicit decision policies rather than a separately trained classification model.

## 9. Evidence-Driven Agent Debate Engine

One of the main features planned for the full prototype is an Evidence-Driven Agent Debate Engine.

The idea is to make agents check one another's outputs instead of automatically accepting every prediction.

For example:

1. Agent 1 predicts a refund-related intent.
2. Agent 2 retrieves mostly subscription-cancellation examples.
3. The system detects a mismatch between the predicted intent and retrieved evidence.
4. Agent 3 considers whether clarification is needed.
5. Agent 4 checks whether the candidate answer is sufficiently supported.
6. Agent 5 makes the final decision.

This is a structured disagreement-resolution process, not an unrestricted conversation between multiple language models.

### Why It Matters

A standard retrieval system may return the closest matching answer even when the match is weak or misleading.

Our approach is designed to identify these situations and avoid forcing an answer.

### Planned Features

- Intent-evidence disagreement detection
- Evidence sufficiency checks
- Risk-aware routing
- Response auditing
- Explainable final decisions
- Agent decision replay in the frontend

The impact of this mechanism should be measured by comparing the system with and without disagreement detection.

## 10. Complete Query Workflow

A customer query passes through the following stages:

```text
1. Customer enters a query
              |
              v
2. Agent 1 predicts intent and category
              |
              v
3. Agent 1 calculates confidence and uncertainty
              |
              v
4. Agent 2 retrieves relevant support examples
              |
              v
5. Agent 3 checks evidence, uncertainty, and missing details
              |
              v
6. Agent 4 audits the proposed response
              |
              v
7. Agent 5 selects the final decision
              |
              v
8. System returns response and execution trace
```

### Example Query

```text
"My payment was deducted, but my order failed."
```

**Agent 1**

Predicts a payment-related intent and provides confidence information.

**Agent 2**

Retrieves relevant payment and failed-order support guidance.

**Agent 3**

Checks whether transaction details or other information are required.

**Agent 4**

Ensures the proposed response does not falsely claim that a refund or reversal has already occurred.

**Agent 5**

Selects the appropriate final action based on the available evidence and audit result.

## 11. Training and Inference

Training and inference are kept separate.

### Training Stage

Agent 1 is trained on labeled customer-support examples.

The training process includes:

1. Loading the training dataset.
2. Preparing intent labels and category mappings.
3. Fitting word and character TF-IDF vectorizers.
4. Training Logistic Regression.
5. Evaluating candidate model settings.
6. Selecting a model using validation Macro F1.
7. Saving the trained model and metadata.

Agent 2 fits its retrieval vectorizers and builds a searchable representation of the support knowledge base.

Agents 3, 4, and 5 are designed primarily around decision rules and do not require separate supervised training.

### Inference Stage

During inference:

- Saved models are loaded.
- A customer query is processed.
- The five-agent workflow runs.
- The final decision and response are returned.

The intent model should not be retrained every time the application starts or receives a query.

## 12. Overfitting and Model Evaluation

High validation accuracy does not automatically mean that a model will perform equally well on unfamiliar customer messages.

Customer-support datasets can contain repeated phrases, similar templates, and near-duplicate queries.

These patterns may make a validation split easier than real-world customer conversations.

Our evaluation plan includes:

- Checking exact duplicate queries.
- Checking near-duplicate queries across splits.
- Investigating potential data leakage.
- Comparing training and validation performance.
- Testing spelling mistakes and paraphrases.
- Evaluating ambiguous queries.
- Testing out-of-domain messages.
- Reporting class-wise performance.
- Measuring confidence and uncertainty behavior.

### Classification Metrics

The main metrics include:

- Accuracy
- Macro F1
- Weighted F1
- Precision
- Recall
- Confusion matrix

### System-Level Metrics

The full prototype is also intended to evaluate:

- Correct final decision rate
- Clarification quality
- Unnecessary escalation rate
- Unsupported response claims
- Retrieval relevance
- End-to-end response latency

The project does not target an artificially low or high accuracy value.

All reported results should come from actual evaluation runs.

The organizers' final evaluation data must remain separate from model training and hyperparameter tuning unless the competition rules explicitly allow otherwise.

## 13. Technology Stack

### Machine Learning and NLP

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| scikit-learn | Text vectorization, classification, retrieval, evaluation |
| pandas | Dataset processing |
| NumPy | Numerical calculations |
| SciPy | Sparse matrix operations |
| Joblib | Model serialization |

### Backend and Agent Coordination

| Technology | Purpose |
|---|---|
| FastAPI | REST API for frontend integration |
| Uvicorn | Running the API server |
| Pydantic | Request, response, and agent schemas |
| Python logging | Error and execution logging |
| JSON | Agent outputs and decision traces |

### Testing and Integration

| Technology | Purpose |
|---|---|
| pytest | Automated testing |
| httpx | Optional external API requests |
| python-dotenv | Optional local environment configuration |

Additional semantic models may be evaluated later, but they are not required for the baseline classification and retrieval pipeline.

## 14. Proposed Project Structure

The final repository is intended to follow a modular structure similar to:

```text
customer-support-system/
|
|-- agents/
|   |-- agent1_intent.py
|   |-- agent2_response.py
|   |-- agent3_routing.py
|   |-- agent4_auditor.py
|   |-- agent5_coordinator.py
|
|-- core/
|   |-- orchestrator.py
|   |-- schemas.py
|   |-- decision_policy.py
|
|-- integrations/
|   |-- organizer_api.py
|
|-- evaluation/
|   |-- run_evaluation.py
|   |-- metrics.py
|
|-- models/
|   |-- intent_model.joblib
|   |-- intent_category_mapping.json
|
|-- outputs/
|   |-- predictions.csv
|   |-- traces.jsonl
|
|-- tests/
|
|-- app.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- README.md
```

This structure is a target layout. Actual filenames should be updated to match the implemented repository.

## 15. Evaluation Dataset Integration

The organizers provide an evaluation dataset that the system must process.

The evaluation runner is designed to use the same five-agent workflow as the live application.

The expected prediction output is:

```csv
query_id,predicted_intent,predicted_category,decision
```

### Evaluation Requirements

- Preserve the original query IDs.
- Preserve the required output column names.
- Use the dataset's exact intent and category labels.
- Generate one prediction for each valid input query.
- Avoid retraining during evaluation.
- Store detailed agent traces separately.
- Validate the final CSV before submission.

If the organizers provide additional evaluation instructions, those requirements take priority.

## 16. Organizer API Integration

The organizers will provide an API key shortly before evaluation.

Because the credentials and API contract are not yet available, the system is being designed to operate locally and support later API integration.

The planned provider interface includes:

- Local processing
- Mock API responses for testing
- Organizer API integration when documentation becomes available

### Environment Configuration

Example environment variables:

```env
HACKATHON_API_KEY=
HACKATHON_API_BASE_URL=
HACKATHON_API_MODEL=
AI_PROVIDER_MODE=local
```

The API key must remain on the backend.

It should never be committed to GitHub or exposed in frontend code.

The external API may be used for additional semantic analysis or auditing if supported by the organizers' service.

The core classification and retrieval components do not require the API key.

## 17. Frontend Integration

The frontend will connect to the backend using REST APIs.

FastAPI is planned as the backend framework.

### Planned Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /ready` | Model readiness |
| `POST /api/v1/analyze` | Intent prediction |
| `POST /api/v1/query` | Complete five-agent workflow |
| `POST /api/v1/batch` | Batch query processing |
| `GET /api/v1/config` | Public system configuration |

### Information Displayed in the UI

The frontend is intended to show:

- Customer query
- Predicted intent and category
- Confidence score
- Retrieved evidence
- Routing recommendation
- Audit findings
- Final decision
- Final response
- Agent execution timeline

The execution timeline is particularly useful during the hackathon demonstration because it shows how the system arrived at its decision.

## 18. Testing

The prototype should be tested using both normal and difficult customer queries.

### Example Test Cases

| Query | Expected behavior |
|---|---|
| "How can I track my order?" | Retrieve relevant tracking guidance |
| "Cancel it." | Request clarification |
| "My payment failed but money was deducted." | Handle payment uncertainty carefully |
| "Has my refund been processed?" | Avoid unverified transaction claims |
| "I forgot my password." | Retrieve account recovery guidance |
| Unrelated or meaningless text | Clarify or escalate as appropriate |

Additional tests cover:

- Empty input
- Misspellings
- Multiple intents
- Conflicting retrieved evidence
- Missing customer information
- API failures
- CSV output validation
- Agent decision consistency

## 19. Current Development Status

The project is being developed in stages.

| Component | Status |
|---|---|
| Agent 1 — Intent Analyst | Existing baseline implementation |
| Agent 2 — Response Specialist | Existing baseline implementation |
| Agent 3 — Routing and Planning | Planned for five-agent prototype |
| Agent 4 — Response Auditor | Planned for five-agent prototype |
| Agent 5 — Coordinator | Planned for five-agent prototype |
| Evidence-Driven Agent Debate | Planned |
| FastAPI Backend | Planned |
| Evaluation Runner | Planned |
| Organizer API Integration | Awaiting API details |
| Frontend Integration | Planned |

This section should be updated as features are implemented and tested.

## 20. What Makes This Project Different?

The main goal is not to generate a response for every customer query.

The system is designed to recognize when it has enough evidence to answer and when it should ask for more information or involve a human.

The key design choices are:

1. **Hybrid text classification:** Word and character TF-IDF features improve coverage of different customer phrasings.

2. **Evidence-based retrieval:** Responses are selected using support examples rather than unsupported generation.

3. **Uncertainty-aware planning:** Confidence, retrieval quality, and missing information influence the next step.

4. **Agent disagreement detection:** The system checks whether predicted intent and retrieved evidence agree.

5. **Response auditing:** Unsupported statements are flagged before the final decision.

6. **Explainable coordination:** The system records how each agent contributed to the outcome.

7. **Reproducible evaluation:** Model performance is assessed using actual metrics and robustness testing.

The overall objective is to build a customer-support system that can make reliable decisions, explain those decisions, and recognize when it should not answer automatically.

## 21. Future Improvements

Potential improvements include:

- Semantic retrieval using sentence embeddings
- Better out-of-domain detection
- Calibrated confidence estimates
- Multi-turn clarification handling
- More detailed human escalation summaries
- Feedback-based improvement of routing rules
- Integration with verified order and payment systems
- API-assisted semantic response auditing
- Performance optimization for larger knowledge bases

These additions will be evaluated based on their actual impact on accuracy, reliability, and latency.

## 22. Conclusion

This project combines machine-learning classification, text retrieval, rule-based planning, response auditing, and multi-agent coordination.

Agent 1 identifies the customer's intent. Agent 2 retrieves relevant support information. Agent 3 determines the next step. Agent 4 checks the proposed response. Agent 5 makes the final decision.

The goal is to build a support system that does more than return the closest matching answer.

It should also recognize uncertainty, check its evidence, and know when clarification or human assistance is needed.

---

**Project:** Intelligent Customer Support & Automated Query Resolution

**Type:** NLP and Multi-Agent Customer Support Hackathon Prototype

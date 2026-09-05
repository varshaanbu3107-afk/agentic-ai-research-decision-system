\# System Architecture



\## Overview



The \*\*Agentic AI Research \& Decision System\*\* is a modular research pipeline that combines \*\*Retrieval-Augmented Generation (RAG)\*\*, evidence evaluation, verification, and decision-making.



The system accepts a research question, retrieves relevant evidence from a persistent FAISS vector store, evaluates and verifies the evidence, and produces a structured research decision.



\---



\## Pipeline



```text

User Research Question

&#x20;         |

&#x20;         v

Research Planner

&#x20;         |

&#x20;         v

Research Sub-Questions

&#x20;         |

&#x20;         v

FAISS Vector Store

&#x20;         |

&#x20;         v

Semantic Retrieval

&#x20;         |

&#x20;         v

Evidence Filtering

&#x20;         |

&#x20;         v

Evidence Quality Scoring

&#x20;         |

&#x20;         v

Source Quality Assessment

&#x20;         |

&#x20;         v

Research Analysis

&#x20;         |

&#x20;         v

Evidence Verification

&#x20;         |

&#x20;         v

Decision Agent

&#x20;         |

&#x20;         v

Final Research Report

```



\---



\## Main Components



\### 1. Research Planner



The planner converts the user's research question into multiple focused research sub-questions.



\*\*Responsibilities:\*\*



\* Understand the research question

\* Identify the research domain

\* Generate focused retrieval questions

\* Provide a fallback plan when the LLM is unavailable



\*\*File:\*\*



```text

app/agents/planner.py

```



\---



\### 2. RAG Pipeline



The RAG pipeline retrieves relevant information from the project's research documents.



\*\*Main components:\*\*



\* PDF loader

\* Document chunker

\* Embedding generation

\* FAISS vector store

\* Research retriever



\*\*Files:\*\*



```text

app/rag/loader.py

app/rag/chunker.py

app/rag/embeddings.py

app/rag/vector\_store.py

app/rag/research\_retriever.py

```



\---



\### 3. Evidence Evaluation



Retrieved evidence is evaluated using multiple signals:



\* Semantic similarity

\* Domain relevance

\* Evidence strength

\* Source quality

\* Source diversity



\*\*Files:\*\*



```text

app/research/evidence.py

app/research/evidence\_builder.py

app/research/scoring.py

app/research/source\_quality.py

```



\---



\### 4. Research Agent



The research component analyzes retrieved evidence and extracts findings relevant to the research question.



\*\*File:\*\*



```text

app/agents/researcher.py

```



A deterministic fallback mechanism is available when the Gemini API is unavailable.



\---



\### 5. Verification Agent



The verification stage evaluates the relationship between retrieved evidence and the research question.



Evidence can be classified as:



\* \*\*Supports\*\*

\* \*\*Partially Supports\*\*

\* \*\*Contradicts\*\*

\* \*\*Neutral\*\*



\*\*File:\*\*



```text

app/agents/verifier.py

```



The system calculates verification confidence and evidence relationships before passing verified evidence to the decision stage.



\---



\### 6. Decision Agent



The decision component evaluates verified evidence and produces:



\* Decision

\* Confidence

\* Status

\* Key findings

\* Recommendation

\* Risks

\* Alternatives

\* Decision reasons

\* Evidence metrics



\*\*File:\*\*



```text

app/agents/decision.py

```



The confidence, status, and evidence metrics are always produced by
deterministic local scoring logic based on the verified evidence. When
the Gemini API is available, it is layered on top solely to generate a
qualitative narrative summary explaining the decision; it is explicitly
instructed not to alter the confidence, status, or scores themselves.
When Gemini is unavailable, only the narrative summary is skipped — the
decision itself is unaffected either way.


The confidence logic considers supporting, partial, neutral, and contradicting evidence before assigning the final confidence level.



\---



\### 7. Orchestrator



The orchestrator coordinates the complete research pipeline.



\*\*File:\*\*



```text

app/core/orchestrator.py

```



\*\*Pipeline:\*\*



1\. Create research plan

2\. Retrieve evidence

3\. Analyze evidence

4\. Verify evidence

5\. Make decision

6\. Build final research result



\---



\### 8. Evaluation



The project contains a dedicated evaluation module for measuring retrieval and evidence performance.



\*\*Directory:\*\*



```text

evaluation/

```



\*\*Includes:\*\*



\* Golden research questions

\* Retrieval metrics

\* Evidence metrics

\* Evaluation results



\---



\## Technology Stack



| Technology               | Purpose                                                        |

| ------------------------ | -------------------------------------------------------------- |

| Python                   | Core programming language                                      |

| Google Gemini API        | LLM-based planning, research, and evidence verification (decision scoring itself is deterministic; Gemini adds narrative only) |

| Sentence Transformers    | Semantic embeddings                                            |

| FAISS                    | Vector similarity search                                       |

| PyPDF                    | PDF document loading                                           |

| LangChain Text Splitters | Document chunking                                              |

| Pytest                   | Automated testing                                              |

| JSON                     | Dataset and evaluation storage                                 |



\---



\## Reliability and Fallback Architecture



The system is designed to continue operating even when the Gemini API is unavailable.



Fallback mechanisms are implemented for:



\* Research planning

\* Evidence analysis

\* Verification

\* Decision making



This allows the retrieval and deterministic reasoning components to continue producing research results without depending entirely on an external LLM API.



\---



\## Testing



The project contains an automated test suite covering the major components of the system.



Run the complete test suite with:



```bash

python -m pytest -q

```



Current test status:



```text

56 passed

```



\---



\## Data Flow



```text

Research Documents

&#x20;       |

&#x20;       v

PDF Loading

&#x20;       |

&#x20;       v

Document Chunking

&#x20;       |

&#x20;       v

Sentence Transformer Embeddings

&#x20;       |

&#x20;       v

FAISS Vector Store

&#x20;       |

&#x20;       v

Research Retrieval

&#x20;       |

&#x20;       v

Evidence Construction

&#x20;       |

&#x20;       v

Evidence Scoring

&#x20;       |

&#x20;       v

Source Quality Assessment

&#x20;       |

&#x20;       v

Research Analysis

&#x20;       |

&#x20;       v

Verification

&#x20;       |

&#x20;       v

Decision

&#x20;       |

&#x20;       v

Final Research Report

```



\---



\## Project Architecture Summary



The system follows a modular architecture where each stage has a specific responsibility:



```text

&#x20;                   +----------------------+

&#x20;                   |   Research Question  |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   |   Research Planner   |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   |   RAG Retrieval      |

&#x20;                   |   + FAISS            |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Evidence Evaluation  |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Research Analysis    |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Evidence Verification|

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Decision Engine      |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Final Research Report|

&#x20;                   +----------------------+

```



This architecture enables the project to combine \*\*semantic retrieval, evidence-based reasoning, verification, and decision-making\*\* into a single research workflow.


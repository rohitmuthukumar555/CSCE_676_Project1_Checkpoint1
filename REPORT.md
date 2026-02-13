# Data Mining Course Project: Dataset Selection & Exploratory Data Analysis

## Three Candidate Datasets (KDD 2025)

| Dataset | Course Topics | Beyond-Course Techniques |
|---------|---------------|--------------------------|
| **Fairness-Aware Graph Learning: A Benchmark** *(selected)* | Graph mining, PageRank, community detection | Fairness-aware graph learning, bias mitigation |
| **EBES: Easy Benchmarking for Event Sequences** | Sequential pattern mining, streams, temporal mining, anomaly detection | Event sequence benchmarking, next-event prediction |
| **Flexible Generation of Preference Data for Recommendation Analysis** | Recommendation, collaborative filtering | Synthetic preference generation, causal/debiasing analysis |

**Source:** All candidate datasets are from the **KDD 2025 Datasets and Benchmarks** track:  
[https://kdd2025.kdd.org/datasets-and-benchmarks-track-papers-2/](https://kdd2025.kdd.org/datasets-and-benchmarks-track-papers-2/)

---

## (A) Identification of Candidate Datasets

### Candidate 1: Fairness-Aware Graph Learning: A Benchmark

| Attribute | Description |
|-----------|-------------|
| **Dataset name and source** | **Fairness-Aware Graph Learning: A Benchmark** — KDD 2025 Datasets and Benchmarks track. Authors: Yushun Dong, Song Wang, Zhenyu Lei, Zaiyi Zheng (University of Virginia); Jing Ma (Case Western Reserve); Chen Chen, Jundong Li (University of Virginia). Find paper via Google Scholar → ACM DL or conference proceedings; dataset/code link in paper (e.g., GitHub, Zenodo). |
| **Course topic alignment** | **Graph mining:** centrality measures, **PageRank**, **community detection** (e.g., Louvain, label propagation), graph structure analysis. |
| **Potential beyond-course techniques** | **Fairness-aware graph learning**; bias measurement and mitigation in graph neural networks; demographic parity / equalized odds on node classification. |
| **Dataset size and structure** | Benchmark typically includes multiple graphs with node/edge lists, node features, and **sensitive attributes** (e.g., gender, race) for fairness evaluation; structure varies by graph (number of nodes, edges, density). |
| **Data types** | Node IDs, edge lists (directed/undirected), node feature vectors, class labels, sensitive attribute(s). |
| **Target variable(s)** | Node classification labels; fairness metrics (e.g., demographic parity difference, equal opportunity difference). |
| **Licensing or usage constraints** | ACM license, For more details, check paper and benchmark repository for license (typically research use). |

---

### Candidate 2: EBES: Easy Benchmarking for Event Sequences

| Attribute | Description |
|-----------|-------------|
| **Dataset name and source** | **EBES: Easy Benchmarking for Event Sequences** — KDD 2025 Datasets and Benchmarks track. Authors: Dmitry Osin, Egor Shvetsov, Evgeny Burnaev (Skolkovo Institute of Science and Technology); Igor Udovichenko (Vega Institute Foundation); Viktor Moskvoretskii (Skolkovo, HSE). Find paper via Google Scholar; dataset/code link in paper or supplementary. |
| **Course topic alignment** | **Streams / sequential data**, **sequential pattern mining**, **temporal mining**; can support **anomaly detection** if benchmark includes anomalous sequences. |
| **Potential beyond-course techniques** | **Event sequence benchmarking** (standardized evaluation); **sequential pattern mining** (e.g., PrefixSpan, SPADE) on event logs; next-event prediction or sequence-to-sequence modeling. |
| **Dataset size and structure** | Event sequences: each sequence is an ordered list of events (e.g., (timestamp, event_type)); multiple sequences per dataset; variable-length sequences. |
| **Data types** | Sequence ID, event type/category, timestamp (or position), optional payload/attributes. |
| **Target variable(s)** | Often **unsupervised** (pattern mining, clustering of sequences); optional supervised tasks (e.g., next-event prediction, sequence classification) if provided by benchmark. |
| **Licensing or usage constraints** | ACM license, For more details, check paper and benchmark repository for license (typically research use). |

---

### Candidate 3: Flexible Generation of Preference Data for Recommendation Analysis

| Attribute | Description |
|-----------|-------------|
| **Dataset name and source** | **Flexible Generation of Preference Data for Recommendation Analysis** — KDD 2025 Datasets and Benchmarks track. Authors: Simone Mungari, Erica Coppolillo (University of Calabria, ICAR-CNR); Ettore Ritacco (University of Udine); Giuseppe Manco (ICAR-CNR). Find paper via Google Scholar; dataset/code for **generating** or **analyzing** preference data in paper/repo. |
| **Course topic alignment** | **Recommendation systems**, **collaborative filtering**, association/preference patterns; matrix completion or ranking. |
| **Potential beyond-course techniques** | **Synthetic preference data generation**; **causal or statistical analysis** of preferences (e.g., confounding, selection bias); debiasing and controlled data generation for RecSys. |
| **Dataset size and structure** | User–item preference data (e.g., user ID, item ID, rating or preference strength); possibly multiple benchmark datasets or a **generation framework** producing configurable preference matrices. |
| **Data types** | User ID, item ID, rating/preference score, optional context (e.g., timestamp, category). |
| **Target variable(s)** | Preference/rating (for prediction or ranking); optionally downstream metrics (e.g., NDCG, fairness). |
| **Licensing or usage constraints** | ACM license, For more details, check paper and benchmark repository for license (typically research use). |

---

## (B) Comparative Analysis of Datasets

| Dimension | **Fairness-Aware Graph Learning** | **EBES (Event Sequences)** | **Flexible Preference Data** |
|-----------|-----------------------------------|----------------------------|------------------------------|
| **Supported data mining tasks** | **Course:** Graph centrality, PageRank, community detection. **External:** Fairness-aware graph learning, bias mitigation in GNNs. | **Course:** Sequential pattern mining, streams, temporal mining, anomaly detection. **External:** Event sequence benchmarking, next-event prediction. | **Course:** Recommendation, collaborative filtering, preference/association patterns. **External:** Synthetic preference generation, causal/debiasing analysis. |
| **Data quality issues** | Depends on benchmark graphs: possible isolated nodes, missing sensitive attributes, imbalanced labels. | Event logs may have missing timestamps, duplicate events, or inconsistent event taxonomies. | Synthetic data may be "too clean"; real preference data can have selection bias, cold start, sparsity. |
| **Algorithmic feasibility** | In-memory graph libraries (NetworkX, PyTorch Geometric) sufficient for benchmark-sized graphs; fairness metrics add moderate compute. | Sequential pattern mining and event stats feasible in Python (pandas, prefixspan); very long or numerous sequences may need sampling or streaming. | Matrix factorization and RecSys baselines feasible; generation framework may require tuning parameters. |
| **Bias considerations** | **Central:** Sensitive attributes (e.g., demographic) can induce **structural bias** and unfair predictions; benchmark designed to measure and mitigate this. | **Temporal/sampling bias:** Certain event types or time windows may be over-represented; benchmark design may control for this. | **Selection bias** in real preferences; **confounding** in observational data; synthetic generation can introduce controlled bias for study. |
| **Ethical considerations** | Fairness analysis can inform equitable systems; misuse of sensitive attributes must be avoided; benchmark supports responsible ML. | Event data may come from sensitive domains (e.g., healthcare, finance); use for research only; avoid re-identification. | Preference data can reflect user behavior; synthetic data reduces privacy risk; use in line with RecSys ethics (filter bubbles, fairness). |

---

## (C) Dataset Selection

**Selected dataset: Fairness-Aware Graph Learning: A Benchmark**

**Reasons:**

1. **Strong course alignment:** Graph mining (centrality, PageRank, community detection) is covered in HW3–HW4; this benchmark provides standardized graphs to apply and extend those techniques.
2. **Clear beyond-course angle:** Fairness-aware graph learning and bias measurement/mitigation in GNNs are not covered in class and offer a meaningful extension into responsible ML.
3. **Interpretable EDA:** Degree distribution, connectivity (SCC/WCC), sensitive-attribute and label balance, and basic fairness metrics are straightforward to analyze and motivate both course and external techniques.
4. **Portfolio-friendly:** Graph learning and fairness are highly relevant for industry (recommendation, social networks, hiring); experience with this benchmark translates well to interviews and projects.
5. **Manageable scope:** Graphs can be loaded with NetworkX or PyTorch Geometric; Python tooling used in HW3–HW4 is sufficient for EDA and initial mining.

**Trade-offs:**

- **No sequential/temporal structure** — sequential pattern mining or stream mining is not applicable on this dataset alone.
- **Sensitive attributes** — require careful handling and ethical use; benchmark is designed for fairness analysis, not discrimination.
- **Data availability** — dataset/code must be obtained from the paper or authors (e.g., GitHub, Zenodo); EDA can be designed against the expected schema (nodes, edges, features, labels, sensitive attribute) and run once data is available.

---

## (D) Exploratory Data Analysis (Selected Dataset: Fairness-Aware Graph Learning)

*(Summary; full EDA is in the Jupyter notebook `project1_checkpoint1.ipynb`.)*

Planned EDA steps:

1. **Data basics:** Load graph data (edge list, node file, or format provided by benchmark); number of nodes, edges; directed vs undirected; missing values in node features, labels, or sensitive attribute(s).
2. **Graph structure:** Degree distribution (in/out if directed); density; number of connected components (weakly/strongly for directed); size of giant component.
3. **Node attributes and labels:** Distribution of class labels; distribution of sensitive attribute(s) (e.g., demographic groups); cross-tabulation of label vs sensitive group (initial view of potential disparity).
4. **Centrality (course alignment):** Optional: compute PageRank or degree centrality on one graph; list top-k nodes; note whether high-centrality nodes have balanced sensitive-attribute distribution.
5. **Data cleaning and bias:** Identify isolated nodes, duplicate edges, missing sensitive attributes; assess label imbalance and sensitive-attribute imbalance; document for fairness metrics later.
6. **Initial observations for techniques:** e.g., "Label and sensitive-attribute imbalance" → need for fairness-aware evaluation; "Sparse/dense regions" → community detection and fairness across communities.

---

## (E) Initial Insights and Direction

**Observation:** Graph benchmarks often exhibit label imbalance and uneven distribution of sensitive attributes across the network. Standard centrality or community detection can surface nodes or communities that are over- or under-represented with respect to a protected group, motivating fairness-aware analysis.

**Hypothesis:** Fairness-aware graph learning (e.g., demographic parity, equalized odds, or bias mitigation in GNNs) can improve equity of outcomes across groups compared to applying standard graph mining alone. The benchmark allows standardized comparison of fairness metrics across methods.

**Potential research questions:**

- How do PageRank rankings correlate with sensitive attribute distribution?
- Do community detection methods (e.g., Louvain) produce communities that are homogeneous with respect to the sensitive attribute, and how does that affect fairness metrics?
- How do fairness-aware GNN or post-processing methods change accuracy–fairness trade-offs compared to standard node classification on this benchmark?

---

## (F) GitHub Portfolio Building

- **Repository:** https://github.com/rohitmuthukumar555/CSCE_676_Project1_Part1

---

## References

**Benchmark (selected dataset)**  
Dong, Y., Wang, S., Lei, Z., Zheng, Z., Ma, J., Chen, C., & Li, J. (2025). Fairness-Aware Graph Learning: A Benchmark. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2025)*, Datasets and Benchmarks track. ACM. https://kdd2025.kdd.org/datasets-and-benchmarks-track-papers-2/

**Data source used for EDA**  
Dong, Y. (n.d.). *Graph-Mining-Fairness-Data* [Dataset and code]. GitHub. https://github.com/yushundong/Graph-Mining-Fairness-Data

---

*End of report.*

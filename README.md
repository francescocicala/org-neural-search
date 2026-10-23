# Organizational Neural Search (`org_neural_search`)

A JAX laboratory for simulating organizations as networks of learning representations that search NK landscapes, with a local web laboratory, a Python API, and a command-line runner. It accompanies the *Organization Science* Special Issue submission (*AI-Enabled Frontiers in Organizational Science: Reusable Research Infrastructure*):

> **Organizational Neural Networks: Distributed Representation Learning, Uncertainty Absorption, and Socialization on Rugged Landscapes**

---

## 1. Overview

Each simulated firm chooses a strategy made of `N` binary choices (2 to 16, default 12), so there are `2^N` possible strategies (from 4 at `N = 2` to 4,096 at `N = 12` and 65,536 at `N = 16`). An organization is a directed graph of **representations**: each node holds a mental model of how strategies pay off. The **dominant coalition (DC)** adopts the firm's strategy; managers and units attend to a subset of the choices (their domain) and pass their views along links. All configurations in a run face the same landscapes, noise draws, and shocks, so the differences between them are paired.

| File | Contents |
| :--- | :--- |
| [`org_graph.py`](org_graph.py) | `MarketConfig`, `OrgGraph`, `OrgNode`, `OrgEdge`, and the 12 templates (`GRAPH_TEMPLATES`) |
| [`engine.py`](engine.py) | JAX engine: `run_graph`, `compare_graphs`, `compare_multi_graphs`, the streaming runners used by the web laboratory, and paired t-tests |
| [`webapp.py`](webapp.py) | Local web laboratory served by the Python standard library: configurations dashboard, organization editor, live charts, and the Model & Parameters Guide |
| [`cli.py`](cli.py) | Command line: `ui` launches the web laboratory; `run` runs an exported experiment file without a browser |
| [`test_smoke.py`](test_smoke.py) | Unit and integration tests |
| `__init__.py`, `pyproject.toml`, `requirements.txt`, `LICENSE` | Package exports, metadata (version 0.1.0), dependencies (JAX and NumPy), MIT license |

---

## 2. The Model

**Landscape.** Payoffs follow an NK landscape: each choice's payoff contribution depends on the choice itself and on `K` other choices (`0 <= K < N`), and the firm payoff is the average contribution, standardized over all `2^N` strategies (Levinthal 1997).

**Before period 1: starting beliefs.** Each node starts either uninformed (random beliefs, the default) or with an initial accuracy `a` between 0 and 1: the correlation between its starting beliefs and the payoff it observes, as far as its own choices can explain it. That explainable part is the observed payoff averaged over the choices outside the node's domain; for the DC and any other node whose domain covers every choice, it is the observed payoff itself. An informed node's representation is fitted, over all `2^N` strategies, to

```
t(s) = 0.5 * (a * z_obs(s) + sqrt(1 - a^2) * z_mis(s))
```

where `z_obs` is the standardized payoff the node will observe (the firm payoff for the DC and for nodes that see firm results, otherwise the average contribution of the choices in its domain) and `z_mis` is the same quantity on a misleading NK landscape with the same interaction structure and freshly drawn contributions: the node knows which choices interact but holds partly wrong beliefs about how. The factor 0.5 keeps starting beliefs weak, so market feedback can overwrite them (Puranam and Swamy 2016).

- When interactions cross domains, a unit's own choices explain only part of its result. For a domain of 6 of the 12 choices at `N = 12`, the correlation between its own-domain payoff and the part its choices explain is about 0.55 with random interactions and `K = 3`, about 0.85 with modular interactions and `K = 3`, and 1 when the domain consists of whole interaction blocks. Even a fully accurate unit therefore cannot predict its results perfectly.
- The misleading landscape is drawn once for each firm, node, and member, not at every learning step: redrawing it would average the misleading part away, and `a` would no longer set the accuracy.
- A node with the same id and settings gets the same starting beliefs in every configuration of a run.
- The engine reports the realized accuracy, measured in the same way and averaged over firms, as `initial_accuracy_realized`. It differs slightly from `a`: for nodes whose domain covers every choice it is about 0.02 to 0.03 lower because starting beliefs are fitted numerically, and for every node it varies with the particular misleading draws, so with few firms it can also come out above `a`.

**Each period** then runs five steps:

1. **Turbulence.** Under random turbulence a shock occurs in each period with probability `p_shock`; under periodic turbulence shocks occur at periods `shock_period`, `2 * shock_period`, and so on. A shock redraws each choice's contribution table with probability 0.5 while the interaction structure stays fixed. All configurations are hit in the same periods with the same redraws (Posen and Levinthal 2012).
2. **Exchange.** Each node moves its representation toward the views of the nodes it learns from. Permeability `P` sets how far it moves; low commensurability `C` adds a mistranslation that is fixed for each pair of nodes, plus noise.
3. **Strategy change.** Every searcher tries `search_steps` candidate changes within its domain (one-bit changes, or distant jumps with probability `rho`; Levinthal 1997) and keeps each candidate only if its own representation predicts a gain over the current strategy. The decision rights set who searches: the coalition alone; units, whose proposals the coalition approves or rejects (Rivkin and Siggelkow 2003); or units in their own domains while the coalition searches the remaining choices (Siggelkow and Levinthal 2003). Candidates are judged on representations, never on true payoffs.
4. **Market feedback.** The market reveals a noisy payoff only for the adopted strategy (Denrell and March 2001). The DC observes the firm payoff; other nodes observe the firm payoff, the payoff contribution of their own domain, or nothing (Gaba and Joseph 2013; Tushman and Scanlan 1981). Unit experiments, track-record learning, and the coalition's aspiration update also happen here (Greve 2003).
5. **Metrics.** Performance and representation metrics are recorded on the current landscape.

**Metrics.** `v_true` (payoff of the adopted strategy, standardized over the `2^N` strategies), `v_pct_max` (position between the worst and best strategy), `v_percentile` (share of the firm's `2^N` strategies with a lower true payoff than the adopted strategy), `v_max_ref` (payoff of the landscape's best strategy, drawn as a dashed gray reference line on the payoff chart), `v_local_ref` (average payoff reached by changing one choice at a time, always to the best one-choice change, while knowing the true payoffs, starting from every strategy; drawn as a dotted gray reference line on the payoff chart), `dc_acc_global` (correlation of the DC's representation with true payoffs), `dc_complexity` (number of perceived local peaks in the DC's representation), `orthogonality` (mean `1 - |corr|` over all pairs of representations), `orthogonality_linked` (the same over linked pairs), `dc_r_local`, `false_stall`, `at_local_peak` (share of firms whose strategy is a local peak of the true landscape: no one-choice change would raise its payoff), `emp_div_local_slope`, and `mean_permeability`. Accuracy, complexity, and the two orthogonality measures evaluate all `2^N` strategies and are recomputed in period 1, every 25 periods, and in the last period (every period when the horizon is 50 periods or less).

**Reading payoffs in absolute terms.** Payoffs are in standard-deviation units relative to all `2^N` strategies of each landscape (4,096 at default `N = 12`), so 0 is an average strategy. Typical values for `N = 12, K = 3` with random interactions: the best strategy is about 3.25, local search with perfect knowledge reaches about 2.5, an average local peak is about 2.2, and the top 5 percent of strategies lie above 1.65. The dashed reference lines on the payoff chart and `v_percentile` put an organization's payoff on this scale. Three additional reading notes: complexity measures how rugged the landscape looks to the DC, not structural complexity in the sense of Csaszar and Ostler (2020); `1 - |corr|` treats opposed representations as similar; the DC is an epistemic leader, not a political coalition with interests of its own (Levinthal and Pham 2024).

---

## 3. Settings

### 3.1 Market environment (`MarketConfig`, shared by all configurations)

| Field | Values | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `n_bits` | 2 to 16 | `12` | Choices `N`: number of binary choices in a strategy (`2^N` strategies; preset chips `8`, `12`, `16` in the web laboratory) |
| `nk_k` | 0 to `n_bits - 1` | `3` | Interdependence `K`: each choice's contribution depends on `K` other choices |
| `nk_pattern` | `"random"`, `"modular"` | `"random"` | Which choices interact; modular uses consecutive blocks of `K + 1` |
| `sigma_obs` | 0 or more (the web laboratory offers 0 to 3) | `0.65` | Standard deviation of the noise in observed market results |
| `turbulence` | `"none"`, `"random"`, `"periodic"` | `"none"` | When shocks occur |
| `p_shock` | 0 to 1 (the web laboratory offers 0 to 0.10) | `0.0` | Random turbulence: shock probability per period |
| `shock_period` | 1 or more | `100` | Periodic turbulence: shocks at periods `shock_period`, `2 * shock_period`, ... |
| `t_steps` | 1 or more | `600` | Horizon in periods |
| `b_firms` | 1 to 2048 | `50` | Number of firms, each with its own landscape, simulated in parallel (preset chips `20`, `50`, `100` in the web laboratory) |
| `seed` | integer | `2026` | Random seed shared by all configurations |
| `device` | `"auto"`, `"cpu"`, `"gpu"`, `"tpu"` | `"auto"` | Hardware |

`landscape` (only `"nk"`) completes the dataclass. `MarketConfig.from_dict` accepts only these field names and raises an error that names any unknown key.

### 3.2 Organization settings (`OrgGraph`, one set per configuration)

| Field | Values | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `feedback_regime` | `"dc_firm_units_domain"`, `"everyone_firm"`, `"dc_only"` | `"dc_firm_units_domain"` | Who observes market results: the DC sees firm results and units their own domain; everyone sees firm results; or only the DC sees results |
| `decision_rights` | `"centralized"`, `"propose_approve"`, `"delegated"` | `"centralized"` | The coalition decides; units propose and the coalition approves; or units decide in their domains |
| `unit_initial_accuracy` | `null` or 0 to 1 | `null` | Initial accuracy of every node other than the DC, unless the node sets its own; `null` means uninformed |
| `experiment_rate` | 0 to 1 | `0.0` | Each period, each non-DC node that sees results tests, with this probability, one random one-choice change in its domain and learns from the result; the firm's strategy does not change |
| `aspiration_sensitivity` | 0 to 5 | `0.0` | `kappa`: the coalition's distant-search probability becomes `min(1, rho + kappa * max(0, A - y))`, where `A` is its aspiration level and `y` its last observed result |
| `lambda_track` | 0 to 5 | `0.0` | Track-record learning: each permeability is multiplied by `exp(lambda_track * (own error - source error))` and kept between 0.001 and 1, so sources more accurate than the node gain weight; nodes that see no market results keep their permeabilities |

An `OrgGraph` also has a `name`, a `description`, `nodes`, and `edges`. It needs exactly one DC (role `"hq"`), a link path between every node and the DC, at most three tiers, and at most 64 members outside the DC.

### 3.3 Nodes and links

| `OrgNode` field | Values | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `id`, `label` | strings | | Identifier and display name |
| `role` | `"hq"`, `"manager"`, `"unit"` | `"unit"` | `"hq"` marks the DC |
| `count` | 1 to 64 (the DC has 1) | `1` | Headcount: parallel representations with identical settings |
| `model_type`, `h_agent` | `"neural"`, `"linear"`, `"tabular"`; 0 to 128 | `"neural"`, `16` | Representation and its capacity (hidden units; 0 means linear) |
| `jurisdiction_bits` | choice indices 0 to `n_bits - 1` | all `n_bits` | Domain: the choices the node attends to |
| `lr`, `sigma_learn` | 0 to 2; 0 or more | `0.15`, `0.05` | Learning rate and learning noise |
| `rho`, `search_steps` | 0 to 1; 1 or more | `0.0`, `1` | Search reach (probability that a candidate change is a distant jump) and candidates per period |
| `upward_agg` | `"mean"`, `"max"`, `"min"` | `"mean"` | How the node combines its sources' views: permeability-weighted mean, most optimistic, or most skeptical |
| `feedback` | `null`, `"firm"`, `"domain"`, `"none"` | `null` | Override of the organization's feedback regime; the DC always sees firm results |
| `initial_accuracy` | `null`, `"uninformed"`, or 0 to 1 | `null` | `null` follows `unit_initial_accuracy` (for the DC: uninformed); `"uninformed"` forces random starting beliefs; a number sets the accuracy |

Each `OrgEdge` from `source` to `target` means that the target learns from the source. `weight` is the directional permeability `P` (0 to 1, default 0.25); `commensurability` is `C` (0 to 1, default 0.85), shared by both directions of a pair. Links may be reciprocal and may form cycles.

---

## 4. Templates

| Key | Design |
| :--- | :--- |
| `coordinated_neural_hierarchy` | Four specialists advise the DC; no downward links |
| `conformity_trap` | Strong downward links from the DC (`P = 0.55`) overwrite the specialists' independent views |
| `bundled_neural_hierarchy` | Two-way links with moderate upward and downward permeability |
| `m_form_3_tier` | DC, two division heads, and four operating units |
| `incommensurable_silos` | Specialized silos with low commensurability (`C = 0.30`) toward the DC |
| `adaptive_meritocracy` | Two informed, low-noise experts (initial accuracy 0.7) and two poorly informed, noisy generalists (0.2), with track-record learning (`lambda_track = 2.5`) |
| `matrix_dual_authority` | Squads report to a functional and a product manager, who both advise the DC |
| `ambidextrous_org` | Low-noise exploitation unit and fast-learning exploration unit with lower commensurability |
| `hybrid_human_ai` | Two human specialists with partial domains and a full-domain AI analyst |
| `boundary_spanners` | Front-line units see own-domain results; internal units see none and talk to their front-line partners |
| `enacted_solo_dc` | The DC alone, local search only |
| `active_solo_dc` | The DC alone, with distant reach |

All templates start uninformed and use the default organization settings unless noted. Templates are defined at `N = 12` and proportionally rescaled when `Choices N` is changed in the web laboratory.

---

## 5. Installation

Works on Linux, macOS (Intel and Apple Silicon), and Windows (or WSL2) with Python 3.10+:

```bash
git clone https://github.com/francescocicala/org-neural-search.git
cd org-neural-search
pip install -r requirements.txt
pip install -e .
```

For NVIDIA GPUs (Linux or Colab, CUDA 12): `pip install -U "jax[cuda12]"`, then `pip install -e .`. Run the tests with `python3 test_smoke.py -v`.

---

## 6. Web Laboratory

```bash
org-neural-search ui                           # opens http://127.0.0.1:8080/
org-neural-search ui --port 8090 --no-browser
python3 cli.py ui                              # from a clone, without installing
```

`org-neural-search-ui` and `python3 webapp.py` start the same server (options `--host`, `--port`, `--no-browser`). The page has five parts:

- **Market environment**: `N` (choices, 2 to 16, default 12), `K` (0 to `N - 1`), interactions, feedback noise, turbulence (none, random with a probability per period, or periodic every `T` periods), firms (1 to 2,048, default 50), horizon, and seed, shared by every configuration. The hardware and the chart update interval are chosen automatically.
- **Configurations**: a bar with up to 8 configurations, each with its own color. Click one to edit it, rename it with the pencil or by double-clicking its name, and use "+ Add configuration" to copy the active one. Run Simulation runs one configuration on its own, or two to eight side by side on the same firms.
- **Design of the active configuration**: template, organization settings, the organization editor, and the node inspector. Every "?" button opens a short explanation of the setting.
- **Results**: six live charts with one line per configuration (plus dashed gray reference lines on the payoff chart for the best strategy and local search with perfect knowledge), a paired-test table that compares every configuration with the first one (shown with two or more configurations), and a checkpoint table.
- **Model & Parameters Guide**: the model, every setting, the templates, the metrics, and references.

### 6.1 Building an organization

- Start from a template and edit it. "+ Unit" adds a unit that reports to the selected manager, or to the DC; "+ Manager" adds a manager that reports to the DC; double-clicking empty canvas adds a unit at that spot.
- Drag from the dot on a node's right edge to another node to add a link; the arrow points from the source to the node that learns from it. Alternatively, select the source and Shift+click the learner.
- Click a link to set its permeability `P` and commensurability `C`, reverse it, or delete it. Click a node to edit it in the inspector below the canvas, including its initial accuracy; its Links list accepts exact values from the keyboard.
- Drag nodes to arrange them. Auto-layout puts the DC on top and every other node by its link distance from the DC; Fit shows the whole graph, and Expand opens a large editor for complex graphs.
- Delete or Backspace removes the selected link or node (the DC stays); Ctrl+Z (Cmd+Z on a Mac) undoes up to 30 changes.

Each card shows the node's label, an `N`-cell strip of its domain, a badge for the market results it sees (F firm, D own domain, a dash for none), its headcount when above 1, and its initial accuracy when set. Thicker links have higher permeability; dashed links have commensurability below 0.6. After a run, the inspector shows each node's realized initial accuracy.

### 6.2 Ready-made experiments

The menu in the configurations bar replaces the configurations with a complete set and clears the results. Experiments set `K = 3` (clamped to `N - 1`) with random interactions and no turbulence unless stated, and keep the current Choices `N`, noise, horizon, firms, and seed.

| Experiment | Configurations |
| :--- | :--- |
| Who sees market results | The coordinated hierarchy under the three feedback regimes |
| Decision rights | The coordinated hierarchy under the three decision-rights modes, on a modular landscape with `K = 2` |
| Cognitive reach under turbulence | The coalition's `rho` at 0, 0.15, 0.30, and 0.50, with random turbulence (probability 0.02 per period) |
| Initial accuracy and conformity | The coordinated hierarchy and the conformity trap, each with uninformed units and with units at initial accuracy 0.7 |
| Boundary spanners versus everyone sees results | `boundary_spanners` as designed, and the same structure with every node seeing firm results |
| Track-record learning under periodic turbulence | The adaptive meritocracy with `lambda_track` at 0, 1, and 2.5, with a shock every 100 periods |

---

## 7. Command Line and Python API

### 7.1 Command line

```bash
org-neural-search run experiment.json                      # print a summary table
org-neural-search run experiment.json --out results.json   # also save the full results
org-neural-search run experiment.json --device cpu         # choose the hardware: cpu, gpu, or auto
```

`run` reads an experiment file exported from the web laboratory (section 8), runs every configuration on the same firms, and prints the terminal payoff of each configuration with paired tests against the first. With `--out` it writes the experiment again with the full output under `simulation_output`, the same key the web laboratory uses; Import JSON in the web laboratory opens that file with its results.

### 7.2 Python API

```python
from dataclasses import replace

from org_neural_search import GRAPH_TEMPLATES, MarketConfig, compare_multi_graphs, run_graph

market = MarketConfig(n_bits=12, nk_k=3, turbulence="periodic", shock_period=100, b_firms=50, t_steps=600, seed=2026)
base = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]

single = run_graph(base, market)
print(single["summary"]["v_true"]["t_final"])  # mean and standard error of the terminal payoff

# Do informed units help? Same firms, landscapes, and shocks, paired against the first configuration.
graphs = [base, replace(base, unit_initial_accuracy=0.7)]
cmp = compare_multi_graphs(graphs, market, labels=["Uninformed units", "Units at 0.7"])
for org in cmp["orgs"]:
  print(org["label"], org["initial_accuracy_realized"])
paired = cmp["orgs"][1]["paired_vs_a"]
print(paired["v_true_diff"], paired["v_true_se"], paired["v_true_p_value"])
```

`compare_multi_graphs` accepts 2 to 8 graphs; `compare_graphs(graph_a, graph_b, market)` is the two-graph shortcut. Paired differences are each configuration minus the first, firm by firm, with two-sided paired t-tests across firms. Graphs can be built from scratch with `OrgGraph`, `OrgNode`, and `OrgEdge`, or loaded with `OrgGraph.from_dict` from the `org_graph` entries of an experiment file.

---

## 8. Experiment Files (schema 0.1.0)

Export JSON in the web laboratory writes one file with the market, every configuration, and the latest results (abbreviated):

```
{
  "schema_version": "0.1.0",
  "market_config": {"n_bits": 12, "nk_k": 3, "nk_pattern": "random", "sigma_obs": 0.65, "turbulence": "none", "p_shock": 0.02,
                    "shock_period": 100, "t_steps": 600, "b_firms": 50, "seed": 2026},
  "configurations": [
    {
      "name": "Configuration 1",
      "org_graph": {"name": "Coordinated Neural Hierarchy", "nodes": [...], "edges": [...], "feedback_regime": "dc_firm_units_domain", ...},
      "layout": {"hq": {"x": -70, "y": 0}, "rep_1": {"x": -310, "y": 166}, ...}
    }
  ],
  "simulation_output": null
}
```

- `market_config` holds `MarketConfig` fields; `org_graph` is the output of `OrgGraph.to_dict()`; `layout` holds the canvas positions, which only the web laboratory uses.
- `simulation_output` holds the latest results (`null` before a run). The web laboratory and `org-neural-search run --out` both write this key, and the web laboratory opens both kinds of file.

---

## 9. References

- Csaszar and Ostler (2020). A contingency theory of representational complexity in organizations. *Organization Science*. https://doi.org/10.1287/orsc.2019.1346
- Denrell and March (2001). Adaptation as information restriction: The hot stove effect. *Organization Science*. https://doi.org/10.1287/orsc.12.5.523.10092
- Gaba and Joseph (2013). Corporate structure and performance feedback: Aspirations and adaptation in M-form firms. *Organization Science*. https://doi.org/10.1287/orsc.1120.0788
- Greve (2003). *Organizational Learning from Performance Feedback*. Cambridge University Press. https://doi.org/10.1017/CBO9780511615139
- Levinthal (1997). Adaptation on rugged landscapes. *Management Science*. https://doi.org/10.1287/mnsc.43.7.934
- Levinthal and Pham (2024). Bringing politics back in: The role of power and coalitions in organizational adaptation. *Organization Science*. https://doi.org/10.1287/orsc.2022.16995
- March (1991). Exploration and exploitation in organizational learning. *Organization Science*. https://doi.org/10.1287/orsc.2.1.71
- Posen and Levinthal (2012). Chasing a moving target: Exploitation and exploration in dynamic environments. *Management Science*. https://doi.org/10.1287/mnsc.1110.1420
- Puranam and Swamy (2016). How initial representations shape coupled learning processes. *Organization Science*. https://doi.org/10.1287/orsc.2015.1033
- Rivkin and Siggelkow (2003). Balancing search and stability: Interdependencies among elements of organizational design. *Management Science*. https://doi.org/10.1287/mnsc.49.3.290.12740
- Siggelkow and Levinthal (2003). Temporarily divide to conquer: Centralized, decentralized, and reintegrated organizational approaches to exploration and adaptation. *Organization Science*. https://doi.org/10.1287/orsc.14.6.650.24840
- Tushman and Scanlan (1981). Boundary spanning individuals: Their role in information transfer and their antecedents. *Academy of Management Journal*. https://doi.org/10.2307/255842

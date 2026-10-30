# Organizational Neural Search (`org_neural_search`)

A JAX laboratory for simulating organizations as networks of learning representations that search NK landscapes, with an interactive local web interface, a Python API, and a command-line runner. It accompanies the *Organization Science* Special Issue submission (*AI-Enabled Frontiers in Organizational Science: Reusable Research Infrastructure*):

> **Organizational Neural Search: Distributed Representation Learning and Socialization on Rugged Landscapes**

Repository: https://github.com/francescocicala/org-neural-search

---

## 1. Quick Start: Launch the Web Dashboard (No Coding Required)

You can launch and explore the interactive laboratory on any computer running **Python 3.10+** (macOS, Windows, or Linux) with three simple commands:

### Step 1: Open a terminal
* **macOS**: Open `Terminal` (press `Cmd + Space`, type `Terminal`, and hit Enter).
* **Windows**: Open `PowerShell` or `Command Prompt`.
* **Linux**: Open your preferred shell terminal.

### Step 2: Download and install dependencies
```bash
git clone https://github.com/francescocicala/org-neural-search.git
cd org-neural-search
pip install -r requirements.txt
```
*(Dependencies are minimal: only standard `jax` and `numpy` for CPU simulation).*

### Step 3: Start the visual interface
```bash
python3 webapp.py
```

### What happens next?
* The local server starts immediately (powered entirely by Python's built-in standard library—no internet connection or cloud accounts required).
* Your default web browser will automatically open to:
  ```
  http://127.0.0.1:8080/
  ```
  *(If the browser does not open automatically, simply click or paste that link into Chrome, Safari, Edge, or Firefox).*
* **How to use the dashboard**:
  1. **Pick an organization**: Choose from 12 pre-configured organizational designs (such as *Coordinated Neural Hierarchy*, *Matrix Dual Authority*, or *Hybrid Human-AI*) from the template dropdown, or choose a ready-made experiment from the top menu.
  2. **Customize your design**: Click and drag nodes on the visual canvas. Drag link arrows to set who learns from whom, adjust sliders for task complexity ($K$) and market turbulence, or inspect individual units.
  3. **Run the simulation**: Click the green **Run Simulation** button. Live adaptation curves, search-state partitions, and statistical comparisons will stream and update in real time.
* **To stop the dashboard**: Return to your terminal window and press `Ctrl + C`.

---

## 2. Overview of the Repository

| File | Contents |
| :--- | :--- |
| [`webapp.py`](webapp.py) | Local visual web laboratory: interactive canvas, node inspector, live streaming charts, and Model & Parameters Guide |
| [`org_graph.py`](org_graph.py) | Organizational graph specifications (`MarketConfig`, `OrgGraph`, `OrgNode`, `OrgEdge`) and 12 built-in organizational templates |
| [`engine.py`](engine.py) | JAX simulation engine: vector-parallel Monte Carlo simulation, representation learning, communication protocols, and paired statistical tests |
| [`cli.py`](cli.py) | Command-line runner: headless batch execution of exported experiment files and CLI shortcuts |
| [`test_smoke.py`](test_smoke.py) | Unit and integration smoke tests |
| `pyproject.toml`, `requirements.txt`, `LICENSE` | Package metadata (v0.3.0), dependencies, and open-source MIT license |

Each simulated firm searches for good strategies across `N` binary policy choices ($N \in \{2, \dots, 16\}$, default 12), spanning $2^N$ possible configurations (4,096 at $N = 12$). An organization is modeled as a network of **learning representations**: each actor holds an internal model of how policies pay off. The **dominant coalition (DC)** adopts the firm's strategic moves; managers and frontline units attend to specialized subsets of choices (their domain) and communicate along directed links. All organizational configurations evaluated in a run confront the exact same paired landscapes, observation noise draws, and environmental shocks.

---

## 3. The Model

**Landscape.** Payoffs follow an NK landscape: each choice's contribution depends on the choice itself and on $K$ other choices ($0 \le K < N$). The firm payoff is the average contribution across choices, standardized to mean zero and unit variance over all $2^N$ strategies.

**Before period 1: starting beliefs.** Every node starts without prior knowledge: neural and linear representations start from small random weights, and tabular representations from a table of zeros.

**Each period runs five sequential steps:**

1. **Turbulence.** Under random turbulence, a shock arrives in each period with probability `p_shock`; under periodic turbulence, shocks occur every `shock_period` periods. A shock redraws choice contributions with probability 0.5 while preserving interaction wiring. All paired configurations face the exact same shock timing and redraws.
2. **Exchange (Socialization & Consultation).** Connected actors exchange predictions along directed communication links. Permeability $P \in [0, 1]$ controls how much the receiver updates toward the sender's beliefs; commensurability $C \in [0, 1]$ controls translation fidelity, injecting systematic distortion and noise when communication links are incommensurable.
3. **Strategy search and decision.** Searching actors sample candidate strategic changes within their domain (1-bit local moves, or distant jumps with probability $ho$) and evaluate them through their internal representation. Under centralized governance, the dominant coalition evaluates and enacts moves; under propose-and-approve, subordinates propose local improvements and headquarters screens them; under delegated rights, autonomous units enact their strongest perceived move directly.
4. **Market feedback.** The market reveals noisy performance feedback for the enacted strategy. The dominant coalition observes overall firm performance; subordinates observe firm payoff, their own domain payoff, or learn purely through peer exchange. Subordinate units also conduct offline prototype testing at rate $e$, isolating directional choice slopes from baseline level error.
5. **Metrics.** Terminal performance, representation accuracy, and organizational diversity are recorded and tracked across time.

**Metrics:**
* `v_true`: Payoff of the enacted strategy, standardized over all $2^N$ strategies.
* `v_pct_max`: Normalized performance between the worst and best possible strategy on the landscape.
* `v_percentile`: Percentage of strategies on the hypercube strictly inferior to the firm's current strategy.
* `v_max_ref`: Global maximum payoff of the landscape (shown as a dashed gray reference line).
* `v_local_ref`: Payoff reachable by greedy 1-bit local search under perfect knowledge (shown as a dotted gray reference line).
* `dc_acc_global`: Global correlation between the dominant coalition's representation and true hypercube payoffs.
* `dc_complexity`: Number of perceived local peaks in the dominant coalition's representation.
* `orthogonality`: Representational heterogeneity (mean $1 - |	ext{corr}|$ across all pairs of actors).
* `orthogonality_linked`: Representational heterogeneity over actively linked pairs.
* `at_local_peak`: Percentage of firms that have reached a true local peak.
* `false_stall`: Percentage of firms stuck on an inferior strategy because leadership mistakenly believes no 1-bit improvement exists.

**Reading payoffs on a standardized scale:**
Payoffs are expressed in standard deviation units relative to all $2^N$ landscape configurations (4,096 at $N = 12$), where 0 is an average strategy. For typical landscapes at $N = 12, K = 3$:
* Global peak $pprox 3.25$
* Local search with perfect knowledge $pprox 2.50$
* Average local peak $pprox 2.20$
* Top 5% of strategies lie above $1.65$.

---

## 4. Settings & Parameters

### 4.1 Market environment (`MarketConfig`, shared by all configurations)

| Field | Values | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `n_bits` | 2 to 16 | `12` | Choices $N$: number of binary choices in a strategy ($2^N$ strategies; preset chips `8`, `12`, `16`) |
| `nk_k` | 0 to `n_bits - 1` | `3` | Interdependence $K$: each choice's contribution depends on $K$ other choices |
| `nk_pattern` | `"random"`, `"modular"` | `"random"` | Which choices interact; modular uses consecutive blocks of $K + 1$ |
| `sigma_obs` | 0 or more | `0.65` | Standard deviation of noise in market performance signals |
| `turbulence` | `"none"`, `"random"`, `"periodic"` | `"none"` | Environmental turbulence regime |
| `p_shock` | 0 to 1 | `0.0` | Random turbulence: shock probability per period |
| `shock_period` | 1 to 20,000 | `100` | Periodic turbulence: shock arrival interval |
| `t_steps` | 10 to 20,000 | `600` | Horizon in periods |
| `eval_every` | 1 to 20,000 | `25` | Interval for full-hypercube representation metrics |
| `b_firms` | 1 to 2,048 | `50` | Monte Carlo landscape replications (firms) evaluated in parallel |
| `seed` | integer | `2026` | Master random seed shared across all paired configurations |

### 4.2 Organization settings (`OrgGraph`, configured per organization)

| Field | Values | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `feedback_regime` | `"dc_firm_units_domain"`, `"everyone_firm"`, `"dc_only"` | `"dc_firm_units_domain"` | Who sees market feedback: DC sees firm and units see their domain; everyone sees firm; or DC only |
| `decision_rights` | `"centralized"`, `"propose_approve"`, `"delegated"` | `"centralized"` | Governance mode: centralized executive control; bottom-up proposals with executive screening; or delegated autonomy |
| `experiment_rate` | 0 to 1 | `0.5` | Probability per period that non-DC units conduct offline 1-bit prototype testing |
| `aspiration_sensitivity` | 0 to 5 | `0.0` | Shortfall sensitivity ($\kappa$): distant search triggers when performance drops below aspirations |
| `lambda_track` | 0 to 5 | `0.0` | Track-record learning ($\lambda$): reweights communication links based on historical prediction accuracy |

### 4.3 Nodes and links

| `OrgNode` field | Values | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `id`, `label` | strings | | Node identifier and display label |
| `role` | `"hq"`, `"manager"`, `"unit"` | `"unit"` | Organizational role (`"hq"` marks the dominant coalition) |
| `count` | 1 to 64 | `1` | Headcount: parallel actors with identical configuration |
| `model_type`, `h_agent` | `"neural"`, `"linear"`, `"tabular"`; 0 to 128 | `"neural"`, `16` | Representation model type and capacity (hidden neurons; 0 is linear) |
| `jurisdiction_bits` | 0 to `n_bits - 1` | all choices | Choice domain ($D_r$): subset of choices attended to |
| `lr`, `sigma_learn` | 0 to 2; 0 or more | `0.15`, `0.05` | Learning rate and internal learning noise |
| `rho`, `search_steps` | 0 to 1; 1 or more | `0.0`, `1` | Exploration reach ($ho$, probability of distant jumps) and search candidate iterations |
| `upward_agg` | `"mean"`, `"max"`, `"min"` | `"mean"` | Method for aggregating incoming peer advice: weighted mean, optimistic, or conservative |
| `feedback` | `null`, `"firm"`, `"domain"`, `"none"` | `null` | Per-node override of the market feedback regime |

Each directed `OrgEdge` from `source` to `target` means the `target` learns from the `source`. `weight` is directional permeability $P \in [0, 1]$ (default 0.25); `commensurability` is shared language $C \in [0, 1]$ (default 0.85).

---

## 5. Built-in Organizational Templates

The laboratory includes 12 canonical organizational archetypes ready to simulate:

| Template Key | Architecture Description |
| :--- | :--- |
| `coordinated_neural_hierarchy` | Four specialized units advise the dominant coalition; upward communication only |
| `conformity_trap` | Strong downward socialization ($P = 0.55$) overwrites specialist independence |
| `bundled_neural_hierarchy` | Reciprocal two-way communication links with moderate upward and downward permeability |
| `m_form_3_tier` | Three-tier multidivisional structure: headquarters, two division heads, and four operating units |
| `incommensurable_silos` | Specialized units separated by semantic barriers ($C = 0.30$) toward headquarters |
| `adaptive_meritocracy` | Deep specialists ($H = 32$) and broad generalists ($H = 4$) with track-record link reweighting ($\lambda = 2.5$) |
| `matrix_dual_authority` | Cross-functional squads report simultaneously to functional and product managers |
| `ambidextrous_org` | Structured separation between low-noise exploitation and fast-learning exploration units |
| `hybrid_human_ai` | Two specialized human units paired with an autonomous, high-capacity AI analyst ($H = 64$) |
| `boundary_spanners` | Frontline units observe external domain signals; internal units learn via boundary-spanning liaisons |
| `enacted_solo_dc` | Baseline solo executive with local incremental search ($ho = 0$) |
| `active_solo_dc` | Solo executive with distant search reach ($ho = 0.20$) |

---

## 6. Ready-Made Experiments

The dropdown menu in the web interface loads complete experimental designs with one click:

| Experiment | Comparison Setup |
| :--- | :--- |
| **Who sees market results** | Compares `coordinated_neural_hierarchy` under own-domain, firm-wide, and DC-only feedback regimes |
| **Decision rights** | Compares centralized authority, propose-and-approve, and delegated autonomy on a modular landscape ($K = 2$) |
| **Cognitive reach under turbulence** | Sweeps executive exploration reach ($ho \in \{0, 0.15, 0.30, 0.50\}$) under random environmental shocks |
| **Boundary spanners versus firm-wide feedback** | Compares specialized boundary spanning against an organization where all units observe firm-wide signals |
| **Track-record learning under periodic turbulence** | Tests dynamic meritocratic link adaptation ($\lambda \in \{0, 1, 2.5\}$) under periodic environmental shocks |

---

## 7. Command Line and Python API

### 7.1 Running from the Command Line

```bash
# Launch the web interface
python3 webapp.py

# Run an exported experiment file headlessly
python3 cli.py run experiment.json

# Run and export full simulation trajectories
python3 cli.py run experiment.json --out results.json
```

### 7.2 Running from Python

```python
from dataclasses import replace
from org_neural_search import GRAPH_TEMPLATES, MarketConfig, compare_multi_graphs, run_graph

# Configure task environment
market = MarketConfig(n_bits=12, nk_k=3, turbulence="periodic", shock_period=100, b_firms=50, t_steps=600, seed=2026)

# Load a canonical template
base = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]

# Run a single organization
result = run_graph(base, market)
print("Terminal payoff:", result["summary"]["v_true"]["t_final"]["mean"])

# Compare centralized vs. propose-and-approve side by side
propose = replace(base, decision_rights="propose_approve")
cmp = compare_multi_graphs([base, propose], market, labels=["Centralized", "Propose and Approve"])

paired = cmp["orgs"][1]["paired_vs_a"]
print("Paired difference:", paired["v_true_diff"], "p-value:", paired["v_true_p_value"])
```

---

## 8. Experiment Files (schema 0.1.0)

Clicking **Export JSON** in the web laboratory downloads a complete, self-contained experiment file containing the market configuration, organizational graphs, canvas layout, and latest simulation outputs:

```json
{
  "schema_version": "0.1.0",
  "market_config": {
    "n_bits": 12,
    "nk_k": 3,
    "nk_pattern": "random",
    "sigma_obs": 0.65,
    "turbulence": "none",
    "p_shock": 0.0,
    "shock_period": 100,
    "t_steps": 600,
    "eval_every": 25,
    "b_firms": 50,
    "seed": 2026
  },
  "configurations": [
    {
      "name": "Configuration 1",
      "org_graph": {
        "name": "Coordinated Neural Hierarchy",
        "nodes": [...],
        "edges": [...],
        "feedback_regime": "dc_firm_units_domain",
        "decision_rights": "centralized"
      },
      "layout": { ... }
    }
  ],
  "simulation_output": null
}
```

Any exported experiment JSON file can be re-imported into the web laboratory or executed headlessly via `python3 cli.py run experiment.json`.

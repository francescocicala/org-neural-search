"""Organization graphs, market configuration, and templates for the NK graph engine."""

from dataclasses import asdict, dataclass, field, fields, replace
import json
import math
from typing import Any, Optional, Sequence, Union

VALID_NODE_ROLES = ("hq", "manager", "unit")
VALID_NODE_MODEL_TYPES = ("neural", "linear", "tabular")
VALID_NODE_CAPACITIES = (0, 4, 8, 16, 32, 64)
VALID_UPWARD_AGGS = ("mean", "max", "min")
VALID_NK_PATTERNS = ("random", "modular")
VALID_TURBULENCE = ("none", "random", "periodic")
VALID_FEEDBACK_REGIMES = ("dc_firm_units_domain", "everyone_firm", "dc_only")
VALID_NODE_FEEDBACK = ("firm", "domain", "none")
VALID_DECISION_RIGHTS = ("centralized", "propose_approve", "delegated")

# Feedback that a non-DC node receives under each organization-level regime
FEEDBACK_REGIME_DEFAULTS: dict[str, str] = {
    "dc_firm_units_domain": "domain",
    "everyone_firm": "firm",
    "dc_only": "none",
}

MAX_B_FIRMS = 2048
MAX_T_STEPS = 20000
MIN_N_BITS = 2
MAX_N_BITS = 16
# Default per-period probability that a unit tests one change off-line (about one test every other period)
DEFAULT_EXPERIMENT_RATE = 0.5
MAX_SEED = 2**32 - 1
MAX_SUBORDINATES = 64
MAX_TIERS = 3
# Node initial accuracy that forces the random initialization, overriding the organization default
UNINFORMED = "uninformed"


def rescale_jurisdiction_bits(bits: Sequence[int], old_n_bits: int, new_n_bits: int) -> tuple[int, ...]:
  """Proportionally rescale a node's decision domain from old_n_bits to new_n_bits."""
  b = sorted(set(int(x) for x in bits))
  if old_n_bits == new_n_bits and all(0 <= x < new_n_bits for x in b) and b:
    return tuple(b)
  if not b or b == list(range(old_n_bits)):
    return tuple(range(new_n_bits))
  if b == list(range(b[0], b[-1] + 1)):
    lo = max(0, min(new_n_bits - 1, round(b[0] * new_n_bits / old_n_bits)))
    hi = max(lo + 1, min(new_n_bits, round((b[-1] + 1) * new_n_bits / old_n_bits)))
    return tuple(range(lo, hi))
  denom = max(1, old_n_bits - 1)
  return tuple(sorted(set(max(0, min(new_n_bits - 1, round(x * (new_n_bits - 1) / denom))) for x in b)))


def _strict_fields(cls: Any, data: dict[str, Any], context: str) -> dict[str, Any]:
  """Return the known fields of a dataclass from a dict; unknown keys raise, null means default."""
  if not isinstance(data, dict):
    raise ValueError(f"{context} must be a JSON object, got {type(data).__name__}")
  valid = [f.name for f in fields(cls) if f.init]
  unknown = [k for k in data if k not in valid]
  if unknown:
    raise ValueError(
        f"Unknown {context} field {unknown[0]!r}; valid fields are: {', '.join(valid)}"
    )
  return {k: v for k, v in data.items() if v is not None}


def _optional_accuracy(value: Any, context: str) -> Optional[float]:
  """Validate an initial accuracy value: None (uninformed) or a float in [0, 1]."""
  if value is None:
    return None
  message = f"{context} must be null or a number in [0, 1], got {value!r}"
  if isinstance(value, bool):
    raise ValueError(message)
  try:
    val = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(message) from exc
  if not 0.0 <= val <= 1.0:
    raise ValueError(message)
  return val


def _node_accuracy(value: Any, context: str) -> Optional[Union[float, str]]:
  """Validate a node's initial accuracy: None (organization default), "uninformed", or a float in [0, 1]."""
  if isinstance(value, str) and value.strip().lower() == UNINFORMED:
    return UNINFORMED
  try:
    return _optional_accuracy(value, context)
  except ValueError as exc:
    raise ValueError(f"{context} must be null, {UNINFORMED!r}, or a number in [0, 1], got {value!r}") from exc


def _as_int(value: Any, name: str) -> int:
  """Coerce integral numbers (including 3.0 or "3") to int; reject fractional values."""
  if isinstance(value, bool):
    raise ValueError(f"{name} must be an integer, got {value!r}")
  try:
    as_float = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be an integer, got {value!r}") from exc
  if not as_float.is_integer():
    raise ValueError(f"{name} must be an integer, got {value!r}")
  return int(as_float)


def _as_float(value: Any, name: str) -> float:
  """Coerce a finite number (including numeric strings such as "0.3") to float; reject NaN and inf."""
  message = f"{name} must be a finite number, got {value!r}"
  if isinstance(value, bool):
    raise ValueError(message)
  try:
    val = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(message) from exc
  if not math.isfinite(val):
    raise ValueError(message)
  return val


def _require(kwargs: dict[str, Any], names: tuple[str, ...], context: str) -> None:
  """Raise a ValueError naming the first required field that is missing or null."""
  for name in names:
    if name not in kwargs:
      raise ValueError(f"{context} is missing the required field {name!r}")


@dataclass(frozen=True)
class MarketConfig:
  """Market environment (NK landscape, feedback noise, turbulence) and simulation scale."""

  n_bits: int = 12
  landscape: str = "nk"
  # NK interdependence K and interaction pattern
  nk_k: int = 3
  nk_pattern: str = "random"
  # Standard deviation of the observation noise on market results
  sigma_obs: float = 0.65
  # Shock timing: none, random (probability p_shock per period), or periodic (every shock_period)
  turbulence: str = "none"
  p_shock: float = 0.0
  shock_period: int = 100
  t_steps: int = 600
  eval_every: int = 25
  b_firms: int = 50
  seed: int = 2026
  device: str = "auto"

  def __post_init__(self) -> None:
    for name in ("landscape", "nk_pattern", "turbulence", "device"):
      object.__setattr__(self, name, str(getattr(self, name)).strip().lower())
    for name in ("n_bits", "nk_k", "shock_period", "t_steps", "eval_every", "b_firms", "seed"):
      object.__setattr__(self, name, _as_int(getattr(self, name), name))
    object.__setattr__(self, "sigma_obs", _as_float(self.sigma_obs, "sigma_obs"))
    object.__setattr__(self, "p_shock", _as_float(self.p_shock, "p_shock"))
    self.validate()

  def validate(self) -> "MarketConfig":
    """Validate the market settings."""
    if self.landscape != "nk":
      raise ValueError(f"landscape {self.landscape!r} is not supported; must be 'nk'")
    if self.n_bits < MIN_N_BITS or self.n_bits > MAX_N_BITS:
      raise ValueError(
          f"n_bits must be in [{MIN_N_BITS}, {MAX_N_BITS}], got {self.n_bits}"
      )
    if self.nk_k < 0 or self.nk_k > self.n_bits - 1:
      raise ValueError(f"nk_k must be in [0, {self.n_bits - 1}] (n_bits - 1), got {self.nk_k}")
    if self.nk_pattern not in VALID_NK_PATTERNS:
      raise ValueError(f"nk_pattern must be one of {VALID_NK_PATTERNS}, got {self.nk_pattern!r}")
    if self.sigma_obs < 0.0:
      raise ValueError(f"sigma_obs must be >= 0, got {self.sigma_obs}")
    if self.turbulence not in VALID_TURBULENCE:
      raise ValueError(f"turbulence must be one of {VALID_TURBULENCE}, got {self.turbulence!r}")
    if self.p_shock < 0.0 or self.p_shock > 1.0:
      raise ValueError(f"p_shock must be in [0, 1], got {self.p_shock}")
    if self.shock_period < 1 or self.shock_period > MAX_T_STEPS:
      raise ValueError(f"shock_period must be between 1 and {MAX_T_STEPS}, got {self.shock_period}")
    if self.t_steps < 1 or self.t_steps > MAX_T_STEPS:
      raise ValueError(f"t_steps must be between 1 and {MAX_T_STEPS}, got {self.t_steps}")
    if self.eval_every < 1 or self.eval_every > MAX_T_STEPS:
      raise ValueError(f"eval_every must be between 1 and {MAX_T_STEPS}, got {self.eval_every}")
    if self.b_firms < 1 or self.b_firms > MAX_B_FIRMS:
      raise ValueError(f"b_firms must be between 1 and {MAX_B_FIRMS}, got {self.b_firms}")
    if self.device not in ("auto", "cpu", "gpu", "tpu"):
      raise ValueError(f"device must be 'auto', 'cpu', 'gpu', or 'tpu', got {self.device!r}")
    if self.seed < 0 or self.seed > MAX_SEED:
      raise ValueError(f"seed must be in [0, {MAX_SEED}] (2**32 - 1), got {self.seed}")
    return self

  def with_overrides(self, **kwargs: Any) -> "MarketConfig":
    """Return a validated copy with updated fields."""
    return replace(self, **kwargs)

  def to_dict(self) -> dict[str, Any]:
    """Serialize to a JSON-compatible dictionary."""
    return asdict(self)

  def to_json(self, indent: int = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "MarketConfig":
    """Construct from a dictionary with current field names only."""
    return cls(**_strict_fields(cls, data, "market_config"))

  @classmethod
  def from_json(cls, json_str: str) -> "MarketConfig":
    return cls.from_dict(json.loads(json_str))

@dataclass(frozen=True)
class OrgNode:
  """One node of an organization: the dominant coalition (hq), a manager, or a unit."""

  id: str
  label: str
  role: str = "unit"
  count: int = 1
  model_type: str = "neural"
  h_agent: int = 16
  jurisdiction_bits: tuple[int, ...] = field(default_factory=lambda: tuple(range(12)))
  lr: float = 0.15
  sigma_learn: float = 0.05
  rho: float = 0.0
  search_steps: int = 1
  # Aggregation of incoming translated views: mean, max, or min
  upward_agg: str = "mean"
  # Market feedback override: None follows the organization's feedback_regime
  feedback: Optional[str] = None
  # Initial accuracy: None follows the organization default (uninformed for the DC), "uninformed"
  # forces the random initialization, and a number in [0, 1] fits starting beliefs to that accuracy
  initial_accuracy: Optional[Union[float, str]] = None

  def __post_init__(self) -> None:
    ctx = f"OrgNode {self.id!r}"
    bits = self.jurisdiction_bits
    if isinstance(bits, (str, bytes, dict)) or not hasattr(bits, "__iter__"):
      raise ValueError(f"{ctx} jurisdiction_bits must be a list of decision indices, got {bits!r}")
    object.__setattr__(self, "jurisdiction_bits", tuple(_as_int(b, f"{ctx} jurisdiction bit") for b in bits))
    for name in ("count", "h_agent", "search_steps"):
      object.__setattr__(self, name, _as_int(getattr(self, name), f"{ctx} {name}"))
    for name in ("lr", "sigma_learn", "rho"):
      object.__setattr__(self, name, _as_float(getattr(self, name), f"{ctx} {name}"))
    if isinstance(self.feedback, str):
      fb = self.feedback.strip().lower()
      object.__setattr__(self, "feedback", None if fb in ("", "default") else fb)
    object.__setattr__(
        self, "initial_accuracy", _node_accuracy(self.initial_accuracy, f"OrgNode {self.id!r} initial_accuracy")
    )
    # A linear representation has no hidden layer; a neural one with h_agent 0 is linear
    if self.model_type == "linear" and self.h_agent != 0:
      object.__setattr__(self, "h_agent", 0)
    elif self.model_type == "neural" and self.h_agent == 0:
      object.__setattr__(self, "model_type", "linear")
    self.validate()

  def validate(self, n_bits: int = MAX_N_BITS) -> "OrgNode":
    """Validate node attributes."""
    if not self.id or not str(self.id).strip():
      raise ValueError("OrgNode id must be a non-empty string")
    if not self.label or not str(self.label).strip():
      raise ValueError(f"OrgNode {self.id!r} label must be a non-empty string")
    if self.role not in VALID_NODE_ROLES:
      raise ValueError(f"OrgNode {self.id!r} role must be one of {VALID_NODE_ROLES}, got {self.role!r}")
    if self.role == "hq" and self.count != 1:
      raise ValueError(f"Dominant coalition node {self.id!r} must have count 1, got {self.count}")
    if self.count < 1 or self.count > MAX_SUBORDINATES:
      raise ValueError(f"OrgNode {self.id!r} count must be in [1, {MAX_SUBORDINATES}], got {self.count}")
    if self.model_type not in VALID_NODE_MODEL_TYPES:
      raise ValueError(
          f"OrgNode {self.id!r} model_type must be one of {VALID_NODE_MODEL_TYPES}, got {self.model_type!r}"
      )
    if self.h_agent < 0 or self.h_agent > 128:
      raise ValueError(f"OrgNode {self.id!r} h_agent must be in [0, 128], got {self.h_agent}")
    if not self.jurisdiction_bits:
      raise ValueError(f"OrgNode {self.id!r} jurisdiction_bits must not be empty")
    if len(set(self.jurisdiction_bits)) != len(self.jurisdiction_bits):
      raise ValueError(f"OrgNode {self.id!r} jurisdiction_bits contains duplicates")
    for b in self.jurisdiction_bits:
      if b < 0 or b >= n_bits:
        raise ValueError(f"OrgNode {self.id!r} jurisdiction bit {b} out of range [0, {n_bits - 1}]")
    if self.lr < 0.0 or self.lr > 2.0:
      raise ValueError(f"OrgNode {self.id!r} lr must be in [0, 2], got {self.lr}")
    if self.sigma_learn < 0.0:
      raise ValueError(f"OrgNode {self.id!r} sigma_learn must be >= 0, got {self.sigma_learn}")
    if self.rho < 0.0 or self.rho > 1.0:
      raise ValueError(f"OrgNode {self.id!r} rho must be in [0, 1], got {self.rho}")
    if self.search_steps < 1:
      raise ValueError(f"OrgNode {self.id!r} search_steps must be at least 1, got {self.search_steps}")
    if self.upward_agg not in VALID_UPWARD_AGGS:
      raise ValueError(
          f"OrgNode {self.id!r} upward_agg must be one of {VALID_UPWARD_AGGS}, got {self.upward_agg!r}"
      )
    if self.feedback is not None and self.feedback not in VALID_NODE_FEEDBACK:
      raise ValueError(
          f"OrgNode {self.id!r} feedback must be null (organization default) or one of "
          f"{VALID_NODE_FEEDBACK}, got {self.feedback!r}"
      )
    if self.role == "hq" and self.feedback not in (None, "firm"):
      raise ValueError(
          f"Dominant coalition node {self.id!r} always sees firm results; feedback "
          f"{self.feedback!r} is not allowed (use null or 'firm')"
      )
    return self

  def with_overrides(self, **kwargs: Any) -> "OrgNode":
    """Return a validated copy with updated attributes (keeps model_type and h_agent consistent)."""
    updates = dict(kwargs)
    if "h_agent" in updates and int(updates["h_agent"]) > 0 and "model_type" not in updates and self.model_type == "linear":
      updates["model_type"] = "neural"
    elif updates.get("model_type") == "neural" and "h_agent" not in updates and self.h_agent == 0:
      updates["h_agent"] = 16
    return replace(self, **updates)

  def to_dict(self) -> dict[str, Any]:
    """Serialize to a JSON-compatible dictionary."""
    d = asdict(self)
    d["jurisdiction_bits"] = list(self.jurisdiction_bits)
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "OrgNode":
    """Construct from a dictionary with current field names only."""
    raw = dict(data)
    has_feedback = "feedback" in raw
    has_accuracy = "initial_accuracy" in raw
    kwargs = _strict_fields(cls, raw, "node")
    _require(kwargs, ("id", "label"), f"node {kwargs['id']!r}" if "id" in kwargs else "node")
    if has_feedback and raw["feedback"] is None:
      kwargs["feedback"] = None
    if has_accuracy and raw["initial_accuracy"] is None:
      kwargs["initial_accuracy"] = None
    return cls(**kwargs)


@dataclass(frozen=True)
class OrgEdge:
  """Directed link: the target node learns from the source node."""

  source: str
  target: str
  # Permeability P: how strongly the target weighs the source's view
  weight: float = 0.25
  # Commensurability C: shared by both directions of a dyad
  commensurability: float = 0.85

  def __post_init__(self) -> None:
    if not self.source or not self.target:
      raise ValueError("OrgEdge source and target must be non-empty node ids")
    if self.source == self.target:
      raise ValueError(f"Self-loop link ({self.source!r} -> {self.target!r}) is not permitted")
    object.__setattr__(self, "weight", _as_float(self.weight, f"Link ({self.source!r} -> {self.target!r}) weight"))
    object.__setattr__(
        self,
        "commensurability",
        _as_float(self.commensurability, f"Link ({self.source!r} -> {self.target!r}) commensurability"),
    )
    if self.weight < 0.0 or self.weight > 1.0:
      raise ValueError(f"Link ({self.source!r} -> {self.target!r}) weight must be in [0, 1], got {self.weight}")
    if self.commensurability < 0.0 or self.commensurability > 1.0:
      raise ValueError(
          f"Link ({self.source!r} -> {self.target!r}) commensurability must be in [0, 1], "
          f"got {self.commensurability}"
      )

  def to_dict(self) -> dict[str, Any]:
    return {
        "source": self.source,
        "target": self.target,
        "weight": self.weight,
        "commensurability": self.commensurability,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "OrgEdge":
    """Construct from a dictionary with current field names only."""
    kwargs = _strict_fields(cls, data, "edge")
    _require(kwargs, ("source", "target"), "edge")
    return cls(**kwargs)

@dataclass(frozen=True)
class OrgGraph:
  """An organization: nodes (one dominant coalition) and directed learning links between them."""

  name: str
  description: str = ""
  nodes: tuple[OrgNode, ...] = field(default_factory=tuple)
  edges: tuple[OrgEdge, ...] = field(default_factory=tuple)
  # Who observes market results: the DC always sees firm results; units follow this regime
  feedback_regime: str = "dc_firm_units_domain"
  # Who changes the strategy: centralized, propose_approve, or delegated
  decision_rights: str = "centralized"
  # Per-period probability that a non-DC node that sees results tests one change in its domain
  experiment_rate: float = DEFAULT_EXPERIMENT_RATE
  # Sensitivity (kappa) of the DC's distant-search probability to shortfalls below aspiration
  aspiration_sensitivity: float = 0.0
  # Track-record learning rate for endogenous permeability
  lambda_track: float = 0.0
  # Default initial accuracy of non-DC nodes; None means uninformed (random) beliefs
  unit_initial_accuracy: Optional[float] = None

  def __post_init__(self) -> None:
    object.__setattr__(self, "nodes", tuple(self.nodes))
    object.__setattr__(self, "feedback_regime", str(self.feedback_regime).strip().lower())
    object.__setattr__(self, "decision_rights", str(self.decision_rights).strip().lower())
    for name in ("experiment_rate", "aspiration_sensitivity", "lambda_track"):
      object.__setattr__(self, name, _as_float(getattr(self, name), name))
    object.__setattr__(
        self, "unit_initial_accuracy", _optional_accuracy(self.unit_initial_accuracy, "unit_initial_accuracy")
    )
    raw_edges = tuple(self.edges)
    # Commensurability is a property of the dyad, so both directions share one value
    pair_comm: dict[frozenset[str], float] = {}
    for e in raw_edges:
      key = frozenset((e.source, e.target))
      if key not in pair_comm or abs(e.commensurability - 0.85) > 1e-6:
        pair_comm[key] = e.commensurability
    object.__setattr__(
        self,
        "edges",
        tuple(replace(e, commensurability=pair_comm[frozenset((e.source, e.target))]) for e in raw_edges),
    )
    self.validate()

  def validate(self, n_bits: int = MAX_N_BITS) -> "OrgGraph":
    """Validate nodes, links, headcount, connectivity, depth, and organization settings."""
    if not self.nodes:
      raise ValueError("An organization needs at least the dominant coalition node (role 'hq')")
    if self.feedback_regime not in VALID_FEEDBACK_REGIMES:
      raise ValueError(
          f"feedback_regime must be one of {VALID_FEEDBACK_REGIMES}, got {self.feedback_regime!r}"
      )
    if self.decision_rights not in VALID_DECISION_RIGHTS:
      raise ValueError(f"decision_rights must be one of {VALID_DECISION_RIGHTS}, got {self.decision_rights!r}")
    if self.experiment_rate < 0.0 or self.experiment_rate > 1.0:
      raise ValueError(f"experiment_rate must be in [0, 1], got {self.experiment_rate}")
    if self.aspiration_sensitivity < 0.0 or self.aspiration_sensitivity > 5.0:
      raise ValueError(f"aspiration_sensitivity must be in [0, 5], got {self.aspiration_sensitivity}")
    if self.lambda_track < 0.0 or self.lambda_track > 5.0:
      raise ValueError(f"lambda_track must be in [0, 5], got {self.lambda_track}")

    node_map: dict[str, OrgNode] = {}
    for node in self.nodes:
      node.validate(n_bits=n_bits)
      if node.id in node_map:
        raise ValueError(f"Duplicate node id {node.id!r}")
      node_map[node.id] = node
    hq_ids = [n.id for n in self.nodes if n.role == "hq"]
    if len(hq_ids) != 1:
      raise ValueError(f"An organization needs exactly one dominant coalition node (role 'hq'), found {len(hq_ids)}")

    seen: set[tuple[str, str]] = set()
    for e in self.edges:
      for end in (e.source, e.target):
        if end not in node_map:
          raise ValueError(f"Link endpoint {end!r} is not a node of the organization")
      if (e.source, e.target) in seen:
        raise ValueError(f"Duplicate link ({e.source!r} -> {e.target!r})")
      seen.add((e.source, e.target))

    total_subordinates = sum(n.count for n in self.nodes if n.role != "hq")
    if total_subordinates > MAX_SUBORDINATES:
      raise ValueError(
          f"Total subordinate headcount ({total_subordinates}) exceeds the maximum of {MAX_SUBORDINATES}"
      )

    hq_id = hq_ids[0]
    adj: dict[str, set[str]] = {n.id: set() for n in self.nodes}
    for e in self.edges:
      adj[e.source].add(e.target)
      adj[e.target].add(e.source)
    reached = {hq_id}
    queue = [hq_id]
    while queue:
      curr = queue.pop(0)
      for nb in adj[curr]:
        if nb not in reached:
          reached.add(nb)
          queue.append(nb)
    for n in self.nodes:
      if n.id not in reached:
        raise ValueError(f"Node {n.id!r} has no link path to the dominant coalition {hq_id!r}")

    depths = self.get_topological_depths()
    if max(depths.values()) > MAX_TIERS - 1:
      raise ValueError(
          f"Organization depth ({max(depths.values()) + 1} tiers) exceeds the maximum of {MAX_TIERS} tiers"
      )
    return self

  def update_node(self, node_id: str, **kwargs: Any) -> "OrgGraph":
    """Return a validated copy with one node updated."""
    if not any(n.id == node_id for n in self.nodes):
      raise ValueError(f"Node id {node_id!r} not found")
    return replace(self, nodes=tuple(n.with_overrides(**kwargs) if n.id == node_id else n for n in self.nodes))

  def with_n_bits(self, n_bits: int, source_n_bits: int = 12) -> "OrgGraph":
    """Return a validated copy with node domains and half-domain labels rescaled to n_bits."""
    n_bits = _as_int(n_bits, "n_bits")
    source_n_bits = _as_int(source_n_bits, "source_n_bits")
    if n_bits < MIN_N_BITS or n_bits > MAX_N_BITS:
      raise ValueError(f"n_bits must be in [{MIN_N_BITS}, {MAX_N_BITS}], got {n_bits}")
    if source_n_bits < MIN_N_BITS or source_n_bits > MAX_N_BITS:
      raise ValueError(f"source_n_bits must be in [{MIN_N_BITS}, {MAX_N_BITS}], got {source_n_bits}")
    old_h = source_n_bits // 2
    new_h = n_bits // 2

    def _rescale_text(text: str) -> str:
      tag_a = "__BITS_FIRST_HALF__"
      tag_b = "__BITS_SECOND_HALF__"
      out = text.replace(f"Bits 1-{old_h}", tag_a).replace(f"Bits {old_h + 1}-{source_n_bits}", tag_b)
      return out.replace(tag_a, f"Bits 1-{new_h}").replace(tag_b, f"Bits {new_h + 1}-{n_bits}")

    new_nodes = tuple(
        replace(
            n,
            label=_rescale_text(n.label),
            jurisdiction_bits=rescale_jurisdiction_bits(n.jurisdiction_bits, source_n_bits, n_bits),
        )
        for n in self.nodes
    )
    updated = replace(self, description=_rescale_text(self.description), nodes=new_nodes)
    return updated.validate(n_bits=n_bits)

  def get_hq_node(self) -> OrgNode:
    """Return the dominant coalition node."""
    for n in self.nodes:
      if n.role == "hq":
        return n
    raise ValueError("No dominant coalition node (role 'hq') found")

  def get_topological_depths(self) -> dict[str, int]:
    """Tier of each node by undirected link distance to the DC (0 = DC); unreached nodes fall back by role."""
    hq = self.get_hq_node()
    neighbors: dict[str, set[str]] = {n.id: set() for n in self.nodes}
    for e in self.edges:
      neighbors[e.target].add(e.source)
      neighbors[e.source].add(e.target)
    dist = {hq.id: 0}
    queue = [hq.id]
    while queue:
      curr = queue.pop(0)
      for nb in neighbors[curr]:
        if nb not in dist:
          dist[nb] = dist[curr] + 1
          queue.append(nb)
    has_managers = any(n.role == "manager" for n in self.nodes)
    for n in self.nodes:
      if n.id not in dist:
        dist[n.id] = 1 if n.role == "manager" else (2 if has_managers else 1)
    return dist

  def effective_feedback(self) -> dict[str, str]:
    """Each node's market feedback ("firm", "domain", or "none"); the DC always sees firm results."""
    default_fb = FEEDBACK_REGIME_DEFAULTS[self.feedback_regime]
    return {
        n.id: "firm" if n.role == "hq" else (n.feedback if n.feedback is not None else default_fb)
        for n in self.nodes
    }

  def effective_initial_accuracy(self) -> dict[str, Optional[float]]:
    """Each node's initial accuracy (None = uninformed); the DC uses only its own setting."""

    def resolve(n: OrgNode) -> Optional[float]:
      if n.initial_accuracy == UNINFORMED:
        return None
      if n.role == "hq" or n.initial_accuracy is not None:
        return n.initial_accuracy
      return self.unit_initial_accuracy

    return {n.id: resolve(n) for n in self.nodes}

  def effective_decision_rights(self) -> str:
    """Decision-rights mode the engine applies (a DC without other nodes decides alone)."""
    if not any(n.role != "hq" for n in self.nodes):
      return "centralized"
    return self.decision_rights

  def dc_search_bits(self) -> tuple[int, ...]:
    """Decision bits the DC searches itself under the effective decision rights."""
    dc_bits = tuple(sorted(set(self.get_hq_node().jurisdiction_bits)))
    mode = self.effective_decision_rights()
    if mode == "propose_approve":
      return ()
    if mode == "delegated":
      unit_bits = {b for n in self.nodes if n.role != "hq" for b in n.jurisdiction_bits}
      return tuple(b for b in dc_bits if b not in unit_bits)
    return dc_bits

  def _explanation_notes(self, node_feedback: dict[str, str], search_bits: tuple[int, ...], n_bits: int) -> list[str]:
    """Plain-language notes about settings that are inert or resolved in a non-obvious way."""
    notes: list[str] = []
    node_map = {n.id: n for n in self.nodes}
    full_bits = set(range(n_bits))
    full_domain = [nid for nid, fb in node_feedback.items() if fb == "domain" and set(node_map[nid].jurisdiction_bits) == full_bits]
    if full_domain:
      notes.append(
          "These nodes cover every decision, so their own-domain results equal firm results "
          f"(same payoff and same noise draw): {', '.join(full_domain)}."
      )
    incoming = {n.id: 0 for n in self.nodes}
    for e in self.edges:
      if e.weight > 0.0:
        incoming[e.target] += 1
    frozen = [nid for nid, fb in node_feedback.items() if fb == "none" and incoming[nid] == 0]
    if frozen:
      notes.append(
          "These nodes see no market results and have no incoming links, so their representations "
          f"never change after initialization: {', '.join(frozen)}."
      )
    eff_mode = self.effective_decision_rights()
    if eff_mode != self.decision_rights:
      notes.append(
          f"Decision rights {self.decision_rights!r} need at least one non-DC node; the DC decides alone (centralized)."
      )
    if eff_mode == "delegated" and not search_bits:
      notes.append("Under delegated decision rights every DC decision lies inside some unit's domain, so the DC does not search.")
    if self.aspiration_sensitivity > 0.0 and not search_bits:
      notes.append(
          "Search after shortfalls changes only the DC's own search, which is inactive under the current decision rights."
      )
    if self.lambda_track > 0.0 and not any(e.weight > 0.0 for e in self.edges):
      notes.append("Track-record learning has no effect without links.")
    has_units = any(n.role != "hq" for n in self.nodes)
    if has_units and self.experiment_rate > 0.0 and not any(node_map[nid].role != "hq" and fb != "none" for nid, fb in node_feedback.items()):
      notes.append("Unit experiments have no effect: no non-DC node sees market results.")
    partial_informed = [
        nid
        for nid, acc in self.effective_initial_accuracy().items()
        if acc is not None and set(node_map[nid].jurisdiction_bits) != full_bits
    ]
    if partial_informed:
      notes.append(
          "Initial accuracy of these nodes is measured against the part of their observed payoff that "
          "their own choices determine (the payoff averaged over the decisions outside their domain): "
          f"{', '.join(partial_informed)}."
      )
    return notes

  def explain_graph(self, market: Optional[MarketConfig] = None) -> dict[str, Any]:
    """Structured summary of the organization and its settings on a market."""
    if market is None:
      max_bit = max((b for n in self.nodes for b in n.jurisdiction_bits), default=0)
      mkt = MarketConfig(n_bits=max(12, max_bit + 1))
    else:
      mkt = market.validate()
    self.validate(n_bits=mkt.n_bits)
    hq = self.get_hq_node()
    depths = self.get_topological_depths()
    managers = [n for n in self.nodes if n.role == "manager"]
    units = [n for n in self.nodes if n.role == "unit"]
    out_links: dict[str, int] = {n.id: 0 for n in self.nodes}
    for e in self.edges:
      out_links[e.source] += 1
    node_feedback = self.effective_feedback()
    node_accuracy = self.effective_initial_accuracy()
    search_bits = self.dc_search_bits()
    return {
        "name": self.name,
        "description": self.description,
        "tiers": max(depths.values()) + 1,
        "n_nodes": len(self.nodes),
        "n_edges": len(self.edges),
        "n_managers": sum(m.count for m in managers),
        "n_operating_units": sum(u.count for u in units),
        "total_subordinates": sum(n.count for n in self.nodes if n.role != "hq"),
        "dual_authority_nodes": [n.id for n in self.nodes if n.role != "hq" and out_links[n.id] > 1],
        "landscape": {
            "family": "nk",
            "n_bits": mkt.n_bits,
            "nk_k": mkt.nk_k,
            "nk_pattern": mkt.nk_pattern,
            "turbulence": mkt.turbulence,
            "p_shock": mkt.p_shock,
            "shock_period": mkt.shock_period,
            "sigma_obs": mkt.sigma_obs,
        },
        "feedback_regime": self.feedback_regime,
        "node_feedback": node_feedback,
        "decision_rights": self.decision_rights,
        "effective_decision_rights": self.effective_decision_rights(),
        "dc_search_bits": list(search_bits),
        "experiment_rate": self.experiment_rate,
        "aspiration_sensitivity": self.aspiration_sensitivity,
        "lambda_track": self.lambda_track,
        "unit_initial_accuracy": self.unit_initial_accuracy,
        "node_initial_accuracy": node_accuracy,
        "notes": self._explanation_notes(node_feedback, search_bits, mkt.n_bits),
        "hq_summary": {
            "id": hq.id,
            "label": hq.label,
            "model_type": hq.model_type,
            "h_agent": hq.h_agent,
            "rho": hq.rho,
            "sigma_learn": hq.sigma_learn,
            "lr": hq.lr,
            "search_steps": hq.search_steps,
            "jurisdiction_bits": list(hq.jurisdiction_bits),
            "feedback": "firm",
            "upward_agg": hq.upward_agg,
            "initial_accuracy": node_accuracy[hq.id],
        },
    }

  def to_dict(self) -> dict[str, Any]:
    """Serialize to a JSON-compatible dictionary."""
    return {
        "name": self.name,
        "description": self.description,
        "feedback_regime": self.feedback_regime,
        "decision_rights": self.decision_rights,
        "experiment_rate": self.experiment_rate,
        "aspiration_sensitivity": self.aspiration_sensitivity,
        "lambda_track": self.lambda_track,
        "unit_initial_accuracy": self.unit_initial_accuracy,
        "nodes": [n.to_dict() for n in self.nodes],
        "edges": [e.to_dict() for e in self.edges],
    }

  def to_json(self, indent: int = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "OrgGraph":
    """Construct from a dictionary with current field names only."""
    kwargs = _strict_fields(cls, data, "org_graph")
    for key in ("nodes", "edges"):
      if not isinstance(data.get(key) or [], (list, tuple)):
        raise ValueError(f"org_graph {key} must be a list of objects, got {type(data[key]).__name__}")
    kwargs["nodes"] = tuple(OrgNode.from_dict(n) for n in data.get("nodes") or [])
    kwargs["edges"] = tuple(OrgEdge.from_dict(e) for e in data.get("edges") or [])
    if "name" not in kwargs:
      kwargs["name"] = "Custom Organization"
    return cls(**kwargs)

  @classmethod
  def from_json(cls, json_str: str) -> "OrgGraph":
    return cls.from_dict(json.loads(json_str))

def _build_graph_templates() -> dict[str, OrgGraph]:
  """Construct the 12 organization templates."""
  all_bits = tuple(range(12))
  first_half = tuple(range(0, 6))
  second_half = tuple(range(6, 12))

  def dc_node(label: str, rho: float, search_steps: int = 1) -> OrgNode:
    return OrgNode(id="hq", label=label, role="hq", jurisdiction_bits=all_bits, lr=0.15, sigma_learn=0.05, rho=rho, search_steps=search_steps)

  def rep_node(
      node_id: str,
      label: str,
      bits: tuple[int, ...],
      lr: float = 0.18,
      sigma_learn: float = 0.04,
      role: str = "unit",
      h_agent: int = 16,
      feedback: Optional[str] = None,
      initial_accuracy: Optional[float] = None,
  ) -> OrgNode:
    return OrgNode(
        id=node_id,
        label=label,
        role=role,
        h_agent=h_agent,
        jurisdiction_bits=bits,
        lr=lr,
        sigma_learn=sigma_learn,
        feedback=feedback,
        initial_accuracy=initial_accuracy,
    )

  specialists = (
      rep_node("rep_1", "Specialist 1 (Bits 1-6)", (0, 1, 2, 3, 4, 5)),
      rep_node("rep_2", "Specialist 2 (Bits 7-12)", (6, 7, 8, 9, 10, 11)),
      rep_node("rep_3", "Specialist 3 (Bits 1-3, 7-9)", (0, 1, 2, 6, 7, 8)),
      rep_node("rep_4", "Specialist 4 (Bits 4-6, 10-12)", (3, 4, 5, 9, 10, 11)),
  )

  def links_to_dc(weight: float, comm: float) -> tuple[OrgEdge, ...]:
    return tuple(OrgEdge(source=f"rep_{k}", target="hq", weight=weight, commensurability=comm) for k in range(1, 5))

  def links_from_dc(weight: float, comm: float) -> tuple[OrgEdge, ...]:
    return tuple(OrgEdge(source="hq", target=f"rep_{k}", weight=weight, commensurability=comm) for k in range(1, 5))

  t: dict[str, OrgGraph] = {}
  t["coordinated_neural_hierarchy"] = OrgGraph(
      name="Coordinated Neural Hierarchy",
      description=(
          "4 domain-specialized representations (6 decisions each, low learning noise) send upward "
          "predictions to the dominant coalition (rho=0.15) with high commensurability (C=0.90) and "
          "no downward links."
      ),
      nodes=(dc_node("Dominant Coalition (HQ)", rho=0.15),) + specialists,
      edges=links_to_dc(0.25, 0.90),
  )
  t["conformity_trap"] = OrgGraph(
      name="Conformity Trap Hierarchy",
      description=(
          "Two-way hierarchy where heavy downward permeability from the dominant coalition "
          "(P_down=0.55) overwrites specialist representations, collapsing cognitive orthogonality."
      ),
      nodes=(dc_node("Dominant Coalition (HQ)", rho=0.05),) + specialists,
      edges=links_to_dc(0.25, 0.90) + links_from_dc(0.55, 0.90),
  )
  t["bundled_neural_hierarchy"] = OrgGraph(
      name="Bundled Socialization Hierarchy",
      description=(
          "Two-way links with moderate upward and downward permeability (P_up=0.25, P_down=0.20, "
          "C=0.85), balancing specialist learning with shared framing."
      ),
      nodes=(dc_node("Dominant Coalition (HQ)", rho=0.10),) + specialists,
      edges=links_to_dc(0.25, 0.85) + links_from_dc(0.20, 0.85),
  )
  t["m_form_3_tier"] = OrgGraph(
      name="Multi-Divisional M-Form (3-Tier)",
      description=(
          "3-tier hierarchy: the dominant coalition integrates 2 division heads (Bits 1-6 and "
          "Bits 7-12), each synthesizing 2 specialized operating representations."
      ),
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.15),
          rep_node("div_a", "Division A Head (Bits 1-6)", first_half, lr=0.16, role="manager"),
          rep_node("div_b", "Division B Head (Bits 7-12)", second_half, lr=0.16, role="manager"),
          rep_node("unit_a1", "Div A Unit 1 (Bits 1-4)", (0, 1, 2, 3), sigma_learn=0.03),
          rep_node("unit_a2", "Div A Unit 2 (Bits 3-6)", (2, 3, 4, 5), sigma_learn=0.03),
          rep_node("unit_b1", "Div B Unit 1 (Bits 7-10)", (6, 7, 8, 9), sigma_learn=0.03),
          rep_node("unit_b2", "Div B Unit 2 (Bits 9-12)", (8, 9, 10, 11), sigma_learn=0.03),
      ),
      edges=(
          OrgEdge(source="div_a", target="hq", weight=0.35, commensurability=0.90),
          OrgEdge(source="div_b", target="hq", weight=0.35, commensurability=0.90),
          OrgEdge(source="unit_a1", target="div_a", weight=0.35, commensurability=0.92),
          OrgEdge(source="unit_a2", target="div_a", weight=0.35, commensurability=0.92),
          OrgEdge(source="unit_b1", target="div_b", weight=0.35, commensurability=0.92),
          OrgEdge(source="unit_b2", target="div_b", weight=0.35, commensurability=0.92),
      ),
  )
  t["incommensurable_silos"] = OrgGraph(
      name="Incommensurable Specialized Silos",
      description=(
          "High-orthogonality specialist representations with low commensurability (C=0.30) toward "
          "the dominant coalition, inducing severe mistranslation."
      ),
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.10),
          rep_node("rep_1", "Silo 1 (Bits 1-4)", (0, 1, 2, 3), sigma_learn=0.03),
          rep_node("rep_2", "Silo 2 (Bits 5-8)", (4, 5, 6, 7), sigma_learn=0.03),
          rep_node("rep_3", "Silo 3 (Bits 9-12)", (8, 9, 10, 11), sigma_learn=0.03),
          rep_node("rep_4", "Silo 4 (Bits 1,4,7,10)", (0, 3, 6, 9), sigma_learn=0.03),
      ),
      edges=links_to_dc(0.25, 0.30),
  )
  t["adaptive_meritocracy"] = OrgGraph(
      name="Adaptive Track-Record Meritocracy",
      description=(
          "Two informed, low-noise experts (initial accuracy 0.7, C=0.95) and two poorly informed, "
          "noisy generalists (initial accuracy 0.2, C=0.40); track-record learning (lambda_track=2.5) "
          "lets the dominant coalition reweight its sources by their accuracy."
      ),
      lambda_track=2.5,
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.15),
          rep_node("rep_expert_1", "Expert Domain A (Accuracy 0.7, C=0.95)", first_half, lr=0.20, sigma_learn=0.02, initial_accuracy=0.7),
          rep_node("rep_expert_2", "Expert Domain B (Accuracy 0.7, C=0.95)", second_half, lr=0.20, sigma_learn=0.02, initial_accuracy=0.7),
          rep_node("rep_noisy_1", "Noisy Generalist 1 (Accuracy 0.2, C=0.40)", all_bits, lr=0.10, sigma_learn=0.25, initial_accuracy=0.2),
          rep_node("rep_noisy_2", "Noisy Generalist 2 (Accuracy 0.2, C=0.40)", all_bits, lr=0.10, sigma_learn=0.25, initial_accuracy=0.2),
      ),
      edges=(
          OrgEdge(source="rep_expert_1", target="hq", weight=0.20, commensurability=0.95),
          OrgEdge(source="rep_expert_2", target="hq", weight=0.20, commensurability=0.95),
          OrgEdge(source="rep_noisy_1", target="hq", weight=0.20, commensurability=0.40),
          OrgEdge(source="rep_noisy_2", target="hq", weight=0.20, commensurability=0.40),
      ),
  )
  t["matrix_dual_authority"] = OrgGraph(
      name="Matrix / Dual-Authority Organization",
      description=(
          "3-tier matrix: two cross-functional squads each report to both a functional manager "
          "(Bits 1-6) and a product manager (Bits 7-12); both managers advise the dominant coalition."
      ),
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.10),
          rep_node("mgr_functional", "Functional Manager (Bits 1-6)", first_half, lr=0.16, role="manager"),
          rep_node("mgr_product", "Product Manager (Bits 7-12)", second_half, lr=0.16, role="manager"),
          rep_node("squad_1", "Matrix Squad 1 (Bits 1-3, 7-9)", (0, 1, 2, 6, 7, 8)),
          rep_node("squad_2", "Matrix Squad 2 (Bits 4-6, 10-12)", (3, 4, 5, 9, 10, 11)),
      ),
      edges=(
          OrgEdge(source="squad_1", target="mgr_functional", weight=0.30, commensurability=0.90),
          OrgEdge(source="squad_1", target="mgr_product", weight=0.30, commensurability=0.90),
          OrgEdge(source="squad_2", target="mgr_functional", weight=0.30, commensurability=0.90),
          OrgEdge(source="squad_2", target="mgr_product", weight=0.30, commensurability=0.90),
          OrgEdge(source="mgr_functional", target="hq", weight=0.35, commensurability=0.90),
          OrgEdge(source="mgr_product", target="hq", weight=0.35, commensurability=0.90),
      ),
  )
  t["ambidextrous_org"] = OrgGraph(
      name="Ambidextrous Organization (Exploitation + Exploration)",
      description=(
          "Structural ambidexterity: a low-noise exploitation unit with a moderate learning rate on "
          "the core decisions (Bits 1-6) and a fast-learning exploration unit on distinct decisions "
          "(Bits 7-12) with lower commensurability (C=0.60) toward the dominant coalition."
      ),
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.10),
          rep_node("exploit_unit", "Exploitation Unit (Bits 1-6)", first_half, lr=0.15, sigma_learn=0.02),
          rep_node("explore_unit", "Exploration Unit (Bits 7-12)", second_half, lr=0.30, sigma_learn=0.06),
      ),
      edges=(
          OrgEdge(source="exploit_unit", target="hq", weight=0.30, commensurability=0.90),
          OrgEdge(source="explore_unit", target="hq", weight=0.30, commensurability=0.60),
      ),
  )
  t["hybrid_human_ai"] = OrgGraph(
      name="Hybrid Human-AI Advisory Structure",
      description=(
          "Two human specialists with partial domains and limited capacity (H=8) plus an AI analyst "
          "covering all decisions with high capacity (H=64), low learning noise, and a high learning "
          "rate; all three advise the dominant coalition."
      ),
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.10),
          rep_node("human_1", "Human Specialist 1 (Bits 1-6)", first_half, lr=0.15, sigma_learn=0.06, h_agent=8),
          rep_node("human_2", "Human Specialist 2 (Bits 7-12)", second_half, lr=0.15, sigma_learn=0.06, h_agent=8),
          rep_node("ai_analyst", "AI Analyst (All Decisions, H=64)", all_bits, lr=0.30, sigma_learn=0.01, h_agent=64),
      ),
      edges=(
          OrgEdge(source="human_1", target="hq", weight=0.25, commensurability=0.85),
          OrgEdge(source="human_2", target="hq", weight=0.25, commensurability=0.85),
          OrgEdge(source="ai_analyst", target="hq", weight=0.30, commensurability=0.85),
      ),
  )
  t["boundary_spanners"] = OrgGraph(
      name="Boundary Spanners (Front Line and Internal Units)",
      description=(
          "Two front-line units see results in their own domains and advise the dominant coalition; "
          "two internal units with the same domains see no market results and exchange predictions "
          "with their front-line partner (Tushman and Scanlan 1981)."
      ),
      nodes=(
          dc_node("Dominant Coalition (HQ)", rho=0.10),
          rep_node("front_1", "Front-Line Unit 1 (Bits 1-6)", first_half, feedback="domain"),
          rep_node("front_2", "Front-Line Unit 2 (Bits 7-12)", second_half, feedback="domain"),
          rep_node("internal_1", "Internal Unit 1 (Bits 1-6)", first_half, feedback="none"),
          rep_node("internal_2", "Internal Unit 2 (Bits 7-12)", second_half, feedback="none"),
      ),
      edges=(
          OrgEdge(source="internal_1", target="front_1", weight=0.20, commensurability=0.85),
          OrgEdge(source="internal_2", target="front_2", weight=0.20, commensurability=0.85),
          OrgEdge(source="front_1", target="internal_1", weight=0.30, commensurability=0.85),
          OrgEdge(source="front_2", target="internal_2", weight=0.30, commensurability=0.85),
          OrgEdge(source="front_1", target="hq", weight=0.30, commensurability=0.90),
          OrgEdge(source="front_2", target="hq", weight=0.30, commensurability=0.90),
      ),
  )
  t["enacted_solo_dc"] = OrgGraph(
      name="Solo Dominant Coalition (Local Search Only)",
      description=(
          "Single dominant coalition representation learning strictly from enacted 1-bit local search "
          "(rho=0.0, no subordinate representations)."
      ),
      nodes=(dc_node("Dominant Coalition (Solo Local)", rho=0.0),),
  )
  t["active_solo_dc"] = OrgGraph(
      name="Solo Dominant Coalition (Distant Reach rho=0.25)",
      description=(
          "Single dominant coalition representation with distant cognitive reach (rho=0.25, "
          "search_steps=2) but no subordinate representations."
      ),
      nodes=(dc_node("Dominant Coalition (Distant Reach)", rho=0.25, search_steps=2),),
  )
  return t


GRAPH_TEMPLATES: dict[str, OrgGraph] = _build_graph_templates()

_GRAPH_PRESET_TITLES: dict[str, tuple[str, str]] = {
    "coordinated_neural_hierarchy": (
        "Coordinated Neural Hierarchy",
        "4 specialists advise the DC (P_up=0.25, no downward links, C=0.90, rho=0.15)",
    ),
    "conformity_trap": (
        "Conformity Trap Hierarchy",
        "Heavy top-down socialization (P_down=0.55) erodes cognitive orthogonality",
    ),
    "bundled_neural_hierarchy": (
        "Bundled Socialization Hierarchy",
        "Two-way links with moderate upward and downward permeability (P_down=0.20, C=0.85)",
    ),
    "m_form_3_tier": ("Multi-Divisional M-Form (3-Tier)", "DC, 2 division heads, and 4 specialized operating units"),
    "incommensurable_silos": (
        "Incommensurable Specialized Silos",
        "Orthogonal domain silos with severe mistranslation (C=0.30)",
    ),
    "adaptive_meritocracy": (
        "Adaptive Track-Record Meritocracy",
        "Informed experts (0.7) and poorly informed generalists (0.2) under track-record learning",
    ),
    "matrix_dual_authority": (
        "Matrix / Dual-Authority Organization",
        "Squads report to a functional and a product manager, who both advise the DC",
    ),
    "ambidextrous_org": (
        "Ambidextrous Organization",
        "Low-noise exploitation unit plus fast-learning exploration unit (C=0.60)",
    ),
    "hybrid_human_ai": (
        "Hybrid Human-AI Advisory Structure",
        "Two human specialists (H=8) and a full-domain AI analyst (H=64) advise the DC",
    ),
    "boundary_spanners": ("Boundary Spanners", "Front-line units see own-domain results; internal units see none"),
    "enacted_solo_dc": (
        "Solo Dominant Coalition (Local Search Only)",
        "Single representation with myopic 1-bit local search (rho=0.0)",
    ),
    "active_solo_dc": (
        "Solo Dominant Coalition (Distant Reach rho=0.25)",
        "Single representation with distant cognitive reach (rho=0.25, search_steps=2)",
    ),
}

GRAPH_PRESET_METADATA: dict[str, dict[str, str]] = {
    key: {
        "title": _GRAPH_PRESET_TITLES[key][0],
        "subtitle": _GRAPH_PRESET_TITLES[key][1],
        "description": GRAPH_TEMPLATES[key].description,
    }
    for key in GRAPH_TEMPLATES
}

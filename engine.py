"""Vectorized JAX engine for organizational search on NK landscapes (organization graphs)."""

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from functools import lru_cache
import math
import time
from typing import Any, NamedTuple, Optional
import zlib

import jax
import jax.numpy as jnp
import numpy as np

try:
  from .org_graph import UNINFORMED, MarketConfig, OrgEdge, OrgGraph, OrgNode
except ImportError:
  from org_graph import UNINFORMED, MarketConfig, OrgEdge, OrgGraph, OrgNode  # type: ignore

# Largest number of configurations in one paired comparison
MAX_CONFIGURATIONS = 8
# Initial beliefs are fit to INIT_BELIEF_SCALE times a standardized payoff mix (held weakly)
INIT_BELIEF_SCALE = 0.5


def _resolve_backend(device: Optional[str] = None) -> str:
  if device is not None and device.lower() not in ("", "auto"):
    return jax.devices(device.lower())[0].platform.upper()
  return jax.default_backend().upper()


def _device_context(device: Optional[str] = None):
  if device is None or device.lower() in ("", "auto"):
    return nullcontext()
  return jax.default_device(jax.devices(device.lower())[0])


@lru_cache(maxsize=16)
def get_hypercube_tables(n_bits: int, backend: str = ""):
  """All 2^n_bits strategies in {-1, +1}^n_bits plus one-bit neighbor index tables."""
  del backend
  num_states = 1 << n_bits
  all_idx = jnp.arange(num_states, dtype=jnp.int32)
  pow2 = (1 << jnp.arange(n_bits - 1, -1, -1)).astype(jnp.int32)
  all_s = 1.0 - 2.0 * ((all_idx[:, None] >> jnp.arange(n_bits - 1, -1, -1)) & 1).astype(jnp.float32)
  flip_eye = 1.0 - 2.0 * jnp.eye(n_bits, dtype=jnp.float32)
  neighbor_idx = all_idx[:, None] ^ pow2[None, :]
  return num_states, all_idx, pow2, all_s, flip_eye, neighbor_idx


class AgentParams(NamedTuple):
  w1: jax.Array
  b1: jax.Array
  w2: jax.Array
  b2: jax.Array


def init_agent(key: jax.Array, n_bits: int, h_agent: int) -> AgentParams:
  """Random initialization of a representation network."""
  k1, k2, k3 = jax.random.split(key, 3)
  w1 = jax.random.normal(k1, (n_bits, h_agent)) * (1.15 / jnp.sqrt(n_bits))
  b1 = jax.random.normal(k2, (h_agent,)) * 0.55
  w2 = jax.random.normal(k3, (h_agent, 1)) * 0.08
  b2 = jnp.zeros((1,), dtype=jnp.float32)
  return AgentParams(w1=w1, b1=b1, w2=w2, b2=b2)


def _mlp_hidden(s: jax.Array, w1: jax.Array, b1: jax.Array, h_mask: jax.Array, inv_sqrt_h: jax.Array) -> jax.Array:
  """Hidden-layer activations with capacity-scaled epistatic resolution."""
  eff_h = 1.0 / jnp.maximum(inv_sqrt_h * inv_sqrt_h, 1e-6)
  log_h = jnp.log2(jnp.maximum(eff_h, 2.0))
  w1_scale = jnp.sqrt(jnp.mean(jnp.square(w1), axis=-2) * float(s.shape[-1]) + 0.20)
  z_mlp = jnp.matmul(s, w1) / w1_scale + 0.65 * b1
  sharp = 0.90 + 0.22 * log_h
  cap_gain = 0.55 + 0.16 * log_h
  a_dir = 0.55 * jax.nn.silu(z_mlp)
  a_loc = 0.50 * jnp.exp(-sharp * jnp.square(z_mlp)) + 0.14 * jnp.cos(sharp * z_mlp)
  a_raw = cap_gain * (a_dir + a_loc)
  return (a_raw * h_mask) * inv_sqrt_h


def eval_agent(params: AgentParams, s: jax.Array, h_mask: jax.Array, inv_sqrt_h: jax.Array, is_linear: jax.Array) -> jax.Array:
  """Representation output for (masked) inputs s with a hidden-width mask and a linear toggle."""
  z_lin = jnp.matmul(s, params.w1[..., :1]) / jnp.sqrt(s.shape[-1])
  w2_raw = params.w2[..., :1, :]
  w2_lin = jnp.where(w2_raw >= 0.0, jnp.maximum(w2_raw, 0.5), jnp.minimum(w2_raw, -0.5))
  out_lin = jnp.squeeze(jnp.matmul(z_lin, w2_lin) + params.b2, axis=-1)
  out_lin_pure = jnp.squeeze(0.78 * z_lin + params.b2, axis=-1)
  a_mlp = _mlp_hidden(s, params.w1, params.b1, h_mask, inv_sqrt_h)
  out_mlp = 0.62 * out_lin + jnp.squeeze(jnp.matmul(a_mlp, params.w2), axis=-1)
  return jnp.where(is_linear, out_lin_pure, out_mlp)


def nlms_update(
    params: AgentParams,
    s_query: jax.Array,
    y_target: jax.Array,
    lr: jax.Array,
    nlms_eps: jax.Array,
    h_mask: jax.Array,
    inv_sqrt_h: jax.Array,
    is_linear: jax.Array,
    active: jax.Array = jnp.array(True),
    point_update: jax.Array = jnp.array(True),
    feat_weights: jax.Array | None = None,
) -> AgentParams:
  """Function-space normalized least mean squares (NLMS) step toward y_target at s_query."""
  def pred_fn(p):
    return eval_agent(p, s_query, h_mask, inv_sqrt_h, is_linear)

  pred, grads = jax.value_and_grad(pred_fn)(params)
  raw_err = jnp.where(active, pred - y_target, 0.0)
  fw = jnp.ones_like(params.w1[:, :1]) if feat_weights is None else feat_weights[:, None]
  did_move = jnp.max(fw) > 0.5
  eff_h = 1.0 / jnp.maximum(inv_sqrt_h * inv_sqrt_h, 1e-6)
  cap_mult = 0.55 + 0.16 * jnp.log2(jnp.maximum(eff_h, 2.0))
  w1_lin_w = jnp.where(
      is_linear,
      jnp.where(point_update, jnp.where(did_move, 0.22, 0.0), 0.52),
      0.72,
  )
  w1_mlp_w = jnp.where(is_linear, 0.0, 0.75 * cap_mult)
  w1_col_scale = jnp.where(jnp.arange(params.w1.shape[1])[None, :] == 0, w1_lin_w, w1_mlp_w)
  w2_weight = jnp.where(is_linear, 0.0, jnp.where(point_update, 0.78, 1.15) * cap_mult)
  b1_weight = jnp.where(is_linear, 0.0, 0.30 * cap_mult)
  scaled_grads = AgentParams(
      w1=grads.w1 * w1_col_scale * fw,
      b1=grads.b1 * b1_weight,
      w2=grads.w2 * w2_weight,
      b2=grads.b2,
  )
  dir_norms = (
      jnp.sum(grads.w1 * scaled_grads.w1)
      + jnp.sum(grads.b1 * scaled_grads.b1)
      + jnp.sum(grads.w2 * scaled_grads.w2)
  )
  # Attribute moved-decision updates to directional weights; when stationary, linear updates b2
  # while neural updates localized hidden readout weights w2 and applies gentle local habituation
  b2_stalled = jnp.where(is_linear, jnp.minimum(0.14, 0.55 * lr), jnp.minimum(0.28, 1.1 * lr))
  b2_moved = jnp.where(is_linear, jnp.minimum(0.10, 0.40 * lr), jnp.minimum(0.25, 1.0 * lr))
  b2_frac = jnp.where(
      point_update,
      jnp.where(did_move, b2_moved, b2_stalled),
      jnp.where(did_move, 0.0, jnp.minimum(0.35, 1.5 * lr)),
  )
  b2_step = b2_frac * raw_err
  rem_err = raw_err - b2_step
  dir_gain = jnp.where(point_update, jnp.where(did_move, 1.15, jnp.where(is_linear, 0.0, 0.90)), 0.95)
  reg_eps = jnp.where(is_linear, 0.30, 0.12)
  dir_step = (lr * dir_gain * rem_err) / (dir_norms + reg_eps + nlms_eps)
  stalled = active & point_update & (~did_move)
  habituation = jnp.where(stalled & (~is_linear), 0.045 * lr * cap_mult, 0.0)
  w1_decay = jnp.where(stalled, jnp.where(is_linear, 0.025 * lr, 0.015 * lr), 0.0)
  return AgentParams(
      w1=params.w1 * (1.0 - w1_decay) - dir_step * scaled_grads.w1,
      b1=params.b1 - dir_step * scaled_grads.b1,
      w2=params.w2 - dir_step * scaled_grads.w2 - habituation * grads.w2,
      b2=params.b2 - b2_step * scaled_grads.b2,
  )


# PRNG stream tags. Shared streams are keyed only by the firm's period key, so every
# configuration run on the same market sees identical draws (common random numbers).
_TAG_SHOCK = 101  # shared: turbulence timing (random mode) and redraws
_TAG_FIRM_NOISE = 102  # shared: firm-level observation noise
_TAG_DOMAIN_NOISE = 103  # shared: domain observation noise, keyed by feature subset
_TAG_METRICS = 104  # shared: random probe states for accuracy metrics
_TAG_SOCIAL = 201
_TAG_SEARCH = 202
_TAG_LEARN = 203
_TAG_EXPERIMENT = 204
_TAG_EXPERIMENT_NOISE = 205
_TAG_UNIT_SEARCH = 206
_TAG_NK_LANDSCAPE = 0x4E4B
_TAG_MISLEADING = 0x4D49  # informed nodes: misleading landscape and starting weights, keyed by node id hash and member

# Canonical hidden width for initialization, so a node's initial weights do not depend on the
# widest node elsewhere in the organization
_H_CANON = 128
_ASPIRATION_ALPHA = 0.1
_FEEDBACK_CODES = {"none": 0, "firm": 1, "domain": 2}
# Predictions computed in one batch can differ by float32 rounding (about 1e-7) across batch
# rows, so a candidate must beat the current strategy by more than this margin to count as an
# improvement (keeps changes that a representation's mask makes neutral from being adopted)
_DECISION_TOL = 1e-6
# Neural initial beliefs: full-batch Adam over the node's domain cells (deterministic, fixed steps)
# with a 10x rate for the biases the network scales by 0.1, then least squares for the output layer
_INIT_FIT_STEPS = 120
_INIT_FIT_LR = 0.05
_INIT_FIT_BIAS_LR = 0.10
_ADAM_B1 = 0.9
_ADAM_B2 = 0.999


class NKLandscape(NamedTuple):
  """Per-firm NK landscape state."""

  all_v: jax.Array  # [2^N] standardized firm payoff by strategy index
  v_max: jax.Array  # payoff of the best strategy
  v_min: jax.Array
  v_local_ref: jax.Array  # mean payoff reached by steepest-ascent local search from every strategy
  nk_tables: jax.Array  # [N, 2^(K+1)] contribution tables
  nk_inputs: jax.Array  # [N, K+1] decisions feeding each contribution (own decision first)
  c_mean: jax.Array  # [N] mean contribution over all strategies
  c_cov: jax.Array  # [N, N] covariance of contributions over all strategies


def _nk_modular_rows(n_bits: int, nk_k: int) -> list[list[int]]:
  """Consecutive blocks of size K+1; a short last block borrows the nearest preceding decisions."""
  size = nk_k + 1
  rows = []
  for i in range(n_bits):
    start = (i // size) * size
    if start + size > n_bits:
      start = n_bits - size
    rows.append([i] + [j for j in range(start, start + size) if j != i])
  return rows


def _nk_input_rows(key: jax.Array, n_bits: int, nk_k: int, pattern: str) -> jax.Array:
  """Return [n_bits, K+1] decision indices: each decision first, then its K partners."""
  own = jnp.arange(n_bits, dtype=jnp.int32)[:, None]
  if nk_k == 0:
    return own
  if pattern == "modular":
    return jnp.array(_nk_modular_rows(n_bits, nk_k), dtype=jnp.int32)

  def partners_for(i):
    perm = jax.random.permutation(jax.random.fold_in(key, i), n_bits - 1)[:nk_k]
    return jnp.where(perm >= i, perm + 1, perm).astype(jnp.int32)

  partners = jax.vmap(partners_for)(jnp.arange(n_bits, dtype=jnp.int32))
  return jnp.concatenate([own, partners], axis=1)


def _nk_contributions(tables: jax.Array, inputs: jax.Array, s: jax.Array) -> jax.Array:
  """Contributions c_i(s_i, s_{P_i}) for strategies s [..., N]; returns [..., N]."""
  n_in = inputs.shape[-1]
  pow2_in = (1 << jnp.arange(n_in - 1, -1, -1)).astype(jnp.int32)
  bits = (1.0 - s).astype(jnp.int32) >> 1
  idx = jnp.sum(bits[..., inputs] * pow2_in, axis=-1)
  return tables[jnp.arange(tables.shape[0]), idx]


def _local_search_reference(all_v: jax.Array) -> jax.Array:
  """Mean payoff reached by steepest-ascent one-bit hill climbing with perfect knowledge.

  Every strategy moves to its best one-bit neighbor while that neighbor pays strictly more.
  Pointer jumping finds each start's peak: after n_bits doublings a start has taken 2^n_bits
  steps, more than any strictly increasing path over 2^n_bits strategies can take. The reference
  is the mean peak payoff over all starts.
  """
  num_states = int(all_v.shape[0])
  n_bits = num_states.bit_length() - 1
  idx = jnp.arange(num_states, dtype=jnp.int32)
  nbrs = idx[:, None] ^ jnp.left_shift(jnp.int32(1), jnp.arange(n_bits, dtype=jnp.int32))[None, :]
  nbr_v = all_v[nbrs]
  best = jnp.take_along_axis(nbrs, jnp.argmax(nbr_v, axis=1)[:, None], axis=1)[:, 0]
  nxt = jnp.where(jnp.max(nbr_v, axis=1) > all_v, best, idx)
  for _ in range(n_bits):
    nxt = nxt[nxt]
  return jnp.mean(all_v[nxt])


def _payoff_percentile(all_v: jax.Array, v: jax.Array) -> jax.Array:
  """Share of the other strategies whose true payoff is lower than v (1 at the global optimum)."""
  return jnp.sum((all_v < v).astype(jnp.float32)) / float(all_v.shape[0] - 1)


def _nk_landscape_stats(tables: jax.Array, inputs: jax.Array, all_s: jax.Array):
  """Standardized payoffs, best and worst payoffs, the local-search reference, and contribution moments."""
  contrib = _nk_contributions(tables, inputs, all_s)
  f_raw = jnp.mean(contrib, axis=-1)
  all_v = (f_raw - jnp.mean(f_raw)) / (jnp.std(f_raw) + 1e-6)
  c_mean = jnp.mean(contrib, axis=0)
  c_dev = contrib - c_mean
  c_cov = jnp.matmul(c_dev.T, c_dev) / float(contrib.shape[0])
  return all_v, jnp.max(all_v), jnp.min(all_v), _local_search_reference(all_v), c_mean, c_cov


def init_nk_landscape(key: jax.Array, all_s: jax.Array, n_bits: int, nk_k: int, pattern: str) -> NKLandscape:
  """Draw one firm's NK landscape: K partners per decision and U(0, 1) contribution tables."""
  k_part, k_tab = jax.random.split(key)
  inputs = _nk_input_rows(k_part, n_bits, nk_k, pattern)
  tables = jax.random.uniform(k_tab, (n_bits, 1 << (nk_k + 1)), dtype=jnp.float32)
  all_v, v_max, v_min, v_local_ref, c_mean, c_cov = _nk_landscape_stats(tables, inputs, all_s)
  return NKLandscape(
      all_v=all_v,
      v_max=v_max,
      v_min=v_min,
      v_local_ref=v_local_ref,
      nk_tables=tables,
      nk_inputs=inputs,
      c_mean=c_mean,
      c_cov=c_cov,
  )


@lru_cache(maxsize=16)
def _landscape_init_fn(n_bits: int, nk_k: int, nk_pattern: str):
  """Compiled per-firm landscape draw, cached by landscape settings so a new seed does not recompile."""
  return jax.jit(jax.vmap(lambda k, s_all: init_nk_landscape(k, s_all, n_bits, nk_k, nk_pattern), in_axes=(0, None)))


@lru_cache(maxsize=32)
def build_landscape_batch(seed: int, b_firms: int, n_bits: int, nk_k: int, nk_pattern: str, backend: str = "") -> NKLandscape:
  """Per-firm NK landscapes; draws depend only on (seed, firm index, landscape settings)."""
  _, _, _, all_s, _, _ = get_hypercube_tables(n_bits, backend=backend)
  base_keys = jax.random.split(jax.random.PRNGKey(seed), b_firms)
  init_keys = jax.vmap(lambda k: jax.random.split(k, 2)[0])(base_keys)
  env_keys = jax.vmap(lambda k: jax.random.split(k, 6)[0])(init_keys)
  nk_keys = jax.vmap(lambda k: jax.random.fold_in(k, _TAG_NK_LANDSCAPE))(env_keys)
  envs = _landscape_init_fn(n_bits, nk_k, nk_pattern)(nk_keys, all_s)
  jax.block_until_ready(envs.v_max)
  return envs


def _shock_landscape(env: NKLandscape, key: jax.Array, all_s: jax.Array) -> NKLandscape:
  """Redraw each decision's contribution table with probability 0.5 (partners stay fixed)."""
  k_mask, k_new = jax.random.split(key)
  redraw = jax.random.uniform(k_mask, (env.nk_tables.shape[0],)) < 0.5
  fresh = jax.random.uniform(k_new, env.nk_tables.shape, dtype=jnp.float32)
  tables = jnp.where(redraw[:, None], fresh, env.nk_tables)
  all_v, v_max, v_min, v_local_ref, c_mean, c_cov = _nk_landscape_stats(tables, env.nk_inputs, all_s)
  return env._replace(
      all_v=all_v, v_max=v_max, v_min=v_min, v_local_ref=v_local_ref, nk_tables=tables, c_mean=c_mean, c_cov=c_cov
  )


class StepMetrics(NamedTuple):
  """Per-period metrics (one value per firm)."""

  v_true: jax.Array
  v_pct_max: jax.Array
  v_percentile: jax.Array  # share of the other strategies with a lower true payoff
  v_max_ref: jax.Array  # payoff of the landscape's best strategy
  v_local_ref: jax.Array  # mean payoff of steepest-ascent local search with perfect knowledge
  dc_mse: jax.Array
  dc_r_local: jax.Array
  at_local_peak: jax.Array  # no one-bit neighbor has a higher true payoff
  false_stall: jax.Array
  emp_mse: jax.Array
  emp_div: jax.Array
  emp_div_local_val: jax.Array
  emp_div_local_slope: jax.Array
  dc_acc_global: jax.Array
  dc_complexity: jax.Array
  orthogonality: jax.Array
  orthogonality_linked: jax.Array
  mean_permeability: jax.Array


class OrgState(NamedTuple):
  """Per-firm state of one organization."""

  s: jax.Array  # [N] enacted strategy
  params: AgentParams  # stacked over representations; index 0 is the dominant coalition
  tables: jax.Array  # [R, table_size] lookup tables of tabular representations
  env: NKLandscape  # current landscape (changes only through turbulence shocks)
  perm_mat: jax.Array  # [R, R] permeability P[a, b] of receiver a to source b
  beta_vec: jax.Array  # [R, R, N] commensurability distortion directions
  beta_scalar: jax.Array  # [R, R] commensurability distortion offsets
  aspiration: jax.Array  # DC aspiration level A
  y_dc_last: jax.Array  # DC's last observed firm payoff
  last_emergent: jax.Array  # [4] dc_acc_global, dc_complexity, orthogonality, orthogonality_linked
  init_acc: jax.Array  # [R] correlation of each initial representation with its observed payoff projected on its domain


class RunParams(NamedTuple):
  """Numeric parameters passed at run time (changing them does not recompile the kernel)."""

  sigma_obs: jax.Array
  p_shock: jax.Array
  shock_period: jax.Array
  rho: jax.Array
  kappa: jax.Array
  experiment_rate: jax.Array
  lambda_track: jax.Array
  rep_lr: jax.Array
  rep_sigma_learn: jax.Array
  rep_rho: jax.Array
  p_init: jax.Array
  c_mat: jax.Array


class _RepStatics(NamedTuple):
  """Per-representation constants used inside the kernel (leading axis = representation)."""

  feat_mask: jax.Array
  in_mask_int: jax.Array
  h_mask: jax.Array
  inv_sqrt_h: jax.Array
  is_lin: jax.Array
  is_tab: jax.Array


class _Kernels(NamedTuple):
  full: Any
  init: Any
  chunk: Any
  predict: Any


class _StaticSpec(NamedTuple):
  """Hashable settings fixed at compile time (the cache key of the organization kernels)."""

  t_steps: int
  eval_every: int
  n_bits: int
  turbulence_mode: str
  m_actors: int
  n_reps: int
  h_max: int
  decision_mode: str
  dc_search_bits: tuple[int, ...]
  search_steps: int
  has_track_record: bool
  has_experiments: bool
  feat_masks: tuple[tuple[float, ...], ...]
  in_masks: tuple[tuple[float, ...], ...]
  sub_rows: tuple[tuple[int, ...], ...]
  n_sub: tuple[int, ...]
  h_masks: tuple[tuple[float, ...], ...]
  inv_sqrt_h: tuple[float, ...]
  is_lin: tuple[bool, ...]
  is_tab: tuple[bool, ...]
  fb_codes: tuple[int, ...]
  mask_ints: tuple[int, ...]
  agg_codes: tuple[int, ...]
  rep_steps: tuple[int, ...]
  informed: tuple[bool, ...]
  src_inv_counts: tuple[float, ...]


class _PriorFit(NamedTuple):
  """Settings that fully determine one informed representation's fitted starting beliefs."""

  kind: str  # "neural", "linear", or "tabular"
  eff_h: int  # hidden width of the fitted network (1 for linear and tabular)
  feat_mask: tuple[float, ...]
  in_mask: tuple[float, ...]
  fb_code: int  # payoff the node learns from: 1 firm, otherwise its own domain
  node_hash: int  # keys the misleading landscape and the starting weights
  member: int  # member index within the node (count > 1)
  acc: float  # initial accuracy setting a


def _ordered_rep_nodes(graph: OrgGraph) -> list[OrgNode]:
  """Representation order: the DC, then the members of each unit, then the members of each manager."""
  ordered = [n for n in graph.nodes if n.role == "unit"] + [n for n in graph.nodes if n.role == "manager"]
  reps = [graph.get_hq_node()]
  for node in ordered:
    reps.extend([node] * node.count)
  return reps


def graph_representation_ids(graph: OrgGraph) -> list[str]:
  """Node id behind each representation index (index 0 is the DC; a node with count n appears n times)."""
  return [n.id for n in _ordered_rep_nodes(graph)]


def _node_hash(node_id: str) -> int:
  """Stable 31-bit hash of a node id (keys the node's misleading landscape and starting weights)."""
  return zlib.crc32(node_id.encode("utf-8")) & 0x7FFFFFFF


def _compile_spec(
    graph: OrgGraph, market: MarketConfig
) -> tuple[_StaticSpec, RunParams, tuple[tuple[int, _PriorFit], ...]]:
  """Split an organization and market into a static spec, run-time parameters, and prior fits."""
  graph.validate(n_bits=market.n_bits)
  n_bits = market.n_bits
  hq = graph.get_hq_node()
  rep_nodes = _ordered_rep_nodes(graph)
  n_reps = len(rep_nodes)
  node_reps: dict[str, list[int]] = {}
  members: list[int] = []
  for idx, node in enumerate(rep_nodes):
    members.append(len(node_reps.setdefault(node.id, [])))
    node_reps[node.id].append(idx)
  h_max = max([4] + [n.h_agent for n in rep_nodes])

  # Initial permeability P_init[a, b] (b -> a) and symmetric commensurability C[a, b]
  p_init = [[0.0] * n_reps for _ in range(n_reps)]
  c_mat = [[0.85 if a != b else 1.0 for b in range(n_reps)] for a in range(n_reps)]
  for edge in graph.edges:
    src_reps = node_reps[edge.source]
    w_per_src = edge.weight / float(len(src_reps))
    for a in node_reps[edge.target]:
      for b in src_reps:
        if a != b:
          p_init[a][b] = w_per_src
          c_mat[a][b] = edge.commensurability
          c_mat[b][a] = edge.commensurability

  node_feedback = graph.effective_feedback()
  node_accuracy = graph.effective_initial_accuracy()
  full_set = set(range(n_bits))
  feat_masks, in_masks, sub_rows, n_sub, h_masks, inv_sqrt_h = [], [], [], [], [], []
  is_lin, is_tab, fb_codes, mask_ints, agg_codes = [], [], [], [], []
  for r_node in rep_nodes:
    bits = sorted(set(r_node.jurisdiction_bits))
    bset = set(bits)
    scale_in = math.sqrt(float(n_bits) / float(len(bits)))
    feat_masks.append(tuple(scale_in if b in bset else 0.0 for b in range(n_bits)))
    in_masks.append(tuple(1.0 if b in bset else 0.0 for b in range(n_bits)))
    sub_rows.append(tuple(bits[j % len(bits)] for j in range(n_bits)))
    n_sub.append(len(bits))
    lin = r_node.model_type == "linear"
    eff_h = 1 if lin else max(1, r_node.h_agent)
    h_masks.append(tuple(1.0 if k < eff_h else 0.0 for k in range(h_max)))
    inv_sqrt_h.append(1.0 / math.sqrt(eff_h))
    is_lin.append(lin)
    is_tab.append(r_node.model_type == "tabular")
    fb = node_feedback[r_node.id]
    # A domain covering every decision is the firm payoff, so it shares the firm signal and noise
    if fb == "domain" and bset == full_set:
      fb = "firm"
    fb_codes.append(_FEEDBACK_CODES[fb])
    mask_ints.append(sum(1 << b for b in bits))
    agg_codes.append(1 if r_node.upward_agg == "max" else (2 if r_node.upward_agg == "min" else 0))

  informed = [node_accuracy[n.id] is not None for n in rep_nodes]
  # (representation index, fit settings) of every informed representation, in representation order
  prior_jobs = tuple(
      (
          r,
          _PriorFit(
              kind="tabular" if is_tab[r] else ("linear" if is_lin[r] else "neural"),
              eff_h=1 if (is_tab[r] or is_lin[r]) else max(1, r_node.h_agent),
              feat_mask=feat_masks[r],
              in_mask=in_masks[r],
              fb_code=fb_codes[r],
              node_hash=_node_hash(r_node.id),
              member=members[r],
              acc=float(node_accuracy[r_node.id]),
          ),
      )
      for r, r_node in enumerate(rep_nodes)
      if informed[r]
  )
  decision_mode = graph.effective_decision_rights()
  # Non-DC searchers (propose_approve, delegated) use the DC's search rule with their own settings
  rep_steps = (
      tuple(n.search_steps for n in rep_nodes) if decision_mode != "centralized" else (1,) * n_reps
  )
  if market.turbulence == "random" and market.p_shock > 0.0:
    turbulence_mode = "random"
  elif market.turbulence == "periodic":
    turbulence_mode = "periodic"
  else:
    turbulence_mode = "none"
  has_edges = any(p_init[a][b] > 0.0 for a in range(n_reps) for b in range(n_reps))
  static_spec = _StaticSpec(
      t_steps=market.t_steps,
      eval_every=int(market.eval_every),
      n_bits=n_bits,
      turbulence_mode=turbulence_mode,
      m_actors=n_reps - 1,
      n_reps=n_reps,
      h_max=h_max,
      decision_mode=decision_mode,
      dc_search_bits=tuple(graph.dc_search_bits()),
      search_steps=hq.search_steps,
      has_track_record=graph.lambda_track > 0.0 and has_edges,
      has_experiments=graph.experiment_rate > 0.0 and any(c > 0 for c in fb_codes[1:]),
      feat_masks=tuple(feat_masks),
      in_masks=tuple(in_masks),
      sub_rows=tuple(sub_rows),
      n_sub=tuple(n_sub),
      h_masks=tuple(h_masks),
      inv_sqrt_h=tuple(inv_sqrt_h),
      is_lin=tuple(is_lin),
      is_tab=tuple(is_tab),
      fb_codes=tuple(fb_codes),
      mask_ints=tuple(mask_ints),
      agg_codes=tuple(agg_codes),
      rep_steps=rep_steps,
      informed=tuple(informed),
      src_inv_counts=tuple(1.0 / float(n.count) for n in rep_nodes),
  )
  run_params = RunParams(
      sigma_obs=jnp.asarray(market.sigma_obs, dtype=jnp.float32),
      p_shock=jnp.asarray(market.p_shock, dtype=jnp.float32),
      shock_period=jnp.asarray(market.shock_period, dtype=jnp.int32),
      rho=jnp.asarray(hq.rho, dtype=jnp.float32),
      kappa=jnp.asarray(graph.aspiration_sensitivity, dtype=jnp.float32),
      experiment_rate=jnp.asarray(graph.experiment_rate, dtype=jnp.float32),
      lambda_track=jnp.asarray(graph.lambda_track, dtype=jnp.float32),
      rep_lr=jnp.asarray([n.lr for n in rep_nodes], dtype=jnp.float32),
      rep_sigma_learn=jnp.asarray([n.sigma_learn for n in rep_nodes], dtype=jnp.float32),
      rep_rho=jnp.asarray([n.rho for n in rep_nodes], dtype=jnp.float32),
      p_init=jnp.asarray(p_init, dtype=jnp.float32),
      c_mat=jnp.asarray(c_mat, dtype=jnp.float32),
  )
  return static_spec, run_params, prior_jobs


# Starting beliefs of informed representations. Each representation is fit by its own compiled
# program, cached by the node's settings and fed only inputs that do not depend on the rest of the
# organization (firm key, landscape, node id hash, member index, accuracy). A node with the same
# id and settings therefore starts from bit-identical beliefs in every organization, at any
# position, and in every run path (full runs and streams).


def _fit_network(p: AgentParams, x: jax.Array, h_mask: jax.Array, inv_sqrt_h: jax.Array, target: jax.Array) -> AgentParams:
  """Least-squares fit of a neural representation at inputs x [M, N] to target [M] (deterministic).

  Full-batch Adam for a fixed number of steps, with a larger rate for the biases (the network
  scales them by 0.1), then an exact least-squares solve of the output layer given the hidden layer.
  """
  not_linear = jnp.array(False)
  rates = AgentParams(w1=_INIT_FIT_LR, b1=_INIT_FIT_BIAS_LR, w2=_INIT_FIT_LR, b2=_INIT_FIT_BIAS_LR)

  def loss(q):
    return jnp.mean(jnp.square(eval_agent(q, x, h_mask, inv_sqrt_h, not_linear) - target))

  grad_fn = jax.grad(loss)
  zeros = jax.tree_util.tree_map(jnp.zeros_like, p)

  def body(i, carry):
    q, m, v = carry
    g = grad_fn(q)
    m = jax.tree_util.tree_map(lambda a, b: _ADAM_B1 * a + (1.0 - _ADAM_B1) * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: _ADAM_B2 * a + (1.0 - _ADAM_B2) * b * b, v, g)
    step = i.astype(jnp.float32) + 1.0
    c1 = 1.0 - jnp.power(_ADAM_B1, step)
    c2 = 1.0 - jnp.power(_ADAM_B2, step)
    q = jax.tree_util.tree_map(
        lambda w, a, b, lr: w - lr * (a / c1) / (jnp.sqrt(b / c2) + 1e-8), q, m, v, rates
    )
    return q, m, v

  fitted, _, _ = jax.lax.fori_loop(0, _INIT_FIT_STEPS, body, (p, zeros, zeros))
  # Output layer: exact least squares given the hidden layer (a tiny ridge keeps it well posed)
  hidden = _mlp_hidden(x, fitted.w1, fitted.b1, h_mask, inv_sqrt_h)
  design = jnp.concatenate([hidden, jnp.ones((x.shape[0], 1), dtype=hidden.dtype)], axis=1)
  gram = jnp.matmul(design.T, design)
  gram = gram + 1e-6 * (jnp.trace(gram) / gram.shape[0]) * jnp.eye(gram.shape[0], dtype=gram.dtype)
  coef = jnp.linalg.solve(gram, jnp.matmul(design.T, target))
  return fitted._replace(w2=coef[:-1, None], b2=coef[-1:])


def _least_squares_linear(p: AgentParams, x_all: jax.Array, target: jax.Array, n_bits: int) -> AgentParams:
  """Exact least squares for a linear representation (domain columns are orthogonal over all strategies)."""
  denom = jnp.sum(x_all * x_all, axis=0)
  beta = jnp.where(denom > 0.0, jnp.matmul(target, x_all) / jnp.maximum(denom, 1e-12), 0.0)
  slope = (beta / 0.78) * jnp.sqrt(float(n_bits))
  return p._replace(
      w1=p.w1.at[:, 0].set(slope),
      b2=p.b2.at[0].set(jnp.mean(target)),
  )


def _segment_mean_table(target: jax.Array, idx: jax.Array, num_states: int) -> jax.Array:
  """Table entries equal the target averaged over strategies that share the node's domain values."""
  sums = jax.ops.segment_sum(target, idx, num_segments=num_states)
  counts = jax.ops.segment_sum(jnp.ones_like(target), idx, num_segments=num_states)
  return jnp.where(counts > 0.0, sums / jnp.maximum(counts, 1.0), 0.0)


@lru_cache(maxsize=64)
def _build_prior_fit(n_bits: int, kind: str, eff_h: int, fb_code: int, n_sub: int, backend: str = ""):
  """Compile the starting-belief fit for one kind of informed representation, vectorized over firms.

  The compiled function maps (base keys [B], landscapes [B], feature mask [N], domain mask [N],
  accuracy, node id hash, member) to the fitted network (hidden width eff_h) and table [B, 2^N]
  (tabular nodes; [B, 1] otherwise). Masks and keys are inputs, so nodes that differ only in
  their domain, id, or accuracy share one compiled program; n_sub is the domain size.
  """
  num_states, _, pow2, all_s, _, _ = get_hypercube_tables(n_bits, backend=backend)
  bits_all = (1.0 - all_s).astype(jnp.int32) >> 1
  n_cells = 1 << n_sub
  cell_ids = jnp.arange(n_cells, dtype=jnp.int32)
  h_mask = jnp.ones((eff_h,), dtype=jnp.float32)
  inv_sqrt_h = jnp.asarray(1.0 / math.sqrt(eff_h), dtype=jnp.float32)
  no_table = jnp.zeros((1,), dtype=jnp.float32)

  def observed(env: NKLandscape, dom_w: jax.Array) -> jax.Array:
    """Standardized payoff the node learns from: the firm payoff, or its own domain average."""
    if fb_code == 1:
      return env.all_v
    contrib = _nk_contributions(env.nk_tables, env.nk_inputs, all_s)
    mu = jnp.dot(dom_w, env.c_mean)
    sd = jnp.sqrt(jnp.maximum(jnp.dot(jnp.matmul(dom_w, env.c_cov), dom_w), 0.0)) + 1e-6
    return (jnp.matmul(contrib, dom_w) - mu) / sd

  def misleading(env: NKLandscape, dom_w: jax.Array, key: jax.Array) -> jax.Array:
    """Fresh U(0, 1) contribution tables on the same partners, aggregated like observed()."""
    tables = jax.random.uniform(key, env.nk_tables.shape, dtype=jnp.float32)
    contrib = _nk_contributions(tables, env.nk_inputs, all_s)
    raw = jnp.mean(contrib, axis=-1) if fb_code == 1 else jnp.matmul(contrib, dom_w)
    return (raw - jnp.mean(raw)) / (jnp.std(raw) + 1e-6)

  def domain_cells(feat_mask: jax.Array, in_mask: jax.Array, target: jax.Array, k_sub: jax.Array):
    """Masked inputs of the 2^n_sub domain configurations and the target averaged within each.

    A masked representation depends only on its domain decisions, so its squared error over all
    strategies equals the error over these cells against the cell means plus a constant (equal
    cell sizes give the same gradient). When 2^n_sub > 4096 (n_bits >= 13 with a large domain),
    subsamples 4096 cells to bound memory and compile/run time.
    """
    if n_cells > 4096 and n_sub == n_bits:
      sub_idx = jax.random.choice(k_sub, num_states, shape=(4096,), replace=False)
      return all_s[sub_idx] * feat_mask, target[sub_idx]
    in_int = in_mask.astype(jnp.int32)
    rank = jnp.maximum(jnp.cumsum(in_int) - 1, 0)
    weight = jnp.where(in_int > 0, jnp.left_shift(jnp.int32(1), rank), 0)
    cell_of_s = jnp.sum(bits_all * weight, axis=-1)
    cell_target = jax.ops.segment_sum(target, cell_of_s, num_segments=n_cells) / float(num_states // n_cells)
    cell_bits = jnp.right_shift(cell_ids[:, None], rank[None, :]) & 1
    x_cells = (1.0 - 2.0 * cell_bits.astype(jnp.float32)) * feat_mask
    if n_cells > 4096:
      sub_idx = jax.random.choice(k_sub, n_cells, shape=(4096,), replace=False)
      return x_cells[sub_idx], cell_target[sub_idx]
    return x_cells, cell_target

  def fit_one(base_key, env, feat_mask, in_mask, acc, node_hash, member):
    dom_w = in_mask / jnp.sum(in_mask)
    # Same firm key derivation as the organization kernels (init key, then its fifth subkey)
    k_init = jax.random.split(base_key, 2)[0]
    k_mis = jax.random.fold_in(jax.random.split(k_init, 8)[4], _TAG_MISLEADING)
    k_node = jax.random.fold_in(jax.random.fold_in(k_mis, node_hash), member)
    mix = jnp.sqrt(jnp.maximum(1.0 - acc * acc, 0.0))
    target = INIT_BELIEF_SCALE * (acc * observed(env, dom_w) + mix * misleading(env, dom_w, jax.random.fold_in(k_node, 0)))
    start = init_agent(jax.random.fold_in(k_node, 1), n_bits, _H_CANON)
    start = AgentParams(w1=start.w1[:, :eff_h], b1=start.b1[:eff_h], w2=start.w2[:eff_h], b2=start.b2)
    if kind == "tabular":
      domain_idx = jnp.sum(bits_all * pow2 * in_mask.astype(jnp.int32), axis=-1)
      return start, _segment_mean_table(target, domain_idx, num_states)
    if kind == "linear":
      fitted_lin = _least_squares_linear(start, all_s * feat_mask, target, n_bits)
      amp = 0.20 + 0.80 * acc * acc
      return fitted_lin._replace(w1=fitted_lin.w1 * amp, b2=fitted_lin.b2 * amp), no_table
    x_cells, cell_target = domain_cells(feat_mask, in_mask, target, jax.random.fold_in(k_node, 2))
    fitted_net = _fit_network(start, x_cells, h_mask, inv_sqrt_h, cell_target)
    # Match uninformed weight scale at acc = 0 while preserving exact target correlation across acc in [0, 1]
    if n_sub == n_bits:
      blend_rate = 0.22 + (0.56 + 0.06 * jnp.clip((acc - 0.72) / 0.28, 0.0, 1.0)) * acc
      ridge_mult = 1e-4
      amp_num = 0.18 + 0.38 * acc
    else:
      blend_rate = 0.05 + 0.95 * acc * acc
      ridge_mult = 0.06 * (1.0 - acc * acc) + 1e-4
      amp_num = 0.08 + 0.92 * acc * acc
    blend_w1 = start.w1 + blend_rate * (fitted_net.w1 - start.w1)
    blend_b1 = start.b1 + blend_rate * (fitted_net.b1 - start.b1)
    hidden = _mlp_hidden(x_cells, blend_w1, blend_b1, h_mask, inv_sqrt_h)
    design = jnp.concatenate([hidden, jnp.ones((x_cells.shape[0], 1), dtype=hidden.dtype)], axis=1)
    gram = jnp.matmul(design.T, design)
    ridge = ridge_mult * (jnp.trace(gram) / gram.shape[0])
    gram = gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype)
    raw_coef = jnp.linalg.solve(gram, jnp.matmul(design.T, cell_target))
    pred_std = jnp.std(jnp.matmul(design, raw_coef)) + 1e-6
    amp = amp_num / jnp.maximum(pred_std, 0.50)
    if n_sub == n_bits:
      lin_w = jnp.clip((acc - 0.72) / 0.28, 0.0, 1.0)
      s0 = jnp.where(raw_coef[0] >= 0.0, 1.0, -1.0)
      z_lin = jnp.squeeze(jnp.matmul(x_cells, blend_w1[..., :1]), axis=-1) / jnp.sqrt(float(n_bits))
      lin_part = lin_w * 0.62 * (0.5 * s0) * z_lin
      coef = jnp.linalg.solve(gram, jnp.matmul(design.T, amp * cell_target - lin_part))
      coef_0 = jnp.where(s0 >= 0.0, jnp.clip(coef[0], 1e-4, 0.5), jnp.clip(coef[0], -0.5, -1e-4))
      coef = jnp.where(lin_w > 0.0, coef.at[0].set(coef_0), raw_coef * amp)
      b2_scale = 1.0 - lin_w * 0.38
      return AgentParams(w1=blend_w1, b1=blend_b1, w2=coef[:-1, None], b2=coef[-1:] / b2_scale), no_table
    coef = raw_coef * amp
    return AgentParams(w1=blend_w1, b1=blend_b1, w2=coef[:-1, None], b2=coef[-1:]), no_table

  return jax.jit(jax.vmap(fit_one, in_axes=(0, 0, None, None, None, None, None)))


def _compute_prior(fit_spec: _PriorFit, landscape: tuple[Any, ...], backend: str = "") -> tuple[AgentParams, jax.Array]:
  """Fitted network and table of one informed representation on every firm of a market."""
  n_bits, nk_k, nk_pattern, seed, b_firms = landscape
  envs = build_landscape_batch(seed, b_firms, n_bits, nk_k, nk_pattern, backend)
  base_keys = jax.random.split(jax.random.PRNGKey(seed), b_firms)
  n_sub = int(round(sum(fit_spec.in_mask)))
  fit = _build_prior_fit(n_bits, fit_spec.kind, fit_spec.eff_h, fit_spec.fb_code, n_sub, backend)
  return fit(
      base_keys,
      envs,
      jnp.asarray(fit_spec.feat_mask, dtype=jnp.float32),
      jnp.asarray(fit_spec.in_mask, dtype=jnp.float32),
      jnp.float32(fit_spec.acc),
      jnp.int32(fit_spec.node_hash),
      jnp.int32(fit_spec.member),
  )


# Fitted priors in least-recently-used order, bounded by the total size of their arrays
_PRIOR_CACHE: "OrderedDict[tuple[Any, ...], tuple[AgentParams, jax.Array]]" = OrderedDict()
_PRIOR_CACHE_MAX_BYTES = 512 * 1024 * 1024


def _fit_prior(fit_spec: _PriorFit, landscape: tuple[Any, ...], backend: str = "") -> tuple[AgentParams, jax.Array]:
  """Cached _compute_prior (landscape = _landscape_signature of the market).

  The fit is a pure function of its arguments, so configurations, reruns, and sweep points that
  share the node's settings and the landscape reuse the same arrays instead of refitting. The
  least recently used entries are evicted once the cached arrays exceed _PRIOR_CACHE_MAX_BYTES
  (the newest entry is always kept).
  """
  key = (fit_spec, landscape, backend)
  if key in _PRIOR_CACHE:
    _PRIOR_CACHE.move_to_end(key)
    return _PRIOR_CACHE[key]
  result = _compute_prior(fit_spec, landscape, backend)
  _PRIOR_CACHE[key] = result

  def total_bytes() -> int:
    return sum(leaf.nbytes for entry in _PRIOR_CACHE.values() for leaf in jax.tree_util.tree_leaves(entry))

  while len(_PRIOR_CACHE) > 1 and total_bytes() > _PRIOR_CACHE_MAX_BYTES:
    _PRIOR_CACHE.popitem(last=False)
  return result


def _initial_priors(
    spec: _StaticSpec,
    jobs: tuple[tuple[int, _PriorFit], ...],
    market: MarketConfig,
    backend: str = "",
) -> tuple[AgentParams, jax.Array]:
  """Fitted starting beliefs of the informed representations, laid out for the kernels.

  Returns the networks of informed neural and linear representations (stacked on axis 1 in
  representation order, hidden width padded with zeros to the kernel's width) and the tables of
  informed tabular representations [B, n, table_size].
  """
  b_firms = market.b_firms
  table_size = (1 << spec.n_bits) if any(spec.is_tab) else 1
  nets: list[AgentParams] = []
  tabs: list[jax.Array] = []
  for _, fit_spec in jobs:
    params, table = _fit_prior(fit_spec, _landscape_signature(market), backend)
    if fit_spec.kind == "tabular":
      tabs.append(table)
      continue
    pad = spec.h_max - fit_spec.eff_h
    nets.append(
        AgentParams(
            w1=jnp.pad(params.w1, ((0, 0), (0, 0), (0, pad))),
            b1=jnp.pad(params.b1, ((0, 0), (0, pad))),
            w2=jnp.pad(params.w2, ((0, 0), (0, pad), (0, 0))),
            b2=params.b2,
        )
    )
  if nets:
    net_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=1), *nets)
  else:
    h_max = spec.h_max
    net_stack = AgentParams(
        w1=jnp.zeros((b_firms, 0, spec.n_bits, h_max), dtype=jnp.float32),
        b1=jnp.zeros((b_firms, 0, h_max), dtype=jnp.float32),
        w2=jnp.zeros((b_firms, 0, h_max, 1), dtype=jnp.float32),
        b2=jnp.zeros((b_firms, 0, 1), dtype=jnp.float32),
    )
  tab_stack = jnp.stack(tabs, axis=1) if tabs else jnp.zeros((b_firms, 0, table_size), dtype=jnp.float32)
  return net_stack, tab_stack


@lru_cache(maxsize=128)
def _build_kernels(static_spec: _StaticSpec, backend: str = "") -> _Kernels:
  """Compile and cache the vectorized kernels (full run, init, chunked steps, predict)."""
  (
      t_steps,
      eval_every,
      n_bits,
      turbulence_mode,
      m_actors,
      n_reps,
      h_max,
      decision_mode,
      dc_search_bits,
      search_steps,
      has_track_record,
      has_experiments,
      feat_masks_tup,
      in_masks_tup,
      sub_rows_tup,
      n_sub_tup,
      h_masks_tup,
      inv_sqrt_h_tup,
      is_lin_tup,
      is_tab_tup,
      fb_codes_tup,
      mask_ints_tup,
      agg_codes_tup,
      rep_steps_tup,
      informed_tup,
      src_inv_counts_tup,
  ) = static_spec

  num_states, _, pow2, all_s, flip_eye, neighbor_idx = get_hypercube_tables(n_bits, backend=backend)
  any_tabular = any(is_tab_tup)
  table_size = num_states if any_tabular else 1
  any_domain = any(c == 2 for c in fb_codes_tup)
  unit_steps = jnp.array(rep_steps_tup, dtype=jnp.int32)
  unit_steps_max = max(rep_steps_tup[1:]) if m_actors > 0 else 1

  in_masks = jnp.array(in_masks_tup, dtype=jnp.float32)
  rep_consts = _RepStatics(
      feat_mask=jnp.array(feat_masks_tup, dtype=jnp.float32),
      in_mask_int=in_masks.astype(jnp.int32),
      h_mask=jnp.array(h_masks_tup, dtype=jnp.float32),
      inv_sqrt_h=jnp.array(inv_sqrt_h_tup, dtype=jnp.float32),
      is_lin=jnp.array(is_lin_tup, dtype=jnp.bool_),
      is_tab=jnp.array(is_tab_tup, dtype=jnp.bool_),
  )
  sub_rows = jnp.array(sub_rows_tup, dtype=jnp.int32)
  n_sub = jnp.array(n_sub_tup, dtype=jnp.int32)
  fb_codes = jnp.array(fb_codes_tup, dtype=jnp.int32)
  mask_ints = jnp.array(mask_ints_tup, dtype=jnp.int32)
  agg_codes = jnp.array(agg_codes_tup, dtype=jnp.int32)
  src_inv_counts = jnp.array(src_inv_counts_tup, dtype=jnp.float32)
  team_boost = jnp.power(1.0 / jnp.maximum(src_inv_counts, 1e-6), 0.20)
  rep_ids = jnp.arange(n_reps, dtype=jnp.int32)
  sees_market = fb_codes > 0
  dom_w = in_masks / jnp.array(n_sub_tup, dtype=jnp.float32)[:, None]
  dc_in_mask = in_masks[0]
  complexity_scale = 1.0 / float(2 ** (n_bits - n_sub_tup[0]))
  n_search = len(dc_search_bits)
  dc_search_arr = jnp.array(dc_search_bits if dc_search_bits else (0,), dtype=jnp.int32)
  dc_search_mask = jnp.array([1.0 if b in dc_search_bits else 0.0 for b in range(n_bits)], dtype=jnp.float32)
  off_diag = ~jnp.eye(n_reps, dtype=jnp.bool_)
  n_pairs = float(max(1, n_reps * (n_reps - 1)))
  # Informed representations take fitted starting beliefs (see _initial_priors): networks for
  # neural and linear ones, tables for tabular ones, in representation order
  prior_net_reps = tuple(r for r in range(n_reps) if informed_tup[r] and not is_tab_tup[r])
  prior_table_reps = tuple(r for r in range(n_reps) if informed_tup[r] and is_tab_tup[r])
  informed_arr = jnp.array(informed_tup, dtype=jnp.bool_)
  cell_sizes = jnp.array([float(1 << (n_bits - s)) for s in n_sub_tup], dtype=jnp.float32)[:, None]

  def s_to_idx(s: jax.Array) -> jax.Array:
    bits = (1.0 - s).astype(jnp.int32) >> 1
    return jnp.sum(bits * pow2, axis=-1)

  def eval_rep(p: AgentParams, tab: jax.Array, rc: _RepStatics, s_q: jax.Array) -> jax.Array:
    """One representation's prediction at strategies s_q [..., N] through its feature mask."""
    net = eval_agent(p, s_q * rc.feat_mask, rc.h_mask, rc.inv_sqrt_h, rc.is_lin)
    if any_tabular:
      bits = (1.0 - s_q).astype(jnp.int32) >> 1
      idx = jnp.sum(bits * pow2 * rc.in_mask_int, axis=-1)
      return jnp.where(rc.is_tab, tab[idx], net)
    return net

  def eval_all(params: AgentParams, tables: jax.Array, s_q: jax.Array) -> jax.Array:
    """Every representation at shared strategies s_q [..., N]; returns [R, ...]."""
    return jax.vmap(eval_rep, in_axes=(0, 0, 0, None))(params, tables, rep_consts, s_q)

  def eval_dc(params: AgentParams, tables: jax.Array, s_q: jax.Array) -> jax.Array:
    p0 = jax.tree_util.tree_map(lambda x: x[0], params)
    rc0 = jax.tree_util.tree_map(lambda x: x[0], rep_consts)
    return eval_rep(p0, tables[0], rc0, s_q)

  def learn_rep(p, tab, rc, s_q, y, lr, active, point_update=jnp.array(True), feat_weights=None):
    """One masked NLMS step (or table update) toward target y at strategy s_q when active."""
    p_new = nlms_update(
        p,
        s_q * rc.feat_mask,
        y,
        lr,
        1e-4,
        rc.h_mask,
        rc.inv_sqrt_h,
        rc.is_lin,
        active=active & ~rc.is_tab,
        point_update=point_update,
        feat_weights=feat_weights,
    )
    if any_tabular:
      bits = (1.0 - s_q).astype(jnp.int32) >> 1
      idx = jnp.sum(bits * pow2 * rc.in_mask_int)
      old_v = tab[idx]
      tab = tab.at[idx].set(jnp.where(active & rc.is_tab, (1.0 - lr) * old_v + lr * y, old_v))
    return p_new, tab

  # Shared strategy with per-representation targets, and per-representation strategies
  learn_all = jax.vmap(learn_rep, in_axes=(0, 0, 0, None, 0, 0, 0, None, 0))
  learn_all_each = jax.vmap(learn_rep, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0))

  def domain_payoffs(env: NKLandscape, s: jax.Array) -> jax.Array:
    """Standardized own-domain payoffs F_a for each representation at s [N] or [R, N]."""
    contrib = _nk_contributions(env.nk_tables, env.nk_inputs, s)
    raw = jnp.sum(dom_w * contrib, axis=-1)
    mu = jnp.matmul(dom_w, env.c_mean)
    var = jnp.sum(jnp.matmul(dom_w, env.c_cov) * dom_w, axis=-1)
    return (raw - mu) / (jnp.sqrt(jnp.maximum(var, 0.0)) + 1e-6)

  def market_signals(env: NKLandscape, s: jax.Array, k_firm: jax.Array, k_dom: jax.Array, sigma_obs: jax.Array):
    """Firm signal, each representation's observed signal (firm or own domain), and noiseless domain payoff at s."""
    v_curr = env.all_v[s_to_idx(s)]
    y_firm = v_curr + sigma_obs * jax.random.normal(k_firm, ())
    if not any_domain:
      return y_firm, jnp.full((n_reps,), y_firm), jnp.full((n_reps,), v_curr)
    # One draw per distinct feature subset, so members of a node share their domain noise
    eps_dom = jax.vmap(lambda m: jax.random.normal(jax.random.fold_in(k_dom, m), ()))(mask_ints)
    dom_curr = domain_payoffs(env, s)
    y_dom = dom_curr + sigma_obs * eps_dom
    return y_firm, jnp.where(fb_codes == 2, y_dom, y_firm), dom_curr

  def rep_normals(key: jax.Array) -> jax.Array:
    """Independent standard normal per representation, keyed by representation index."""
    return jax.vmap(lambda r: jax.random.normal(jax.random.fold_in(key, r), ()))(rep_ids)

  def init_rep_params(key: jax.Array) -> AgentParams:
    base = init_agent(key, n_bits, _H_CANON)
    return AgentParams(w1=base.w1[:, :h_max], b1=base.b1[:h_max], w2=base.w2[:h_max], b2=base.b2)

  def observed_payoffs_all(env: NKLandscape) -> jax.Array:
    """[R, S] standardized payoff each representation observes: firm payoff, or its own domain."""
    contrib = _nk_contributions(env.nk_tables, env.nk_inputs, all_s)
    mu = jnp.matmul(dom_w, env.c_mean)
    sd = jnp.sqrt(jnp.maximum(jnp.sum(jnp.matmul(dom_w, env.c_cov) * dom_w, axis=-1), 0.0)) + 1e-6
    dom = ((jnp.matmul(contrib, dom_w.T) - mu) / sd).T
    return jnp.where((fb_codes == 1)[:, None], env.all_v[None, :], dom)

  def realized_accuracy(params, tables, env) -> jax.Array:
    """[R] correlation over all strategies between each representation and what it can know.

    The reference is the representation's observed payoff averaged over the decisions outside its
    domain (P z_obs), the best any function of its own decisions can match; for a full domain it
    is the observed payoff itself. Because a representation depends only on its own decisions,
    corr(f, z_obs) = corr(z_obs, P z_obs) * corr(f, P z_obs).
    """
    f_c = eval_all(params, tables, all_s)
    z_obs = observed_payoffs_all(env)
    rep_domain_idx = jnp.sum(
        (((1.0 - all_s).astype(jnp.int32) >> 1) * pow2)[None, :, :] * rep_consts.in_mask_int[:, None, :], axis=-1
    )
    cell_sums = jax.vmap(lambda v, idx: jax.ops.segment_sum(v, idx, num_segments=num_states))(z_obs, rep_domain_idx)
    cell_means = cell_sums / cell_sizes
    z_c = jnp.take_along_axis(cell_means, rep_domain_idx, axis=1)
    f_c = f_c - jnp.mean(f_c, axis=-1, keepdims=True)
    z_c = z_c - jnp.mean(z_c, axis=-1, keepdims=True)
    den = jnp.sqrt(jnp.sum(f_c * f_c, axis=-1) * jnp.sum(z_c * z_c, axis=-1))
    return jnp.where(den > 1e-12, jnp.sum(f_c * z_c, axis=-1) / jnp.maximum(den, 1e-12), 0.0)

  def eval_emergent(state: OrgState, run: RunParams) -> jax.Array:
    """Global accuracy, perceived ruggedness, and orthogonality over all 2^N strategies."""
    f_all = eval_all(state.params, state.tables, all_s)
    f_dc = f_all[0]
    v_all = state.env.all_v
    dc_c = f_dc - jnp.mean(f_dc)
    v_c = v_all - jnp.mean(v_all)
    dc_acc_global = jnp.sum(dc_c * v_c) / (jnp.sqrt(jnp.sum(jnp.square(dc_c)) * jnp.sum(jnp.square(v_c))) + 1e-6)
    # Strict local maxima of the DC's representation over flips inside its own domain
    beats = (f_dc[:, None] > f_dc[neighbor_idx]) | (dc_in_mask[None, :] < 0.5)
    dc_complexity = jnp.sum(jnp.all(beats, axis=-1)).astype(jnp.float32) * complexity_scale
    if m_actors == 0:
      zero = jnp.array(0.0, dtype=jnp.float32)
      return jnp.stack([dc_acc_global, dc_complexity, zero, zero])
    f_mean = jnp.mean(f_all, axis=-1, keepdims=True)
    f_std = jnp.std(f_all, axis=-1, keepdims=True) + 1e-6
    f_norm = (f_all - f_mean) / f_std
    corr_mat = jnp.matmul(f_norm, f_norm.T) / float(num_states)
    dissim = 1.0 - jnp.clip(jnp.abs(corr_mat), 0.0, 1.0)
    orth = jnp.sum(jnp.where(off_diag, dissim, 0.0)) / n_pairs
    linked = ((run.p_init > 0.0) | (run.p_init.T > 0.0)) & off_diag
    n_linked = jnp.sum(linked.astype(jnp.float32))
    orth_linked = jnp.where(n_linked > 0.0, jnp.sum(jnp.where(linked, dissim, 0.0)) / jnp.maximum(n_linked, 1.0), 0.0)
    return jnp.stack([dc_acc_global, dc_complexity, orth, orth_linked])

  def init_state(key: jax.Array, env: NKLandscape, run: RunParams, prior_nets: AgentParams, prior_tables: jax.Array) -> OrgState:
    ks = jax.random.split(key, 8)
    s0 = jax.random.choice(ks[1], jnp.array([-1.0, 1.0]), shape=(n_bits,))
    # The DC is representation 0 and units are keyed by position, so uninformed initial weights
    # are identical across organizations that share these positions
    rep_keys = ks[2][None, :]
    if m_actors > 0:
      unit_keys = jax.vmap(lambda i: jax.random.fold_in(ks[3], i))(jnp.arange(m_actors, dtype=jnp.int32))
      rep_keys = jnp.concatenate([rep_keys, unit_keys], axis=0)
    params = jax.vmap(init_rep_params)(rep_keys)
    tables = jnp.zeros((n_reps, table_size), dtype=jnp.float32)
    # Informed representations start from their fitted beliefs (ks[4] keys those fits)
    if prior_net_reps:
      sel = jnp.array(prior_net_reps, dtype=jnp.int32)
      params = jax.tree_util.tree_map(lambda x, y: x.at[sel].set(y), params, prior_nets)
    if prior_table_reps:
      tables = tables.at[jnp.array(prior_table_reps, dtype=jnp.int32)].set(prior_tables)
    beta_vec = jax.random.normal(ks[6], (n_reps, n_reps, n_bits)) * 0.70
    beta_scalar = jax.random.normal(ks[7], (n_reps, n_reps)) * 0.30
    y0 = env.all_v[s_to_idx(s0)] + run.sigma_obs * jax.random.normal(ks[5], ())
    st0 = OrgState(
        s=s0,
        params=params,
        tables=tables,
        env=env,
        perm_mat=run.p_init,
        beta_vec=beta_vec,
        beta_scalar=beta_scalar,
        aspiration=y0,
        y_dc_last=y0,
        last_emergent=jnp.zeros((4,), dtype=jnp.float32),
        init_acc=realized_accuracy(params, tables, env),
    )
    return st0._replace(last_emergent=eval_emergent(st0, run))

  def step_metrics(state: OrgState, key: jax.Array, run: RunParams) -> StepMetrics:
    env = state.env
    idx_s = s_to_idx(state.s)
    v_true = env.all_v[idx_s]
    v_pct_max = (v_true - env.v_min) / (env.v_max - env.v_min + 1e-6)
    delta_true = env.all_v[neighbor_idx[idx_s]] - v_true

    probe_idx = jax.random.randint(key, shape=(64,), minval=0, maxval=num_states)
    queries = jnp.concatenate([state.s[None, :], state.s[None, :] * flip_eye, all_s[probe_idx]], axis=0)
    f_q = eval_all(state.params, state.tables, queries)
    f_center = f_q[:, 0]
    f_nbrs = f_q[:, 1 : 1 + n_bits]
    f_probe = f_q[:, 1 + n_bits :]

    delta_pred = f_nbrs[0] - f_center[0]
    dt_c = delta_true - jnp.mean(delta_true)
    dp_c = delta_pred - jnp.mean(delta_pred)
    denom = jnp.sqrt(jnp.sum(jnp.square(dt_c)) * jnp.sum(jnp.square(dp_c))) + 1e-6
    dc_r_local = jnp.sum(dt_c * dp_c) / denom
    n_pos = jnp.sum(delta_true > 0.0)
    at_local_peak = (n_pos == 0).astype(jnp.float32)
    if decision_mode == "delegated" and m_actors > 0 and n_search == 0:
      unit_deltas = (f_nbrs[1:] - f_center[1:, None]) * in_masks[1:]
      believes_peak = jnp.all(unit_deltas <= _DECISION_TOL)
    else:
      believes_peak = jnp.all(delta_pred <= _DECISION_TOL)
    false_stall = ((n_pos > 0) & believes_peak).astype(jnp.float32)

    v_probe = env.all_v[probe_idx]
    dc_mse = jnp.mean(jnp.square(f_probe[0] - v_probe))
    if m_actors > 0:
      emp_slopes = f_nbrs[1:] - f_center[1:, None]
      emp_div_local_val = jnp.mean(jnp.var(f_nbrs[1:], axis=0))
      emp_div_local_slope = jnp.mean(jnp.var(emp_slopes, axis=0))
      emp_mse = jnp.mean(jnp.square(f_probe[1:] - v_probe[None, :]))
      emp_div = jnp.mean(jnp.var(f_probe[1:], axis=0))
    else:
      emp_div_local_val = jnp.array(0.0, dtype=jnp.float32)
      emp_div_local_slope = jnp.array(0.0, dtype=jnp.float32)
      emp_mse = dc_mse
      emp_div = jnp.array(0.0, dtype=jnp.float32)

    active = run.p_init > 0.0
    n_active = jnp.sum(active.astype(jnp.float32) * src_inv_counts[None, :])
    mean_perm = jnp.where(n_active > 0.0, jnp.sum(jnp.where(active, state.perm_mat, 0.0)) / jnp.maximum(n_active, 1e-6), 0.0)
    return StepMetrics(
        v_true=v_true,
        v_pct_max=v_pct_max,
        v_percentile=_payoff_percentile(env.all_v, v_true),
        v_max_ref=env.v_max,
        v_local_ref=env.v_local_ref,
        dc_mse=dc_mse,
        dc_r_local=dc_r_local,
        at_local_peak=at_local_peak,
        false_stall=false_stall,
        emp_mse=emp_mse,
        emp_div=emp_div,
        emp_div_local_val=emp_div_local_val,
        emp_div_local_slope=emp_div_local_slope,
        dc_acc_global=state.last_emergent[0],
        dc_complexity=state.last_emergent[1],
        orthogonality=state.last_emergent[2],
        orthogonality_linked=state.last_emergent[3],
        mean_permeability=mean_perm,
    )

  def social_exchange(state: OrgState, key: jax.Array, run: RunParams):
    """Dyadic exchange of translated predictions on 6 local neighborhood query strategies."""
    k_nb, k_rd, k_xi = jax.random.split(key, 3)
    b0 = jax.random.randint(k_nb, (), 0, n_bits)
    nb_bits = (b0 + jnp.array([0, max(1, n_bits // 4), max(2, n_bits // 2), max(3, (3 * n_bits) // 4)], dtype=jnp.int32)) % n_bits
    s_nbrs = state.s[None, :] * flip_eye[nb_bits]
    k_2a, k_2b = jax.random.split(k_rd, 2)
    b2_1 = jax.random.randint(k_2a, shape=(1,), minval=0, maxval=n_bits)
    b2_2 = (b2_1 + jax.random.randint(k_2b, shape=(1,), minval=1, maxval=max(2, n_bits))) % n_bits
    s_two = state.s[None, :] * flip_eye[b2_1] * flip_eye[b2_2]
    queries = jnp.concatenate([state.s[None, :], s_nbrs, s_two], axis=0)
    preds = jax.lax.stop_gradient(eval_all(state.params, state.tables, queries))

    # Domain-aware advice: anchor baseline level when receiver observes the market and translate specialist slopes
    diff_mask = (queries != state.s[None, :]).astype(jnp.float32)
    n_flipped = jnp.maximum(jnp.sum(diff_mask, axis=-1), 1.0)
    overlap = jnp.matmul(in_masks, diff_mask.T) / n_flipped[None, :]
    overlap = overlap.at[:, 0].set(jnp.mean(in_masks, axis=-1))
    delta_preds = preds - preds[:, 0:1]
    prior_q = jnp.where(informed_arr, jnp.square(jnp.clip(state.init_acc - 0.15, 0.0, 1.0)), 0.0)
    spec_gain = (jnp.sqrt(float(n_bits) / jnp.maximum(n_sub.astype(jnp.float32), 1.0)) * (1.0 + 0.60 * prior_q))[:, None]
    has_mkt_lr = (sees_market & (run.rep_lr > 0.0))[:, None, None]

    xi_raw = jax.random.normal(k_xi, (n_reps, n_reps, 6))
    bias_s0 = jnp.einsum("d,abd->ab", state.s, state.beta_vec) / jnp.sqrt(float(n_bits)) + state.beta_scalar
    y_level_ab = run.c_mat * preds[None, :, 0] + (1.0 - run.c_mat) * (bias_s0 + 0.20 * xi_raw[:, :, 0])
    anchor_ab = jnp.where(has_mkt_lr, preds[:, None, 0:1], y_level_ab[:, :, None])

    delta_bias = jnp.einsum("qd,abd->abq", queries - state.s[None, :], state.beta_vec) / jnp.sqrt(float(n_bits))
    xi_slope = (0.28 / jnp.sqrt(float(n_bits))) * (xi_raw - xi_raw[:, :, 0:1])
    c_3d = run.c_mat[:, :, None]
    trans_slope_b = c_3d * (spec_gain * delta_preds)[None, :, :] + (1.0 - c_3d) * (0.45 * (1.0 - c_3d) * delta_bias + xi_slope)
    y_trans = (
        anchor_ab
        + overlap[None, :, :] * trans_slope_b
        + (1.0 - overlap[None, :, :]) * delta_preds[:, None, :]
    )

    rel_spec = jnp.where(n_sub[None, :] <= n_sub[:, None], (1.0 + 0.65 * prior_q)[None, :], 0.06)
    p_mat = jnp.where(off_diag, state.perm_mat * team_boost * rel_spec, 0.0)
    has_inc = jnp.sum(p_mat, axis=1) > 1e-5
    w_dom = p_mat[:, :, None] * (0.04 + 0.96 * overlap[None, :, :])
    w_dom_norm = w_dom / jnp.maximum(1e-6, jnp.sum(w_dom, axis=1, keepdims=True))
    y_mean = jnp.sum(w_dom_norm * y_trans, axis=1)
    dev = y_trans - y_mean[:, None, :]
    y_max = y_mean + 0.85 * jnp.sum(w_dom_norm * jnp.maximum(dev, 0.0), axis=1)
    y_min = y_mean + 1.25 * jnp.sum(w_dom_norm * jnp.minimum(dev, 0.0), axis=1)
    targets = jnp.where(agg_codes[:, None] == 1, y_max, jnp.where(agg_codes[:, None] == 2, y_min, y_mean))
    p_dom_q = jnp.sum(p_mat[:, :, None] * (0.10 + 0.90 * overlap[None, :, :]), axis=1)
    p_eff_q = p_dom_q / (0.85 + 0.35 * p_dom_q)
    lr_soc_q = 0.14 * p_eff_q
    needs_level = ~(sees_market & (run.rep_lr > 0.0))

    def body_q(q_i, carry):
      p_c, t_c = carry
      fw_diff = diff_mask[q_i][None, :] * rep_consts.feat_mask
      in_r_dom = jnp.any(fw_diff > 0.5, axis=-1)
      active_q = has_inc & jnp.where(q_i == 0, needs_level, jnp.where(q_i == 5, in_r_dom & (~rep_consts.is_lin), in_r_dom))
      fw_q = jnp.where(q_i == 0, jnp.full((n_reps, n_bits), 0.10), fw_diff)
      return learn_all(p_c, t_c, rep_consts, queries[q_i], targets[:, q_i], lr_soc_q[:, q_i], active_q, jnp.array(False), fw_q)

    return jax.lax.fori_loop(0, 6, body_q, (state.params, state.tables))

  def dc_search(s_start, params, tables, key, rho_t):
    """DC search over its searchable bits: one-bit local moves or multi-candidate distant jumps (prob rho_t)."""
    if n_search == 0:
      return s_start
    search_keys = jax.random.split(key, search_steps)

    def body(i_step, s_curr):
      k_u, k_b1, k_b2, k_rd = jax.random.split(search_keys[i_step], 4)
      j1 = jax.random.randint(k_b1, (), 0, n_search)
      s_local_1 = s_curr * flip_eye[dc_search_arr[j1]]
      if n_search >= 2:
        j_alt = (j1 + jax.random.randint(k_b2, (), 1, n_search)) % n_search
        j_alt2 = (j1 + max(1, n_search // 2)) % n_search
        loc_cands = jnp.stack([s_local_1, s_curr * flip_eye[dc_search_arr[j_alt]], s_curr * flip_eye[dc_search_arr[j_alt2]]])

        def sample_dist(k_c):
          ka, kb, kc, ku = jax.random.split(k_c, 4)
          i1 = jax.random.randint(ka, (), 0, n_search)
          i2 = (i1 + jax.random.randint(kb, (), 1, n_search)) % n_search
          s2 = s_curr * flip_eye[dc_search_arr[i1]] * flip_eye[dc_search_arr[i2]]
          if n_search >= 3:
            i3 = (i2 + jax.random.randint(kc, (), 1, n_search - 1)) % n_search
            s3 = s2 * flip_eye[dc_search_arr[i3]]
            return jnp.where(jax.random.uniform(ku, ()) < 0.5, s3, s2)
          return s2

        dist_cands = jax.vmap(sample_dist)(jax.random.split(k_rd, 4))
        f_all = eval_dc(params, tables, jnp.concatenate([s_curr[None, :], loc_cands, dist_cands], axis=0))
        f_curr = f_all[0]
        idx_loc = jnp.argmax(f_all[1:4])
        s_local = loc_cands[idx_loc]
        f_loc = f_all[1 + idx_loc]
        idx_dist = jnp.argmax(f_all[4:8])
        s_dist = dist_cands[idx_dist]
        f_dist = f_all[4 + idx_dist]
      else:
        f_pair = eval_dc(params, tables, jnp.stack([s_curr, s_local_1]))
        f_curr = f_pair[0]
        s_local = s_local_1
        f_loc = f_pair[1]
        s_dist = s_local_1
        f_dist = f_pair[1]
      local_stalled = f_loc <= f_curr + _DECISION_TOL
      use_dist = (jax.random.uniform(k_u, ()) < rho_t) | (local_stalled & (jax.random.uniform(jax.random.fold_in(k_u, 1), ()) < jnp.minimum(1.0, 2.5 * rho_t)))
      s_cand = jnp.where(use_dist, s_dist, s_local)
      f_cand = jnp.where(use_dist, f_dist, f_loc)
      tol = jnp.where(use_dist, _DECISION_TOL - 0.14 * rho_t, _DECISION_TOL)
      return jnp.where(f_cand > f_curr + tol, s_cand, s_curr)

    return jax.lax.fori_loop(0, search_steps, body, s_start)

  def unit_search(s, params, tables, key, rho_t, run):
    """Each non-DC node searches its own domain with the DC's rule and its own rho and search_steps.

    Returns a [R, N] mask of the bits each representation changed relative to s (DC row False).
    """
    k_units = jax.random.fold_in(key, _TAG_UNIT_SEARCH)

    def search_one(r_idx, p_r, t_r, rc, rho_r, steps_r):
      k_r = jax.random.fold_in(k_units, r_idx)
      n_r = n_sub[r_idx]
      eff_rho_r = jnp.minimum(1.0, rho_r + 0.5 * rho_t)

      def body(i_step, s_curr):
        k_u, k_b1, k_b2, k_rd = jax.random.split(jax.random.fold_in(k_r, i_step), 4)
        j1 = jax.random.randint(k_b1, (), 0, n_r)
        j2 = (j1 + jax.random.randint(k_b2, (), 1, jnp.maximum(n_r, 2))) % n_r
        s_loc1 = s_curr * flip_eye[sub_rows[r_idx, j1]]
        s_loc2 = s_curr * flip_eye[sub_rows[r_idx, j2]]

        def sample_unit_dist(k_c):
          ka, kb = jax.random.split(k_c, 2)
          i1 = jax.random.randint(ka, (), 0, n_r)
          i2 = (i1 + jax.random.randint(kb, (), 1, jnp.maximum(n_r, 2))) % n_r
          return s_curr * flip_eye[sub_rows[r_idx, i1]] * flip_eye[sub_rows[r_idx, i2]]

        dist_cands = jax.vmap(sample_unit_dist)(jax.random.split(k_rd, 4))
        f_all = eval_rep(p_r, t_r, rc, jnp.concatenate([s_curr[None, :], s_loc1[None, :], s_loc2[None, :], dist_cands], axis=0))
        f_curr = f_all[0]
        use_loc2 = (n_r >= 2) & (f_all[2] > f_all[1])
        s_local = jnp.where(use_loc2, s_loc2, s_loc1)
        f_loc = jnp.where(use_loc2, f_all[2], f_all[1])
        idx_dist = jnp.argmax(f_all[3:7])
        s_dist = jnp.where(n_r >= 2, dist_cands[idx_dist], s_local)
        f_dist = jnp.where(n_r >= 2, f_all[3 + idx_dist], f_loc)
        local_stalled = f_loc <= f_curr + _DECISION_TOL
        use_dist = (jax.random.uniform(k_u, ()) < eff_rho_r) | (local_stalled & (jax.random.uniform(jax.random.fold_in(k_u, 1), ()) < jnp.minimum(1.0, 2.5 * eff_rho_r)))
        s_cand = jnp.where(use_dist, s_dist, s_local)
        f_cand = jnp.where(use_dist, f_dist, f_loc)
        tol = jnp.where(use_dist, _DECISION_TOL - 0.14 * eff_rho_r, _DECISION_TOL)
        accept = (i_step < steps_r) & (f_cand > f_curr + tol)
        return jnp.where(accept, s_cand, s_curr)

      return jax.lax.fori_loop(0, unit_steps_max, body, s)

    s_units = jax.vmap(search_one)(rep_ids, params, tables, rep_consts, run.rep_rho, unit_steps)
    return (s_units != s[None, :]) & (rep_ids[:, None] > 0)

  def choose_strategy(s, params, tables, key, rho_t, run):
    """Strategy change under the organization's decision rights."""
    if decision_mode == "centralized":
      return dc_search(s, params, tables, key, rho_t)
    changed = unit_search(s, params, tables, key, rho_t, run)
    proposes = jnp.any(changed, axis=-1)
    s_single = s[None, :] * jnp.where(changed[1:], -1.0, 1.0)
    if decision_mode == "delegated":
      # Under delegated decision rights, the proposing unit with the strongest domain gain enacts its move
      k_pick = jax.random.fold_in(key, 77)
      f_base = eval_all(params, tables, s)[1:]
      f_prop = jax.vmap(eval_rep, in_axes=(0, 0, 0, 0))(
          jax.tree_util.tree_map(lambda x: x[1:], params),
          tables[1:],
          jax.tree_util.tree_map(lambda x: x[1:], rep_consts),
          s_single,
      )
      unit_gain = (f_prop - f_base) + 0.05 * jax.random.uniform(k_pick, (m_actors,))
      scores = jnp.where(proposes[1:], unit_gain, -1e9)
      chosen = jnp.argmax(scores)
      s_del = jnp.where(jnp.any(proposes[1:]), s_single[chosen], s)
      return dc_search(s_del, params, tables, key, rho_t)
    s_comp = s * jnp.where(jnp.any(changed, axis=0), -1.0, 1.0)
    cands = jnp.concatenate([s_comp[None, :], s_single], axis=0)
    valid = jnp.concatenate([jnp.any(proposes)[None], proposes[1:]])
    f_c = eval_dc(params, tables, jnp.concatenate([s[None, :], cands], axis=0))
    f_cands = jnp.where(valid, f_c[1:], -1e9)
    best = jnp.argmax(f_cands)
    return jnp.where(f_cands[best] > f_c[0] + _DECISION_TOL, cands[best], s)

  def run_experiments(params, tables, env, s_next, dom_curr, perm_mat, init_acc, step_key, run):
    """Non-DC representations probe 1-bit domain flips with directional payoff feedback."""
    if m_actors == 0:
      return params, tables
    k_exp = jax.random.fold_in(step_key, _TAG_EXPERIMENT)
    k_noise = jax.random.fold_in(step_key, _TAG_EXPERIMENT_NOISE)
    p_down = jnp.where(rep_ids > 0, perm_mat[:, 0] * run.c_mat[:, 0], 0.0)
    prior_q = jnp.where(informed_arr, jnp.square(jnp.clip(init_acc - 0.08, 0.0, 1.0)), 0.0)
    # Every non-DC node that sees results experiments at the same organization-wide rate.
    eff_rate = run.experiment_rate * (rep_ids > 0).astype(jnp.float32)

    def draw(r):
      k_do, k_bit, k_lrn = jax.random.split(jax.random.fold_in(k_exp, r), 3)
      do_exp = jax.random.uniform(k_do, ()) < eff_rate[r]
      j = jax.random.randint(k_bit, (), 0, n_sub[r])
      b_idx = sub_rows[r, j]
      s_e = s_next * flip_eye[b_idx]
      k_ctx = jax.random.fold_in(k_lrn, 1)
      ctx_flip = jnp.where((jnp.arange(n_bits) != b_idx) & (jax.random.uniform(k_ctx, (n_bits,)) < 0.42), -1.0, 1.0)
      s_ctx = s_next * ctx_flip
      return do_exp, s_e, s_ctx, s_ctx * flip_eye[b_idx], jax.random.normal(jax.random.fold_in(k_noise, r), ()), jax.random.normal(k_lrn, ())

    do_exp, s_exp, s_ctx, s_e_lin, eps_obs, eps_learn = jax.vmap(draw)(rep_ids)
    do_exp = do_exp & sees_market & (rep_ids > 0)
    sig_exp_v = env.all_v[s_to_idx(s_exp)]
    sig_curr_v = env.all_v[s_to_idx(s_next)]
    v_diff_lin = env.all_v[s_to_idx(s_e_lin)] - env.all_v[s_to_idx(s_ctx)]
    if any_domain:
      dom_exp = domain_payoffs(env, s_exp)
      dom_diff_lin = domain_payoffs(env, s_e_lin) - domain_payoffs(env, s_ctx)
      w_hol = (0.45 * p_down + 0.25 * prior_q) / (1.0 + 0.45 * p_down + 0.25 * prior_q)
      sig_raw = jnp.where(fb_codes == 2, dom_exp, sig_exp_v)
      sig_raw_lin = jnp.where(fb_codes == 2, dom_curr + dom_diff_lin, sig_curr_v + v_diff_lin)
      delta_sig = jnp.where(
          fb_codes == 2,
          (1.0 - w_hol) * (dom_exp - dom_curr) + w_hol * (sig_exp_v - sig_curr_v),
          sig_exp_v - sig_curr_v,
      )
      delta_sig_lin = jnp.where(
          fb_codes == 2,
          (1.0 - w_hol) * dom_diff_lin + w_hol * v_diff_lin,
          v_diff_lin,
      )
    else:
      sig_raw = sig_exp_v
      sig_raw_lin = sig_curr_v + v_diff_lin
      delta_sig = sig_exp_v - sig_curr_v
      delta_sig_lin = v_diff_lin
    noise_lin = (0.45 * run.sigma_obs * eps_obs + 0.95 * run.rep_sigma_learn * eps_learn) / (1.0 + 0.45 * p_down + 0.45 * prior_q)
    noise_mlp = (run.sigma_obs * eps_obs + run.rep_sigma_learn * eps_learn) / (1.0 + 0.40 * p_down + 0.40 * prior_q)
    noise = jnp.where(rep_consts.is_lin, noise_lin, noise_mlp)
    pred_curr = jax.lax.stop_gradient(eval_all(params, tables, s_next))
    y_dir = jnp.where(rep_consts.is_lin, 0.78 * sig_raw_lin + 0.22 * (pred_curr + delta_sig_lin), pred_curr + delta_sig)
    y_exp = jnp.where(rep_consts.is_tab, sig_raw + noise, y_dir + noise)
    exp_diff = (s_exp != s_next[None, :]).astype(jnp.float32)
    exp_lr = run.rep_lr * (jnp.where(rep_consts.is_lin, 0.70, 0.75) + 0.25 * prior_q + 0.25 * p_down + 0.35 * run.experiment_rate)
    return learn_all_each(params, tables, rep_consts, s_exp, y_exp, exp_lr, do_exp, jnp.array(False), exp_diff)

  def track_record(perm, state_prev: OrgState, s_next, y_rep, run):
    """Receivers reweight sources by start-of-period translated accuracy against each source's target signal."""
    pred0 = jax.lax.stop_gradient(eval_all(state_prev.params, state_prev.tables, s_next))
    bias_s = jnp.einsum("d,abd->ab", s_next, state_prev.beta_vec) / jnp.sqrt(float(n_bits)) + state_prev.beta_scalar
    pred_trans = run.c_mat * pred0[None, :] + (1.0 - run.c_mat) * bias_s
    y_target_b = jnp.where(sees_market[None, :], y_rep[None, :], y_rep[:, None])
    err_src = jnp.abs(pred_trans - y_target_b) + 0.45 * (1.0 - run.c_mat)
    active_links = run.p_init > 0.0
    n_inc = jnp.sum(active_links.astype(jnp.float32), axis=1, keepdims=True)
    mean_src_err = jnp.sum(jnp.where(active_links, err_src, 0.0), axis=1, keepdims=True) / jnp.maximum(n_inc, 1.0)
    err_self = jnp.abs(pred0 - y_rep)[:, None]
    ref_err = 0.80 * mean_src_err + 0.20 * err_self
    adv = ref_err - err_src
    upd = active_links & sees_market[:, None]
    return jnp.where(upd, jnp.clip(perm * jnp.exp(0.20 * run.lambda_track * adv), 0.005, 1.0), perm)

  def org_step(state: OrgState, step_key: jax.Array, t_idx: jax.Array, run: RunParams):
    # Turbulence: shared keys, so every configuration is shocked in the same periods
    env = state.env
    if turbulence_mode != "none":
      k_shock = jax.random.fold_in(step_key, _TAG_SHOCK)
      if turbulence_mode == "random":
        hit = jax.random.uniform(jax.random.fold_in(k_shock, 0), ()) < run.p_shock
      else:
        # Periodic shocks at periods shock_period, 2 * shock_period, ... (1-based periods)
        hit = ((t_idx + 1) % run.shock_period) == 0
      env = jax.lax.cond(hit, lambda e: _shock_landscape(e, jax.random.fold_in(k_shock, 1), all_s), lambda e: e, env)

    # Dyadic exchange (start-of-period parameters are kept for the track record)
    if m_actors > 0:
      params1, tables1 = social_exchange(state, jax.random.fold_in(step_key, _TAG_SOCIAL), run)
    else:
      params1, tables1 = state.params, state.tables

    # Problemistic search raises the DC's distant-search probability after shortfalls
    eff_asp = 0.65 * state.aspiration + 0.35 * 0.75
    rho_t = jnp.minimum(1.0, run.rho + run.kappa * jnp.maximum(0.0, eff_asp - state.y_dc_last))
    s_next = choose_strategy(state.s, params1, tables1, jax.random.fold_in(step_key, _TAG_SEARCH), rho_t, run)

    # Market feedback, experiments, track record, aspiration
    y_firm, y_rep, dom_curr = market_signals(
        env, s_next, jax.random.fold_in(step_key, _TAG_FIRM_NOISE), jax.random.fold_in(step_key, _TAG_DOMAIN_NOISE), run.sigma_obs
    )
    prior_q = jnp.where(
        informed_arr,
        jnp.where(n_sub == n_bits, jnp.clip(state.init_acc, 0.0, 1.0), jnp.square(jnp.clip(state.init_acc - 0.15, 0.0, 1.0))),
        0.0,
    )
    p_down = jnp.where(rep_ids > 0, state.perm_mat[:, 0] * run.c_mat[:, 0], 0.0)
    y_learn = y_rep + (1.35 * run.rep_sigma_learn / (1.0 + 0.50 * prior_q + 0.35 * p_down)) * rep_normals(
        jax.random.fold_in(step_key, _TAG_LEARN)
    )
    fw_base = ((s_next != state.s).astype(jnp.float32)[None, :] * rep_consts.feat_mask > 0.5).astype(jnp.float32)
    base_lr = run.rep_lr * (1.0 + 0.15 * prior_q)
    params2, tables2 = learn_all(params1, tables1, rep_consts, s_next, y_learn, base_lr, sees_market, jnp.array(True), fw_base)
    if has_experiments:
      params3, tables3 = run_experiments(params2, tables2, env, s_next, dom_curr, state.perm_mat, state.init_acc, step_key, run)
    else:
      params3, tables3 = params2, tables2
    perm_next = track_record(state.perm_mat, state, s_next, y_rep, run) if has_track_record else state.perm_mat
    mid_state = state._replace(
        s=s_next,
        params=params3,
        tables=tables3,
        env=env,
        perm_mat=perm_next,
        aspiration=(1.0 - _ASPIRATION_ALPHA) * state.aspiration + _ASPIRATION_ALPHA * y_firm,
        y_dc_last=y_firm,
    )

    # Metrics: emergent properties at checkpoints (t_idx is shared across firms, so this cond
    # stays a real branch under vmap)
    if eval_every <= 1:
      emergent_vals = eval_emergent(mid_state, run)
    else:
      t_next = t_idx + 1
      should_eval = (t_next == 1) | (t_next == t_steps) | ((t_next % eval_every) == 0)
      emergent_vals = jax.lax.cond(should_eval, lambda st: eval_emergent(st, run), lambda st: st.last_emergent, mid_state)
    next_state = mid_state._replace(last_emergent=emergent_vals)
    return next_state, step_metrics(next_state, jax.random.fold_in(step_key, _TAG_METRICS), run)

  def run_single_org(key: jax.Array, env: NKLandscape, run: RunParams, prior_nets: AgentParams, prior_tables: jax.Array):
    k_init, k_loop = jax.random.split(key, 2)
    init_st = init_state(k_init, env, run, prior_nets, prior_tables)
    step_keys = jax.random.split(k_loop, t_steps)
    t_idx = jnp.arange(t_steps, dtype=jnp.int32)
    return jax.lax.scan(lambda st, xs: org_step(st, xs[0], xs[1], run), init_st, (step_keys, t_idx))

  def init_single_org(key: jax.Array, env: NKLandscape, run: RunParams, prior_nets: AgentParams, prior_tables: jax.Array):
    k_init, k_loop = jax.random.split(key, 2)
    return init_state(k_init, env, run, prior_nets, prior_tables), jax.random.split(k_loop, t_steps)

  def run_chunk_single_org(state: OrgState, chunk_keys: jax.Array, chunk_t: jax.Array, run: RunParams):
    return jax.lax.scan(lambda st, xs: org_step(st, xs[0], xs[1], run), state, (chunk_keys, chunk_t))

  # Kernel inputs: base keys [B], landscapes [B], run parameters, and the fitted priors of the
  # informed representations [B, ...] from _initial_priors
  return _Kernels(
      full=jax.jit(jax.vmap(run_single_org, in_axes=(0, 0, None, 0, 0))),
      init=jax.jit(jax.vmap(init_single_org, in_axes=(0, 0, None, 0, 0))),
      chunk=jax.jit(jax.vmap(run_chunk_single_org, in_axes=(0, 0, None, None))),
      predict=jax.jit(jax.vmap(eval_all, in_axes=(0, 0, None))),
  )


def _extract_chunk_dict(history: StepMetrics, t_start: int, t_end: int) -> dict[str, list[Any]]:
  """Per-period means (and SEs for the main metrics) across firms for one block of periods."""
  sqrt_b = math.sqrt(int(history.v_true.shape[0]))

  def mean_of(arr):
    return [round(float(x), 4) for x in np.asarray(jnp.mean(arr, axis=0), dtype=np.float64)]

  def se_of(arr):
    return [round(float(x), 4) for x in np.asarray(jnp.std(arr, axis=0) / sqrt_b, dtype=np.float64)]

  return {
      "steps": list(range(t_start + 1, t_end + 1)),
      "v_true_mean": mean_of(history.v_true),
      "v_true_se": se_of(history.v_true),
      "v_pct_max_mean": mean_of(history.v_pct_max),
      "v_percentile_mean": mean_of(history.v_percentile),
      "v_percentile_se": se_of(history.v_percentile),
      "v_max_ref_mean": mean_of(history.v_max_ref),
      "v_local_ref_mean": mean_of(history.v_local_ref),
      "dc_r_local_mean": mean_of(history.dc_r_local),
      "false_stall_mean": mean_of(history.false_stall),
      "at_local_peak_mean": mean_of(history.at_local_peak),
      "emp_div_local_slope_mean": mean_of(history.emp_div_local_slope),
      "dc_acc_global_mean": mean_of(history.dc_acc_global),
      "dc_acc_global_se": se_of(history.dc_acc_global),
      "dc_complexity_mean": mean_of(history.dc_complexity),
      "dc_complexity_se": se_of(history.dc_complexity),
      "orthogonality_mean": mean_of(history.orthogonality),
      "orthogonality_se": se_of(history.orthogonality),
      "orthogonality_linked_mean": mean_of(history.orthogonality_linked),
      "orthogonality_linked_se": se_of(history.orthogonality_linked),
      "mean_permeability_mean": mean_of(history.mean_permeability),
  }


def _concat_histories(chunks: list[StepMetrics]) -> StepMetrics:
  if len(chunks) == 1:
    return chunks[0]
  return StepMetrics(*(jnp.concatenate([getattr(c, f) for c in chunks], axis=1) for f in StepMetrics._fields))


def summarize_trajectories(history: StepMetrics) -> dict[str, list[Any]]:
  """Per-period mean and SE trajectories across the full horizon."""
  return _extract_chunk_dict(history, 0, int(history.v_true.shape[1]))


def summarize_history(history: StepMetrics, eval_every: int = 25) -> dict[str, Any]:
  """Checkpoint means and SEs (t1, intermediate checkpoints, final) for every metric."""
  b_firms, t_steps = history.v_true.shape
  step = max(1, int(eval_every))
  if step == 25 and t_steps <= 600:
    checkpoints = [idx for idx in (0, 49, 99, 199, 299, 449) if idx < t_steps - 1] + [t_steps - 1]
  else:
    mults = [idx for idx in range(step - 1, t_steps - 1, step) if idx > 0]
    if len(mults) > 6:
      mults = [mults[round(i * (len(mults) - 1) / 5)] for i in range(6)]
    checkpoints = ([0] if t_steps > 1 else []) + mults + [t_steps - 1]

  def stat_at(arr):
    out = {
        f"t{idx + 1}": {
            "mean": round(float(jnp.mean(arr[:, idx])), 4),
            "se": round(float(jnp.std(arr[:, idx]) / jnp.sqrt(b_firms)), 4),
        }
        for idx in checkpoints
    }
    out["t_final"] = dict(out[f"t{t_steps}"])
    return out

  r_traj_per_firm = jnp.mean(history.dc_r_local, axis=-1)
  summary: dict[str, Any] = {name: stat_at(getattr(history, name)) for name in StepMetrics._fields}
  summary["dc_r_local_traj"] = {
      "mean": round(float(jnp.mean(r_traj_per_firm)), 4),
      "se": round(float(jnp.std(r_traj_per_firm) / jnp.sqrt(b_firms)), 4),
  }
  return summary


def _beta_cont_frac(a: float, b: float, x: float) -> float:
  """Continued fraction of the regularized incomplete beta function (modified Lentz method)."""
  tiny, eps = 1e-300, 1e-15
  c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
  d = 1.0 / (d if abs(d) > tiny else tiny)
  h = d
  for m in range(1, 1001):
    for num in (m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0))):
      d = 1.0 + num * d
      d = 1.0 / (d if abs(d) > tiny else tiny)
      c = 1.0 + num / c
      c = c if abs(c) > tiny else tiny
      h *= d * c
    if abs(d * c - 1.0) < eps:
      break
  return h


def _student_t_p_two_sided(t_stat: float, df: int) -> float:
  """Two-sided p-value of a t-statistic with df degrees of freedom (float64, no dependencies).

  Uses p = I_x(df / 2, 1 / 2) with x = df / (df + t^2), the regularized incomplete beta function;
  x and 1 - x are formed separately so large df and small t keep full precision.
  """
  t2 = t_stat * t_stat
  if t2 == 0.0:
    return 1.0
  if not math.isfinite(t2):
    return 0.0
  a, b = df / 2.0, 0.5
  x, y = df / (df + t2), t2 / (df + t2)
  log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
  if x < (a + 1.0) / (a + b + 2.0):
    p = math.exp(log_front) * _beta_cont_frac(a, b, x) / a
  else:
    p = 1.0 - math.exp(log_front) * _beta_cont_frac(b, a, y) / b
  return min(1.0, max(0.0, p))


def paired_ttest(v_a: jax.Array, v_b: jax.Array) -> tuple[float, float, float, float]:
  """Paired within-firm difference (A - B): mean, SE, t-statistic, and two-sided p-value.

  The p-value uses the Student t distribution with n - 1 degrees of freedom (n firms). With a
  single firm the test is undefined, so it reports t = 0 and p = 1 (no evidence).
  """
  diff = v_a - v_b
  n = int(diff.shape[0])
  mean_d = float(jnp.mean(diff))
  if n < 2:
    return mean_d, 0.0, 0.0, 1.0
  se_d = float(jnp.std(diff, ddof=1) / jnp.sqrt(n)) + 1e-9
  t_stat = mean_d / se_d
  return mean_d, se_d, t_stat, _student_t_p_two_sided(t_stat, n - 1)


def _paired_metrics_dict(hist_a: StepMetrics, hist_b: StepMetrics) -> dict[str, float]:
  """Paired tests at the final period between two histories on matched firms (A - B)."""
  rows = [
      ("v_true", hist_a.v_true[:, -1], hist_b.v_true[:, -1]),
      ("v_percentile", hist_a.v_percentile[:, -1], hist_b.v_percentile[:, -1]),
      ("r_local", hist_a.dc_r_local[:, -1], hist_b.dc_r_local[:, -1]),
      ("r_local_traj", jnp.mean(hist_a.dc_r_local, axis=-1), jnp.mean(hist_b.dc_r_local, axis=-1)),
      ("false_stall", hist_a.false_stall[:, -1], hist_b.false_stall[:, -1]),
      ("dc_acc_global", hist_a.dc_acc_global[:, -1], hist_b.dc_acc_global[:, -1]),
      ("dc_complexity", hist_a.dc_complexity[:, -1], hist_b.dc_complexity[:, -1]),
      ("orthogonality", hist_a.orthogonality[:, -1], hist_b.orthogonality[:, -1]),
      ("orthogonality_linked", hist_a.orthogonality_linked[:, -1], hist_b.orthogonality_linked[:, -1]),
  ]
  out: dict[str, float] = {}
  for name, arr_a, arr_b in rows:
    diff, se, t_stat, _ = paired_ttest(arr_a, arr_b)
    n_firms = int(arr_a.shape[0])
    p_val = _student_t_p_two_sided(round(t_stat, 4), n_firms - 1) if n_firms >= 2 else 1.0
    out[f"{name}_diff"] = round(diff, 4)
    out[f"{name}_se"] = round(se, 4)
    out[f"{name}_t_stat"] = round(t_stat, 2)
    out[f"{name}_p_value"] = round(p_val, 6)
  return out


def _landscape_for_market(mkt: MarketConfig, backend: str = "") -> NKLandscape:
  return build_landscape_batch(mkt.seed, mkt.b_firms, mkt.n_bits, mkt.nk_k, mkt.nk_pattern, backend)


def _landscape_signature(mkt: MarketConfig) -> tuple[Any, ...]:
  """Settings that determine the initial landscapes (paired tests require equal signatures)."""
  return (mkt.n_bits, mkt.nk_k, mkt.nk_pattern, mkt.seed, mkt.b_firms)


def _accuracy_by_node(graph: OrgGraph, init_acc: jax.Array) -> dict[str, float]:
  """Realized initial accuracy per node, averaged over firms and the node's members."""
  per_rep = np.asarray(jnp.mean(init_acc, axis=0), dtype=np.float64)
  rep_ids = graph_representation_ids(graph)
  return {
      n.id: round(float(np.mean([per_rep[i] for i, rid in enumerate(rep_ids) if rid == n.id])), 4)
      for n in graph.nodes
  }


def _resolve_device(device: Optional[str], mkt: MarketConfig) -> str:
  return device if device not in (None, "", "auto") else mkt.device


def run_graph(
    graph: OrgGraph,
    market: Optional[MarketConfig] = None,
    return_raw: bool = False,
    device: Optional[str] = None,
) -> dict[str, Any]:
  """Run one organization on a market (all firms in parallel) and summarize the results."""
  mkt = (market or MarketConfig()).validate()
  eff_device = _resolve_device(device, mkt)
  backend = _resolve_backend(eff_device)
  with _device_context(eff_device):
    t0 = time.perf_counter()
    static_spec, run_params, prior_jobs = _compile_spec(graph, mkt)
    kernels = _build_kernels(static_spec, backend=backend)
    base_keys = jax.random.split(jax.random.PRNGKey(mkt.seed), mkt.b_firms)
    envs = _landscape_for_market(mkt, backend)
    priors = _initial_priors(static_spec, prior_jobs, mkt, backend)
    final_state, history = kernels.full(base_keys, envs, run_params, *priors)
    jax.block_until_ready(history.v_true)
    wall_time = round(time.perf_counter() - t0, 4)
    summary = summarize_history(history, eval_every=mkt.eval_every)
    accuracy = _accuracy_by_node(graph, final_state.init_acc)

  out: dict[str, Any] = {
      "device_used": backend,
      "wall_time_sec": wall_time,
      "org_graph": graph.to_dict(),
      "market_config": mkt.to_dict(),
      "graph_explanation": graph.explain_graph(mkt),
      "summary": summary,
      "initial_accuracy_realized": accuracy,
  }
  if return_raw:
    out["history"] = history
    out["final_state"] = final_state
  return out


def predict_graph_representations(
    graph: OrgGraph,
    market: MarketConfig,
    state: OrgState,
    strategies: Any,
    device: Optional[str] = None,
) -> jax.Array:
  """Predictions of every representation at strategies [Q, N]; returns [b_firms, n_reps, Q].

  Uses the kernel's own feature masks, so each prediction depends only on the bits in the
  representation's domain. Representation order follows graph_representation_ids(graph).
  """
  mkt = market.validate()
  eff_device = _resolve_device(device, mkt)
  backend = _resolve_backend(eff_device)
  with _device_context(eff_device):
    static_spec, _, _ = _compile_spec(graph, mkt)
    kernels = _build_kernels(static_spec, backend=backend)
    return kernels.predict(state.params, state.tables, jnp.asarray(strategies, dtype=jnp.float32))


def _slots_and_labels(graphs: list[OrgGraph], labels: Optional[list[str]], slot_ids: Optional[list[str]]):
  if len(graphs) < 2 or len(graphs) > MAX_CONFIGURATIONS:
    raise ValueError(f"A comparison needs between 2 and {MAX_CONFIGURATIONS} configurations, got {len(graphs)}")
  default_slots = [chr(ord("a") + i) for i in range(MAX_CONFIGURATIONS)]
  eff_slots = list(slot_ids) if slot_ids and len(slot_ids) == len(graphs) else default_slots[: len(graphs)]
  if len(set(eff_slots)) != len(eff_slots):
    raise ValueError(f"Configuration slot ids must be unique, got {eff_slots}")
  eff_labels = (
      list(labels)
      if labels and len(labels) == len(graphs)
      else [g.name or f"Configuration {i + 1}" for i, g in enumerate(graphs)]
  )
  return eff_slots, eff_labels


def compare_multi_graphs(
    graphs: list[OrgGraph],
    market: Optional[MarketConfig] = None,
    labels: Optional[list[str]] = None,
    slot_ids: Optional[list[str]] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
  """Run 2 to 8 organizations on the same firms and landscapes; paired tests vs the first."""
  mkt = (market or MarketConfig()).validate()
  eff_slots, eff_labels = _slots_and_labels(graphs, labels, slot_ids)
  eff_device = _resolve_device(device, mkt)
  backend = _resolve_backend(eff_device)
  t0 = time.perf_counter()
  raw = [run_graph(g, mkt, return_raw=True, device=eff_device) for g in graphs]
  base_hist = raw[0]["history"]
  orgs = [
      {
          "slot_id": slot_id,
          "label": lbl,
          "org_graph": res["org_graph"],
          "graph_explanation": res["graph_explanation"],
          "summary": summarize_history(res["history"], eval_every=mkt.eval_every),
          "trajectories": summarize_trajectories(res["history"]),
          "initial_accuracy_realized": res["initial_accuracy_realized"],
          "paired_vs_a": _paired_metrics_dict(res["history"], base_hist) if idx > 0 else None,
      }
      for idx, (slot_id, lbl, res) in enumerate(zip(eff_slots, eff_labels, raw))
  ]
  return {
      "device_used": backend,
      "wall_time_sec": round(time.perf_counter() - t0, 4),
      "market_config": mkt.to_dict(),
      "slot_ids": eff_slots,
      "labels": eff_labels,
      "orgs": orgs,
  }


def compare_graphs(
    graph_a: OrgGraph,
    graph_b: OrgGraph,
    market: Optional[MarketConfig] = None,
    labels: Optional[list[str]] = None,
    device: Optional[str] = None,
) -> dict[str, Any]:
  """Run two organizations on the same firms and landscapes (same output as compare_multi_graphs)."""
  return compare_multi_graphs([graph_a, graph_b], market=market, labels=labels, device=device)


def _graph_stream_closures(graph: OrgGraph, market: MarketConfig, backend: str):
  """Return (init_closure, chunk_closure); chunk_closure(state, keys, t_start) advances periods."""
  static_spec, run_params, prior_jobs = _compile_spec(graph, market)
  kernels = _build_kernels(static_spec, backend=backend)

  def init_closure(base_keys: jax.Array, envs: NKLandscape):
    # Priors depend on the market's landscape signature, which also determines base_keys and envs
    priors = _initial_priors(static_spec, prior_jobs, market, backend)
    return kernels.init(base_keys, envs, run_params, *priors)

  def chunk_closure(state: OrgState, chunk_keys: jax.Array, t_start: int):
    chunk_t = jnp.arange(t_start, t_start + int(chunk_keys.shape[1]), dtype=jnp.int32)
    return kernels.chunk(state, chunk_keys, chunk_t, run_params)

  return init_closure, chunk_closure


def _initial_states(graph: OrgGraph, market: MarketConfig, backend: str = "") -> OrgState:
  """Initial states of all firms, computed exactly as the run and stream paths compute them."""
  envs = _landscape_for_market(market, backend)
  base_keys = jax.random.split(jax.random.PRNGKey(market.seed), market.b_firms)
  init_closure, _ = _graph_stream_closures(graph, market, backend)
  return init_closure(base_keys, envs)[0]


def stream_graph(
    graph: OrgGraph,
    market: Optional[MarketConfig] = None,
    chunk_size: int = 25,
    device: Optional[str] = "auto",
):
  """Stream progress events and the final summary for one organization."""
  mkt = (market or MarketConfig()).validate()
  eff_device = _resolve_device(device, mkt)
  eff_chunk = max(1, min(int(chunk_size), mkt.t_steps))
  backend = _resolve_backend(eff_device)
  graph_exp = graph.explain_graph(mkt)
  yield {
      "event": "start",
      "mode": "single",
      "t_steps": mkt.t_steps,
      "b_firms": mkt.b_firms,
      "chunk_size": eff_chunk,
      "device_used": backend,
      "org_graph": graph.to_dict(),
      "market_config": mkt.to_dict(),
      "graph_explanation": graph_exp,
  }
  with _device_context(eff_device):
    t0 = time.perf_counter()
    init_closure, chunk_closure = _graph_stream_closures(graph, mkt, backend)
    base_keys = jax.random.split(jax.random.PRNGKey(mkt.seed), mkt.b_firms)
    state, all_step_keys = init_closure(base_keys, _landscape_for_market(mkt, backend))
    jax.block_until_ready(state.s)
    accuracy = _accuracy_by_node(graph, state.init_acc)
    chunks: list[StepMetrics] = []
    for t_start in range(0, mkt.t_steps, eff_chunk):
      t_end = min(t_start + eff_chunk, mkt.t_steps)
      state, chunk_hist = chunk_closure(state, all_step_keys[:, t_start:t_end, :], t_start)
      jax.block_until_ready(chunk_hist.v_true)
      chunks.append(chunk_hist)
      yield {
          "event": "progress",
          "mode": "single",
          "step": t_end,
          "t_start": t_start + 1,
          "t_end": t_end,
          "t_steps": mkt.t_steps,
          "progress": round(t_end / mkt.t_steps, 4),
          "elapsed_sec": round(time.perf_counter() - t0, 4),
          "device_used": backend,
          "chunk": _extract_chunk_dict(chunk_hist, t_start, t_end),
      }
    full_history = _concat_histories(chunks)
    wall_time = round(time.perf_counter() - t0, 4)
    summary = summarize_history(full_history, eval_every=mkt.eval_every)
    trajectories = summarize_trajectories(full_history)
  yield {
      "event": "complete",
      "mode": "single",
      "device_used": backend,
      "wall_time_sec": wall_time,
      "org_graph": graph.to_dict(),
      "market_config": mkt.to_dict(),
      "graph_explanation": graph_exp,
      "summary": summary,
      "trajectories": trajectories,
      "initial_accuracy_realized": accuracy,
  }


def stream_multi_graph_comparison(
    graphs: list[OrgGraph],
    market: Optional[MarketConfig] = None,
    labels: Optional[list[str]] = None,
    slot_ids: Optional[list[str]] = None,
    chunk_size: int = 25,
    device: Optional[str] = "auto",
):
  """Stream lockstep progress for 2 to 8 organizations on the same firms; paired tests vs the first."""
  mkt = (market or MarketConfig()).validate()
  eff_slots, eff_labels = _slots_and_labels(graphs, labels, slot_ids)
  eff_device = _resolve_device(device, mkt)
  t_steps = mkt.t_steps
  eff_chunk = max(1, min(int(chunk_size), t_steps))
  backend = _resolve_backend(eff_device)
  start_orgs = [
      {"slot_id": slot_id, "label": lbl, "org_graph": g.to_dict(), "graph_explanation": g.explain_graph(mkt)}
      for slot_id, lbl, g in zip(eff_slots, eff_labels, graphs)
  ]
  yield {
      "event": "start",
      "mode": "comparison",
      "slot_ids": eff_slots,
      "labels": eff_labels,
      "t_steps": t_steps,
      "b_firms": mkt.b_firms,
      "chunk_size": eff_chunk,
      "device_used": backend,
      "market_config": mkt.to_dict(),
      "orgs": start_orgs,
  }
  with _device_context(eff_device):
    t0 = time.perf_counter()
    closures = [_graph_stream_closures(g, mkt, backend) for g in graphs]
    envs = _landscape_for_market(mkt, backend)
    base_keys = jax.random.split(jax.random.PRNGKey(mkt.seed), mkt.b_firms)
    states, step_keys = [], []
    for init_fn, _ in closures:
      st_i, k_i = init_fn(base_keys, envs)
      states.append(st_i)
      step_keys.append(k_i)
    accuracies = [_accuracy_by_node(g, st.init_acc) for g, st in zip(graphs, states)]
    chunks_per_org: list[list[StepMetrics]] = [[] for _ in graphs]
    for t_start in range(0, t_steps, eff_chunk):
      t_end = min(t_start + eff_chunk, t_steps)
      chunk_dicts: dict[str, dict[str, Any]] = {}
      for idx, (slot_id, (_, chunk_fn)) in enumerate(zip(eff_slots, closures)):
        states[idx], hist_i = chunk_fn(states[idx], step_keys[idx][:, t_start:t_end, :], t_start)
        jax.block_until_ready(hist_i.v_true)
        chunks_per_org[idx].append(hist_i)
        chunk_dicts[slot_id] = _extract_chunk_dict(hist_i, t_start, t_end)
      yield {
          "event": "progress",
          "mode": "comparison",
          "step": t_end,
          "t_start": t_start + 1,
          "t_end": t_end,
          "t_steps": t_steps,
          "progress": round(t_end / t_steps, 4),
          "elapsed_sec": round(time.perf_counter() - t0, 4),
          "device_used": backend,
          "chunks": chunk_dicts,
      }
    histories = [_concat_histories(ch) for ch in chunks_per_org]
    wall_time = round(time.perf_counter() - t0, 4)
    orgs = [
        {
            **info,
            "summary": summarize_history(hist, eval_every=mkt.eval_every),
            "trajectories": summarize_trajectories(hist),
            "initial_accuracy_realized": acc,
            "paired_vs_a": _paired_metrics_dict(hist, histories[0]) if idx > 0 else None,
        }
        for idx, (info, hist, acc) in enumerate(zip(start_orgs, histories, accuracies))
    ]
  yield {
      "event": "complete",
      "mode": "comparison",
      "slot_ids": eff_slots,
      "labels": eff_labels,
      "device_used": backend,
      "wall_time_sec": wall_time,
      "market_config": mkt.to_dict(),
      "orgs": orgs,
  }


def stream_graph_comparison(
    graph_a: OrgGraph,
    graph_b: OrgGraph,
    market: Optional[MarketConfig] = None,
    label_a: str = "Configuration 1",
    label_b: str = "Configuration 2",
    chunk_size: int = 25,
    device: Optional[str] = "auto",
):
  """Stream lockstep progress for two organizations (same events as stream_multi_graph_comparison)."""
  yield from stream_multi_graph_comparison(
      [graph_a, graph_b], market=market, labels=[label_a, label_b], chunk_size=chunk_size, device=device
  )


GRAPH_SWEEP_PARAMS = (
    "nk_k",
    "p_shock",
    "shock_period",
    "sigma_obs",
    "experiment_rate",
    "aspiration_sensitivity",
    "lambda_track",
    "decision_rights",
    "feedback_regime",
    "unit_initial_accuracy",
    "rho",
    "sigma_learn",
    "h_agent",
    "commensurability",
    "permeability",
    "upward_permeability",
    "downward_permeability",
)
_NODE_SWEEP_ATTRS = (
    "count",
    "h_agent",
    "search_steps",
    "upward_agg",
    "model_type",
    "feedback",
    "lr",
    "sigma_learn",
    "rho",
    "initial_accuracy",
)


def _accuracy_value(raw_val: Any) -> Optional[float]:
  """Sweep value for unit_initial_accuracy: a number, or null / "none" / "uninformed" for uninformed."""
  if raw_val is None or str(raw_val).strip().lower() in ("none", "null", "", UNINFORMED):
    return None
  return float(raw_val)


def _node_accuracy_value(raw_val: Any) -> Any:
  """Sweep value for node_id.initial_accuracy: a number, "uninformed", or null / "default"."""
  text = "" if raw_val is None else str(raw_val).strip().lower()
  if text == UNINFORMED:
    return UNINFORMED
  if text in ("default", "none", "null", ""):
    return None
  return float(raw_val)


def _apply_graph_sweep_override(
    graph: OrgGraph, market: MarketConfig, sweep_param: str, raw_val: Any
) -> tuple[OrgGraph, MarketConfig, Any]:
  """Apply one comparative-statics value to an (OrgGraph, MarketConfig) pair."""
  param = str(sweep_param).strip()
  if param == "nk_k":
    val = int(float(raw_val))
    return graph, market.with_overrides(nk_k=val), val
  if param == "p_shock":
    val = float(raw_val)
    return graph, market.with_overrides(turbulence="random", p_shock=val), val
  if param == "shock_period":
    val = int(float(raw_val))
    return graph, market.with_overrides(turbulence="periodic", shock_period=val), val
  if param == "sigma_obs":
    val = float(raw_val)
    return graph, market.with_overrides(sigma_obs=val), val
  if param in ("experiment_rate", "aspiration_sensitivity", "lambda_track"):
    val = float(raw_val)
    return replace(graph, **{param: val}), market, val
  if param in ("decision_rights", "feedback_regime"):
    val = str(raw_val).strip().lower()
    return replace(graph, **{param: val}), market, val
  if param == "unit_initial_accuracy":
    val = _accuracy_value(raw_val)
    return replace(graph, unit_initial_accuracy=val), market, val
  if param == "rho":
    val = float(raw_val)
    return replace(graph, nodes=tuple(n.with_overrides(rho=val) if n.role == "hq" else n for n in graph.nodes)), market, val
  if param == "sigma_learn":
    val = float(raw_val)
    nodes = tuple(n.with_overrides(sigma_learn=val) if n.role != "hq" else n for n in graph.nodes)
    return replace(graph, nodes=nodes), market, val
  if param == "h_agent":
    val = int(float(raw_val))
    return replace(graph, nodes=tuple(n.with_overrides(h_agent=val) for n in graph.nodes)), market, val
  if param == "commensurability":
    val = float(raw_val)
    return replace(graph, edges=tuple(replace(e, commensurability=val) for e in graph.edges)), market, val
  if param in ("permeability", "upward_permeability"):
    val = float(raw_val)
    hq_id = graph.get_hq_node().id
    edges = tuple(
        replace(e, weight=val) if (param == "permeability" or e.source != hq_id) else e for e in graph.edges
    )
    return replace(graph, edges=edges), market, val
  if param == "downward_permeability":
    val = float(raw_val)
    hq_id = graph.get_hq_node().id
    up_links = {e.source: e for e in graph.edges if e.target == hq_id}
    edges = [e for e in graph.edges if e.source != hq_id]
    if val > 0.0:
      edges += [
          OrgEdge(source=hq_id, target=sid, weight=val, commensurability=up.commensurability)
          for sid, up in up_links.items()
      ]
    return replace(graph, edges=tuple(edges)), market, val
  if "." in param:
    target_id, attr = param.split(".", 1)
    if attr not in _NODE_SWEEP_ATTRS:
      raise ValueError(f"Unsupported node sweep attribute {attr!r}; use one of {', '.join(_NODE_SWEEP_ATTRS)}")
    if not any(n.id == target_id for n in graph.nodes):
      raise ValueError(f"Sweep node {target_id!r} not found")
    if attr in ("count", "h_agent", "search_steps"):
      val = int(float(raw_val))
    elif attr in ("upward_agg", "model_type"):
      val = str(raw_val).strip()
    elif attr == "feedback":
      fb = str(raw_val).strip().lower()
      val = None if fb in ("", "default", "null") else fb
    elif attr == "initial_accuracy":
      val = _node_accuracy_value(raw_val)
    else:
      val = float(raw_val)
    return graph.update_node(target_id, **{attr: val}), market, val
  raise ValueError(
      f"Unsupported sweep parameter {param!r}. Supported: {', '.join(GRAPH_SWEEP_PARAMS)}, "
      f"or node attributes as 'node_id.attr' with attr in {', '.join(_NODE_SWEEP_ATTRS)}."
  )


def stream_graph_sweep(
    graph: OrgGraph,
    market: Optional[MarketConfig],
    sweep_param: str,
    sweep_values: list[Any],
    chunk_size: int = 50,
    device: Optional[str] = "auto",
):
  """Stream comparative statics for one organization across parameter values on matched firms."""
  if not sweep_values:
    raise ValueError("sweep_values must contain at least one value")
  mkt_base = (market or MarketConfig()).validate()
  eff_device = _resolve_device(device, mkt_base)
  backend = _resolve_backend(eff_device)
  points = []
  for raw_val in sweep_values:
    g_pt, m_pt, val = _apply_graph_sweep_override(graph, mkt_base, sweep_param, raw_val)
    points.append((g_pt, m_pt, val, f"{sweep_param}={val}"))
  sweep_vals = [p[2] for p in points]
  labels = [p[3] for p in points]
  eff_chunk = max(1, min(int(chunk_size), mkt_base.t_steps))
  yield {
      "event": "start",
      "mode": "sweep",
      "sweep_param": sweep_param,
      "sweep_values": sweep_vals,
      "point_labels": labels,
      "n_points": len(points),
      "t_steps": mkt_base.t_steps,
      "b_firms": mkt_base.b_firms,
      "chunk_size": eff_chunk,
      "device_used": backend,
  }
  with _device_context(eff_device):
    t0 = time.perf_counter()
    completed: list[dict[str, Any]] = []
    baseline: Optional[StepMetrics] = None
    baseline_sig = _landscape_signature(points[0][1])
    for idx, (g_pt, m_pt, val, lbl) in enumerate(points):
      init_closure, chunk_closure = _graph_stream_closures(g_pt, m_pt, backend)
      base_keys = jax.random.split(jax.random.PRNGKey(m_pt.seed), m_pt.b_firms)
      state, all_step_keys = init_closure(base_keys, _landscape_for_market(m_pt, backend))
      accuracy = _accuracy_by_node(g_pt, state.init_acc)
      pt_chunks: list[StepMetrics] = []
      for t_start in range(0, m_pt.t_steps, eff_chunk):
        t_end = min(t_start + eff_chunk, m_pt.t_steps)
        state, chunk_hist = chunk_closure(state, all_step_keys[:, t_start:t_end, :], t_start)
        jax.block_until_ready(chunk_hist.v_true)
        pt_chunks.append(chunk_hist)
        yield {
            "event": "progress",
            "mode": "sweep",
            "point_index": idx,
            "point_value": val,
            "point_label": lbl,
            "step": t_end,
            "t_steps": m_pt.t_steps,
            "progress": round((idx + t_end / m_pt.t_steps) / len(points), 4),
            "elapsed_sec": round(time.perf_counter() - t0, 4),
            "device_used": backend,
            "chunk": _extract_chunk_dict(chunk_hist, t_start, t_end),
        }
      full_hist = _concat_histories(pt_chunks)
      if idx == 0:
        baseline = full_hist
      paired = None
      if idx > 0 and baseline is not None and _landscape_signature(m_pt) == baseline_sig:
        full_paired = _paired_metrics_dict(full_hist, baseline)
        paired = {k: v for k, v in full_paired.items() if k.startswith(("v_true", "false_stall"))}
      record = {
          "point_index": idx,
          "point_value": val,
          "point_label": lbl,
          "org_graph": g_pt.to_dict(),
          "market_config": m_pt.to_dict(),
          "graph_explanation": g_pt.explain_graph(m_pt),
          "summary": summarize_history(full_hist, eval_every=m_pt.eval_every),
          "trajectories": summarize_trajectories(full_hist),
          "initial_accuracy_realized": accuracy,
          "paired_vs_baseline": paired,
      }
      completed.append(record)
      yield {"event": "point_complete", "mode": "sweep", "point": record, "elapsed_sec": round(time.perf_counter() - t0, 4)}
    wall_time = round(time.perf_counter() - t0, 4)
  yield {
      "event": "complete",
      "mode": "sweep",
      "sweep_param": sweep_param,
      "sweep_values": sweep_vals,
      "point_labels": labels,
      "device_used": backend,
      "wall_time_sec": wall_time,
      "points": completed,
  }

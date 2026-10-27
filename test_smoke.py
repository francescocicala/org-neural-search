"""Smoke and regression tests for the organizational neural search laboratory."""

from dataclasses import replace
import importlib.util
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
  from .cli import main as cli_main
  from .org_graph import (
      GRAPH_TEMPLATES,
      MAX_B_FIRMS,
      MAX_N_BITS,
      MAX_T_STEPS,
      MIN_N_BITS,
      MarketConfig,
      OrgEdge,
      OrgGraph,
      OrgNode,
      rescale_jurisdiction_bits,
  )
  from .webapp import create_server, format_server_url
except ImportError:
  from cli import main as cli_main  # type: ignore
  from org_graph import (  # type: ignore
      GRAPH_TEMPLATES,
      MAX_B_FIRMS,
      MAX_N_BITS,
      MAX_T_STEPS,
      MIN_N_BITS,
      MarketConfig,
      OrgEdge,
      OrgGraph,
      OrgNode,
      rescale_jurisdiction_bits,
  )
  from webapp import create_server, format_server_url  # type: ignore

HAS_JAX = importlib.util.find_spec("jax") is not None


def _engine():
  if __package__:
    from . import engine
  else:
    import engine  # type: ignore
  return engine


class _LabServer:
  """Run the web laboratory on a free local port for the duration of a with-block."""

  def __enter__(self):
    self.server = create_server(host="127.0.0.1", port=0)
    threading.Thread(target=self.server.serve_forever, daemon=True).start()
    self.base_url = format_server_url(self.server)
    return self

  def __exit__(self, *exc):
    self.server.shutdown()
    self.server.server_close()

  def post(self, path, payload):
    req = urllib.request.Request(
        f"{self.base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req)

  def post_json(self, path, payload):
    with self.post(path, payload) as resp:
      return resp.status, json.loads(resp.read().decode("utf-8"))

  def stream(self, payload):
    with self.post("/api/stream", payload) as resp:
      assert resp.status == 200
      return [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]


def _same_rep_params(state_a, state_b, rep_a: int, rep_b: int) -> bool:
  """Whether representation rep_a in state_a and rep_b in state_b have identical parameters."""
  import numpy as np

  leaves = zip(state_a.params, state_b.params)
  return all(np.array_equal(np.asarray(a[:, rep_a]), np.asarray(b[:, rep_b])) for a, b in leaves)


class OrgNeuralSearchSmokeTest(unittest.TestCase):
  """Unit and integration tests for the graph engine, web laboratory, and CLI."""

  def test_market_and_graph_validation(self):
    self.assertEqual(len(GRAPH_TEMPLATES), 12)
    self.assertEqual(MarketConfig().b_firms, 50)
    market = MarketConfig(b_firms=8, t_steps=12)
    for key, g in GRAPH_TEMPLATES.items():
      g.validate()
      self.assertEqual(OrgGraph.from_json(g.to_json()), g, key)
      info = g.explain_graph(market)
      for field_name in (
          "landscape",
          "feedback_regime",
          "node_feedback",
          "decision_rights",
          "effective_decision_rights",
          "dc_search_bits",
          "experiment_rate",
          "aspiration_sensitivity",
          "lambda_track",
          "notes",
          "hq_summary",
      ):
        self.assertIn(field_name, info, f"{key} explain_graph lacks {field_name}")
      self.assertEqual(
          set(info["landscape"]),
          {"family", "n_bits", "nk_k", "nk_pattern", "turbulence", "p_shock", "shock_period", "sigma_obs"},
      )
      self.assertEqual(info["node_feedback"]["hq"], "firm")
    self.assertEqual(MarketConfig.from_json(market.to_json()), market)

    # Unknown keys or unsupported values raise and name the invalid field
    for bad in ({"unknown_field": 3.0}, {"lambda_track": 0.1}):
      with self.assertRaisesRegex(ValueError, list(bad)[0]):
        MarketConfig.from_dict(bad)
    with self.assertRaisesRegex(ValueError, "must be 'nk'"):
      MarketConfig(landscape="invalid")
    with self.assertRaisesRegex(ValueError, "unknown_field"):
      OrgGraph.from_dict(dict(GRAPH_TEMPLATES["enacted_solo_dc"].to_dict(), unknown_field=5))
    with self.assertRaisesRegex(ValueError, "unknown_field"):
      OrgNode.from_dict({"id": "u", "label": "U", "unknown_field": 0.2})
    with self.assertRaisesRegex(ValueError, "unknown_field"):
      OrgEdge.from_dict({"source": "u", "target": "hq", "unknown_field": 0.3})
    with self.assertRaises(ValueError):
      OrgNode(id="bad", label="Bad", model_type="unknown_type")

    # Market limits and turbulence settings
    self.assertEqual(MAX_B_FIRMS, 2048)
    self.assertEqual(MAX_T_STEPS, 20000)
    self.assertEqual(MarketConfig().eval_every, 25)
    MarketConfig(b_firms=2048)
    long_mkt = MarketConfig(t_steps=20000, shock_period=20000, eval_every=10)
    self.assertEqual((long_mkt.t_steps, long_mkt.shock_period, long_mkt.eval_every), (20000, 20000, 10))
    for bad_kwargs in (
        {"b_firms": 2049},
        {"b_firms": 0},
        {"t_steps": 0},
        {"t_steps": 20001},
        {"turbulence": "chaotic"},
        {"shock_period": 0},
        {"shock_period": 20001},
        {"eval_every": 0},
        {"eval_every": 20001},
        {"p_shock": 1.5},
        {"nk_k": 12},
        {"nk_pattern": "blocky"},
    ):
      with self.assertRaises(ValueError, msg=str(bad_kwargs)):
        MarketConfig(**bad_kwargs)
    periodic = MarketConfig.from_dict({"turbulence": "periodic", "shock_period": "50", "nk_k": 4.0})
    self.assertEqual((periodic.turbulence, periodic.shock_period, periodic.nk_k), ("periodic", 50, 4))

    # Feedback resolution
    self.assertEqual(
        GRAPH_TEMPLATES["boundary_spanners"].effective_feedback(),
        {"hq": "firm", "front_1": "domain", "front_2": "domain", "internal_1": "none", "internal_2": "none"},
    )
    dc_only = replace(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], feedback_regime="dc_only").effective_feedback()
    self.assertEqual({k: v for k, v in dc_only.items() if k != "hq"}, {f"rep_{k}": "none" for k in range(1, 5)})
    with self.assertRaises(ValueError):
      OrgNode(id="hq", label="DC", role="hq", feedback="none")

    # Structure: exactly one DC, reciprocal links and cycles allowed, depth, headcount, connectivity
    hq = OrgNode(id="hq", label="DC", role="hq")
    with self.assertRaises(ValueError):
      OrgGraph(name="Two DCs", nodes=(hq, OrgNode(id="hq2", label="DC 2", role="hq")), edges=(OrgEdge(source="hq2", target="hq"),))
    cyclic = OrgGraph(
        name="Peer cycle",
        nodes=(hq, OrgNode(id="u1", label="U1"), OrgNode(id="u2", label="U2"), OrgNode(id="u3", label="U3")),
        edges=(
            OrgEdge(source="u1", target="hq"),
            OrgEdge(source="u2", target="hq"),
            OrgEdge(source="u3", target="hq"),
            OrgEdge(source="u1", target="u2"),
            OrgEdge(source="u2", target="u3"),
            OrgEdge(source="u3", target="u1"),
            OrgEdge(source="hq", target="u1"),
        ),
    )
    self.assertEqual(cyclic.explain_graph()["tiers"], 2)
    with self.assertRaises(ValueError):
      OrgGraph(
          name="Too deep",
          nodes=(hq, OrgNode(id="m1", label="M1", role="manager"), OrgNode(id="m2", label="M2", role="manager"), OrgNode(id="u1", label="U1")),
          edges=(OrgEdge(source="u1", target="m2"), OrgEdge(source="m2", target="m1"), OrgEdge(source="m1", target="hq")),
      )
    with self.assertRaises(ValueError):
      OrgGraph(
          name="Too large",
          nodes=(hq, OrgNode(id="u1", label="U1", count=33), OrgNode(id="u2", label="U2", count=32)),
          edges=(OrgEdge(source="u1", target="hq"), OrgEdge(source="u2", target="hq")),
      )
    with self.assertRaises(ValueError):
      OrgGraph(name="Orphan", nodes=(hq, OrgNode(id="u1", label="U1", count=2)))
    with self.assertRaises(ValueError):
      OrgEdge(source="u1", target="u1")
    with self.assertRaises(ValueError):
      OrgGraph(name="Duplicate link", nodes=(hq, OrgNode(id="u1", label="U1")), edges=(OrgEdge(source="u1", target="hq"), OrgEdge(source="u1", target="hq")))

    # Node helpers keep model type and capacity consistent
    lin = OrgNode(id="u", label="U", model_type="linear")
    self.assertEqual((lin.with_overrides(h_agent=16).model_type, lin.with_overrides(h_agent=16).h_agent), ("neural", 16))
    self.assertEqual(OrgNode(id="u", label="U", h_agent=0).model_type, "linear")
    updated = GRAPH_TEMPLATES["m_form_3_tier"].update_node("div_a", upward_agg="max", lr=0.2)
    self.assertEqual([n.upward_agg for n in updated.nodes if n.id == "div_a"], ["max"])

  def test_webapp_http_endpoints(self):
    with _LabServer() as lab:
      with urllib.request.urlopen(f"{lab.base_url}/") as resp:
        self.assertEqual(resp.status, 200)
        html = resp.read().decode("utf-8")
      for text in ("Organizational Neural Search Laboratory", "Model &amp; Parameters Guide", "Add configuration", "config-bar"):
        self.assertIn(text, html)

      with urllib.request.urlopen(f"{lab.base_url}/api/catalog") as resp:
        self.assertEqual(resp.status, 200)
        cat = json.loads(resp.read().decode("utf-8"))
      self.assertEqual(cat["schema_version"], "0.1.0")
      self.assertEqual(len(cat["graph_templates"]), 12)
      self.assertEqual(set(cat["graph_preset_metadata"]), set(GRAPH_TEMPLATES))
      self.assertIn("market_defaults", cat)
      for choice in (
          "node_roles",
          "node_model_types",
          "node_capacities",
          "node_feedback",
          "upward_aggs",
          "feedback_regimes",
          "decision_rights",
          "nk_patterns",
          "turbulence",
      ):
        self.assertIn(choice, cat["field_choices"])
      self.assertEqual(cat["limits"]["max_firms"], 2048)

      market = {"nk_k": 3, "turbulence": "periodic", "shock_period": 50, "b_firms": 8, "t_steps": 12, "seed": 2026}
      status, comp = lab.post_json("/api/compile", {"org_graph": GRAPH_TEMPLATES["m_form_3_tier"].to_dict(), "market_config": market})
      self.assertEqual(status, 200)
      self.assertEqual(len(comp["org_graph"]["nodes"]), 7)
      self.assertEqual(comp["market_config"]["turbulence"], "periodic")
      self.assertEqual(comp["graph_explanation"]["tiers"], 3)

      bad_payloads = [
          {"unknown_field": "bad"},
          {"org_graph": GRAPH_TEMPLATES["m_form_3_tier"].to_dict(), "unknown_field": {}},
          {"org_graph": dict(GRAPH_TEMPLATES["enacted_solo_dc"].to_dict(), unknown_field=True)},
          {"org_graph": GRAPH_TEMPLATES["enacted_solo_dc"].to_dict(), "market_config": {"unknown_field": 3.0}},
      ]
      for payload in bad_payloads:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
          lab.post_json("/api/compile", payload)
        self.assertEqual(ctx.exception.code, 400, payload)

      with self.assertRaises(urllib.error.HTTPError) as ctx_get:
        urllib.request.urlopen(f"{lab.base_url}/api/stream")
      self.assertEqual(ctx_get.exception.code, 404)
      with self.assertRaises(urllib.error.HTTPError) as ctx_mode:
        lab.post("/api/stream", {"mode": "invalid", "org_graph": GRAPH_TEMPLATES["enacted_solo_dc"].to_dict()})
      self.assertEqual(ctx_mode.exception.code, 400)

  @unittest.skipUnless(HAS_JAX, "JAX is required for engine tests")
  def test_engine_runs_comparisons_and_streams(self):
    engine = _engine()
    market = MarketConfig(b_firms=8, t_steps=12)

    for key in ("m_form_3_tier", "matrix_dual_authority", "boundary_spanners"):
      res = engine.run_graph(GRAPH_TEMPLATES[key], market)
      self.assertEqual(
          set(res),
          {"wall_time_sec", "org_graph", "market_config", "graph_explanation", "summary"},
      )
      self.assertIn("orthogonality_linked", res["summary"])

    # A paired comparison and its streamed counterpart agree exactly
    pair = [GRAPH_TEMPLATES["m_form_3_tier"], GRAPH_TEMPLATES["coordinated_neural_hierarchy"]]
    cmp_out = engine.compare_graphs(*pair, market=market)
    self.assertIsNone(cmp_out["orgs"][0]["paired_vs_a"])
    self.assertIn("orthogonality_linked_p_value", cmp_out["orgs"][1]["paired_vs_a"])
    ev_cmp = list(engine.stream_graph_comparison(*pair, market=market, chunk_size=6))
    self.assertEqual([e["event"] for e in ev_cmp], ["start", "progress", "progress", "complete"])
    self.assertEqual(set(ev_cmp[1]["chunks"]), {"a", "b"})
    for streamed, direct in zip(ev_cmp[-1]["orgs"], cmp_out["orgs"]):
      self.assertEqual(streamed["summary"], direct["summary"])
      self.assertEqual(streamed["paired_vs_a"], direct["paired_vs_a"])

    # A single streamed run equals the full run
    ev_single = list(engine.stream_graph(GRAPH_TEMPLATES["matrix_dual_authority"], market, chunk_size=5))
    self.assertEqual([e["step"] for e in ev_single if e["event"] == "progress"], [5, 10, 12])
    self.assertIn("orthogonality_linked_mean", ev_single[1]["chunk"])
    self.assertNotIn("v_true_mean", ev_single[1])
    done = ev_single[-1]
    self.assertEqual(done["summary"], engine.run_graph(GRAPH_TEMPLATES["matrix_dual_authority"], market)["summary"])
    self.assertEqual(len(done["trajectories"]["v_true_mean"]), 12)

    # Up to 8 configurations on the same firms
    tiny = MarketConfig(b_firms=4, t_steps=6)
    eight = [GRAPH_TEMPLATES[k] for k in list(GRAPH_TEMPLATES)[:8]]
    multi = engine.compare_multi_graphs(eight, market=tiny)
    self.assertEqual(multi["slot_ids"], list("abcdefgh"))
    self.assertEqual(len(multi["orgs"]), 8)
    self.assertTrue(all(org["paired_vs_a"] is not None for org in multi["orgs"][1:]))
    ev_multi = list(engine.stream_multi_graph_comparison(eight, market=tiny, chunk_size=3))
    self.assertEqual(ev_multi[-1]["event"], "complete")
    self.assertEqual([o["summary"] for o in ev_multi[-1]["orgs"]], [o["summary"] for o in multi["orgs"]])
    with self.assertRaises(ValueError):
      engine.compare_multi_graphs(eight + [GRAPH_TEMPLATES["enacted_solo_dc"]], market=tiny)

    import jax.numpy as jnp
    import numpy as np

    # Configurable eval_every updates emergent metrics on multiples of eval_every (plus t1 and t_steps)
    mkt_ev5 = MarketConfig(b_firms=4, t_steps=16, eval_every=5)
    mkt_ev25 = MarketConfig(b_firms=4, t_steps=16, eval_every=25)
    res_ev5 = engine.run_graph(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], mkt_ev5, return_raw=True)
    res_ev25 = engine.run_graph(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], mkt_ev25, return_raw=True)
    acc5 = np.asarray(res_ev5["history"].dc_acc_global)
    acc25 = np.asarray(res_ev25["history"].dc_acc_global)
    # Period 1 (idx 0) and period 16 (idx 15) are evaluated under both settings
    self.assertTrue(np.allclose(acc5[:, 0], acc25[:, 0], atol=1e-6))
    self.assertTrue(np.allclose(acc5[:, 15], acc25[:, 15], atol=1e-6))
    # Under eval_every=25, periods 1..15 (indices 0..14) hold the period-1 evaluation
    for idx in range(1, 15):
      self.assertTrue(np.array_equal(acc25[:, idx], acc25[:, 0]), idx)
    # Under eval_every=5, period 5 (idx 4), period 10 (idx 9), and period 15 (idx 14) update
    self.assertFalse(np.array_equal(acc5[:, 4], acc5[:, 0]))
    self.assertTrue(np.array_equal(acc5[:, 3], acc5[:, 0]))
    self.assertFalse(np.array_equal(acc5[:, 9], acc5[:, 4]))
    self.assertTrue(np.array_equal(acc5[:, 8], acc5[:, 4]))
    self.assertEqual(set(res_ev5["summary"]["v_true"]), {"t1", "t5", "t10", "t15", "t16", "t_final"})
    self.assertEqual(set(res_ev25["summary"]["v_true"]), {"t1", "t16", "t_final"})

    self.assertEqual(engine.paired_ttest(jnp.array([0.3]), jnp.array([0.1]))[2:], (0.0, 1.0))
    # p-values follow the Student t distribution with n - 1 degrees of freedom (reference values
    # from scipy.stats.t: 2 * t.sf(|t|, df))
    for t_stat, df, ref in ((2.0, 7, 0.0856193286), (2.262157, 9, 0.0500000133), (1.0, 1, 0.5), (4.0, 63, 1.6905252e-4), (0.5, 2047, 0.6171288205)):
      self.assertAlmostEqual(engine._student_t_p_two_sided(t_stat, df) / ref, 1.0, places=7)
    self.assertEqual(engine._student_t_p_two_sided(0.0, 7), 1.0)
    _, _, t_8, p_8 = engine.paired_ttest(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), jnp.zeros(8))
    self.assertAlmostEqual(t_8, 4.5 / (2.449489743 / 8**0.5), places=4)
    self.assertAlmostEqual(p_8, engine._student_t_p_two_sided(t_8, 7), places=12)

  @unittest.skipUnless(HAS_JAX, "JAX is required for NK landscape tests")
  def test_nk_landscape_properties(self):
    import numpy as np

    engine = _engine()
    nbr = np.asarray(engine.get_hypercube_tables(12)[5])
    for pattern in ("random", "modular"):
      for nk_k in (0, 3, 11):
        v = np.asarray(engine.build_landscape_batch(2026, 8, 12, nk_k, pattern).all_v)
        self.assertLess(float(np.max(np.abs(v.mean(axis=-1)))), 1e-4)
        self.assertLess(float(np.max(np.abs(v.std(axis=-1) - 1.0))), 1e-3)
        n_opt = (v[:, :, None] > v[:, nbr]).all(axis=-1).sum(axis=-1)
        if nk_k == 0:
          self.assertTrue(np.all(n_opt == 1), f"K=0 must have exactly one local optimum, got {n_opt}")
        else:
          self.assertTrue(np.all(n_opt > 1))
      inputs = np.asarray(engine.build_landscape_batch(2026, 8, 12, 3, pattern).nk_inputs)
      self.assertEqual(inputs.shape, (8, 12, 4))
      for firm_rows in inputs:
        for i, row in enumerate(firm_rows):
          self.assertEqual(int(row[0]), i)
          self.assertEqual(len(set(row.tolist())), 4)
    modular = np.asarray(engine.build_landscape_batch(2026, 2, 12, 3, "modular").nk_inputs)[0]
    self.assertEqual(sorted(modular[5].tolist()), [4, 5, 6, 7])

    # Landscapes depend only on the market, so all organizations face identical landscapes
    market = MarketConfig(b_firms=4, t_steps=8)
    env_a = engine.run_graph(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], market, return_raw=True)["final_state"].env
    env_b = engine.run_graph(GRAPH_TEMPLATES["enacted_solo_dc"], market, return_raw=True)["final_state"].env
    env_0 = engine.build_landscape_batch(market.seed, 4, 12, 3, "random")
    self.assertTrue(np.array_equal(np.asarray(env_a.all_v), np.asarray(env_b.all_v)))
    self.assertTrue(np.array_equal(np.asarray(env_a.all_v), np.asarray(env_0.all_v)))

  @unittest.skipUnless(HAS_JAX, "JAX is required for payoff reference tests")
  def test_payoff_references(self):
    import jax
    import numpy as np

    engine = _engine()
    flips = np.arange(4096)[:, None] ^ (1 << np.arange(12))[None, :]

    # Local search with perfect knowledge ends between the average local peak and the best strategy
    env = engine.build_landscape_batch(2026, 16, 12, 3, "random")
    all_v = np.asarray(env.all_v)
    v_local = np.asarray(env.v_local_ref)
    is_peak = np.all(all_v[:, :, None] >= all_v[:, flips], axis=-1)
    peak_mean = np.array([all_v[f][is_peak[f]].mean() for f in range(16)])
    self.assertTrue(np.all(v_local > peak_mean), (v_local, peak_mean))
    self.assertTrue(np.all(v_local < np.asarray(env.v_max)))
    self.assertTrue(np.array_equal(np.asarray(env.v_max), all_v.max(axis=1)))
    # Steepest ascent from every start, step by step, agrees with the pointer-jumping reference
    cur = np.arange(4096)
    for _ in range(4096):
      cand = flips[cur, np.argmax(all_v[0][flips[cur]], axis=1)]
      nxt = np.where(all_v[0][cand] > all_v[0][cur], cand, cur)
      if np.array_equal(nxt, cur):
        break
      cur = nxt
    self.assertAlmostEqual(float(all_v[0][cur].mean()), float(v_local[0]), places=5)

    # K = 0 has a single peak, so local search reaches the best strategy from every start
    env_0 = engine.build_landscape_batch(2026, 8, 12, 0, "random")
    self.assertTrue(np.allclose(np.asarray(env_0.v_local_ref), np.asarray(env_0.v_max), atol=1e-5))

    # Share of strategies beaten: in [0, 1], 1 at the best strategy and 0 at the worst
    self.assertEqual(float(engine._payoff_percentile(env.all_v[0], env.v_max[0])), 1.0)
    self.assertEqual(float(engine._payoff_percentile(env.all_v[0], env.v_min[0])), 0.0)
    market = MarketConfig(b_firms=8, t_steps=12)
    res = engine.run_graph(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], market, return_raw=True)
    hist = res["history"]
    landscapes = engine._landscape_for_market(market)
    pct = np.asarray(hist.v_percentile)
    self.assertTrue(np.all((pct >= 0.0) & (pct <= 1.0)))
    below = (np.asarray(landscapes.all_v)[:, None, :] < np.asarray(hist.v_true)[:, :, None]).sum(axis=-1)
    self.assertTrue(np.allclose(pct, below / 4095.0, atol=1e-6))
    self.assertTrue(np.array_equal(np.asarray(hist.v_max_ref), np.repeat(np.asarray(landscapes.v_max)[:, None], 12, axis=1)))
    self.assertTrue(np.array_equal(np.asarray(hist.v_local_ref), np.repeat(np.asarray(landscapes.v_local_ref)[:, None], 12, axis=1)))

    # Summaries, trajectories, and paired tests carry the reference and local-peak metrics
    traj = engine.summarize_trajectories(hist)
    for key in ("v_percentile", "v_max_ref", "v_local_ref", "at_local_peak"):
      self.assertIn(key, res["summary"])
      self.assertIn(f"{key}_mean", traj)
    self.assertIn("v_percentile_p_value", engine._paired_metrics_dict(hist, hist))

    # A shock recomputes the references; between shocks they stay fixed
    periodic = MarketConfig(b_firms=4, t_steps=6, turbulence="periodic", shock_period=3)
    init_c, chunk_c = engine._graph_stream_closures(GRAPH_TEMPLATES["enacted_solo_dc"], periodic)
    keys = jax.random.split(jax.random.PRNGKey(periodic.seed), periodic.b_firms)
    state, step_keys = init_c(keys, engine._landscape_for_market(periodic))
    prev_ref = np.asarray(state.env.v_local_ref)
    for t in range(periodic.t_steps):
      state, hist_t = chunk_c(state, step_keys[:, t : t + 1, :], t)
      ref_t = np.asarray(state.env.v_local_ref)
      fresh = np.asarray(jax.vmap(engine._local_search_reference)(state.env.all_v))
      self.assertTrue(np.allclose(ref_t, fresh, atol=1e-6), t)
      self.assertTrue(np.array_equal(np.asarray(state.env.v_max), np.asarray(state.env.all_v).max(axis=1)), t)
      self.assertEqual(not np.array_equal(ref_t, prev_ref), (t + 1) % 3 == 0, t)
      self.assertTrue(np.array_equal(np.asarray(hist_t.v_local_ref[:, 0]), ref_t), t)
      prev_ref = ref_t

  @unittest.skipUnless(HAS_JAX, "JAX is required for turbulence tests")
  def test_turbulence_modes(self):
    import jax
    import numpy as np

    engine = _engine()

    def landscape_path(graph, market):
      """all_v after every period, stepping one period at a time."""
      init_c, chunk_c = engine._graph_stream_closures(graph, market)
      keys = jax.random.split(jax.random.PRNGKey(market.seed), market.b_firms)
      env0 = engine._landscape_for_market(market)
      state, step_keys = init_c(keys, env0)
      path = [np.asarray(env0.all_v)]
      for t in range(market.t_steps):
        state, _ = chunk_c(state, step_keys[:, t : t + 1, :], t)
        path.append(np.asarray(state.env.all_v))
      return path, np.asarray(step_keys)

    org_a = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]
    org_b = GRAPH_TEMPLATES["boundary_spanners"]

    # Periodic: shocks exactly at periods 4, 8, 12 and nowhere else, identical for both organizations
    periodic = MarketConfig(b_firms=4, t_steps=12, turbulence="periodic", shock_period=4)
    path_a, _ = landscape_path(org_a, periodic)
    path_b, _ = landscape_path(org_b, periodic)
    for v_a, v_b in zip(path_a, path_b):
      self.assertTrue(np.array_equal(v_a, v_b))
    for t in range(periodic.t_steps):
      changed = np.any(path_a[t + 1] != path_a[t], axis=-1)
      if (t + 1) % 4 == 0:
        self.assertTrue(np.all(changed), f"period {t + 1} should be shocked for every firm")
      else:
        self.assertFalse(np.any(changed), f"period {t + 1} must not be shocked")

    # Random: shocks occur exactly where the shared shock stream says
    p_shock = 0.3
    random_mkt = MarketConfig(b_firms=4, t_steps=12, turbulence="random", p_shock=p_shock)
    path_r, step_keys = landscape_path(org_a, random_mkt)
    path_rb, _ = landscape_path(org_b, random_mkt)
    expected = np.asarray(
        jax.vmap(jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(jax.random.fold_in(k, 101), 0), ()) < p_shock))(
            step_keys
        )
    )
    observed = np.stack([np.any(path_r[t + 1] != path_r[t], axis=-1) for t in range(12)], axis=1)
    self.assertTrue(np.array_equal(expected, observed))
    self.assertTrue(expected.any())
    for v_a, v_b in zip(path_r, path_rb):
      self.assertTrue(np.array_equal(v_a, v_b))

    # No turbulence: p_shock alone does not shock
    calm, _ = landscape_path(org_a, MarketConfig(b_firms=4, t_steps=6, p_shock=0.5))
    self.assertTrue(all(np.array_equal(calm[0], v) for v in calm))

  @unittest.skipUnless(HAS_JAX, "JAX is required for feedback tests")
  def test_feedback_and_track_record(self):
    import jax
    import numpy as np

    engine = _engine()
    market = MarketConfig(b_firms=4, t_steps=12)

    def states(graph):
      init_closure, chunk_closure = engine._graph_stream_closures(graph, market)
      keys = jax.random.split(jax.random.PRNGKey(market.seed), market.b_firms)
      st0, step_keys = init_closure(keys, engine._landscape_for_market(market))
      st1, _ = chunk_closure(st0, step_keys, 0)
      return st0, st1

    # Units that see no results and have no incoming links never change, even with experiments on
    dc_only = replace(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], feedback_regime="dc_only", experiment_rate=1.0)
    st0, st1 = states(dc_only)
    for r in range(1, 5):
      self.assertTrue(_same_rep_params(st0, st1, r, r), f"representation {r} changed without feedback or links")
    self.assertFalse(_same_rep_params(st0, st1, 0, 0))
    st0_d, st1_d = states(replace(dc_only, feedback_regime="dc_firm_units_domain"))
    self.assertFalse(_same_rep_params(st0_d, st1_d, 1, 1))

    # A domain covering every decision is the firm payoff with the firm noise draw
    full_unit = OrgGraph(
        name="Full-domain advisor",
        nodes=(OrgNode(id="hq", label="DC", role="hq"), OrgNode(id="u", label="Advisor")),
        edges=(OrgEdge(source="u", target="hq"),),
    )
    self.assertEqual(
        engine.run_graph(full_unit, market)["summary"],
        engine.run_graph(replace(full_unit, feedback_regime="everyone_firm"), market)["summary"],
    )

    # Track record updates a single-source receiver; receivers without feedback keep their weights
    tr_graph = OrgGraph(
        name="Single advisor",
        nodes=(
            OrgNode(id="hq", label="DC", role="hq"),
            OrgNode(id="u", label="Advisor", jurisdiction_bits=(0, 1, 2, 3, 4, 5)),
            OrgNode(id="w", label="Internal", jurisdiction_bits=(0, 1, 2, 3, 4, 5), feedback="none"),
        ),
        edges=(OrgEdge(source="u", target="hq", weight=0.3), OrgEdge(source="u", target="w", weight=0.4)),
        lambda_track=2.0,
    )
    perm_on = np.asarray(engine.run_graph(tr_graph, market, return_raw=True)["final_state"].perm_mat)
    perm_off = np.asarray(engine.run_graph(replace(tr_graph, lambda_track=0.0), market, return_raw=True)["final_state"].perm_mat)
    self.assertTrue(np.all(np.abs(perm_on[:, 0, 1] - np.float32(0.3)) > 1e-5))
    self.assertTrue(np.all(perm_on[:, 2, 1] == np.float32(0.4)))
    self.assertTrue(np.all(perm_off[:, 0, 1] == np.float32(0.3)))

  @unittest.skipUnless(HAS_JAX, "JAX is required for initialization tests")
  def test_random_initialization(self):
    import jax
    import numpy as np

    engine = _engine()
    market = MarketConfig(b_firms=8, t_steps=6)

    # Every representation starts from the standard random initialization, keyed by its position
    coord = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]
    st_plain = engine._initial_states(coord, market)

    def ref_init(base_key):
      ks = jax.random.split(jax.random.split(base_key, 2)[0], 8)
      rep_keys = jax.numpy.stack([ks[2]] + [jax.random.fold_in(ks[3], i) for i in range(4)])
      p = jax.vmap(lambda k: engine.init_agent(k, 12, 128))(rep_keys)
      return engine.AgentParams(w1=p.w1[..., :16], b1=p.b1[..., :16], w2=p.w2[:, :16], b2=p.b2)

    # Jitted like the kernel (eager and fused normal sampling can differ in the last bit)
    expected = jax.jit(jax.vmap(ref_init))(jax.random.split(jax.random.PRNGKey(market.seed), market.b_firms))
    for got, want in zip(st_plain.params, expected):
      self.assertTrue(np.array_equal(np.asarray(got), np.asarray(want)))
    self.assertTrue(np.array_equal(np.asarray(st_plain.tables), np.zeros_like(np.asarray(st_plain.tables))))

  def test_legacy_initial_accuracy_keys(self):
    # Files saved by earlier versions may carry initial-accuracy settings; loading ignores them
    base = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]
    legacy = base.to_dict()
    legacy["unit_initial_accuracy"] = 0.6
    legacy["nodes"][0]["initial_accuracy"] = 0.3
    legacy["nodes"][1]["initial_accuracy"] = "uninformed"
    legacy["nodes"][2]["initial_accuracy"] = None
    loaded = OrgGraph.from_dict(legacy)
    self.assertEqual(loaded, base)
    self.assertEqual(loaded.to_dict(), base.to_dict())
    self.assertEqual(OrgNode.from_dict({"id": "u", "label": "U", "initial_accuracy": 0.5}), OrgNode(id="u", label="U"))
    # The hardware field of version 0.2.0 market settings is ignored as well
    self.assertEqual(MarketConfig.from_dict({"device": "gpu", "seed": 7}), MarketConfig(seed=7))
    # Other unknown keys and non-object values still raise
    with self.assertRaises(ValueError):
      OrgNode.from_dict({"id": "u", "label": "U", "prior": 0.5})
    with self.assertRaises(ValueError):
      OrgGraph.from_dict(dict(base.to_dict(), unknown_setting=1))
    for cls in (MarketConfig, OrgGraph, OrgNode):
      for bad in (None, "abc", [["seed", 7]]):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
          cls.from_dict(bad)

  @unittest.skipUnless(HAS_JAX, "JAX is required for decision-rights and mask tests")
  def test_decision_rights_search_and_masks(self):
    import jax
    import numpy as np

    engine = _engine()
    market = MarketConfig(b_firms=4, t_steps=10)

    # With a solo DC every decision-rights mode reduces to centralized search
    solo = GRAPH_TEMPLATES["enacted_solo_dc"]
    s_central = engine.run_graph(solo, market)["summary"]
    for mode in ("propose_approve", "delegated"):
      self.assertEqual(engine.run_graph(replace(solo, decision_rights=mode), market)["summary"], s_central)

    # Each mode runs on a 3-tier organization; delegation leaves the DC no bits of its own here
    m_form = GRAPH_TEMPLATES["m_form_3_tier"]
    for mode in ("centralized", "propose_approve", "delegated"):
      res = engine.run_graph(replace(m_form, decision_rights=mode, aspiration_sensitivity=2.0), market)
      exp = res["graph_explanation"]
      self.assertEqual(exp["effective_decision_rights"], mode)
      self.assertTrue(np.isfinite(res["summary"]["v_true"]["t_final"]["mean"]))
      if mode == "delegated":
        self.assertEqual(exp["dc_search_bits"], [])
        self.assertTrue(any("does not search" in note for note in exp["notes"]))

    # Units search with the DC's rule: with rho = 0 a unit with search_steps = 1 changes at most one
    # bit of its domain per period, one with search_steps = 3 at most three
    two_units = OrgGraph(
        name="Two searching units",
        nodes=(
            OrgNode(id="hq", label="DC", role="hq"),
            OrgNode(id="u1", label="Unit 1", jurisdiction_bits=(0, 1, 2, 3, 4, 5)),
            OrgNode(id="u2", label="Unit 2", jurisdiction_bits=(6, 7, 8, 9, 10, 11), search_steps=3),
        ),
        edges=(OrgEdge(source="u1", target="hq"), OrgEdge(source="u2", target="hq")),
    )
    keys = jax.random.split(jax.random.PRNGKey(market.seed), market.b_firms)
    for mode in ("delegated", "propose_approve"):
      init_c, chunk_c = engine._graph_stream_closures(replace(two_units, decision_rights=mode), market)
      state, step_keys = init_c(keys, engine._landscape_for_market(market))
      prev_s = np.asarray(state.s)
      moved = 0
      for t in range(market.t_steps):
        state, _ = chunk_c(state, step_keys[:, t : t + 1, :], t)
        cur_s = np.asarray(state.s)
        changed = prev_s != cur_s
        self.assertLessEqual(int(changed[:, :6].sum(axis=1).max()), 1, mode)
        self.assertLessEqual(int(changed[:, 6:].sum(axis=1).max()), 3, mode)
        moved += int(changed[:, :6].sum())
        prev_s = cur_s
      self.assertGreater(moved, 0, mode)

    # Mask consistency: a prediction never responds to a bit outside the node's domain
    res_m = engine.run_graph(m_form, market, return_raw=True)
    rep_ids = engine.graph_representation_ids(m_form)
    node_bits = {n.id: set(n.jurisdiction_bits) for n in m_form.nodes}
    base_s = np.random.default_rng(7).choice([-1.0, 1.0], size=(12,)).astype(np.float32)
    queries = [base_s]
    for b in range(12):
      q = base_s.copy()
      q[b] = -q[b]
      queries.append(q)
    preds = np.asarray(engine.predict_graph_representations(m_form, market, res_m["final_state"], np.stack(queries)))
    self.assertEqual(preds.shape, (4, len(rep_ids), 13))
    for r, nid in enumerate(rep_ids):
      for b in sorted(set(range(12)) - node_bits[nid]):
        # Rows of one batched evaluation may differ by float32 rounding only
        self.assertLess(float(np.max(np.abs(preds[:, r, 0] - preds[:, r, 1 + b]))), 1e-6, f"{nid} reacted to bit {b}")
      self.assertTrue(any(float(np.max(np.abs(preds[:, r, 0] - preds[:, r, 1 + b]))) > 1e-3 for b in node_bits[nid]))

  @unittest.skipUnless(HAS_JAX, "JAX is required for sweep tests")
  def test_sweeps(self):
    engine = _engine()
    market = MarketConfig(b_firms=4, t_steps=6)
    base = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]
    cases = [
        ("decision_rights", ["centralized", "delegated"], True),
        ("p_shock", [0.0, 0.2], True),
        ("shock_period", [3], None),
        ("rep_1.sigma_learn", [0.12], None),
        ("nk_k", [1, 4], False),
    ]
    results = {}
    for param, values, paired in cases:
      events = list(engine.stream_graph_sweep(base, market, sweep_param=param, sweep_values=values, chunk_size=3))
      self.assertEqual(events[-1]["event"], "complete", param)
      points = events[-1]["points"]
      self.assertEqual(len(points), len(values), param)
      results[param] = points
      if paired is not None:
        self.assertEqual(points[1]["paired_vs_baseline"] is not None, paired, param)
    self.assertEqual([p["market_config"]["turbulence"] for p in results["p_shock"]], ["random", "random"])
    self.assertEqual(results["shock_period"][0]["market_config"]["turbulence"], "periodic")
    rep_1 = next(n for n in results["rep_1.sigma_learn"][0]["org_graph"]["nodes"] if n["id"] == "rep_1")
    self.assertEqual(rep_1["sigma_learn"], 0.12)
    for bad_param in ("unknown_param", "rep_1.unknown_attr", "unit_initial_accuracy", "rep_1.initial_accuracy"):
      with self.assertRaises(ValueError, msg=bad_param):
        list(engine.stream_graph_sweep(base, market, sweep_param=bad_param, sweep_values=[0.1]))

  @unittest.skipUnless(HAS_JAX, "JAX is required for HTTP streaming tests")
  def test_api_multi_stream(self):
    graph = replace(GRAPH_TEMPLATES["boundary_spanners"], decision_rights="propose_approve", experiment_rate=0.5)
    market = {"nk_k": 2, "nk_pattern": "modular", "turbulence": "periodic", "shock_period": 3, "b_firms": 4, "t_steps": 8, "seed": 2026}
    with _LabServer() as lab:
      lines = lab.stream({"mode": "single", "org_graph": graph.to_dict(), "market_config": market, "chunk_size": 4})
      self.assertEqual([e["event"] for e in lines], ["start", "progress", "progress", "complete"])
      self.assertEqual(lines[0]["graph_explanation"]["node_feedback"]["front_1"], "domain")
      self.assertEqual(lines[0]["graph_explanation"]["landscape"]["turbulence"], "periodic")
      done = lines[-1]
      self.assertEqual(done["market_config"]["shock_period"], 3)
      self.assertEqual(set(done), {"event", "mode", "wall_time_sec", "org_graph", "market_config", "graph_explanation", "summary", "trajectories"})
      self.assertEqual(len(done["trajectories"]["orthogonality_linked_mean"]), 8)

      regimes = ("dc_firm_units_domain", "everyone_firm", "dc_only")
      orgs = [
          {"slot_id": f"s{i}", "label": regime, "org_graph": replace(GRAPH_TEMPLATES["coordinated_neural_hierarchy"], feedback_regime=regime).to_dict()}
          for i, regime in enumerate(regimes)
      ]
      c_lines = lab.stream({"mode": "comparison", "market_config": market, "orgs": orgs, "chunk_size": 4})
      self.assertEqual(c_lines[0]["slot_ids"], ["s0", "s1", "s2"])
      self.assertEqual(set(c_lines[1]["chunks"]), {"s0", "s1", "s2"})
      final = c_lines[-1]
      self.assertEqual(final["labels"], list(regimes))
      self.assertIsNone(final["orgs"][0]["paired_vs_a"])
      self.assertIn("v_true_diff", final["orgs"][2]["paired_vs_a"])
      self.assertEqual({v for k, v in final["orgs"][2]["graph_explanation"]["node_feedback"].items() if k != "hq"}, {"none"})

  @unittest.skipUnless(HAS_JAX, "JAX is required for the CLI run test")
  def test_cli_run(self):
    engine = _engine()
    experiment = {
        "schema_version": "0.1.0",
        "market_config": MarketConfig(b_firms=4, t_steps=6).to_dict(),
        "configurations": [
            {"name": "Coordinated", "org_graph": GRAPH_TEMPLATES["coordinated_neural_hierarchy"].to_dict(), "layout": {"hq": {"x": 10, "y": 20}}},
            {"name": "Conformity trap", "org_graph": GRAPH_TEMPLATES["conformity_trap"].to_dict(), "layout": {}},
        ],
        "simulation_output": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "experiment.json")
      out = os.path.join(tmp, "results.json")
      with open(path, "w") as f:
        json.dump(experiment, f)
      self.assertEqual(cli_main(["run", path, "--out", out]), 0)
      with open(out) as f:
        saved = json.load(f)
      self.assertEqual([c["name"] for c in saved["configurations"]], ["Coordinated", "Conformity trap"])
      self.assertEqual(saved["configurations"][0]["layout"], {"hq": {"x": 10, "y": 20}})
      self.assertEqual(saved["schema_version"], "0.1.0")
      sim = saved["simulation_output"]
      self.assertEqual((sim["event"], sim["mode"]), ("complete", "comparison"))
      self.assertEqual(sim["labels"], ["Coordinated", "Conformity trap"])
      self.assertEqual(len(sim["orgs"]), 2)
      self.assertIsNone(sim["orgs"][0]["paired_vs_a"])
      self.assertIn("v_true_t_stat", sim["orgs"][1]["paired_vs_a"])
      # The output equals the web laboratory's comparison stream on the same file
      market = MarketConfig.from_dict(dict(experiment["market_config"]))
      streamed = list(
          engine.stream_multi_graph_comparison(
              [GRAPH_TEMPLATES["coordinated_neural_hierarchy"], GRAPH_TEMPLATES["conformity_trap"]],
              market=market,
              labels=["Coordinated", "Conformity trap"],
          )
      )[-1]
      self.assertEqual([o["summary"] for o in sim["orgs"]], [o["summary"] for o in streamed["orgs"]])
      # The output file is itself a valid experiment file
      self.assertEqual(cli_main(["run", out]), 0)

      # One configuration: the complete event of a single run
      single = dict(experiment, configurations=experiment["configurations"][:1])
      with open(path, "w") as f:
        json.dump(single, f)
      self.assertEqual(cli_main(["run", path, "--out", out]), 0)
      with open(out) as f:
        sim_single = json.load(f)["simulation_output"]
      self.assertEqual((sim_single["event"], sim_single["mode"]), ("complete", "single"))
      self.assertEqual(sim_single["summary"], sim["orgs"][0]["summary"])

      old = dict(experiment, schema_version="0.0.0")
      with open(path, "w") as f:
        json.dump(old, f)
      self.assertEqual(cli_main(["run", path]), 2)

  @unittest.skipUnless(HAS_JAX, "JAX is required for sweep override tests")
  def test_every_sweep_param_applies(self):
    # Every listed sweep parameter and node attribute changes the graph or market (no simulation)
    engine = _engine()
    market = MarketConfig(b_firms=2, t_steps=10)
    base = GRAPH_TEMPLATES["bundled_neural_hierarchy"]
    samples = {
        "nk_k": 1, "p_shock": 0.1, "shock_period": 7, "sigma_obs": 0.2, "experiment_rate": 0.9,
        "aspiration_sensitivity": 1.5, "lambda_track": 2.0, "decision_rights": "delegated",
        "feedback_regime": "dc_only", "rho": 0.4, "sigma_learn": 0.3, "h_agent": 4,
        "commensurability": 0.3, "permeability": 0.6, "upward_permeability": 0.6, "downward_permeability": 0.0,
    }
    self.assertEqual(set(samples), set(engine.GRAPH_SWEEP_PARAMS))
    for param, value in samples.items():
      g, m, val = engine._apply_graph_sweep_override(base, market, param, str(value))
      self.assertTrue(g != base or m != market, param)
      g.validate(n_bits=m.n_bits)
    node_samples = {
        "count": 3, "h_agent": 8, "search_steps": 2, "upward_agg": "max", "model_type": "tabular",
        "feedback": "none", "lr": 0.5, "sigma_learn": 0.3, "rho": 0.5,
    }
    self.assertEqual(set(node_samples), set(engine._NODE_SWEEP_ATTRS))
    for attr, value in node_samples.items():
      g, _, val = engine._apply_graph_sweep_override(base, market, f"rep_1.{attr}", str(value))
      self.assertEqual(getattr(next(n for n in g.nodes if n.id == "rep_1"), attr), val, attr)
    for removed in ("unit_initial_accuracy", "rep_1.initial_accuracy", "device"):
      with self.assertRaises(ValueError, msg=removed):
        engine._apply_graph_sweep_override(base, market, removed, "0.5")

  @unittest.skipUnless(HAS_JAX, "JAX is required for HTTP streaming tests")
  def test_http_sweep_sse_legacy_and_bad_bodies(self):
    solo = GRAPH_TEMPLATES["enacted_solo_dc"].to_dict()
    market = {"b_firms": 2, "t_steps": 4, "seed": 3}
    with _LabServer() as lab:
      # Sweep mode as server-sent events
      req = urllib.request.Request(
          f"{lab.base_url}/api/stream",
          data=json.dumps({"mode": "sweep", "org_graph": solo, "market_config": market, "sweep_param": "rho", "sweep_values": "0, 0.3", "chunk_size": 2}).encode("utf-8"),
          headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
          method="POST",
      )
      with urllib.request.urlopen(req) as resp:
        self.assertIn("text/event-stream", resp.headers["Content-Type"])
        events = [json.loads(line[5:]) for line in resp.read().decode("utf-8").splitlines() if line.startswith("data:")]
      self.assertEqual(events[0]["event"], "start")
      self.assertEqual([e["event"] for e in events].count("point_complete"), 2)
      self.assertEqual(events[-1]["event"], "complete")
      self.assertEqual(events[-1]["sweep_values"], [0.0, 0.3])
      self.assertIsNotNone(events[-1]["points"][1]["paired_vs_baseline"])

      # Fields of earlier versions (device, initial accuracy) are dropped, not rejected
      legacy = json.loads(json.dumps(GRAPH_TEMPLATES["coordinated_neural_hierarchy"].to_dict()))
      legacy["unit_initial_accuracy"] = 0.6
      legacy["nodes"][1]["initial_accuracy"] = 0.7
      status, comp = lab.post_json("/api/compile", {"org_graph": legacy, "market_config": dict(market, device="gpu")})
      self.assertEqual(status, 200)
      self.assertNotIn("device", comp["market_config"])
      self.assertNotIn("initial_accuracy", json.dumps(comp))

      # Malformed bodies and removed sweep parameters get 400 responses with a JSON error message
      for raw in (b"{not json", b"[1, 2]"):
        for path in ("/api/compile", "/api/stream"):
          req = urllib.request.Request(f"{lab.base_url}{path}", data=raw, headers={"Content-Type": "application/json"}, method="POST")
          with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
          self.assertEqual(ctx.exception.code, 400, (path, raw))
          self.assertIn("error", json.loads(ctx.exception.read().decode("utf-8")))
      for param in ("unit_initial_accuracy", "hq.initial_accuracy"):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
          lab.post("/api/stream", {"mode": "sweep", "org_graph": solo, "market_config": market, "sweep_param": param, "sweep_values": [0.5]})
        self.assertEqual(ctx.exception.code, 400, param)

  def test_cli_and_server_arg_parsers(self):
    import contextlib
    import io

    if __package__:
      from . import cli, webapp
    else:
      import cli  # type: ignore
      import webapp  # type: ignore

    args = cli.build_arg_parser().parse_args(["ui", "--port", "8090", "--no-browser"])
    self.assertEqual((args.command, args.port, args.no_browser), ("ui", 8090, True))
    args = cli.build_arg_parser().parse_args(["run", "experiment.json", "--out", "results.json"])
    self.assertEqual((args.command, args.experiment, args.out), ("run", "experiment.json", "results.json"))
    args = webapp.build_arg_parser().parse_args(["--host", "0.0.0.0", "--port", "9000", "--no-browser"])
    self.assertEqual((args.host, args.port, args.no_browser), ("0.0.0.0", 9000, True))
    # The removed hardware option is no longer accepted
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit):
        cli.build_arg_parser().parse_args(["run", "experiment.json", "--device", "gpu"])

    # An experiment file saved by an earlier version loads with its legacy fields dropped
    base = GRAPH_TEMPLATES["coordinated_neural_hierarchy"]
    legacy_graph = base.to_dict()
    legacy_graph["unit_initial_accuracy"] = None
    for node in legacy_graph["nodes"]:
      node["initial_accuracy"] = None
    experiment = {
        "schema_version": "0.1.0",
        "market_config": dict(MarketConfig(b_firms=2, t_steps=10).to_dict(), device="auto"),
        "configurations": [{"name": "Old", "org_graph": legacy_graph, "layout": {}}],
        "simulation_output": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "old.json")
      with open(path, "w") as f:
        json.dump(experiment, f)
      market, named, _ = cli.load_experiment(path)
    self.assertEqual(market, MarketConfig(b_firms=2, t_steps=10))
    self.assertEqual(named, [("Old", base)])

  def test_input_bounds_and_cli_errors(self):
    hq = OrgNode(id="hq", label="DC", role="hq")
    unit = OrgNode(id="u", label="Unit")

    # Non-finite numbers raise instead of propagating NaN or inf into the engine
    with self.assertRaises(ValueError):
      OrgEdge(source="u", target="hq", weight=float("nan"))
    with self.assertRaises(ValueError):
      MarketConfig(sigma_obs=float("inf"))

    # A node without an id raises an error that names the field
    with self.assertRaisesRegex(ValueError, "id"):
      OrgNode.from_dict({"label": "No id"})

    # Market bounds: seeds must fit in 32 bits, and n_bits must lie in [2, 16]
    self.assertEqual((MIN_N_BITS, MAX_N_BITS), (2, 16))
    with self.assertRaises(ValueError):
      MarketConfig(seed=2**32)
    with self.assertRaises(ValueError):
      MarketConfig(n_bits=1)
    with self.assertRaises(ValueError):
      MarketConfig(n_bits=17)
    self.assertEqual(MarketConfig(n_bits=2, nk_k=1).n_bits, 2)
    self.assertEqual(MarketConfig(n_bits=16, nk_k=3).n_bits, 16)
    with self.assertRaises(ValueError):
      GRAPH_TEMPLATES["m_form_3_tier"].validate(n_bits=8)
    with self.assertRaises(ValueError):
      GRAPH_TEMPLATES["m_form_3_tier"].explain_graph(MarketConfig(n_bits=8, nk_k=2))

    # Depth is the undirected link distance from the DC, so a downward chain of 4 tiers is rejected
    chain = (
        hq,
        OrgNode(id="a", label="A", role="manager"),
        OrgNode(id="b", label="B"),
        OrgNode(id="c", label="C"),
    )
    with self.assertRaisesRegex(ValueError, "tiers"):
      OrgGraph(
          name="Downward chain",
          nodes=chain,
          edges=(OrgEdge(source="hq", target="a"), OrgEdge(source="a", target="b"), OrgEdge(source="b", target="c")),
      )

    # The CLI reports malformed experiment files with exit code 2 instead of raising
    good_graph = OrgGraph(name="Pair", nodes=(hq, unit), edges=(OrgEdge(source="u", target="hq"),)).to_dict()
    no_id = json.loads(json.dumps(good_graph))
    del no_id["nodes"][1]["id"]
    nan_weight = json.loads(json.dumps(good_graph))
    nan_weight["edges"][0]["weight"] = float("nan")
    with tempfile.TemporaryDirectory() as tmp:
      for bad_graph in (no_id, nan_weight):
        path = os.path.join(tmp, "bad.json")
        experiment = {
            "schema_version": "0.1.0",
            "market_config": MarketConfig(b_firms=2, t_steps=2).to_dict(),
            "configurations": [{"name": "Bad", "org_graph": bad_graph}],
        }
        with open(path, "w") as f:
          json.dump(experiment, f)
        self.assertEqual(cli_main(["run", path]), 2)

  @unittest.skipUnless(HAS_JAX, "JAX is required for configurable N tests")
  def test_configurable_n_bits(self):
    import numpy as np

    engine = _engine()

    # Proportional domain rescaling across N = 2, 8, 12, 16
    self.assertEqual(rescale_jurisdiction_bits(range(6), 12, 8), (0, 1, 2, 3))
    self.assertEqual(rescale_jurisdiction_bits(range(6, 12), 12, 8), (4, 5, 6, 7))
    self.assertEqual(rescale_jurisdiction_bits(range(6), 12, 16), tuple(range(8)))
    self.assertEqual(rescale_jurisdiction_bits(range(6, 12), 12, 16), tuple(range(8, 16)))
    self.assertEqual(rescale_jurisdiction_bits(range(6), 12, 2), (0,))
    self.assertEqual(rescale_jurisdiction_bits(range(6, 12), 12, 2), (1,))
    self.assertEqual(rescale_jurisdiction_bits((0, 1, 2, 6, 7, 8), 12, 2), (0, 1))

    for n_bits in (2, 8, 16):
      for key, tmpl in GRAPH_TEMPLATES.items():
        rescaled = tmpl.with_n_bits(n_bits)
        rescaled.validate(n_bits=n_bits)
        self.assertEqual(rescaled.get_hq_node().jurisdiction_bits, tuple(range(n_bits)), (key, n_bits))
      m8 = GRAPH_TEMPLATES["m_form_3_tier"].with_n_bits(8)
      div_a = next(n for n in m8.nodes if n.id == "div_a")
      div_b = next(n for n in m8.nodes if n.id == "div_b")
      self.assertIn("Bits 1-4", div_a.label)
      self.assertIn("Bits 5-8", div_b.label)
      self.assertIn("Bits 1-4", m8.description)
      self.assertIn("Bits 5-8", m8.description)

    # Engine runs and comparisons at n_bits = 2 (k = 0 and k = 1), n_bits = 8 (k = 2), and n_bits = 16 (k = 3)
    cases = (
        (2, 0, 6, 8),
        (2, 1, 6, 8),
        (8, 2, 6, 8),
        (16, 3, 4, 10),
    )
    for n_bits, nk_k, b_firms, t_steps in cases:
      mkt = MarketConfig(n_bits=n_bits, nk_k=nk_k, b_firms=b_firms, t_steps=t_steps)
      coord = GRAPH_TEMPLATES["coordinated_neural_hierarchy"].with_n_bits(n_bits)
      m_form = GRAPH_TEMPLATES["m_form_3_tier"].with_n_bits(n_bits)
      res = engine.run_graph(coord, mkt, return_raw=True)
      self.assertEqual(res["market_config"]["n_bits"], n_bits)
      self.assertEqual(res["graph_explanation"]["landscape"]["n_bits"], n_bits)
      for metric in ("v_true", "v_percentile", "v_max_ref", "v_local_ref", "at_local_peak"):
        val = res["summary"][metric]["t_final"]["mean"]
        self.assertTrue(np.isfinite(val), (n_bits, nk_k, metric, val))
      pct = np.asarray(res["history"].v_percentile)
      self.assertTrue(np.all((pct >= 0.0) & (pct <= 1.0)), (n_bits, nk_k))
      if nk_k == 0:
        self.assertAlmostEqual(
            res["summary"]["v_local_ref"]["t_final"]["mean"],
            res["summary"]["v_max_ref"]["t_final"]["mean"],
            places=4,
        )
      cmp_res = engine.compare_multi_graphs([coord, m_form], market=mkt)
      self.assertEqual(len(cmp_res["orgs"]), 2)
      self.assertIn("v_percentile_p_value", cmp_res["orgs"][1]["paired_vs_a"])


if __name__ == "__main__":
  unittest.main()

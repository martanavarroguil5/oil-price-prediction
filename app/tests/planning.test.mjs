import test from "node:test";
import assert from "node:assert/strict";
import { calculatePlan } from "../static/planning.js";

const base = { volume: 100, budget: 11000, reference: 100, median: 110,
  low: 80, high: 140, share: 0.5, shock: 20, fee: 2 };

test("costs apply the unit fee once and preserve total volume", () => {
  const result = calculatePlan(base);
  const byId = Object.fromEntries(result.strategies.map(s => [s.id, s]));
  assert.equal(byId.now.costs.low, 10200);
  assert.equal(byId.now.costs.high, 10200);
  assert.equal(byId.later.costs.central, 11200);
  assert.equal(byId.split.costs.central, 10700);
  assert.equal(byId.split.costs.low, 9200);
  assert.equal(byId.split.costs.high, 12200);
  assert.equal(byId.split.margin, 300);
  assert.equal(byId.split.sensitivity, 50);
  assert.equal(result.prices.stress, 132);
  assert.equal(result.ceiling, 108);
});

test("allocating zero or all volume matches the corresponding strategy", () => {
  for (const share of [0, 1]) {
    const strategies = calculatePlan({ ...base, share }).strategies;
    assert.deepEqual(strategies[1].costs, strategies[share === 0 ? 2 : 0].costs);
  }
});

test("a manual decline changes only the manual scenario, not the forecast limits", () => {
  const a = calculatePlan(base), b = calculatePlan({ ...base, shock: -50 });
  assert.equal(b.prices.stress, 55);
  for (let i = 0; i < a.strategies.length; i++)
    for (const key of ["low", "central", "high"])
      assert.equal(a.strategies[i].costs[key], b.strategies[i].costs[key]);
});

test("invalid values cannot produce an apparently valid budget", () => {
  for (const override of [{ volume: 0 }, { share: 1.1 }, { fee: -1 },
    { median: NaN }, { shock: -100 }, { low: 120 }, { high: 90 }])
    assert.throws(() => calculatePlan({ ...base, ...override }));
});

import test from "node:test";
import assert from "node:assert/strict";
import { forecastMetrics, scatterData } from "../static/scatter-stats.js";

test("predictive R² is negative even when points are perfectly correlated",()=>{
  const actual=[90,100,110], predicted=[80,90,100];
  assert.equal(forecastMetrics(actual,predicted).r2,-.5);
  assert.equal(forecastMetrics(actual,actual).r2,1);
  assert.equal(forecastMetrics(actual,[100,100,100]).r2,0);
});
test("undefined R² and inconsistent arrays are handled explicitly",()=>{
  assert.equal(forecastMetrics([1,1],[2,2]).r2,null);
  assert.throws(()=>forecastMetrics([1],[1,2]));
});
test("return scatter uses the origin price and the zero-return baseline",()=>{
  const result=scatterData([["2026-01-01",110,105,100],["2026-01-02",180,190,200]],"retorno");
  assert.equal(result.points[0].x,10);
  assert.equal(result.points[0].y,5);
  assert.equal(result.points[1].x,-10);
  assert.equal(result.points[1].y,-5);
  assert.deepEqual(result.naive.map(p=>p.y),[0,0]);
  assert.equal(result.metrics.mae,5);
});

/** R² of the original forecasts, never the R² of a fitted trend line. */
export function forecastMetrics(actual, predicted) {
  if (!actual.length || actual.length !== predicted.length ||
      ![...actual, ...predicted].every(Number.isFinite)) throw Error("Datos de evaluación no válidos.");
  const mean = actual.reduce((a,b)=>a+b,0)/actual.length;
  const sse = predicted.reduce((sum,p,i)=>sum+(p-actual[i])**2,0);
  const sst = actual.reduce((sum,a)=>sum+(a-mean)**2,0);
  return { r2: sst > 0 ? 1-sse/sst : null,
    mae: predicted.reduce((sum,p,i)=>sum+Math.abs(p-actual[i]),0)/actual.length,
    rmse: Math.sqrt(sse/actual.length) };
}

export function scatterData(rows, unit="precio") {
  const real=[], predicted=[], base=[];
  for (const row of rows) {
    const factor = unit === "retorno" ? 100/row[3] : 1;
    const offset = unit === "retorno" ? 100 : 0;
    if (unit === "retorno" && row[3] <= 0) throw Error("Precio de origen no positivo.");
    real.push(row[1]*factor-offset);
    predicted.push(row[2]*factor-offset);
    base.push(row[3]*factor-offset);
  }
  return { points:real.map((x,i)=>({x,y:predicted[i],date:rows[i][0]})),
    naive:real.map((x,i)=>({x,y:base[i],date:rows[i][0]})),
    metrics:forecastMetrics(real,predicted), baseline:forecastMetrics(real,base) };
}

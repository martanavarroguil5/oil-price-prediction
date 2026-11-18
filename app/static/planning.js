/** Transparent endpoint scenarios, not an optimiser or a trading recommendation. */
export function calculatePlan({ volume, budget, reference, median, low, high, share, shock, fee }) {
  const values = [volume, budget, reference, median, low, high, share, shock, fee];
  if (!values.every(Number.isFinite) || volume <= 0 || budget < 0 || reference <= 0 ||
      low <= 0 || median < low || high < median || share < 0 || share > 1 ||
      shock <= -100 || fee < 0) throw new Error("Revisa los valores del escenario.");
  const prices = { low, central: median, high, stress: median * (1 + shock / 100) };
  const strategies = [
    { id: "now", label: "Todo al precio de referencia", locked: 1 },
    { id: "split", label: "Repartir la compra", locked: share },
    { id: "later", label: "Todo al final del plazo", locked: 0 },
  ].map((strategy) => {
    const costs = Object.fromEntries(Object.entries(prices).map(([key, price]) =>
      [key, volume * (strategy.locked * reference + (1 - strategy.locked) * price + fee)]));
    return { ...strategy, costs, margin: budget - costs.central,
      sensitivity: volume * (1 - strategy.locked) };
  });
  return { prices, strategies, ceiling: budget / volume - fee };
}

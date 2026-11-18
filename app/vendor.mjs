import { build } from "esbuild";
import { mkdir, copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

process.chdir(fileURLToPath(new URL(".", import.meta.url)));
const output = "static/vendor";
await mkdir(output, { recursive: true });
for (const [name, contents] of Object.entries({
  three:
    "export * from 'three'; export { OrbitControls } from 'three/addons/controls/OrbitControls.js';",
  chart: "export { default } from 'chart.js/auto';",
})) {
  await build({
    stdin: { contents, resolveDir: process.cwd() },
    outfile: output + "/" + name + ".js",
    bundle: true,
    minify: true,
    format: "esm",
    legalComments: "inline",
  });
}
for (const [family, weights] of [
  ["barlow", [400, 500, 600, 700]],
  ["ibm-plex-mono", [400, 500]],
]) {
  const root = "node_modules/@fontsource/" + family;
  for (const weight of weights) {
    const file = family + "-latin-" + weight + "-normal.woff2";
    await copyFile(root + "/files/" + file, output + "/" + file);
  }
  await copyFile(root + "/LICENSE", output + "/" + family + "-LICENSE.txt");
}
for (const name of ["three", "chart.js"]) {
  await copyFile(
    "node_modules/" + name + "/LICENSE" + (name === "chart.js" ? ".md" : ""),
    output + "/" + name + "-LICENSE.txt",
  );
}

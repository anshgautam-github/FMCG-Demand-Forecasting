// Builds reports/FreshBasket_Forecasting_Presentation.pptx from outputs/figures.
// Run:  node tools/build_deck.js   (needs pptxgenjs; skill theme helper optional)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (n) => path.join(ROOT, "outputs", "figures", n);
const OUT = path.join(ROOT, "reports", "FreshBasket_Forecasting_Presentation.pptx");
fs.mkdirSync(path.dirname(OUT), { recursive: true });

const THEME = {
  name: "FreshBasket",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A41", lt1: "FFFFFF", dk2: "1F5F8B", lt2: "F3F6F9",
    accent1: "1F5F8B", accent2: "E07A1F", accent3: "3A9D5D", accent4: "C0392B",
    accent5: "7B6FB0", accent6: "9AA5B1", hlink: "1F5F8B", folHlink: "7B6FB0",
  },
};
const NAVY = "1B2A41", BLUE = "1F5F8B", ORANGE = "E07A1F", GREEN = "3A9D5D", RED = "C0392B", GREY = "5B6B7B", LIGHT = "F3F6F9";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "FreshBasket FMCG demand forecasting";

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: NAVY },
  objects: [
    { rect: { x: 0, y: 5.25, w: 10, h: 0.375, fill: { color: ORANGE } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.6, w: 8.6, h: 1.3, fontFace: "Calibri", fontSize: 38, bold: true, color: "FFFFFF", valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.7, y: 3.05, w: 8.6, h: 1.0, fontFace: "Calibri", fontSize: 18, color: "CFE0EE", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  slideNumber: { x: 9.2, y: 5.3, fontFace: "Calibri", fontSize: 9, color: GREY },
  objects: [
    { rect: { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: BLUE } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.22, w: 9.1, h: 0.75, fontFace: "Calibri", fontSize: 24, bold: true, color: NAVY, valign: "middle" }, text: "" } },
    { line: { x: 0.45, y: 0.98, w: 9.1, h: 0, line: { color: "D5DDE5", width: 1 } } },
  ],
});

// ---------- helpers ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, w, h, name) {
  const s = pngSize(file);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, altText: name || path.basename(file), objectName: name || path.basename(file) });
}
function content(title, section, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function bullets(slide, items, x, y, w, h, size = 14) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  slide.addText(runs, { x, y, w, h, fontFace: "Calibri", fontSize: size, color: NAVY, valign: "top", isTextBox: true, margin: 0.05 });
}
function stat(slide, x, y, w, big, small, color) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h: 1.25, fill: { color: LIGHT }, line: { color: "D5DDE5", width: 0.75 }, rectRadius: 0.08, objectName: "stat card " + big });
  slide.addText(big, { x, y: y + 0.08, w, h: 0.62, align: "center", fontFace: "Calibri", fontSize: 28, bold: true, color: color || BLUE, isTextBox: true, margin: 0 });
  slide.addText(small, { x: x + 0.1, y: y + 0.68, w: w - 0.2, h: 0.5, align: "center", fontFace: "Calibri", fontSize: 11, color: GREY, isTextBox: true, margin: 0, valign: "top" });
}

// ---------- 1 title ----------
pres.addSection({ title: "Problem & data" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Problem & data" });
s.addText("FreshBasket demand forecasting", { placeholder: "title" });
s.addText("13-week forecasts for 1,782 SKU-market series: hybrid global gradient boosting, calibrated uncertainty, driver attribution and scenario simulation", { placeholder: "subtitle" });
s.addNotes("Introduce the problem: weekly unit forecasts, 13 weeks ahead, for every store x product-family series. Set the tone: leakage-safe validation, honest uncertainty, explainable drivers. About 12 minutes in total; roughly 45 seconds per slide.");

// ---------- 2 task and data mapping ----------
s = content("The task, and what the data actually allowed", "Problem & data",
  "Be upfront: the supplied files are the Store Sales retail dataset, not the FMCG schema in the brief. We mapped store to market and product family to SKU. Price, weather, media and distribution are not in the data, so they cannot be modelled or ablated - we use proxies and say so.");
bullets(s, [
  "Goal: weekly unit forecasts for the next 13 weeks, with uncertainty, drivers and what-if scenarios",
  "Supplied files: train, test, stores, transactions, oil, holidays - not the FMCG schema in the brief",
  "Mapping: store = market (54), product family = SKU (33) giving 1,782 series; daily data rolled up to weeks",
  "Not available: price, weather, media, true distribution. Proxies: promo item count, availability proxy, traffic, oil price",
  "Consequence: the price and distribution questions in the brief are answered with proxies only",
], 0.45, 1.15, 5.0, 4.0, 14);
s.addShape(pres.ShapeType.roundRect, { x: 5.7, y: 1.2, w: 3.85, h: 3.85, fill: { color: LIGHT }, line: { color: "D5DDE5", width: 0.75 }, rectRadius: 0.08, objectName: "mapping card" });
s.addText([
  { text: "Brief concept  ->  used here", options: { bold: true, breakLine: true, color: BLUE } },
  { text: "market_id  ->  store_nbr", options: { breakLine: true } },
  { text: "sku_id  ->  product family", options: { breakLine: true } },
  { text: "promotion  ->  items on promo", options: { breakLine: true } },
  { text: "price  ->  not available", options: { breakLine: true, color: RED } },
  { text: "distribution  ->  availability proxy", options: { breakLine: true } },
  { text: "weather  ->  oil price (macro proxy)", options: { breakLine: true } },
  { text: "media  ->  not available", options: { breakLine: true, color: RED } },
  { text: "calendar  ->  holidays, events, paydays", options: { breakLine: true } },
  { text: "extra  ->  store transactions (lag-only)" },
], { x: 5.85, y: 1.3, w: 3.6, h: 3.6, fontFace: "Calibri", fontSize: 13, color: NAVY, valign: "top", paraSpaceAfter: 5, isTextBox: true });

// ---------- 3 data quality ----------
s = content("Data quality: five findings that shape the model", "Problem & data",
  "The biggest surprise: whole product families vanish for months and then return - about 5 percent of active data, mostly 2014-15. We treat these as reporting gaps, not zero demand, and mask them from training targets using only knowledge available at each fold origin. Promotion tracking only starts 31 March 2014, so earlier promo values are missing, not zero.");
stat(s, 0.45, 1.15, 1.7, "5%", "of series-weeks are reporting gaps (8+ zero weeks, then sales resume)", RED);
stat(s, 2.3, 1.15, 1.7, "786", "series launched after the first two months (new stores / families)", BLUE);
stat(s, 4.15, 1.15, 1.7, "Mar 2014", "promo tracking starts - earlier values are missing, not zero", ORANGE);
stat(s, 6.0, 1.15, 1.7, "54", "whole-week store closures, masked from training and scoring", BLUE);
stat(s, 7.85, 1.15, 1.7, "4", "missing 25-Dec days: stores close, weeks have 6 days", BLUE);
fitImage(s, FIG("eda01_total_weekly_sales.png"), 0.45, 2.6, 5.3, 2.6, "total weekly sales");
bullets(s, [
  "Pre-launch zeros are treated as missing, not demand",
  "Delisted series (8+ trailing zero weeks) forecast zero, unless seasonal",
  "Anomalies are kept in the targets when the calendar explains them",
], 5.9, 2.7, 3.7, 2.4, 12);

// ---------- 4 method ----------
pres.addSection({ title: "Method" });
s = content("Method: one global model, direct 13-week horizon", "Method",
  "One global gradient-boosting model serves all series and horizons; the horizon is a feature. The target is the log deviation from each series own recent run-rate, so the trees learn drivers rather than 1,782 volume scales. A 10 percent blend of seasonal-naive adds a little robustness. Rules handle never-sold and delisted series; conformal calibration gives intervals.");
const steps = [["Weekly panel", "240 weeks, masks for launch, closure, gaps"], ["Leakage-safe features", "lags, seasonality, promo plan, calendar, traffic"], ["Global GBM", "log deviation from run-rate, horizon 1..13"], ["Hybrid + rules", "0.9 GBM + 0.1 seasonal naive; cold-start; delist rule"], ["Intervals", "conformal, by horizon x volume tier"]];
steps.forEach((st, i) => {
  const x = 0.45 + i * 1.84;
  s.addShape(pres.ShapeType.roundRect, { x, y: 1.4, w: 1.7, h: 1.7, fill: { color: i === 2 ? BLUE : LIGHT }, line: { color: BLUE, width: 1 }, rectRadius: 0.08, objectName: "step " + st[0] });
  s.addText(st[0], { x: x + 0.05, y: 1.5, w: 1.6, h: 0.6, align: "center", bold: true, fontSize: 13, fontFace: "Calibri", color: i === 2 ? "FFFFFF" : NAVY, isTextBox: true, margin: 0 });
  s.addText(st[1], { x: x + 0.08, y: 2.1, w: 1.54, h: 0.95, align: "center", fontSize: 10.5, fontFace: "Calibri", color: i === 2 ? "FFFFFF" : GREY, isTextBox: true, margin: 0, valign: "top" });
  if (i < 4) s.addText(">", { x: x + 1.68, y: 2.0, w: 0.18, h: 0.4, align: "center", fontSize: 16, bold: true, color: ORANGE, isTextBox: true, margin: 0 });
});
bullets(s, [
  "Future inputs: calendar is known; the promo plan is known for 2.6 weeks (test file) then assumed = last 28 days; traffic and oil are lag-only, so no future assumption is needed",
  "Baselines for comparison: naive 4-week mean, seasonal naive x growth, damped-trend Holt",
  "Cold start: series with under 13 weeks of history use the best-validated fallback (damped Holt)",
], 0.45, 3.35, 9.1, 1.8, 13);

// ---------- 5 validation ----------
s = content("Validation: four rolling origins, no random splits", "Method",
  "Four expanding-window folds, each forecasting 13 weeks, tile the last 52 observed weeks, so every season including Christmas and Carnival is tested once out-of-sample. The model for each fold only sees targets up to its origin. A unit test scrambles all post-origin data and checks that features do not change.");
const W = 240, X0 = 1.3, XW = 8.2;
const origins = [187, 200, 213, 226];
origins.forEach((o, i) => {
  const y = 1.35 + i * 0.62;
  s.addText("Fold " + (i + 1), { x: 0.45, y, w: 0.8, h: 0.4, fontSize: 12, bold: true, fontFace: "Calibri", color: NAVY, isTextBox: true, margin: 0, valign: "middle" });
  s.addShape(pres.ShapeType.rect, { x: X0, y, w: XW * (o + 1) / W, h: 0.4, fill: { color: BLUE }, line: { color: "FFFFFF", width: 0.5 }, objectName: "train fold " + (i + 1) });
  s.addShape(pres.ShapeType.rect, { x: X0 + XW * (o + 1) / W, y, w: XW * 13 / W, h: 0.4, fill: { color: ORANGE }, line: { color: "FFFFFF", width: 0.5 }, objectName: "test fold " + (i + 1) });
});
s.addText("2013-01", { x: X0 - 0.2, y: 3.85, w: 0.9, h: 0.3, fontSize: 10, color: GREY, isTextBox: true, margin: 0 });
s.addText("2017-08 (last observed week)", { x: X0 + XW - 2.2, y: 3.85, w: 2.2, h: 0.3, fontSize: 10, color: GREY, align: "right", isTextBox: true, margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 0.45, y: 4.3, w: 0.25, h: 0.2, fill: { color: BLUE }, line: { color: BLUE, width: 0.5 }, objectName: "legend train" });
s.addText("training rows (targets up to the origin only)", { x: 0.75, y: 4.25, w: 3.4, h: 0.3, fontSize: 11, color: NAVY, isTextBox: true, margin: 0 });
s.addShape(pres.ShapeType.rect, { x: 4.3, y: 4.3, w: 0.25, h: 0.2, fill: { color: ORANGE }, line: { color: ORANGE, width: 0.5 }, objectName: "legend test" });
s.addText("13-week test window (Aug-Nov 2016, Nov-Feb, Feb-May, May-Aug 2017)", { x: 4.6, y: 4.25, w: 5.0, h: 0.3, fontSize: 11, color: NAVY, isTextBox: true, margin: 0 });
s.addText("Scoring excludes whole-week closures and reporting gaps (2% of series-weeks); a sensitivity table includes them.", { x: 0.45, y: 4.75, w: 9.1, h: 0.35, fontSize: 11, italic: true, color: GREY, isTextBox: true, margin: 0 });

// ---------- 6 results ----------
pres.addSection({ title: "Results" });
s = content("Accuracy: the global model beats every baseline", "Results",
  "Pooled over four folds and 85,695 series-weeks: naive 11.4 percent WMAPE, global GBM 9.7, ensemble 9.6. RMSLE falls from 0.51 to 0.40. Be honest: the ensemble adds little over the GBM alone, and in fold 3 the margin over naive is thin. Bias is a mild under-forecast of about 3 percent.");
fitImage(s, FIG("res01_model_comparison.png"), 0.45, 1.1, 6.1, 3.0, "model comparison");
stat(s, 6.75, 1.15, 1.35, "-16%", "WMAPE vs naive (11.4% to 9.6%)", GREEN);
stat(s, 8.2, 1.15, 1.35, "0.40", "RMSLE vs 0.51 for naive", GREEN);
bullets(s, [
  "32 of 33 families and 53 of 54 stores beat naive",
  "Median per-series WMAPE 18% vs 21% for naive",
  "Ensemble 0.9 GBM + 0.1 seasonal naive; weight chosen on folds 1-3 only",
  "Fold 3 is close (10.1% vs 10.4%): gains are not uniform",
], 0.45, 4.15, 9.1, 1.1, 12);
s.addText("Bias is -2.8% (mild under-forecast); small-volume series stay hard (bottom-quartile WMAPE about 49%).", { x: 6.75, y: 2.55, w: 2.8, h: 1.4, fontSize: 11, color: GREY, isTextBox: true, margin: 0, valign: "top" });

s = content("Accuracy by forecast horizon", "Results",
  "Error grows with horizon, from about 7 percent in week 1 to 12 to 14 percent by weeks 12 and 13. The model equals naive at week 1 and wins at most horizons, but is slightly worse than naive at weeks 10 and 11. Bias stays within a few percent, mostly a mild under-forecast.");
fitImage(s, FIG("res02_by_horizon.png"), 0.45, 1.1, 6.3, 3.9, "error by horizon");
stat(s, 6.95, 1.2, 2.6, "7.0% to 13.8%", "ensemble WMAPE, week 1 to week 13", BLUE);
bullets(s, [
  "Equal to naive in week 1, better at most later weeks",
  "Biggest gains vs naive at weeks 3, 5, 6 and 7 (3-4.5 points)",
  "Weeks 10-11: ensemble is slightly WORSE than naive (10.1% vs 9.3%, 10.0% vs 9.6%)",
], 6.95, 2.6, 2.6, 2.5, 11.5);

s = content("Accuracy by segment: broad, not just in the big lines", "Results",
  "Left: the 15 largest families; the ensemble beats naive in nearly all of them. Right: every store as a dot against its naive error; points below the diagonal are better than naive - 53 of 54 stores. Store types A to E all improve by 10 to 20 percent relative. Small-volume series are the weak spot.");
fitImage(s, FIG("res03_segments.png"), 0.45, 1.1, 6.6, 3.5, "segments");
bullets(s, [
  "32 of 33 families beat naive; 53 of 54 stores",
  "Store types A-E improve 11-17% relative",
  "Bottom volume quartile: WMAPE about 49%, barely better than naive (50%)",
  "Cold-start series (<13 weeks): damped Holt 6.6% WMAPE vs GBM 15.7% (881 rows only)",
], 7.2, 1.2, 2.4, 3.9, 11);

s = content("Aggregates are more accurate: coherent bottom-up totals", "Results",
  "Summing the series forecasts bottom-up gives coherent totals. Total weekly units are forecast within 5.2 percent WMAPE; family-week 6.3; store-week 7.4. This matters for inventory planning, where aggregate error is what drives stock decisions.");
fitImage(s, FIG("res04_aggregate_backtest.png"), 0.45, 1.1, 6.0, 3.2, "aggregate backtest");
stat(s, 6.75, 1.2, 2.8, "5.2%", "WMAPE on total weekly units (naive 6.8%)", GREEN);
stat(s, 6.75, 2.6, 2.8, "6.3% / 7.4%", "family-week / store-week WMAPE (naive 8.0% / 9.1%)", GREEN);
s.addText("Dotted lines mark the four fold origins. Forecasts are out-of-sample throughout.", { x: 0.45, y: 4.5, w: 9.1, h: 0.4, fontSize: 12, color: GREY, isTextBox: true, margin: 0 });

s = content("Uncertainty: conformal intervals that hold up out-of-sample", "Results",
  "Intervals are calibrated on folds 1 to 3 residuals, by horizon bin and volume tier, and tested on the untouched fold 4: 90 percent intervals cover 91.4 percent, 50 percent intervals 51.8 percent. Coverage is slightly low at week 13 for the 50 percent band, and the large-volume tier is mildly conservative.");
fitImage(s, FIG("res05_interval_calibration.png"), 0.45, 1.1, 5.2, 3.6, "interval calibration");
stat(s, 6.0, 1.2, 1.7, "91.4%", "coverage of 90% intervals (fold 4)", GREEN);
stat(s, 7.85, 1.2, 1.7, "51.8%", "coverage of 50% intervals", GREEN);
bullets(s, [
  "Calibrated on folds 1-3, scored on fold 4 only",
  "Cells: horizon bin x series-volume tier",
  "Every forecast row carries 50% and 90% bands",
], 6.0, 2.7, 3.6, 2.0, 12);

// ---------- ablation + explain ----------
s = content("What drives accuracy: ablation across all four folds", "Results",
  "Each bar is the cost of removing one driver group and re-training. Seasonal lags matter most (9 percent relative WMAPE), then promotions and calendar. Store traffic is neutral. Oil helps WMAPE a little but not RMSLE - treat it cautiously. The availability proxy slightly hurts WMAPE but helps small series on RMSLE: a mixed result we report as is. Price and weather cannot be tested because they are not in the data.");
fitImage(s, FIG("res06_ablation.png"), 0.45, 1.1, 5.9, 3.4, "ablation");
bullets(s, [
  "Seasonal lags: +9.2% WMAPE if removed (all 4 folds)",
  "Promotions: +3.1%, and RMSLE +0.04 (3 of 4 folds)",
  "Calendar and holidays: +2.3% (3 of 4 folds)",
  "Store traffic: about 0 - not useful beyond sales history",
  "Availability proxy: -1.8% WMAPE but worse RMSLE - mixed",
  "Autoregressive only: +17% - exogenous drivers do add value",
], 6.45, 1.2, 3.1, 3.9, 11.5);

s = content("Explaining individual forecasts: exact group-Shapley drivers", "Results",
  "For five automatically selected series we compute exact Shapley values over eight driver groups. Read each bar as the effect on that series' forecast relative to a typical series' average. Attributions sum exactly to the model output - verified. Permutation importance (global) ranks promotions first on RMSLE, then momentum and scale; oil shows zero by construction, so the ablation is the evidence for oil.");
fitImage(s, FIG("res08_local_driver_attribution.png"), 0.45, 1.05, 6.6, 4.2, "local attribution");
bullets(s, [
  "Promotions add about +5% to the two largest grocery series and +8% to the seasonal beverage series",
  "Seasonality (year-ago pattern) pulls the beverage series down about 11%",
  "Global permutation importance: promotions, then momentum, scale, seasonality",
  "Effects are relative to a typical series; they sum exactly to the model output",
], 7.2, 1.2, 2.4, 4.0, 11);

s = content("What the forecast looks like: examples with 50% / 90% bands", "Results",
  "Five illustrative series: the highest-volume, the most promotion-exposed, the strongest seasonal swing, a cold-start series (store 52 opened in April 2017) and a mid-volume series. Note how bands widen with horizon and how the cold-start series gets a sensible level from its short history.");
fitImage(s, FIG("res09_forecast_examples.png"), 0.45, 1.05, 6.6, 4.2, "forecast examples");
bullets(s, [
  "Total: about 77.8M units over 13 weeks (14 Aug to 12 Nov 2017), peaking the week of 30 Oct",
  "23,166 rows with 50% and 90% intervals; flags for cold-start and delisted series",
  "Bands widen with horizon; the cold-start series (store 52) gets a sensible level from 16 weeks of history",
], 7.2, 1.2, 2.4, 4.0, 11);

// ---------- business ----------
pres.addSection({ title: "Business insight & next steps" });
s = content("Scenarios: what changes if promotion intensity changes", "Business insight & next steps",
  "Re-running the trained model with modified known inputs. Withdrawing all promotions lowers demand about 7 percent; +25, +50 and double intensity lift demand 1.1, 2.2 and 4.0 percent - roughly linear, mildly diminishing (extrapolation is capped). Caveats: these are associations from history, not causal experiments. At extremes the model can overreach: school supplies drops 96 percent with no promotions, which is not credible. The traffic shock barely moves the forecast because the model hardly uses traffic.");
fitImage(s, FIG("res10_scenarios.png"), 0.45, 1.1, 6.2, 2.3, "scenarios");
bullets(s, [
  "No promotions: -7.3% units; +25% / +50% / x2 intensity: +1.1% / +2.2% / +4.0%",
  "Response is roughly linear and mildly diminishing; extrapolation is capped at 1.5x each series' history",
  "Strongest responders: Produce, Beverages, Grocery I; seasonal lines are most sensitive",
], 0.45, 3.5, 6.2, 1.7, 12);
s.addShape(pres.ShapeType.roundRect, { x: 6.85, y: 1.2, w: 2.7, h: 3.9, fill: { color: "FDF1E6" }, line: { color: ORANGE, width: 1 }, rectRadius: 0.08, objectName: "caution card" });
s.addText([
  { text: "Use with care", options: { bold: true, color: ORANGE, breakLine: true } },
  { text: "Promotions were not randomised: these are associations, not causal lift.", options: { breakLine: true } },
  { text: "Extreme cases can be implausible (school supplies -96% with no promo).", options: { breakLine: true } },
  { text: "Traffic -10% moves demand only -0.06%: the model barely uses traffic, so this is a sensitivity check, not insight." },
], { x: 6.95, y: 1.3, w: 2.5, h: 3.7, fontSize: 12, fontFace: "Calibri", color: NAVY, valign: "top", paraSpaceAfter: 6, isTextBox: true });

s = content("Beyond the minimum: evidence-based extras", "Business insight & next steps",
  "Highlight what goes beyond the brief and was evidenced: the leakage test, the gap-aware training, the seasonal exception to the delisting rule which fixed a measured problem, conformal intervals checked on a held-out fold, exact Shapley, and a scenario simulator.");
bullets(s, [
  "Global vs per-series: one model for 1,782 series, direct horizon, relative log target with volume and recency weighting",
  "Hybrid statistical + ML ensemble, with weights chosen on earlier folds only",
  "Reporting-gap detection that is leakage-safe (only gaps already over at each origin)",
  "Business rules checked against data: the first delisting rule zeroed 311 forecasts that later had sales; a seasonal exception fixed RMSLE 0.417 to 0.398",
  "Conformal intervals by horizon x volume tier, verified on an untouched fold",
  "Exact group-Shapley local explanations (no external library) and a promo scenario simulator with response curves",
  "A unit test that scrambles all post-origin data to prove no leakage",
], 0.45, 1.2, 9.1, 4.0, 15);

s = content("Limitations and recommendations for business use", "Business insight & next steps",
  "Close with a clear view of what to trust. Use the forecasts for aggregate planning and for large and mid-volume lines; treat small-volume series with more caution. Provide the promotion plan to improve results. Run a controlled test before relying on scenario lifts. Retrain monthly and monitor bias and interval coverage.");
s.addText("Limitations", { x: 0.45, y: 1.1, w: 4.4, h: 0.35, fontSize: 15, bold: true, color: RED, isTextBox: true, margin: 0 });
bullets(s, [
  "No price, weather, media or true distribution data: proxies only",
  "Product families, not individual SKUs",
  "Promo plan beyond week 3 is assumed (last 28 days)",
  "Small-volume series: WMAPE about 49%, barely better than naive",
  "Scenario effects are associations, not causal",
  "Fold 3 margin thin; ensemble gain over GBM is small",
  "Settings (volume weight, cold-start choice) chosen on the CV folds: results slightly optimistic",
], 0.45, 1.5, 4.5, 3.7, 11.5);
s.addText("Recommendations", { x: 5.15, y: 1.1, w: 4.4, h: 0.35, fontSize: 15, bold: true, color: GREEN, isTextBox: true, margin: 0 });
bullets(s, [
  "Plan inventory on the aggregate and large-line forecasts; apply the 90% band as safety-stock input",
  "Feed the real trade-promotion calendar: with the plan unknown, RMSLE worsens from 0.40 to 0.45",
  "Run a promo test before using lift numbers; collect price and discount depth",
  "Retrain monthly; monitor bias (-3%) and interval coverage",
  "Flag reporting gaps and launches at source so zeros are not read as demand",
  "Add price, distribution and weather data to unlock the missing drivers",
], 5.15, 1.5, 4.4, 3.7, 11.5);

s = content("Appendix: reproducibility and deliverables", "Business insight & next steps",
  "Everything runs end to end from one command on a single CPU. Tests use a tiny synthetic dataset so they run in about a minute.");
bullets(s, [
  "pip install -r requirements.txt, then python run_pipeline.py (about 8-10 minutes on 1 CPU)",
  "python tests/test_pipeline.py runs six tests including the future-leakage test",
  "Forecast file: outputs/forecast_13w.csv (date, market_id, sku_id, forecast, 50% / 90% bounds)",
  "Reports: data-quality and EDA, project report, this deck; all tables as CSV in outputs/tables",
  "Seeds fixed; stages cached; synthetic-data generator included",
], 0.45, 1.3, 9.1, 3.8, 16);

pres.writeFile({ fileName: OUT }).then(async () => {
  try {
    const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");
    await applyTheme(OUT, THEME);
  } catch (e) { console.log("theme helper not applied:", e.message); }
  console.log("wrote", OUT);
});

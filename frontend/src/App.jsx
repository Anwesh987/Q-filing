import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, AlertTriangle, BarChart3, CheckCircle2, Cloud, FileText, Loader2 } from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const DEFAULT_TEXT = `PORTFOLIO COMPLIANCE REQUIREMENTS:
The portfolio must select exactly 10 assets.
Technology sector exposure must not exceed 4 assets.
Healthcare sector must represent at least 2 assets.
Energy sector is capped at 2 assets maximum.
Financial sector allocation is limited to at most 3 assets.`;

function MetricCard({ label, value, icon }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <h3>{value}</h3>
      </div>
    </div>
  );
}

function AllocationChart({ weights }) {
  const rows = Object.entries(weights || {}).sort((a, b) => b[1] - a[1]);
  if (!rows.length) return null;

  return (
    <div className="panel">
      <div className="panel-title">
        <BarChart3 size={20} />
        Portfolio allocation
      </div>
      <div className="bars">
        {rows.map(([ticker, weight]) => (
          <div className="bar-row" key={ticker}>
            <span className="ticker">{ticker}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.max(weight * 100, 2)}%` }} />
            </div>
            <span className="weight">{(weight * 100).toFixed(2)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConstraintList({ constraints }) {
  if (!constraints?.length) {
    return (
      <div className="empty">
        No constraints extracted. The app used default portfolio allocation.
      </div>
    );
  }

  return (
    <div className="constraints">
      {constraints.map((c, idx) => (
        <div className="constraint" key={`${c.description}-${idx}`}>
          <div className="constraint-header">
            <span>{c.constraint_type}</span>
            <strong>{c.threshold_value}</strong>
          </div>
          <p>{c.description}</p>
          <small>{c.target_tickers?.slice(0, 8).join(", ")}{c.target_tickers?.length > 8 ? "..." : ""}</small>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [horizon, setHorizon] = useState(90);
  const [objective, setObjective] = useState("SORTINO");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const metrics = result?.portfolio_result?.risk_metrics || {};
  const constraints = result?.compliance_payload?.constraints || [];
  const portfolio = result?.portfolio_result;

  const statusLabel = useMemo(() => {
    if (!result) return "Not run yet";
    return `${result.execution_status} · ${result.mode}`;
  }, [result]);

  async function runOptimization() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/optimize/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          regulatory_text: text,
          horizon_days: Number(horizon),
          weight_objective: objective,
          use_existing_pipeline: false
        })
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || `Request failed with ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function downloadJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "q-filing-result.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main>
      <section className="hero">
        <div>
          <div className="badge">
            <Cloud size={16} />
            Frontend + Cloud Deployment Layer
          </div>
          <h1>Q-Filing</h1>
          <p>
            AI + quantum-inspired portfolio compliance optimizer. Paste filing or policy text,
            extract constraints, and visualize portfolio allocation and risk metrics.
          </p>
        </div>
        <div className="status-card">
          <span>API</span>
          <strong>{API_BASE_URL}</strong>
          <small>{statusLabel}</small>
        </div>
      </section>

      <section className="grid">
        <div className="panel input-panel">
          <div className="panel-title">
            <FileText size={20} />
            Compliance / filing text
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste compliance rules, filing text, investment policy constraints..."
          />

          <div className="form-row">
            <label>
              Horizon
              <select value={horizon} onChange={(e) => setHorizon(e.target.value)}>
                <option value={7}>7 days</option>
                <option value={30}>30 days</option>
                <option value={90}>90 days</option>
                <option value={365}>365 days</option>
              </select>
            </label>

            <label>
              Objective
              <select value={objective} onChange={(e) => setObjective(e.target.value)}>
                <option value="SORTINO">SORTINO</option>
                <option value="SHARPE">SHARPE</option>
                <option value="MAXRET">MAXRET</option>
              </select>
            </label>
          </div>

          <button onClick={runOptimization} disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <Activity size={18} />}
            {loading ? "Running..." : "Run optimization"}
          </button>

          {error && (
            <div className="error">
              <AlertTriangle size={18} />
              {error}
            </div>
          )}
        </div>

        <div className="right-stack">
          <div className="metrics-grid">
            <MetricCard
              label="Expected return"
              value={metrics.expected_return_pct ? `${metrics.expected_return_pct}%` : "—"}
              icon={<Activity size={22} />}
            />
            <MetricCard
              label="Volatility"
              value={metrics.expected_volatility_pct ? `${metrics.expected_volatility_pct}%` : "—"}
              icon={<BarChart3 size={22} />}
            />
            <MetricCard
              label="Sharpe"
              value={metrics.sharpe_ratio ?? "—"}
              icon={<CheckCircle2 size={22} />}
            />
            <MetricCard
              label="CVaR 95%"
              value={metrics.cvar_95_pct ? `${metrics.cvar_95_pct}%` : "—"}
              icon={<AlertTriangle size={22} />}
            />
          </div>

          <AllocationChart weights={portfolio?.weights} />
        </div>
      </section>

      <section className="grid bottom-grid">
        <div className="panel">
          <div className="panel-title">
            <CheckCircle2 size={20} />
            Extracted constraints
          </div>
          <ConstraintList constraints={constraints} />
        </div>

        <div className="panel">
          <div className="panel-title">
            <Activity size={20} />
            Run details
          </div>

          {!result ? (
            <div className="empty">Run the optimizer to see logs, warnings, and downloadable output.</div>
          ) : (
            <>
              <div className="summary">
                <p><strong>Selected assets:</strong> {portfolio.selected_assets.join(", ")}</p>
                <p><strong>Objective:</strong> {portfolio.weight_objective}</p>
                <p><strong>Horizon:</strong> {portfolio.horizon_days} days</p>
                <p><strong>Max drawdown:</strong> {metrics.max_drawdown_pct}%</p>
                <p><strong>Sortino:</strong> {metrics.sortino_ratio}</p>
              </div>

              {result.warnings?.length > 0 && (
                <div className="warning-box">
                  {result.warnings.map((w, idx) => <p key={idx}>⚠ {w}</p>)}
                </div>
              )}

              <button className="secondary" onClick={downloadJson}>Download JSON report</button>
            </>
          )}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);

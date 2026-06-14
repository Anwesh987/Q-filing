"""
train_agent.py  —  FinQ RL Agent Trainer  v2.0
================================================
Target hardware : Intel i5-13420H (12 threads) + NVIDIA RTX 4050 Laptop GPU
Training budget : ~10 minutes → ~500 000 timesteps
Output          : ppo_trading_agent.zip  +  rl_weights.json

HOW IT CONNECTS TO quantum_core.py
------------------------------------
  RLStateAgent.get_dynamic_weights(mode, market_volatility)  currently uses
  hard-coded if/else tables.  After training, rl_weights.json replaces those
  tables so the live dashboard benefits from learned weight schedules.

  To activate in quantum_core.py, change RLStateAgent to:

      import json, os
      _RL_W = json.load(open("rl_weights.json")) if os.path.exists("rl_weights.json") else None

      @staticmethod
      def get_dynamic_weights(mode, market_vol):
          if _RL_W and mode in _RL_W:
              t = min(int(market_vol / 0.10), 9)   # bucket 0-9
              return tuple(_RL_W[mode][t])
          # ... existing fallback ...

OBSERVATION SPACE  (15 features, all float32)
----------------------------------------------
  [0]  log_return_1d         — daily log-return
  [1]  log_return_5d         — 5-day log-return
  [2]  log_return_20d        — 20-day log-return
  [3]  rsi_14                — RSI(14), normalised 0→1
  [4]  momentum_short        — tanh-scaled 5d momentum
  [5]  momentum_med          — tanh-scaled 20d momentum
  [6]  realised_vol_20       — 20-day annualised realised vol
  [7]  ewma_vol              — EWMA (λ=0.94) daily vol, annualised
  [8]  cvar_proxy            — rolling 95-pctile loss proxy (annualised)
  [9]  max_drawdown          — rolling 20-day max-drawdown (negative)
  [10] sortino_proxy         — 20-day Sortino proxy
  [11] sharpe_proxy          — 20-day Sharpe proxy
  [12] current_exposure      — fraction of portfolio in equities (0→1)
  [13] market_vol_regime     — cross-asset vol regime flag (0=low, 1=high)
  [14] norm_portfolio_value  — portfolio value / initial_balance

ACTION SPACE (continuous)
--------------------------
  Single scalar in [-1, 1]:
      -1 = fully cash  (0% exposure)
       0 = half equity (50% exposure)
      +1 = fully long  (100% exposure)

  Mapped to exposure = (action + 1) / 2
  This is the same convention as the original SimpleTradingEnv.

REWARD
------
  Sortino-shaped:  r = Δportfolio / portfolio  capped at ±10%
  Downside is penalised 2× vs upside to teach risk-awareness.
  An episode-end bonus rewards Sharpe > 1 to encourage consistency.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import yfinance as yf

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE CONFIG — tune these if you change hardware
# ─────────────────────────────────────────────────────────────────────────────

N_ENVS          = 8          # parallel workers  (i5-13420H has 12 threads; leave 4 free)
TOTAL_TIMESTEPS = 1_000_000  # 2× longer — needed to see enough volatile + calm regimes
DEVICE          = "cuda"     # "cuda" for RTX 4050, "cpu" fallback handled below
BATCH_SIZE      = 1024       # bigger batch → better GPU utilisation on RTX 4050
N_STEPS         = 2048       # longer rollout → better advantage estimates
N_EPOCHS        = 10         # PPO update epochs
LEARNING_RATE   = 2e-4       # slightly lower → more stable on financial noise
CLIP_RANGE      = 0.15       # tighter clip → less aggressive updates on noisy rewards
ENT_COEF        = 0.02       # higher entropy → more exploration across vol regimes
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
GAMMA           = 0.995      # higher discount → better long-episode credit assignment
GAE_LAMBDA      = 0.95
NET_ARCH        = [512, 256, 128]  # deeper net for 15 features across 4 regimes

# ─────────────────────────────────────────────────────────────────────────────
# TICKER UNIVERSE  — diversified for robust generalisation
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TICKERS = [
    # Mega-cap / stable (long-term regime)
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "COST", "WMT",
    # High-beta tech (medium-term regime)
    "NVDA", "AMD", "META", "TSLA", "NFLX", "CRM", "SNOW", "PLTR",
    # Healthcare / Biotech (mixed vol)
    "LLY", "ABBV", "REGN", "MRNA",
    # Financials / Energy
    "GS", "CVX", "CAT",
    # Small/mid-cap high-volatility (teaches penny-stock-like dynamics)
    "RIVN", "LCID", "SOFI", "UPST", "HOOD", "CLOV",
]

EVAL_TICKERS = ["MSFT", "LLY", "XOM", "RIVN", "SNOW"]   # diverse eval set

# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA  — cached to disk so parallel workers don't re-download
# ─────────────────────────────────────────────────────────────────────────────

DATA_CACHE = "rl_price_cache.npy"
META_CACHE = "rl_price_meta.json"


def _download_prices(tickers: list, period: str = "3y") -> dict:
    """Download adjusted close prices, return dict ticker→np.ndarray."""
    print(f"  Downloading {len(tickers)} tickers ({period})…")
    raw = yf.download(
        tickers, period=period, interval="1d",
        progress=False, auto_adjust=True, threads=True,
    )
    if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "get_level_values"):
        try:
            close = raw["Close"]
        except KeyError:
            close = raw.iloc[:, :len(tickers)]
    else:
        close = raw
    close = close.dropna(axis=1, thresh=int(0.80 * len(close)))
    result = {}
    for col in close.columns:
        arr = close[col].dropna().values.astype(np.float32)
        if len(arr) >= 252:
            result[str(col)] = arr
    return result


def load_or_download(tickers: list) -> dict:
    """Load from disk cache or download fresh."""
    all_tickers = sorted(set(tickers))
    if os.path.exists(DATA_CACHE) and os.path.exists(META_CACHE):
        with open(META_CACHE) as f:
            meta = json.load(f)
        if set(meta["tickers"]) == set(all_tickers):
            arrays = np.load(DATA_CACHE, allow_pickle=True).item()
            print(f"  Loaded {len(arrays)} tickers from cache.")
            return arrays
    arrays = _download_prices(all_tickers)
    np.save(DATA_CACHE, arrays)
    with open(META_CACHE, "w") as f:
        json.dump({"tickers": sorted(arrays.keys())}, f)
    print(f"  Cached {len(arrays)} tickers to disk.")
    return arrays


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (pure numpy, no pandas overhead in env)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 2:
        return 0.5
    deltas = np.diff(prices[-period - 2:])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains[:period].mean()
    avg_l  = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    rs = avg_g / avg_l if avg_l > 1e-10 else 100.0
    return (100.0 - 100.0 / (1.0 + rs)) / 100.0   # normalised 0→1


def _ewma_vol(log_rets: np.ndarray, lam: float = 0.94) -> float:
    if len(log_rets) < 2:
        return 0.01
    v = log_rets[0] ** 2
    for r in log_rets[1:]:
        v = lam * v + (1 - lam) * r ** 2
    return float(np.sqrt(v * 252))  # annualised


def _cvar_proxy(log_rets: np.ndarray, conf: float = 0.95) -> float:
    if len(log_rets) < 5:
        return 0.0
    cutoff = max(1, int((1.0 - conf) * len(log_rets)))
    tail = np.sort(log_rets)[:cutoff]
    return float(-tail.mean() * np.sqrt(252))   # annualised, positive = bad


def _max_drawdown(log_rets: np.ndarray) -> float:
    cum = np.cumprod(1.0 + log_rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(peak, 1e-10)
    return float(dd.min())   # negative


def _build_obs(prices: np.ndarray, step: int, exposure: float,
               market_vol_regime: float) -> np.ndarray:
    """Build the 15-feature observation vector."""
    w = min(step, 20)
    window = prices[step - w: step + 1]
    log_rets_full = np.diff(np.log(np.maximum(window, 1e-8)))

    price_t    = prices[step]
    price_1d   = prices[step - 1]   if step >= 1  else price_t
    price_5d   = prices[step - 5]   if step >= 5  else price_t
    price_20d  = prices[step - 20]  if step >= 20 else price_t

    lr_1d  = float(np.log(price_t / price_1d))
    lr_5d  = float(np.log(price_t / price_5d))
    lr_20d = float(np.log(price_t / price_20d))

    rsi = _rsi(prices[max(0, step - 30): step + 1])

    mom_5  = float(np.tanh(lr_5d  * 3))
    mom_20 = float(np.tanh(lr_20d * 1.5))

    rv_20 = float(log_rets_full.std() * np.sqrt(252)) if len(log_rets_full) >= 2 else 0.01
    ewma  = _ewma_vol(log_rets_full)
    cvar  = _cvar_proxy(log_rets_full)
    mdd   = _max_drawdown(log_rets_full) if len(log_rets_full) >= 2 else 0.0

    # Sortino proxy:  mean / downside_std  (daily, not annualised for stability)
    downside = log_rets_full[log_rets_full < 0]
    ds_std   = float(downside.std()) if len(downside) >= 2 else rv_20
    sortino  = float(np.clip(log_rets_full.mean() / (ds_std + 1e-8), -3, 3))

    # Sharpe proxy
    sharpe = float(np.clip(
        log_rets_full.mean() / (log_rets_full.std() + 1e-8), -3, 3
    )) if len(log_rets_full) >= 2 else 0.0

    obs = np.array([
        np.clip(lr_1d,  -0.15, 0.15),
        np.clip(lr_5d,  -0.30, 0.30),
        np.clip(lr_20d, -0.50, 0.50),
        rsi,
        mom_5,
        mom_20,
        np.clip(rv_20, 0.0, 2.0),
        np.clip(ewma,  0.0, 2.0),
        np.clip(cvar,  0.0, 3.0),   # raised: penny stocks exceed 100% CVaR
        np.clip(mdd,  -1.0, 0.0),
        np.clip(sortino, -3.0, 3.0),
        np.clip(sharpe,  -3.0, 3.0),
        float(exposure),
        float(market_vol_regime),
        0.0,   # norm_portfolio_value — filled in by env
    ], dtype=np.float32)

    return obs


# ─────────────────────────────────────────────────────────────────────────────
# TRADING ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioTradingEnv(gym.Env):
    """
    Single-asset trading env with rich feature set.
    Trained across many tickers → policy generalises to portfolio weighting.
    Episodes sample a random ticker + random start date on each reset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_arrays: dict,
        episode_len:  int   = 63,    # ~3 months per episode (randomised at reset for diversity)
        tc_bps:       float = 5.0,   # 5 bps transaction cost each trade
        seed:         int   = 0,
    ):
        super().__init__()
        self.price_arrays  = price_arrays
        self.ticker_list   = list(price_arrays.keys())
        self.episode_len   = episode_len
        self.tc_rate       = tc_bps / 10_000.0
        self._seed         = seed

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self._rng      = np.random.default_rng(seed)
        self._prices   = None
        self._step     = 0
        self._start    = 0
        self._end      = 0
        self._balance  = 1.0
        self._shares   = 0.0
        self._exposure = 0.0
        self._episode_rets = []

    # ── Gym interface ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random ticker for this episode
        ticker = self._rng.choice(self.ticker_list)
        self._prices = self.price_arrays[ticker]

        # Sample a random start within the price array (need at least episode_len + 20 bars)
        margin = self.episode_len + 21
        if len(self._prices) <= margin:
            ticker = self.ticker_list[0]
            self._prices = self.price_arrays[ticker]
        max_start = len(self._prices) - margin
        self._start = int(self._rng.integers(20, max(21, max_start)))
        self._step  = self._start

        self._balance  = 1.0
        self._shares   = 0.0
        self._exposure = 0.0
        self._episode_rets = []

        # Randomise episode length each reset: 21–126 days
        # Teaches the policy to handle both short-term and medium-term horizons
        self._episode_len_actual = int(self._rng.integers(21, self.episode_len + 1))
        self._end = self._step + self._episode_len_actual

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        current_price = float(self._prices[self._step])
        port_value    = self._balance + self._shares * current_price

        # Map action [-1,1] → target exposure [0,1]
        target_exp = float(np.clip((float(action[0]) + 1.0) / 2.0, 0.0, 1.0))

        # Rebalance to target exposure with transaction cost
        target_shares_value   = port_value * target_exp
        current_shares_value  = self._shares * current_price
        trade_value           = target_shares_value - current_shares_value
        tc                    = abs(trade_value) * self.tc_rate

        self._balance -= (trade_value + tc)
        self._shares  += trade_value / (current_price + 1e-8)
        self._exposure = target_exp

        # Advance time
        self._step += 1
        new_price  = float(self._prices[self._step])
        new_value  = self._balance + self._shares * new_price

        # Compute return
        ret = (new_value - port_value) / (port_value + 1e-8)
        self._episode_rets.append(float(ret))

        # ── Sortino-shaped reward ─────────────────────────────────────────
        # Penalise downside 2× — teaches the agent to avoid drawdowns
        if ret >= 0:
            reward = float(np.clip(ret, 0.0, 0.10))
        else:
            reward = float(np.clip(ret * 2.0, -0.20, 0.0))

        done     = self._step >= self._end  # _end set per-episode at reset
        truncated = False

        # Real-time drawdown penalty: punish if current portfolio is more than
        # 5% below its episode peak — teaches capital preservation mid-episode
        if len(self._episode_rets) >= 2:
            ep_arr = np.array(self._episode_rets)
            cum = np.cumprod(1.0 + ep_arr)
            peak = float(np.maximum.accumulate(cum)[-1])
            current_dd = (cum[-1] - peak) / (peak + 1e-8)
            if current_dd < -0.05:
                reward += float(np.clip(current_dd * 0.5, -0.05, 0.0))  # extra dd penalty

        # Episode-end Sharpe bonus — encourages consistent returns
        if done and len(self._episode_rets) >= 10:
            ep_rets = np.array(self._episode_rets)
            ep_sharpe = float(ep_rets.mean() / (ep_rets.std() + 1e-8)) * np.sqrt(252)
            if ep_sharpe > 1.0:
                reward += 0.02 * min(ep_sharpe, 3.0)   # capped bonus
            # Penalty for ending with a large drawdown — consistency matters
            ep_cum = np.cumprod(1.0 + ep_rets)
            ep_peak = np.maximum.accumulate(ep_cum)
            ep_mdd = float((ep_cum - ep_peak).min() / (ep_peak.max() + 1e-8))
            if ep_mdd < -0.15:
                reward += float(np.clip(ep_mdd * 0.3, -0.05, 0.0))

        obs = self._get_obs(norm_value=new_value)
        info = {
            "portfolio_value": new_value,
            "exposure": self._exposure,
            "return": ret,
        }
        return obs, reward, done, truncated, info

    # ── Internals ────────────────────────────────────────────────────────────

    def _get_obs(self, norm_value: float = 1.0) -> np.ndarray:
        # Market vol regime: rolling 60-day vol vs long-run median
        lo = max(0, self._step - 60)
        recent_log_r = np.diff(np.log(np.maximum(self._prices[lo: self._step + 1], 1e-8)))
        rv = float(recent_log_r.std() * np.sqrt(252)) if len(recent_log_r) >= 2 else 0.20
        # Continuous regime: tanh-scaled around 0.40 anchor
        # Normal ~0.20 vol → 0.27, high-beta ~0.60 → 0.73, penny ~1.0 → 0.93
        vol_regime = float(np.tanh((rv - 0.40) * 2.5) * 0.5 + 0.5)

        obs = _build_obs(self._prices, self._step, self._exposure, vol_regime)
        # Fill portfolio value feature
        port_val = self._balance + self._shares * float(self._prices[self._step])
        obs[14]  = float(np.clip(port_val / 1.0, 0.5, 2.0))   # normalised
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# RL WEIGHT EXTRACTOR
# Reads the trained policy and writes rl_weights.json for RLStateAgent
# ─────────────────────────────────────────────────────────────────────────────

def extract_rl_weights(model: PPO, price_arrays: dict, n_probe: int = 2000) -> dict:
    """
    Probe the trained policy across (mode, market_vol) buckets.
    Records the mean action (= preferred exposure) per bucket.
    Translates into (w_ret, w_mom, w_cvar) triples that RLStateAgent consumes.
    """
    print("\nExtracting RL weight schedule from trained policy…")

    modes = {
        "intraday": {"vol_regime": 0.93, "mom_boost": 0.6},  # ~1.0 vol tanh → 0.93
        "short":    {"vol_regime": 0.73, "mom_boost": 0.4},  # ~0.6 vol tanh → 0.73
        "medium":   {"vol_regime": 0.37, "mom_boost": 0.2},  # ~0.2 vol tanh → 0.37
        "long":     {"vol_regime": 0.12, "mom_boost": 0.1},  # ~0.1 vol tanh → 0.12
    }
    vol_buckets = np.linspace(0.05, 2.00, 20)   # 20 buckets up to 200% vol (penny stocks)
    rng = np.random.default_rng(0)
    ticker_list = list(price_arrays.keys())

    rl_weights = {}
    for mode, cfg in modes.items():
        bucket_actions = []
        for vol in vol_buckets:
            actions = []
            for _ in range(n_probe // 10):
                # Build a synthetic obs for this (mode, vol) combination
                ticker = rng.choice(ticker_list)
                prices = price_arrays[ticker]
                step   = int(rng.integers(25, len(prices) - 2))
                obs    = _build_obs(prices, step, rng.uniform(0, 1), cfg["vol_regime"])
                # Override vol features to match bucket
                obs[6]  = float(np.clip(vol, 0, 2))   # rv_20
                obs[7]  = float(np.clip(vol, 0, 2))   # ewma_vol
                obs[13] = float(cfg["vol_regime"])
                obs    = obs.reshape(1, -1)
                action, _ = model.predict(obs, deterministic=True)
                exposure   = float((action[0][0] + 1.0) / 2.0)
                actions.append(exposure)
            bucket_actions.append(float(np.mean(actions)))

        # Map mean exposure per bucket → (w_ret, w_mom, w_cvar)
        # High exposure  → agent is confident → weight towards return
        # Low exposure   → agent is defensive  → weight towards cvar
        triples = []
        for exp in bucket_actions:
            w_ret  = round(float(np.clip(0.10 + 0.50 * exp, 0.10, 0.60)), 3)
            w_mom  = round(float(np.clip(
                cfg["mom_boost"] * exp * 0.8 + 0.10, 0.10, 0.50
            )), 3)
            w_cvar = round(float(np.clip(1.0 - w_ret - w_mom, 0.10, 0.70)), 3)
            # Renormalise to sum = 1.0 exactly
            total = w_ret + w_mom + w_cvar
            triples.append([
                round(w_ret  / total, 3),
                round(w_mom  / total, 3),
                round(w_cvar / total, 3),
            ])
        rl_weights[mode] = triples

    return rl_weights


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    def __init__(self, total_steps: int, log_interval: int = 25_000):
        super().__init__()
        self.total_steps  = total_steps
        self.log_interval = log_interval
        self._start_time  = None
        self._last_log    = 0

    def _on_training_start(self):
        self._start_time = time.time()
        print(f"\n{'─'*60}")
        print(f"  FinQ RL Trainer  |  Target: {self.total_steps:,} steps")
        print(f"  Workers: {N_ENVS}  |  Device: {DEVICE}  |  Batch: {BATCH_SIZE}")
        print(f"{'─'*60}")

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n - self._last_log >= self.log_interval:
            elapsed = time.time() - self._start_time
            sps     = n / elapsed
            eta     = (self.total_steps - n) / sps
            pct     = n / self.total_steps * 100
            print(
                f"  [{pct:5.1f}%]  {n:>8,} steps  |  "
                f"{sps:>6,.0f} sps  |  ETA {eta/60:.1f} min"
            )
            self._last_log = n
        return True


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORY (needed by SubprocVecEnv)
# ─────────────────────────────────────────────────────────────────────────────

def make_env(price_arrays: dict, seed: int, episode_len: int = 63):
    def _init():
        set_random_seed(seed)
        env = PortfolioTradingEnv(price_arrays, episode_len=episode_len, seed=seed)
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Check GPU ────────────────────────────────────────────────────────────
    global DEVICE
    try:
        import torch
        if not torch.cuda.is_available():
            print("  WARNING: CUDA not available — falling back to CPU.")
            DEVICE = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    except ImportError:
        print("  PyTorch not found — using CPU. Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        DEVICE = "cpu"

    # ── Load market data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading market data…")
    all_tickers   = sorted(set(TRAIN_TICKERS + EVAL_TICKERS))
    price_arrays  = load_or_download(all_tickers)

    train_arrays  = {t: price_arrays[t] for t in TRAIN_TICKERS if t in price_arrays}
    eval_arrays   = {t: price_arrays[t] for t in EVAL_TICKERS  if t in price_arrays}
    if not eval_arrays:
        eval_arrays = train_arrays   # fallback

    print(f"  Train: {len(train_arrays)} tickers  |  Eval: {len(eval_arrays)} tickers")

    # ── Build vectorised training env ────────────────────────────────────────
    print("\n[2/5] Building environments…")
    train_fns = [make_env(train_arrays, seed=i) for i in range(N_ENVS)]

    # Use SubprocVecEnv for true parallelism across CPU cores
    # "fork" is Linux-only; Windows requires "spawn"
    _start_method = "fork" if sys.platform != "win32" else "spawn"
    vec_env = SubprocVecEnv(train_fns, start_method=_start_method)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,   # normalise observations (stable training)
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    # Eval env (single, deterministic)
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([lambda: Monitor(PortfolioTradingEnv(eval_arrays, episode_len=126, seed=999))])
    eval_env = VecNormalize(
        eval_env,
        training=False,      # Do not update normalization stats during evaluation
        norm_reward=False,   # Do not normalize rewards during evaluation
        clip_obs=10.0,
        gamma=GAMMA,
    )

    # ── Build PPO model ──────────────────────────────────────────────────────
    print("\n[3/5] Building PPO model…")
    policy_kwargs = dict(
        net_arch=NET_ARCH,
        # Orthogonal init — empirically better for financial RL
        ortho_init=True,
    )
    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = LEARNING_RATE,
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        gamma           = GAMMA,
        gae_lambda      = GAE_LAMBDA,
        clip_range      = CLIP_RANGE,
        ent_coef        = ENT_COEF,
        vf_coef         = VF_COEF,
        max_grad_norm   = MAX_GRAD_NORM,
        policy_kwargs   = policy_kwargs,
        verbose         = 0,
        device          = DEVICE,
        seed            = 42,
        tensorboard_log = "./tb_logs/",
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}  |  Device: {model.device}")

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(50_000 // N_ENVS, 1),
        save_path   = "./checkpoints/",
        name_prefix = "ppo_finq",
        verbose     = 0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = "./best_model/",
        log_path             = "./eval_logs/",
        eval_freq            = max(50_000 // N_ENVS, 1),
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 0,
    )
    progress_cb = ProgressCallback(TOTAL_TIMESTEPS, log_interval=25_000)

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {TOTAL_TIMESTEPS:,} timesteps…")
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./best_model",  exist_ok=True)
    os.makedirs("./eval_logs",   exist_ok=True)

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, eval_cb, progress_cb],
        reset_num_timesteps = True,
        progress_bar    = False,
    )

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed/60:.1f} min  ({TOTAL_TIMESTEPS/elapsed:,.0f} sps)")

    # ── Save model + VecNormalize stats ──────────────────────────────────────
    model.save("ppo_trading_agent")
    vec_env.save("vecnorm_stats.pkl")
    print("  Saved: ppo_trading_agent.zip  +  vecnorm_stats.pkl")

    # ── Extract RL weight schedule for RLStateAgent ──────────────────────────
    print("\n[5/5] Extracting RL weight schedule…")
    rl_weights = extract_rl_weights(model, train_arrays)
    with open("rl_weights.json", "w") as f:
        json.dump(rl_weights, f, indent=2)
    print("  Saved: rl_weights.json")

    # Print sample extracted weights
    print("\n  Sample extracted weights (first 3 vol buckets):")
    for mode, triples in rl_weights.items():
        print(f"    {mode:10s}: vol=low  → w_ret={triples[0][0]:.3f}  "
              f"w_mom={triples[0][1]:.3f}  w_cvar={triples[0][2]:.3f}")
        print(f"    {' ':10s}  vol=high → w_ret={triples[-1][0]:.3f}  "
              f"w_mom={triples[-1][1]:.3f}  w_cvar={triples[-1][2]:.3f}")

    print(f"\n{'─'*60}")
    print("  DONE. Load the agent:")
    print("    model = PPO.load('ppo_trading_agent')")
    print("    stats = VecNormalize.load('vecnorm_stats.pkl', DummyVecEnv(...))")
    print("  rl_weights.json → drop next to quantum_core.py for live use")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
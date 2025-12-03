# -*- coding: utf-8 -*-
# tennis_ai_plus_streamlit.py — Versión Streamlit
#
# - Batch por múltiples match_key (api-tennis.com)
# - Barra de progreso
# - Log de avance tipo consola
# - Timer total del lote
# - Velocidad promedio por match (segundos desde API hasta JSON final)
# - Soporte de múltiples API keys (round-robin, hasta 6)
# - Misma lógica de cálculo que el script original (compute_from_fixture, etc.)

import os
import json
import math
import time
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from unidecode import unidecode

import pandas as pd
import numpy as np

import streamlit as st

# ===================== CONFIGURACIÓN GLOBAL =====================

BASE_URL = "https://api.api-tennis.com/tennis/"

RANK_BUCKETS = {
    "GS": 1.30,      # Grand Slam
    "ATP/WTA": 1.15,
    "Challenger": 1.00,
    "ITF": 0.85
}
RANK_BUCKETS.setdefault("Other", 0.95)


# ===================== MANEJO DE API KEYS =====================

def _load_api_keys_from_env():
    """Carga hasta 6 API keys desde variables de entorno."""
    keys = []
    single = (os.getenv("API_TENNIS_KEY") or "").strip()
    if single:
        keys.append(single)
    # Ahora hasta 6 adicionales: API_TENNIS_KEY_1..6
    for i in range(1, 7):
        k = (os.getenv(f"API_TENNIS_KEY_{i}") or "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


API_KEYS = _load_api_keys_from_env()
_API_IDX = 0


def set_api_keys_from_string(s: str):
    """
    Permite escribir varias API keys separadas por coma/espacio/;.
    Ej: "KEY1,KEY2,KEY3,KEY4,KEY5,KEY6"
    """
    global API_KEYS, _API_IDX
    parts = []
    if s:
        tmp = s.replace(";", ",")
        for token in tmp.split(","):
            token = token.strip()
            if token:
                parts.append(token)
    # máximo 6 keys
    parts = parts[:6]
    if parts:
        API_KEYS = parts
        _API_IDX = 0


def get_next_api_key():
    """Devuelve la siguiente API key en round-robin, o None si no hay."""
    global _API_IDX
    if not API_KEYS:
        return None
    key = API_KEYS[_API_IDX % len(API_KEYS)]
    _API_IDX += 1
    return key


# ===================== UTILIDADES =====================

def normalize(s: str) -> str:
    return unidecode(s or "").strip().lower()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def clamp(v, a, b):
    return max(a, min(b, v))


def make_session():
    """requests.Session con reintentos para 5xx/timeout."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = make_session()
HTTP_TIMEOUT = 25  # segundos


# ===================== API WRAPPER =====================

def call_api(method: str, params: dict):
    """
    Llama a la API rotando entre las API keys disponibles.
    """
    params = {k: v for k, v in params.items() if v is not None}
    params.pop("APIkey", None)

    api_key = get_next_api_key()
    if not api_key:
        raise RuntimeError(
            "No hay API keys configuradas.\n"
            "Define API_TENNIS_KEY o API_TENNIS_KEY_1..6 en el entorno\n"
            "o escribe 1–6 keys en el campo 'API Keys'."
        )

    params["APIkey"] = api_key

    r = SESSION.get(BASE_URL, params={"method": method, **params}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if str(data.get("success")) == "1":
        return data.get("result", {})

    if str(data.get("error")) == "1":
        try:
            detail = (data.get("result") or [{}])[0]
            cod = detail.get("cod")
            msg = detail.get("msg") or "API error"
        except Exception:
            cod, msg = None, "API error"
        raise RuntimeError(f"{method} → {msg} (cod={cod})")

    raise RuntimeError(f"{method} → Respuesta no esperada: {data}")


# ===================== ODDS (Bet365) =====================

def get_bet365_odds_for_match(match_key: int):
    """
    Devuelve (home_odds, away_odds) de Bet365 para ganador del partido (Home/Away),
    o (None, None) si no hay datos. Formato decimal.
    """
    try:
        res = call_api("get_odds", {"match_key": match_key}) or {}
        m = res.get(str(match_key)) or res.get(int(match_key))
        if not isinstance(m, dict):
            return (None, None)

        ha = m.get("Home/Away") or {}
        home = (ha.get("Home") or {})
        away = (ha.get("Away") or {})

        def pick_b365(d):
            if not isinstance(d, dict):
                return None
            for k in d.keys():
                if str(k).strip().lower() == "bet365":
                    return d[k]
            return None

        home_b365 = pick_b365(home)
        away_b365 = pick_b365(away)

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        return (to_float(home_b365), to_float(away_b365))
    except Exception:
        return (None, None)


def get_bet365_setscore_odds_for_match(match_key: int):
    """
    Devuelve un dict con momios Bet365 de marcador de sets (best-of-3).
    """
    out = {"2:0": None, "2:1": None, "1:2": None, "0:2": None}
    try:
        res = call_api("get_odds", {"match_key": match_key}) or {}
        m = res.get(str(match_key)) or res.get(int(match_key))
        if not isinstance(m, dict):
            return out

        for market_name, market_data in m.items():
            if not isinstance(market_data, dict):
                continue
            for sel_name, sel_data in market_data.items():
                if not isinstance(sel_data, dict):
                    continue

                price = None
                for bk, val in sel_data.items():
                    if str(bk).strip().lower() == "bet365":
                        try:
                            price = float(val)
                        except Exception:
                            price = None
                        break

                if price is None:
                    continue

                name_clean = str(sel_name).lower().replace(" ", "")
                name_clean = name_clean.replace(":", "-")
                if "2-0" in name_clean:
                    out["2:0"] = price
                elif "2-1" in name_clean:
                    out["2:1"] = price
                elif "1-2" in name_clean:
                    out["1:2"] = price
                elif "0-2" in name_clean:
                    out["0:2"] = price

        return out
    except Exception:
        return out


# ===================== FIXTURE HELPERS =====================

def list_fixtures(date_start: str, date_stop: str, tz: str, player_key=None):
    params = {
        "date_start": date_start,
        "date_stop": date_stop,
        "timezone": tz,
    }
    if player_key:
        params["player_key"] = player_key
    res = call_api("get_fixtures", params) or []
    return res


def get_fixture_by_key(match_key: int, tz: str = "Europe/Berlin", center_date: str | None = None):
    """
    Obtiene el fixture por match_key de forma robusta (get_events + ventanas de fixtures).
    """
    # 1) Intento directo
    try:
        res = call_api("get_events", {"event_key": match_key}) or []
        if isinstance(res, list):
            for m in res:
                if safe_int(m.get("event_key")) == int(match_key):
                    return m
        elif isinstance(res, dict) and safe_int(res.get("event_key")) == int(match_key):
            return res
    except Exception:
        pass

    # 2) Escanear ventanas de fixtures
    if center_date:
        try:
            base = datetime.strptime(center_date, "%Y-%m-%d").date()
        except Exception:
            base = datetime.utcnow().date()
    else:
        base = datetime.utcnow().date()

    CHUNK_SIZES = [7, 3, 1]
    RINGS = [14, 28, 56, 112, 200]

    for ring in RINGS:
        start_global = base - timedelta(days=ring)
        stop_global = base + timedelta(days=10)
        cur_start = start_global
        while cur_start <= stop_global:
            hit_this_window = False
            for chunk in CHUNK_SIZES:
                cur_stop = min(cur_start + timedelta(days=chunk - 1), stop_global)
                try:
                    fixtures = list_fixtures(
                        cur_start.strftime("%Y-%m-%d"),
                        cur_stop.strftime("%Y-%m-%d"),
                        tz,
                    ) or []
                    for m in fixtures:
                        if safe_int(m.get("event_key")) == int(match_key):
                            return m
                    hit_this_window = True
                    break
                except requests.HTTPError as http_err:
                    if http_err.response is not None and http_err.response.status_code == 500:
                        continue
                    else:
                        raise
                except Exception:
                    continue
            step = max(CHUNK_SIZES) if hit_this_window else 1
            cur_start = cur_start + timedelta(days=step)

    raise ValueError(f"No se encontró el match_key={match_key} alrededor de {base}.")


# ===================== FEATURE ENGINEERING =====================

def get_player_matches(player_key: int, days_back=365, ref_date: str | None = None):
    """
    Partidos ya finalizados de un jugador (hasta el día anterior).
    """
    if ref_date:
        try:
            ref = datetime.strptime(ref_date, "%Y-%m-%d").date()
        except Exception:
            ref = datetime.utcnow().date()
    else:
        ref = datetime.utcnow().date()

    stop = ref - timedelta(days=1)
    start = stop - timedelta(days=days_back)

    start_str = start.strftime("%Y-%m-%d")
    stop_str = stop.strftime("%Y-%m-%d")

    res = list_fixtures(start_str, stop_str, "Europe/Berlin", player_key=player_key) or []
    clean = []
    for m in res:
        status = (m.get("event_status") or "").lower()
        if "finished" in status or m.get("event_winner") in ("First Player", "Second Player"):
            clean.append(m)
    return clean


def is_win_for_name(match, player_name_norm: str):
    fp = normalize(match.get("event_first_player"))
    sp = normalize(match.get("event_second_player"))
    w = match.get("event_winner")
    if w == "First Player":
        return fp == player_name_norm
    if w == "Second Player":
        return sp == player_name_norm
    res = (match.get("event_final_result") or "").strip().lower()
    if fp == player_name_norm and (res.startswith("2 - 0") or res.startswith("2 - 1")):
        return True
    if sp == player_name_norm and (res.startswith("0 - 2") or res.startswith("1 - 2")):
        return True
    return False


def winrate_60d_and_lastN(matches, player_name_norm: str, N=10, days=60, ref_date: str | None = None):
    if ref_date:
        try:
            base_dt = datetime.strptime(ref_date, "%Y-%m-%d")
        except Exception:
            base_dt = datetime.utcnow()
    else:
        base_dt = datetime.utcnow()

    def days_ago(m):
        try:
            d = datetime.strptime(m["event_date"], "%Y-%m-%d")
            return (base_dt - d).days
        except Exception:
            return 10 ** 6

    recent = [m for m in matches if days_ago(m) <= days]
    wr60 = (sum(is_win_for_name(m, player_name_norm) for m in recent) / len(recent)) if recent else 0.5

    sorted_all = sorted(
        matches,
        key=lambda x: (x.get("event_date") or "", x.get("event_time") or "00:00"),
        reverse=True,
    )
    lastN = sorted_all[:N]
    wrN = (sum(is_win_for_name(m, player_name_norm) for m in lastN) / len(lastN)) if lastN else 0.5

    last_date = sorted_all[0]["event_date"] if sorted_all else None
    return wr60, wrN, last_date, sorted_all


def compute_momentum(sorted_matches, player_name_norm: str):
    streak = 0
    for m in sorted_matches:
        w = is_win_for_name(m, player_name_norm)
        if w:
            streak = +1 if streak < 0 else streak + 1
        else:
            streak = -1 if streak > 0 else -1
        if streak >= 4:
            return +1
        if streak <= -3:
            return -1
    return 0


def rest_days(last_date_str: str | None, ref_date_str: str | None = None):
    if not last_date_str:
        return None
    try:
        d = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except Exception:
        return None

    if ref_date_str:
        try:
            base = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
        except Exception:
            base = datetime.utcnow().date()
    else:
        base = datetime.utcnow().date()

    return (base - d).days


def rest_score(days):
    if days is None:
        return 0.0
    return clamp(1.0 - abs(days - 7) / 21.0, 0.0, 1.0)


def league_bucket(league_name: str):
    s = (league_name or "").lower()
    if any(k in s for k in ["grand slam", "roland", "wimbledon", "us open", "australian open"]):
        return "GS"
    if any(k in s for k in ["atp", "wta"]):
        return "ATP/WTA"
    if "challenger" in s:
        return "Challenger"
    if "itf" in s:
        return "ITF"
    return "Other"


def surface_winrate(matches, player_name_norm: str, surface: str):
    if not surface:
        return 0.5
    sur = surface.lower()
    hist = [m for m in matches if (m.get("event_tournament_surface") or "").lower() == sur]
    if not hist:
        return 0.5
    return sum(is_win_for_name(m, player_name_norm) for m in hist) / len(hist)


def travel_penalty(last_match_country, current_country, days_since):
    if not last_match_country or not current_country or days_since is None:
        return 0.0
    if last_match_country.strip().lower() == current_country.strip().lower():
        return 0.0
    if days_since <= 3:
        return 0.15
    if days_since <= 5:
        return 0.07
    return 0.0


def elo_synth_from_opposition(matches, player_name_norm: str):
    if not matches:
        return 0.0
    score = 0.0
    for m in matches[:20]:
        bucket = league_bucket(m.get("league_name", ""))
        weight = RANK_BUCKETS.get(bucket, 1.0)
        w = is_win_for_name(m, player_name_norm)
        score += (1.0 if w else -1.0) * weight
    score = score / (20.0 * 1.30)
    return clamp(score, -1.0, 1.0)


def compute_h2h(player_key_a, player_key_b, years_back=5, ref_date: str | None = None):
    if ref_date:
        try:
            ref = datetime.strptime(ref_date, "%Y-%m-%d").date()
        except Exception:
            ref = datetime.utcnow().date()
    else:
        ref = datetime.utcnow().date()

    stop = ref - timedelta(days=1)
    start = stop - timedelta(days=365 * years_back)

    start_str = start.strftime("%Y-%m-%d")
    stop_str = stop.strftime("%Y-%m-%d")

    res_a = list_fixtures(start_str, stop_str, "Europe/Berlin", player_key=player_key_a) or []
    res_b = list_fixtures(start_str, stop_str, "Europe/Berlin", player_key=player_key_b) or []

    def key_of(m):
        return (
            normalize(m.get("event_first_player")),
            normalize(m.get("event_second_player")),
            m.get("event_date"),
        )

    idx_b = {key_of(m): m for m in res_b}
    wins_a = wins_b = 0

    for ma in res_a:
        k = key_of(ma)
        mb = idx_b.get(k)
        if not mb:
            continue
        w = ma.get("event_winner")
        if w == "First Player":
            wins_a += 1
        elif w == "Second Player":
            wins_b += 1

    total = wins_a + wins_b
    pct_a = wins_a / total if total else 0.5
    return wins_a, wins_b, pct_a


# ===================== CACHÉ =====================

@lru_cache(maxsize=2000)
def cached_player_matches(player_key: int, days_back: int, ref_date: str | None):
    return tuple(get_player_matches(player_key, days_back=days_back, ref_date=ref_date))


@lru_cache(maxsize=2000)
def cached_h2h(player_key_a: int, player_key_b: int, years_back: int, ref_date: str | None):
    return compute_h2h(player_key_a, player_key_b, years_back=years_back, ref_date=ref_date)


@lru_cache(maxsize=5000)
def cached_bet365_match(match_key: int):
    return get_bet365_odds_for_match(match_key)


@lru_cache(maxsize=5000)
def cached_bet365_sets(match_key: int):
    return get_bet365_setscore_odds_for_match(match_key)


# ===================== MODELO =====================

def calibrate_probability(diff, weights, gamma=3.0, bias=0.0, bonus=0.0, malus=0.0):
    wsum = sum(weights.values()) or 1.0
    w = {k: v / wsum for k, v in weights.items()}
    z = (
        w.get("wr60", 0) * diff.get("wr60", 0)
        + w.get("wr10", 0) * diff.get("wr10", 0)
        + w.get("h2h", 0) * diff.get("h2h", 0)
        + w.get("rest", 0) * diff.get("rest", 0)
        + w.get("surface", 0) * diff.get("surface", 0)
        + w.get("elo", 0) * diff.get("elo", 0)
        + w.get("momentum", 0) * diff.get("momentum", 0)
        - w.get("travel", 0) * diff.get("travel", 0)
        + bias
    )
    p = logistic(gamma * z + bonus - malus)
    return clamp(p, 0.05, 0.95)


def invert_bo3_set_prob(pm):
    lo, hi = 0.05, 0.95
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pm_mid = mid * mid * (3 - 2 * mid)
        if pm_mid < pm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bo3_distribution(p_set):
    s = p_set
    q = 1 - s
    p20 = s * s
    p21 = 2 * s * s * q
    p12 = 2 * q * q * s
    p02 = q * q
    tot = p20 + p21 + p12 + p02
    return {"2:0": p20 / tot, "2:1": p21 / tot, "1:2": p12 / tot, "0:2": p02 / tot}


def to_decimal(p):
    p = clamp(p, 0.01, 0.99)
    return round(1.0 / p, 3)


# ===================== compute_from_fixture =====================

def compute_from_fixture(meta: dict, surface_hint: str,
                         weights: dict, gamma: float, bias: float):
    match_key = safe_int(meta.get("event_key"))
    tz = meta.get("timezone") or "Europe/Berlin"
    date_str = meta.get("event_date") or datetime.utcnow().strftime("%Y-%m-%d")

    api_p1 = meta.get("event_first_player")
    api_p2 = meta.get("event_second_player")
    api_p1n = normalize(api_p1)
    api_p2n = normalize(api_p2)

    p1k = safe_int(meta.get("first_player_key"))
    p2k = safe_int(meta.get("second_player_key"))

    surface_api = (meta.get("event_tournament_surface") or "").strip() or None
    surface_final = (surface_hint or "").strip().lower() or (surface_api.lower() if surface_api else None)

    lastA = list(cached_player_matches(p1k, 365, date_str)) if p1k else []
    lastB = list(cached_player_matches(p2k, 365, date_str)) if p2k else []

    wr60_A, wr10_A, lastA_date, sortedA = winrate_60d_and_lastN(lastA, api_p1n, N=10, days=60, ref_date=date_str)
    wr60_B, wr10_B, lastB_date, sortedB = winrate_60d_and_lastN(lastB, api_p2n, N=10, days=60, ref_date=date_str)

    momA = compute_momentum(sortedA, api_p1n)
    momB = compute_momentum(sortedB, api_p2n)

    rA_days = rest_days(lastA_date, ref_date_str=date_str)
    rB_days = rest_days(lastB_date, ref_date_str=date_str)
    rA = rest_score(rA_days)
    rB = rest_score(rB_days)

    surf_wrA = surface_winrate(lastA, api_p1n, surface_final)
    surf_wrB = surface_winrate(lastB, api_p2n, surface_final)

    lastA_country = lastA and (lastA[0].get("country") or lastA[0].get("event_tournament_country"))
    lastB_country = lastB and (lastB[0].get("country") or lastB[0].get("event_tournament_country"))
    tourn_country = meta.get("country") or meta.get("event_tournament_country")
    travA = travel_penalty(lastA_country, tourn_country, rA_days or 999)
    travB = travel_penalty(lastB_country, tourn_country, rB_days or 999)

    if p1k and p2k:
        _, _, h2h_pct_a = cached_h2h(p1k, p2k, 5, date_str)
    else:
        h2h_pct_a = 0.5
    h2h_pct_b = 1.0 - h2h_pct_a

    eloA = elo_synth_from_opposition(sortedA, api_p1n)
    eloB = elo_synth_from_opposition(sortedB, api_p2n)

    total_obs = len(sortedA) + len(sortedB)
    reg_alpha = 0.0
    if total_obs < 6:
        reg_alpha = 0.6
    elif total_obs < 12:
        reg_alpha = 0.35
    elif total_obs < 20:
        reg_alpha = 0.2

    wr60_A = (1 - reg_alpha) * wr60_A + reg_alpha * 0.5
    wr60_B = (1 - reg_alpha) * wr60_B + reg_alpha * 0.5
    wr10_A = (1 - reg_alpha) * wr10_A + reg_alpha * 0.5
    wr10_B = (1 - reg_alpha) * wr10_B + reg_alpha * 0.5
    surf_wrA = (1 - reg_alpha) * surf_wrA + reg_alpha * 0.5
    surf_wrB = (1 - reg_alpha) * surf_wrB + reg_alpha * 0.5
    h2h_pct_a = (1 - reg_alpha) * h2h_pct_a + reg_alpha * 0.5
    h2h_pct_b = 1 - h2h_pct_a
    eloA = (1 - reg_alpha) * eloA
    eloB = (1 - reg_alpha) * eloB

    diff = {
        "wr60": wr60_A - wr60_B,
        "wr10": wr10_A - wr10_B,
        "h2h": h2h_pct_a - h2h_pct_b,
        "rest": rA - rB,
        "surface": surf_wrA - surf_wrB,
        "elo": eloA - eloB,
        "momentum": (0.03 if momA > 0 else (-0.03 if momA < 0 else 0.0))
                    - (0.03 if momB > 0 else (-0.03 if momB < 0 else 0.0)),
        "travel": travA - travB,
    }

    pA = calibrate_probability(diff=diff, weights=weights, gamma=gamma, bias=bias)
    pB = 1 - pA

    p_set_A = invert_bo3_set_prob(pA)
    dist = bo3_distribution(p_set_A)

    event_status = (meta.get("event_status") or "").strip()
    event_winner_side = meta.get("event_winner")
    if event_winner_side == "First Player":
        winner_name = api_p1
    elif event_winner_side == "Second Player":
        winner_name = api_p2
    else:
        winner_name = None
    final_sets_str = (meta.get("event_final_result") or "").strip() or None

    if match_key:
        b365_home, b365_away = cached_bet365_match(match_key)
        bet365_cs = cached_bet365_sets(match_key)
    else:
        b365_home, b365_away = (None, None)
        bet365_cs = {"2:0": None, "2:1": None, "1:2": None, "0:2": None}

    bet365_p1 = b365_home
    bet365_p2 = b365_away

    out = {
        "match_key": int(match_key) if match_key is not None else None,
        "inputs": {
            "date": date_str,
            "player1": api_p1,
            "player2": api_p2,
            "timezone": tz,
            "surface_used": surface_final or "(no especificada)",
        },
        "probabilities": {
            "match": {"player1": round(pA, 4), "player2": round(pB, 4)},
            "final_sets": {k: round(v, 4) for k, v in dist.items()},
        },
        "synthetic_odds_decimal": {
            "player1": to_decimal(pA),
            "player2": to_decimal(pB),
            "2:0": to_decimal(dist["2:0"]),
            "2:1": to_decimal(dist["2:1"]),
            "1:2": to_decimal(dist["1:2"]),
            "0:2": to_decimal(dist["0:2"]),
        },
        "bet365_odds_decimal": {
            "player1": bet365_p1,
            "player2": bet365_p2,
        },
        "bet365_setscore_odds_decimal": {
            "2:0": bet365_cs.get("2:0"),
            "2:1": bet365_cs.get("2:1"),
            "1:2": bet365_cs.get("1:2"),
            "0:2": bet365_cs.get("0:2"),
        },
        "official_result": {
            "status": event_status,
            "winner_side": event_winner_side,
            "winner_name": winner_name,
            "final_sets": final_sets_str,
        },
    }
    return out


# ===================== STREAMLIT APP =====================

st.set_page_config(page_title="🎾 Tennis AI+ Batch (Streamlit)", layout="wide")
st.title("🎾 Tennis AI+ — Momios sintéticos (api-tennis.com) · Batch por match_key")

st.markdown(
    """
Escribe tus **API keys** de api-tennis (1–6, separadas por coma),
la superficie (opcional) y la lista de `match_key` (uno por línea o separados por coma).
"""
)

# --------- Sidebar: configuración ---------
with st.sidebar:
    st.header("🔑 API & Modelo")

    api_keys_input = st.text_input(
        "API Keys (1–6, separadas por coma)",
        value=",".join(API_KEYS) if API_KEYS else "",
        type="password",
    )

    tz = st.text_input("Timezone IANA", value="Europe/Berlin")
    surface_hint = st.text_input("Superficie (hard/clay/grass/indoor, opcional)", value="")

    st.markdown("**Pesos (se normalizan a suma 1):**")
    w_wr60 = st.slider("wr60 (forma 60 días)", 0.0, 1.0, 0.12, 0.01)
    w_wr10 = st.slider("wr10 (últimos 10)", 0.0, 1.0, 0.33, 0.01)
    w_h2h = st.slider("h2h", 0.0, 1.0, 0.01, 0.01)
    w_rest = st.slider("rest (descanso)", 0.0, 1.0, 0.19, 0.01)
    w_surf = st.slider("surface", 0.0, 1.0, 0.00, 0.01)
    w_elo = st.slider("elo sintético", 0.0, 1.0, 0.31, 0.01)
    w_mom = st.slider("momentum", 0.0, 1.0, 0.05, 0.01)
    w_trav = st.slider("travel (malus)", 0.0, 1.0, 0.00, 0.01)

    gamma = st.slider("gamma (agresividad)", 0.5, 5.0, 3.0, 0.1)
    bias = st.slider("bias (sesgo)", -0.5, 0.5, 0.0, 0.01)

    max_workers = st.slider("Hilos simultáneos máximos", 1, 16, 4, 1)

    center_date_for_key = st.text_input(
        "Fecha estimada para buscar fixtures (YYYY-MM-DD, opcional)",
        value="",
        help="Se usa para acotar la búsqueda de fixtures alrededor de esa fecha.",
    )

# --------- Main: input de match_keys ---------

st.subheader("Lista de match_key")
raw_keys = st.text_area(
    "Pega aquí los match_key (uno por línea o separados por espacios/comas):",
    height=160,
)


def parse_batch_keys(raw: str):
    parts = [p.strip() for p in raw.replace(",", " ").replace("\n", " ").split(" ") if p.strip()]
    keys = []
    for p in parts:
        if p.isdigit():
            keys.append(int(p))
    # quitar duplicados manteniendo orden
    seen = set()
    dedup = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


# --------- Placeholders para progreso / log ---------

progress_bar = st.progress(0, text="Esperando lote…")
status_placeholder = st.empty()
timer_placeholder = st.empty()
speed_placeholder = st.empty()
log_placeholder = st.empty()
results_placeholder = st.empty()

if "last_results_batch" not in st.session_state:
    st.session_state.last_results_batch = []
if "last_errors" not in st.session_state:
    st.session_state.last_errors = []


def run_batch():
    # Configurar API keys
    set_api_keys_from_string(api_keys_input)
    if not API_KEYS:
        st.error("Faltan API keys. Escríbelas en la barra lateral.")
        return

    keys = parse_batch_keys(raw_keys)
    if not keys:
        st.warning("No hay match_keys válidos en el cuadro de texto.")
        return

    weights = {
        "wr60": w_wr60,
        "wr10": w_wr10,
        "h2h": w_h2h,
        "rest": w_rest,
        "surface": w_surf,
        "elo": w_elo,
        "momentum": w_mom,
        "travel": w_trav,
    }

    logs: list[str] = []
    durations: list[float] = []
    results: list[dict] = []
    errors: list[tuple[int, str]] = []

    total = len(keys)
    status_placeholder.write(f"🔄 Lote en progreso… {total} partidos")
    progress_bar.progress(0, text="0%")

    batch_start = time.perf_counter()

    def fmt_time(seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update_log():
        log_text = "\n".join(logs[-500:])  # limitar para que no crezca infinito
        log_placeholder.text_area(
            "Resultados (JSON / Log)",
            value=log_text,
            height=300,
        )

    center = center_date_for_key.strip() or None

    def process_one(idx: int, mk: int):
        """
        Devuelve: (status, mk, out, err, elapsed)
        status ∈ {"ok","err"}
        """
        t0 = time.perf_counter()
        meta = get_fixture_by_key(mk, tz=tz or "Europe/Berlin", center_date=center)
        out = compute_from_fixture(
            meta,
            surface_hint=surface_hint,
            weights=weights,
            gamma=gamma,
            bias=bias,
        )
        elapsed = time.perf_counter() - t0
        return "ok", mk, out, None, elapsed

    # Ejecutar en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(process_one, idx, mk): (idx, mk)
            for idx, mk in enumerate(keys, start=1)
        }

        done_count = 0
        for future in as_completed(future_to_info):
            idx, mk = future_to_info[future]
            try:
                status, mk_ret, out, err, elapsed = future.result()
            except Exception as e:
                status, mk_ret, out, err, elapsed = "err", mk, None, str(e), 0.0

            if status == "ok" and out is not None:
                results.append(out)
                durations.append(elapsed)
                logs.append(
                    f"OK [{idx}/{total}]: {out['inputs']['player1']} vs {out['inputs']['player2']}  "
                    f"(date: {out['inputs']['date']})  tiempo={elapsed:.2f}s"
                )
            else:
                errors.append((mk_ret, err))
                logs.append(f"ERROR [{idx}/{total}] match_key {mk_ret}: {err}")

            done_count += 1
            pct = int(100 * done_count / total)
            progress_bar.progress(pct / 100, text=f"{pct}% completado")
            elapsed_total = int(time.perf_counter() - batch_start)
            timer_placeholder.write(f"⏱ Tiempo transcurrido: {fmt_time(elapsed_total)}")
            if durations:
                avg = sum(durations) / len(durations)
                speed_placeholder.write(f"⚡ Velocidad promedio por match: {avg:.2f} s")
            update_log()

    st.session_state.last_results_batch = results
    st.session_state.last_errors = errors

    elapsed_total = int(time.perf_counter() - batch_start)
    status_placeholder.write(f"✅ Lote finalizado ({total} partidos, tiempo total {fmt_time(elapsed_total)}).")

    if errors:
        st.warning(f"Hubo {len(errors)} errores. Revisa el log para detalles.")

    # Mostrar un resumen JSON opcional
    results_placeholder.json(
        {
            "count": len(results),
            "errors": errors,
        }
    )


if st.button("🚀 Calcular lote"):
    run_batch()

# Mostrar tabla resumida cuando ya hay resultados
if st.session_state.last_results_batch:
    st.subheader("Resumen rápido (match_key, jugadores, probabilidades)")
    rows = []
    for r in st.session_state.last_results_batch:
        mk = r.get("match_key")
        inp = r.get("inputs", {})
        probs = r.get("probabilities", {}).get("match", {})
        odds = r.get("synthetic_odds_decimal", {})
        rows.append(
            {
                "match_key": mk,
                "date": inp.get("date"),
                "player1": inp.get("player1"),
                "player2": inp.get("player2"),
                "p_player1": probs.get("player1"),
                "p_player2": probs.get("player2"),
                "odds_player1": odds.get("player1"),
                "odds_player2": odds.get("player2"),
            }
        )
    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, use_container_width=True)

    # Descargar JSON
    json_bytes = json.dumps(st.session_state.last_results_batch, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Descargar resultados (JSON)",
        data=json_bytes,
        file_name="tennis_batch_results.json",
        mime="application/json",
    )

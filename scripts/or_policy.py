#!/usr/bin/env python3
"""OpenRouter run policy: which endpoints a model may run on.

Pure logic over two OpenRouter documents that the shell side fetches and caches
(`/api/v1/models/<id>/endpoints` and `/api/v1/endpoints/zdr`), so it can be tested
with fixtures and never touches the network itself.

  decide  one model: the `provider` request object, the model_info facts, or an exclusion
  table   one model: its endpoints (provider, quantization, price, context, uptime, ZDR)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: OpenRouter's `provider.quantizations` values that say what the weights are;
#: `unknown` is the one it uses for "not disclosed"
DISCLOSED = ("int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32")


def _load(path: str | None) -> Any:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _price(value: Any) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def normalize(endpoints_doc: Any, zdr_doc: Any) -> list[dict[str, Any]] | None:
    """Endpoints of one model with their disclosed quantization and ZDR flag; None when
    either document is missing or unreadable (nothing can be said about eligibility)."""
    if not isinstance(endpoints_doc, dict) or not isinstance(zdr_doc, dict):
        return None
    data = endpoints_doc.get("data")
    zdr_rows = zdr_doc.get("data")
    if not isinstance(data, dict) or not isinstance(zdr_rows, list):
        return None
    model_name = data.get("name")
    out: list[dict[str, Any]] = []
    for ep in data.get("endpoints") or []:
        if not isinstance(ep, dict):
            continue
        tag, name = ep.get("tag"), ep.get("name")
        in_zdr = any(
            isinstance(z, dict)
            and ((tag and z.get("tag") == tag and z.get("model_name") == model_name)
                 or (name and z.get("name") == name))
            for z in zdr_rows
        )
        quant = str(ep.get("quantization") or "unknown").lower()
        pricing = ep.get("pricing") or {}
        out.append({
            "provider": str(ep.get("provider_name") or ""),
            "tag": str(tag or ""),
            "quantization": quant,
            "disclosed": quant in DISCLOSED,
            "zdr": in_zdr,
            "context": ep.get("context_length"),
            "price_in": _price(pricing.get("prompt")),
            "price_out": _price(pricing.get("completion")),
            "uptime": ep.get("uptime_last_30m"),
        })
    return out


def _is(ep: dict[str, Any], wanted: str) -> bool:
    w = wanted.strip().casefold()
    return bool(w) and (ep["provider"].casefold() == w or ep["tag"].casefold() == w
                        or ep["tag"].casefold().split("/")[0] == w)


def decide(slug: str, eps: list[dict[str, Any]] | None, *, open_weight: bool, zdr_only: bool,
           pin: str, quantizations: tuple[str, ...] = DISCLOSED) -> dict[str, Any]:
    """The routing decision for one OpenRouter model.

    `open_weight` is the strict policy applying to it (disclosed quantization and ZDR only);
    `zdr_only` is the global ZDR switch; `pin` is `Provider[|Provider...]` or empty.
    """
    pins = [p.strip() for p in pin.split("|") if p.strip()]
    filt: dict[str, Any] = {}
    if open_weight:
        filt = {"zdr": True, "quantizations": list(quantizations), "allow_fallbacks": True}
    elif zdr_only:
        filt = {"zdr": True}
    policy = "open_weight_strict" if open_weight else "zdr_only" if zdr_only else "pinned" if pins else "none"
    result: dict[str, Any] = {
        "slug": slug, "policy": policy, "exclude": False, "reason": "", "provider": filt or None,
        "pinned": None, "quantization": None, "zdr": True if filt.get("zdr") else None,
        "eligible": None, "warnings": [],
    }
    if eps is None:
        # Nothing to check against: the request filter still enforces the policy upstream
        result["warnings"].append(
            f"{slug}: endpoint data unavailable — filter applied, eligibility not checked"
            + (f", pin '{pin}' ignored" if pins else ""))
        return result

    if open_weight:
        eligible = [e for e in eps if e["disclosed"] and e["zdr"]]
    elif zdr_only:
        eligible = [e for e in eps if e["zdr"]]
    else:
        eligible = list(eps)
    result["eligible"] = len(eligible)
    if (open_weight or zdr_only) and not eligible:
        result["exclude"] = True
        result["reason"] = ("no endpoint with disclosed quantization and ZDR" if open_weight
                            else "no ZDR endpoint")
        return result

    chosen: list[dict[str, Any]] = []
    for p in pins:
        found = [e for e in eps if _is(e, p)]
        usable = [e for e in found if e in eligible]
        if usable:
            chosen.extend(e for e in usable if e not in chosen)
        elif not found:
            result["warnings"].append(f"{slug}: pinned provider '{p}' does not serve this model — pin ignored")
        elif open_weight:
            result["warnings"].append(
                f"{slug}: pinned provider '{p}' has no disclosed quantization and ZDR endpoint — pin ignored")
        else:
            result["warnings"].append(
                f"{slug}: pinned provider '{p}' is not ZDR, which OR_ZDR_ONLY requires — pin ignored")
    if chosen:
        names: list[str] = []
        for e in chosen:
            if e["provider"] not in names:
                names.append(e["provider"])
        result["provider"] = {**filt, "order": names, "allow_fallbacks": False}
        result["pinned"] = "|".join(names)
        quants = {e["quantization"] for e in chosen}
        result["quantization"] = quants.pop() if len(quants) == 1 else None
        result["zdr"] = True if all(e["zdr"] for e in chosen) else (
            result["zdr"] if result["zdr"] else False)
    elif open_weight:
        quants = {e["quantization"] for e in eligible}
        result["quantization"] = quants.pop() if len(quants) == 1 else None
    return result


def table(eps: list[dict[str, Any]] | None, *, only_eligible: bool) -> str:
    if eps is None:
        return "(endpoint data unavailable)"
    rows = [e for e in eps if not only_eligible or (e["disclosed"] and e["zdr"])]
    head = ("provider", "quantization", "in $/M", "out $/M", "context", "uptime 30m", "ZDR")

    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.3g}"

    body = [(e["provider"], e["quantization"] if e["disclosed"] else "not disclosed",
             fmt(e["price_in"]), fmt(e["price_out"]),
             "-" if e["context"] is None else str(e["context"]),
             "-" if e["uptime"] is None else f"{float(e['uptime']):.1f}%",
             "yes" if e["zdr"] else "no") for e in rows]
    if not body:
        return "(no eligible endpoint)" if only_eligible else "(no endpoint)"
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in [head, *body])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--slug", required=True)
    d.add_argument("--endpoints")
    d.add_argument("--zdr")
    d.add_argument("--open-weight", action="store_true")
    d.add_argument("--zdr-only", action="store_true")
    d.add_argument("--pin", default="")
    d.add_argument("--quantizations", default=",".join(DISCLOSED))
    t = sub.add_parser("table")
    t.add_argument("--endpoints")
    t.add_argument("--zdr")
    t.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    eps = normalize(_load(a.endpoints), _load(a.zdr))
    if a.cmd == "decide":
        quants = tuple(q.strip().lower() for q in a.quantizations.split(",") if q.strip())
        json.dump(decide(a.slug, eps, open_weight=a.open_weight, zdr_only=a.zdr_only, pin=a.pin,
                         quantizations=quants or DISCLOSED), sys.stdout)
        print()
    else:
        print(table(eps, only_eligible=not a.all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime, date
from zoneinfo import ZoneInfo
import io, re, time
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ================== Sabitler ==================
IST = ZoneInfo("Europe/Istanbul")
APP_VERSION = "1.2.0"

TR_MONTHS = {
    "ocak":1,"şubat":2,"subat":2,"mart":3,"nisan":4,"mayıs":5,"mayis":5,
    "haziran":6,"temmuz":7,"ağustos":8,"agustos":8,"eylül":9,"eylul":9,
    "ekim":10,"kasım":11,"kasim":11,"aralık":12,"aralik":12
}
ENTRY_KEYS = ["giriş","girış","tahsilat","alacak","maaş","maas","kira tahsilatı","borç tahsilatı","prim","satış","satis"]
EXIT_KEYS  = ["çıkış","cıkış","cikis","ödeme","odeme","harcama","kredi kartı","kira ödemesi","konut kredisi","umre","fatura","taksit"]

NUM_MAP = {
    "sıfır":0,"sifir":0,"bir":1,"iki":2,"üç":3,"uc":3,"dört":4,"dort":4,"beş":5,"bes":5,"altı":6,"alti":6,
    "yedi":7,"sekiz":8,"dokuz":9,"on":10,"yirmi":20,"otuz":30,"kırk":40,"kirk":40,"elli":50,"altmış":60,"altmis":60,"yetmiş":70,"yetmis":70,"seksen":80,"doksan":90,
    "yüz":100,"yuz":100,"bin":1000,"milyon":1_000_000,"milyar":1_000_000_000
}

# Kur cache
_RATE_CACHE: dict[tuple[str,str], tuple[float, float]] = {}

# ================== Yardımcılar ==================
def normalize_ccy(ccy: Optional[str]) -> str:
    if not ccy: return "TRY"
    c = ccy.strip().upper().replace("TL.", "TL")
    return "TRY" if c in ["TL","TRY"] else c

def tr_int_format(n: int) -> str:
    s = f"{abs(int(n)):,}".replace(",", ".")
    return f"-{s}" if int(n) < 0 else s

def parse_amount(x: Union[str,int,float]) -> float:
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().lower()
    for token in ["₺"," tl"," tl."," try"," eur","€"," usd","$"," gbp","£"]:
        s = s.replace(token, "")
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\d", s):
        s2 = s.replace(".", "").replace(",", ".")
        m = re.findall(r"[-+]?\d*\.?\d+", s2)
        if m: return float(m[0])
    tokens = re.findall(r"[a-zçğıöşü]+", s, flags=re.UNICODE)
    total, current = 0, 0
    for t in tokens:
        if t not in NUM_MAP: continue
        val = NUM_MAP[t]
        if val >= 100:
            if current == 0: current = 1
            current *= val
        else:
            current += val
        if val in (1000, 1_000_000, 1_000_000_000):
            total += current; current = 0
    total += current
    return float(total)

def parse_date_tr(s: str, now: date) -> date:
    s = s.strip().lower()
    m = re.match(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{4}))?$", s)
    if m:
        d, mth = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else now.year
        dt = date(y, mth, d);  dt = date(y+1, mth, d) if dt < now else dt
        return dt
    m2 = re.match(r"^(\d{1,2})\s+([a-zçğıöşü]+)(?:\s+(\d{4}))?$", s, flags=re.UNICODE)
    if m2:
        d, mon = int(m2.group(1)), m2.group(2)
        mth = TR_MONTHS.get(mon);  y = int(m2.group(3)) if m2.group(3) else now.year
        if not mth: raise ValueError(f"Ay adı anlaşılamadı: {s}")
        dt = date(y, mth, d);  dt = date(y+1, mth, d) if dt < now else dt
        return dt
    try:
        dt = datetime.fromisoformat(s).date()
        dt = date(dt.year+1, dt.month, dt.day) if dt < now else dt
        return dt
    except: pass
    raise ValueError(f"Tarih anlaşılamadı: {s}")

def detect_type(desc: str, given: Optional[str]) -> str:
    if given:
        g = given.strip().lower()
        return "Giriş" if "giriş" in g or g.startswith("gir") else "Çıkış"
    d = desc.lower()
    if any(k in d for k in EXIT_KEYS): return "Çıkış"
    return "Giriş"

# ---------- TCMB ----------
def _tcmb_today_xml() -> Optional[ET.Element]:
    try:
        r = requests.get("https://www.tcmb.gov.tr/kurlar/today.xml", timeout=6)
        if r.status_code == 200:
            r.encoding = "utf-8"
            return ET.fromstring(r.text)
    except: return None
    return None

def _tcmb_try_per(code: str, root: ET.Element) -> Optional[float]:
    for c in root.findall("Currency"):
        if c.attrib.get("CurrencyCode") == code:
            v = (c.findtext("ForexSelling") or "").replace(",", ".")
            try:
                x = float(v);  return x if x > 0 else None
            except: return None
    return None

def _rate_tcmb(frm: str, to: str) -> Optional[float]:
    frm, to = normalize_ccy(frm), normalize_ccy(to)
    if frm == to: return 1.0
    root = _tcmb_today_xml()
    if not root: return None
    def x_to_try(code: str) -> Optional[float]:
        return 1.0 if code == "TRY" else _tcmb_try_per(code, root)
    if to == "TRY": return x_to_try(frm)
    if frm == "TRY":
        v = x_to_try(to);  return (1.0/v) if v else None
    a, b = x_to_try(frm), x_to_try(to)
    return (a/b) if (a and b) else None

# ---------- Yahoo ----------
def _rate_yahoo(frm: str, to: str) -> Optional[float]:
    try:
        t = f"{frm}{to}=X"
        hist = yf.Ticker(t).history(period="1d")
        if not hist.empty:
            v = float(hist["Close"].iloc[-1])
            if v > 0: return v
        info = yf.Ticker(t).fast_info
        px = info.get("last_price")
        if px and float(px) > 0: return float(px)
    except: return None
    return None

def get_rate(frm: str, to: str, _depth: int = 0) -> float:
    frm, to = normalize_ccy(frm), normalize_ccy(to)
    if frm == to: return 1.0
    key = (frm, to);  now = time.time()
    if key in _RATE_CACHE and now - _RATE_CACHE[key][1] < 600:
        return _RATE_CACHE[key][0]
    for fn in (_rate_tcmb, _rate_yahoo):
        r = fn(frm, to)
        if r:
            _RATE_CACHE[key] = (r, now);  return r
    if _depth == 0 and frm != "USD" and to != "USD":
        a = get_rate(frm, "USD", 1); b = get_rate("USD", to, 1)
        v = a*b; _RATE_CACHE[key] = (v, now); return v
    raise RuntimeError(f"Kur alınamadı: {frm}->{to}")

# ================== Modeller ==================
class Transaction(BaseModel):
    date: str
    desc: str
    amount: Union[str, float, int]
    currency: Optional[str]
    type: Optional[str]

class CashflowRequest(BaseModel):
    report_currency: Optional[str]
    opening_cash: Optional[Union[str, float, int]]
    transactions: List[Transaction] = []

# ================== FastAPI ==================
app = FastAPI(title="Cashflow API", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"service": "cashflow-api", "version": APP_VERSION}
@app.get("/health")
def health(): return {"status": "ok"}

# ================== Çekirdek İşlem ==================
def _process(payload: CashflowRequest):
    now = datetime.now(IST).date()
    RPB = normalize_ccy(payload.report_currency or "TRY")
    RPB_DISP = "TL" if RPB == "TRY" else RPB

    rows = []
    if payload.opening_cash is not None:
        amt = round(parse_amount(payload.opening_cash))
        rows.append({
            "Tarih": now.strftime("%d.%m.%Y"),
            "Tarih2": now.strftime("%Y.%m.%d"),
            "Açıklama": "Başlangıç Nakit",
            "Orijinal Tutar": amt,
            "Orijinal Para Birimi": RPB_DISP,
            "Hareket": "Giriş",
            "_ccy": RPB, "_amt": float(amt),
        })

    for t in payload.transactions:
        tx_ccy = normalize_ccy(t.currency or "TRY")
        tx_date = parse_date_tr(t.date, now)
        amt = parse_amount(t.amount)
        hareket = detect_type(t.desc, t.type)
        rows.append({
            "Tarih": tx_date.strftime("%d.%m.%Y"),
            "Tarih2": tx_date.strftime("%Y.%m.%d"),
            "Açıklama": t.desc,
            "Orijinal Tutar": round(amt),
            "Orijinal Para Birimi": "TL" if tx_ccy=="TRY" else tx_ccy,
            "Hareket": hareket,
            "_ccy": tx_ccy, "_amt": float(amt),
        })

    if not rows:
        return RPB_DISP, pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Tarih2").reset_index(drop=True)
    def to_rpb(row):
        if row["_ccy"] == RPB: return row["_amt"]
        return row["_amt"] * get_rate(row["_ccy"], RPB)
    df["_RPB"] = df.apply(to_rpb, axis=1)
    df["_RPB_signed"] = np.where(df["Hareket"]=="Giriş", df["_RPB"], -df["_RPB"])
    df["Rapor Tutarı"] = df["_RPB_signed"].round().astype(int)
    df["Net Nakit"] = df["Rapor Tutarı"].cumsum()
    grp = df.groupby("Tarih", sort=False).agg(
        Giriş=("Rapor Tutarı", lambda s: int(s[s>0].sum())),
        Çıkış=("Rapor Tutarı", lambda s: int(-s[s<0].sum())),
    ).reset_index()
    grp["Tarih2"] = pd.to_datetime(grp["Tarih"], format="%d.%m.%Y").dt.strftime("%Y.%m.%d")
    grp = grp.sort_values("Tarih2").reset_index(drop=True)
    grp["Kümülatif"] = grp["Giriş"] - grp["Çıkış"]
    grp["Kümülatif"] = grp["Kümülatif"].cumsum()
    return RPB_DISP, df, grp

# ================== Çıktılar ==================
@app.post("/api/cashflow")
def cashflow(req: CashflowRequest):
    RPB_DISP, _, grp = _process(req)
    if grp.empty:
        return JSONResponse(content={
            "report_currency": RPB_DISP,
            "summary_table": [],
            "detail_csv_endpoint": "/api/cashflow/detail.csv",
            "chart_png_endpoint": "/api/cashflow/chart.png"
        })
    R_IN, R_OUT, R_NET = f"Giriş ({RPB_DISP})", f"Çıkış ({RPB_DISP})", f"Net Nakit ({RPB_DISP})"
    disp = grp.copy()
    disp.rename(columns={"Giriş": R_IN, "Çıkış": R_OUT, "Kümülatif": R_NET}, inplace=True)
    for c in [R_IN, R_OUT, R_NET]:
        disp[c] = disp[c].astype(int).apply(tr_int_format)
    summary = disp[["Tarih", R_IN, R_OUT, R_NET]].to_dict(orient="records")
    summary.append({
        "Tarih": "Toplam",
        R_IN: tr_int_format(int(grp["Giriş"].sum())),
        R_OUT: tr_int_format(int(grp["Çıkış"].sum())),
        R_NET: tr_int_format(int(grp["Kümülatif"].iloc[-1] if not grp.empty else 0)),
    })
    return JSONResponse(content={
        "report_currency": RPB_DISP,
        "summary_table": summary,
        "detail_csv_endpoint": "/api/cashflow/detail.csv",
        "chart_png_endpoint": "/api/cashflow/chart.png"
    })

@app.post("/api/cashflow/detail.csv")
def cashflow_detail_csv(req: CashflowRequest):
    _, df, _ = _process(req)
    cols = ["Tarih","Açıklama","Orijinal Tutar","Orijinal Para Birimi","Hareket","Rapor Tutarı","Net Nakit"]
    if df.empty:
        csv = ";".join(cols) + "\n"
        return StreamingResponse(io.BytesIO(csv.encode("utf-8-sig")), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=detay.csv"})
    out = df[cols].copy()
    out["Orijinal Tutar"] = out["Orijinal Tutar"].astype(int)
    buf = io.StringIO()
    out.to_csv(buf, sep=";", index=False)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode("utf-8-sig")), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=detay.csv"})

@app.post("/api/cashflow/chart.png")
def cashflow_chart_png(req: CashflowRequest):
    RPB_DISP, _, grp = _process(req)
    fig, ax = plt.subplots(figsize=(10,5))
    if grp.empty:
        ax.text(0.5, 0.5, "Veri yok", ha="center", va="center")
        ax.axis("off")
    else:
        dates = grp["Tarih"].tolist()
        giris = grp["Giriş"].tolist()
        cikis = [-v for v in grp["Çıkış"].tolist()]
        kum = grp["Kümülatif"].tolist()
        x = np.arange(len(dates))
        ax.bar(x-0.2, giris, width=0.4, label=f"Giriş ({RPB_DISP})", color="#2dba1e")
        ax.bar(x+0.2, cikis, width=0.4, label=f"Çıkış ({RPB_DISP})", color="#fe0101")
        ax.plot(x, kum, label=f"Net Nakit ({RPB_DISP})", color="#424dc6", linewidth=2.0, marker="o")
        ax.set_xticks(x); ax.set_xticklabels(dates)
        ax.axhline(0, color="#999", linestyle="--", linewidth=1)
        def fmt(y,_): return tr_int_format(int(round(y)))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.set_title(f"Nakit Akışı – Özet ({RPB_DISP})")
        ax.legend(); ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    p = io.BytesIO()
    plt.savefig(p, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    p.seek(0)
    return StreamingResponse(p, media_type="image/png",
                             headers={"Content-Disposition":"inline; filename=nakit_akisi.png"})

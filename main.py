from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime, date
from zoneinfo import ZoneInfo
import io, base64, re, time
import pandas as pd
import numpy as np

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import requests
import yfinance as yf

# ================== Sabitler ==================
IST = ZoneInfo("Europe/Istanbul")

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

# Kur cache (hafıza)
_RATE_CACHE: dict[tuple[str,str], tuple[float, float]] = {}  # (from,to) -> (rate, ts)

# ================== Yardımcılar ==================
def normalize_ccy(ccy: Optional[str]) -> str:
    if not ccy: return "TRY"
    c = ccy.strip().upper().replace("TL.", "TL")
    if c in ["TL","TRY"]: return "TRY"
    return c

def tr_int_format(n: int) -> str:
    s = f"{abs(int(n)):,}".replace(",", ".")
    return f"-{s}" if int(n) < 0 else s

def parse_amount(x: Union[str,int,float]) -> float:
    """Rakam + yazıyla rakam desteği. Ondalıklar yok sayılır; tam sayıya yakınsama rapor katmanında yapılacak."""
    if isinstance(x, (int,float)):
        return float(x)
    s = str(x).strip().lower()
    # semboller & kısaltmalar
    for token in ["₺"," tl"," tl."," try"," eur","€"," usd","$"," gbp","£"]:
        s = s.replace(token, "")
    s = re.sub(r"\s+", " ", s).strip()

    # içinde rakam varsa: 1.234.567,89 / 1.234.567 / 1234567
    if re.search(r"\d", s):
        s2 = s.replace(".", "").replace(",", ".")
        m = re.findall(r"[-+]?\d*\.?\d+", s2)
        if m:
            try:
                return float(m[0])
            except:
                pass

    # yazıyla rakam
    tokens = re.findall(r"[a-zçğıöşü]+", s, flags=re.UNICODE)
    if not tokens:
        raise ValueError(f"Tutar anlaşılamadı: {x}")
    total = 0
    current = 0
    for t in tokens:
        if t not in NUM_MAP:
            continue
        val = NUM_MAP[t]
        if val >= 100:
            if current == 0:
                current = 1
            current *= val
        else:
            current += val
        if val in (1000, 1_000_000, 1_000_000_000):
            total += current
            current = 0
    total += current
    return float(total)

def parse_date_tr(s: str, now: date) -> date:
    """gg.aa(.yyyy), '12 ekim (2025?)', ISO; geçmişse bir sonraki yılı al."""
    s = s.strip().lower()
    # dd.mm(.yyyy)?
    m = re.match(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{4}))?$", s)
    if m:
        d = int(m.group(1)); mth = int(m.group(2)); y = int(m.group(3)) if m.group(3) else now.year
        dt = date(y, mth, d)
        if dt < now:
            dt = date(y+1, mth, d)
        return dt
    # '12 ekim [2025]?'
    m2 = re.match(r"^(\d{1,2})\s+([a-zçğıöşü]+)(?:\s+(\d{4}))?$", s, flags=re.UNICODE)
    if m2:
        d = int(m2.group(1)); mon_word = m2.group(2)
        mth = TR_MONTHS.get(mon_word)
        if not mth:
            raise ValueError(f"Ay adı anlaşılamadı: {s}")
        y = int(m2.group(3)) if m2.group(3) else now.year
        dt = date(y, mth, d)
        if dt < now:
            dt = date(y+1, mth, d)
        return dt
    # ISO
    try:
        dt = datetime.fromisoformat(s).date()
        if dt < now:
            dt = date(dt.year+1, dt.month, dt.day)
        return dt
    except:
        pass
    raise ValueError(f"Tarih anlaşılamadı: {s}")

def detect_type(desc: str, given: Optional[str]) -> str:
    if given:
        g = given.strip().lower()
        if "giriş" in g or g.startswith("gir"):
            return "Giriş"
        return "Çıkış"
    d = desc.lower()
    if any(k in d for k in EXIT_KEYS): return "Çıkış"
    if any(k in d for k in ENTRY_KEYS): return "Giriş"
    # anahtar yoksa: gelir-kokluysa Giriş varsay
    return "Giriş"

def _http_get(url: str, timeout: float = 6.0) -> Optional[str]:
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except:
        return None
    return None

def _fetch_google_fx(frm: str, to: str) -> Optional[float]:
    # Google HTML; değişebilir — sadece ilk tercih
    html = _http_get(f"https://www.google.com/finance/quote/{frm}-{to}")
    if not html:
        return None
    m = re.search(r'data-last-price="([0-9\.,]+)"', html)
    if not m:
        m = re.search(r'class="YMlKec fxKbKc">([0-9\.,]+)<', html)
    if m:
        val = m.group(1).replace(".", "").replace(",", ".")
        try:
            return float(val)
        except:
            return None
    return None

def _fetch_yf_fx(frm: str, to: str) -> Optional[float]:
    try:
        tkr = f"{frm}{to}=X"
        info = yf.Ticker(tkr).fast_info
        px = info.get("last_price")
        if px and float(px) > 0:
            return float(px)
        hist = yf.Ticker(tkr).history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        # ters kotasyon dene
        inv = f"{to}{frm}=X"
        info2 = yf.Ticker(inv).fast_info
        px2 = info2.get("last_price")
        if px2 and float(px2) > 0:
            return 1.0/float(px2)
        hist2 = yf.Ticker(inv).history(period="1d")
        if not hist2.empty:
            v = float(hist2["Close"].iloc[-1])
            if v != 0:
                return 1.0/v
    except:
        return None
    return None

def get_rate(frm: str, to: str, _depth: int = 0) -> float:
    """En güncel kur: Google → Yahoo; olmazsa USD köprüsü (tek sefer). Cache 5dk."""
    frm = normalize_ccy(frm); to = normalize_ccy(to)
    if frm == to: return 1.0
    key = (frm, to)
    now_ts = time.time()
    # cache: 5 dk
    if key in _RATE_CACHE:
        val, ts = _RATE_CACHE[key]
        if now_ts - ts < 300:
            return val

    # 1) Google
    r = _fetch_google_fx(frm, to)
    if r:
        _RATE_CACHE[key] = (r, now_ts)
        return r
    # 2) Yahoo
    r2 = _fetch_yf_fx(frm, to)
    if r2:
        _RATE_CACHE[key] = (r2, now_ts)
        return r2
    # 3) USD köprüsü (tek adım)
    if _depth == 0 and frm != "USD" and to != "USD":
        a = get_rate(frm, "USD", _depth=1)
        b = get_rate("USD", to, _depth=1)
        r3 = a * b
        _RATE_CACHE[key] = (r3, now_ts)
        return r3

    raise RuntimeError(f"Kur alınamadı: {frm}->{to}")

# ================== Pydantic Modeller ==================
class Transaction(BaseModel):
    date: str = Field(..., description="Tarih; ör: '12 ekim', '15.10', '17.10.2025'")
    desc: str = Field(..., description="Açıklama")
    amount: Union[str, float, int] = Field(..., description="Tutar; '50.000', 50000, 'elli bin TL' vb.")
    currency: Optional[str] = Field(None, description="Para birimi; boşsa TL varsayılır (TRY)")
    type: Optional[str] = Field(None, description="'giriş' veya 'çıkış' (boşsa açıklamadan tespit edilir)")

class CashflowRequest(BaseModel):
    report_currency: Optional[str] = Field(None, description="Rapor Para Birimi; boşsa 'TRY' kullanılır")
    opening_cash: Optional[Union[str, float, int]] = Field(None, description="Başlangıç nakit; bugünün tarihiyle 'Giriş' kaydedilir")
    transactions: List[Transaction] = Field(default_factory=list)

# ================== FastAPI Uygulaması ==================
app = FastAPI(title="Cashflow API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================== Rapor Üretimi ==================
def build_reports(payload: CashflowRequest):
    now = datetime.now(IST).date()
    RPB = normalize_ccy(payload.report_currency or "TRY")
    RPB_DISPLAY = "TL" if RPB == "TRY" else RPB

    rows = []

    # 1) Başlangıç nakit → bugünün tarihi, Giriş, RPB cinsinden
    if payload.opening_cash is not None:
        amt = round(parse_amount(payload.opening_cash))
        rows.append({
            "Tarih": now.strftime("%d.%m.%Y"),
            "Tarih2": now.strftime("%Y.%m.%d"),
            "Açıklama": "Başlangıç Nakit",
            "Orijinal Tutar": amt,
            "Orijinal Para Birimi": RPB_DISPLAY,
            "Hareket": "Giriş",
            "_ccy": RPB,
            "_amt": float(amt),
        })

    # 2) İşlemler
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
            "Orijinal Para Birimi": "TL" if tx_ccy == "TRY" else tx_ccy,
            "Hareket": hareket,
            "_ccy": tx_ccy,
            "_amt": float(amt),
        })

    # Boşsa boş dön
    if not rows:
        return {
            "report_currency": RPB_DISPLAY,
            "detail_table": [], "summary_table": [],
            "chart_base64": None
        }

    df = pd.DataFrame(rows).sort_values("Tarih2", ascending=True).reset_index(drop=True)

    # 3) Kur dönüşümü (rapor anındaki güncel kur; oranlarda yuvarlama yok)
    def to_rpb(row):
        if row["_ccy"] == RPB:
            return row["_amt"]
        rate = get_rate(row["_ccy"], RPB)
        return row["_amt"] * rate

    df["_RPB"] = df.apply(to_rpb, axis=1)

    # 4) İşaretler ve rapor tam sayı yuvarlama
    df["_RPB_signed"] = np.where(df["Hareket"] == "Giriş", df["_RPB"], -df["_RPB"])
    df["Rapor Tutarı"] = df["_RPB_signed"].round().astype(int)  # sadece burada yuvarla
    df["Net Nakit"] = df["Rapor Tutarı"].cumsum()

    # 5) Ekran için format (binlik ayraç)
    df_display = df.copy()
    df_display["Orijinal Tutar"] = df_display["Orijinal Tutar"].astype(int).apply(tr_int_format)
    df_display["Rapor Tutarı"] = df_display["Rapor Tutarı"].astype(int).apply(tr_int_format)
    df_display["Net Nakit"] = df_display["Net Nakit"].astype(int).apply(tr_int_format)
    df_display.rename(columns={
        "Rapor Tutarı": f"Rapor Tutarı ({RPB_DISPLAY})",
        "Net Nakit": f"Net Nakit ({RPB_DISPLAY})"
    }, inplace=True)

    detail_cols = ["Tarih","Açıklama","Orijinal Tutar","Orijinal Para Birimi","Hareket",
                   f"Rapor Tutarı ({RPB_DISPLAY})", f"Net Nakit ({RPB_DISPLAY})"]
    detail_table = df_display[detail_cols].to_dict(orient="records")

    # 6) Özet (günlük)
    grp = df.groupby("Tarih", sort=False).agg(
        Giriş=("Rapor Tutarı", lambda s: int(s[s>0].sum())),
        Çıkış=("Rapor Tutarı", lambda s: int(-s[s<0].sum())),
    ).reset_index()
    grp["Tarih2"] = pd.to_datetime(grp["Tarih"], format="%d.%m.%Y").dt.strftime("%Y.%m.%d")
    grp = grp.sort_values("Tarih2").reset_index(drop=True)
    grp["Net Nakit"] = grp["Giriş"] - grp["Çıkış"]
    grp["Kümülatif"] = grp["Net Nakit"].cumsum()

    grp_display = grp.copy()
    grp_display.rename(columns={
        "Giriş": f"Giriş ({RPB_DISPLAY})",
        "Çıkış": f"Çıkış ({RPB_DISPLAY})",
        "Kümülatif": f"Net Nakit ({RPB_DISPLAY})"
    }, inplace=True)
    for col in [f"Giriş ({RPB_DISPLAY})", f"Çıkış ({RPB_DISPLAY})", f"Net Nakit ({RPB_DISPLAY})"]:
        grp_display[col] = grp_display[col].astype(int).apply(tr_int_format)

    summary_cols = ["Tarih", f"Giriş ({RPB_DISPLAY})", f"Çıkış ({RPB_DISPLAY})", f"Net Nakit ({RPB_DISPLAY})"]
    summary_table = grp_display[summary_cols].to_dict(orient="records")

    # Toplam satırı
    total_giris = int(grp["Giriş"].sum()) if not grp.empty else 0
    total_cikis = int(grp["Çıkış"].sum()) if not grp.empty else 0
    total_net = int(grp["Kümülatif"].iloc[-1]) if not grp.empty else 0
    summary_table.append({
        "Tarih": "Toplam",
        f"Giriş ({RPB_DISPLAY})": tr_int_format(total_giris),
        f"Çıkış ({RPB_DISPLAY})": tr_int_format(total_cikis),
        f"Net Nakit ({RPB_DISPLAY})": tr_int_format(total_net),
    })

    # 7) Grafik (özet günleri)
    dates = grp["Tarih"].tolist()
    giris_vals = grp["Giriş"].tolist()
    cikis_vals = [-v for v in grp["Çıkış"].tolist()]  # negatif göster
    kumulatif = grp["Kümülatif"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dates))
    ax.bar(x - 0.2, giris_vals, width=0.4, label=f"Giriş ({RPB_DISPLAY})", color="#2dba1e")
    ax.bar(x + 0.2, cikis_vals, width=0.4, label=f"Çıkış ({RPB_DISPLAY})", color="#fe0101")
    ax.plot(x, kumulatif, label=f"Net Nakit ({RPB_DISPLAY})", linewidth=2.0, color="#424dc6", marker="o")

    ax.set_title(f"Nakit Akışı – Özet ({RPB_DISPLAY})")
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    ax.axhline(0, linewidth=1, color="#999", linestyle="--")

    def tr_thousands(y, _pos):
        return tr_int_format(int(round(y)))
    ax.yaxis.set_major_formatter(FuncFormatter(tr_thousands))

    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    chart_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "report_currency": RPB_DISPLAY,
        "detail_table": detail_table,
        "summary_table": summary_table,
        "chart_base64": chart_b64
    }

# ================== Routes ==================
@app.get("/")
def root():
    return {"service": "cashflow-api", "version": app.version}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/cashflow")
def cashflow(req: CashflowRequest):
    """
    Input JSON:
    {
      "report_currency": "TRY",
      "opening_cash": "50.000 TL",
      "transactions": [
        {"date":"15.10","desc":"Maaş","amount":"50.000","currency":"TRY","type":"giriş"},
        {"date":"16.10","desc":"Kira ödemesi","amount":"20.000","currency":"TRY","type":"çıkış"},
        {"date":"17 ekim","desc":"Satış","amount":"1.000","currency":"EUR","type":"giriş"}
      ]
    }
    """
    return build_reports(req)

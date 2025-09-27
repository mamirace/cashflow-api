from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime, date
from zoneinfo import ZoneInfo
import io, base64, re, time, uuid, os
import pandas as pd
import numpy as np

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import requests
import xml.etree.ElementTree as ET
import yfinance as yf

# ================== Sabitler ==================
IST = ZoneInfo("Europe/Istanbul")
APP_VERSION = "1.0.0"

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
    """Rakam + yazıyla rakam desteği. Ondalık varsa okunur; nihai tam sayıya yuvarlama rapor katmanında."""
    if isinstance(x, (int,float)):
        return float(x)
    s = str(x).strip().lower()
    for token in ["₺"," tl"," tl."," try"," eur","€"," usd","$"," gbp","£"]:
        s = s.replace(token, "")
    s = re.sub(r"\s+", " ", s).strip()

    if re.search(r"\d", s):
        s2 = s.replace(".", "").replace(",", ".")
        m = re.findall(r"[-+]?\d*\.?\d+", s2)
        if m:
            try:
                return float(m[0])
            except:
                pass

    tokens = re.findall(r"[a-zçğıöşü]+", s, flags=re.UNICODE)
    if not tokens:
        raise ValueError(f"Tutar anlaşılamadı: {x}")
    total = 0
    current = 0
    for t in tokens:
        if t not in NUM_MAP:  # bilinmeyen kelimeyi yoksay
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
        if "giriş" in g or g.startswith("gir"): return "Giriş"
        return "Çıkış"
    d = desc.lower()
    if any(k in d for k in EXIT_KEYS): return "Çıkış"
    if any(k in d for k in ENTRY_KEYS): return "Giriş"
    return "Giriş"  # varsayılan

def _http_get(url: str, timeout: float = 6.0) -> Optional[str]:
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except:
        return None
    return None

# ---------- TCMB (Resmi) ----------
def _get_tcmb_today_xml() -> Optional[ET.Element]:
    try:
        url = "https://www.tcmb.gov.tr/kurlar/today.xml"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            resp.encoding = "utf-8"
            return ET.fromstring(resp.text)
    except:
        return None
    return None

def _tcmb_try_per(code: str, root: ET.Element) -> Optional[float]:
    """TCMB ForexSelling TRY fiyatı: 1 birim 'code' kaç TRY eder."""
    for currency in root.findall("Currency"):
        if currency.attrib.get("CurrencyCode") == code:
            txt = (currency.findtext("ForexSelling") or "").replace(",", ".")
            try:
                v = float(txt)
                return v if v > 0 else None
            except:
                return None
    return None

def get_tcmb_rate(frm: str, to: str) -> Optional[float]:
    frm = normalize_ccy(frm); to = normalize_ccy(to)
    if frm == to:
        return 1.0
    root = _get_tcmb_today_xml()
    if not root:
        return None

    def x_to_try(code: str) -> Optional[float]:
        if code == "TRY": return 1.0
        return _tcmb_try_per(code, root)

    if to == "TRY":
        return x_to_try(frm)
    if frm == "TRY":
        v = x_to_try(to)
        return (1.0 / v) if v and v != 0 else None

    a = x_to_try(frm)
    b = x_to_try(to)
    if a and b and b != 0:
        return a / b
    return None

# ---------- Yahoo & Google yedek ----------
def _fetch_yf_fx(frm: str, to: str) -> Optional[float]:
    try:
        tkr = f"{frm}{to}=X"
        hist = yf.Ticker(tkr).history(period="1d")
        if not hist.empty:
            v = float(hist["Close"].iloc[-1])
            if v > 0: return v
        info = yf.Ticker(tkr).fast_info
        px = info.get("last_price")
        if px and float(px) > 0:
            return float(px)
        # ters kotasyon
        inv = f"{to}{frm}=X"
        hist2 = yf.Ticker(inv).history(period="1d")
        if not hist2.empty:
            v2 = float(hist2["Close"].iloc[-1])
            return (1.0/v2) if v2 else None
        info2 = yf.Ticker(inv).fast_info
        px2 = info2.get("last_price")
        if px2 and float(px2) > 0:
            return 1.0/float(px2)
    except:
        return None
    return None

def _fetch_google_fx(frm: str, to: str) -> Optional[float]:
    html = _http_get(f"https://www.google.com/finance/quote/{frm}-{to}")
    if not html:
        return None
    m = re.search(r'data-last-price="([0-9\.,]+)"', html)
    if not m:
        m = re.search(r'class="YMlKec fxKbKc">([0-9\.,]+)<', html)
    if m:
        val = m.group(1).strip()
        # Google bazen 34,1234 gibi verir → virgül nokta dönüşümü yaparken 34.1234'e çevir
        val = val.replace(".", "").replace(",", ".")
        try:
            v = float(val)
            return v if v > 0 else None
        except:
            return None
    return None

def get_rate(frm: str, to: str, _depth: int = 0) -> float:
    """Önce TCMB → sonra Yahoo → sonra Google. Olmazsa USD köprüsü (tek adım). 10 dk cache."""
    frm = normalize_ccy(frm); to = normalize_ccy(to)
    if frm == to: return 1.0

    key = (frm, to)
    now_ts = time.time()
    if key in _RATE_CACHE:
        val, ts = _RATE_CACHE[key]
        if now_ts - ts < 600:  # 10 dk
            return val

    # 1) TCMB
    r = get_tcmb_rate(frm, to)
    if r:
        _RATE_CACHE[key] = (r, now_ts)
        return r

    # 2) Yahoo
    r = _fetch_yf_fx(frm, to)
    if r:
        _RATE_CACHE[key] = (r, now_ts)
        return r

    # 3) Google
    r = _fetch_google_fx(frm, to)
    if r:
        _RATE_CACHE[key] = (r, now_ts)
        return r

    # 4) USD köprüsü (sadece 1 adım)
    if _depth == 0 and frm != "USD" and to != "USD":
        a = get_rate(frm, "USD", _depth=1)
        b = get_rate("USD", to, _depth=1)
        v = a * b
        _RATE_CACHE[key] = (v, now_ts)
        return v

    raise RuntimeError(f"Kur alınamadı: {frm}->{to}")

# ================== Modeller ==================
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
app = FastAPI(title="Cashflow API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================== Çekirdek İşlem ==================
def _process(payload: CashflowRequest):
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

    if not rows:
        # Boş rapor
        empty_detail = []
        empty_summary = []
        return RPB, RPB_DISPLAY, empty_detail, empty_summary, None

    df = pd.DataFrame(rows).sort_values("Tarih2", ascending=True).reset_index(drop=True)

    # 3) Kur dönüşümü (rapor anındaki kur; oranlarda yuvarlama yok)
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

    # 5) Detay tablo (ekran için format)
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
    # CSV için ham veriyi ayrıca döndüreceğiz (df), ekranda df_display kullanılır.

    # 6) Özet (günlük)
    grp = df.groupby("Tarih", sort=False).agg(
        Giriş=("Rapor Tutarı", lambda s: int(s[s>0].sum())),
        Çıkış=("Rapor Tutarı", lambda s: int(-s[s<0].sum())),
    ).reset_index()
    grp["Tarih2"] = pd.to_datetime(grp["Tarih"], format="%d.%m.%Y").dt.strftime("%Y.%m.%d")
    grp = grp.sort_values("Tarih2").reset_index(drop=True)
    grp["Net Nakit"] = grp["Giriş"] - grp["Çıkış"]
    grp["Kümülatif"] = grp["Net Nakit"].cumsum()

    # Özet tablo (ekranda gösterim için binlik)
    grp_display = grp.copy()
    R_IN, R_OUT, R_NET = f"Giriş ({RPB_DISPLAY})", f"Çıkış ({RPB_DISPLAY})", f"Net Nakit ({RPB_DISPLAY})"
    grp_display.rename(columns={"Giriş": R_IN, "Çıkış": R_OUT, "Kümülatif": R_NET}, inplace=True)
    for col in [R_IN, R_OUT, R_NET]:
        grp_display[col] = grp_display[col].astype(int).apply(tr_int_format)
    summary_cols = ["Tarih", R_IN, R_OUT, R_NET]
    summary_table = grp_display[summary_cols].to_dict(orient="records")

    # Toplam satırı
    total_giris = int(grp["Giriş"].sum()) if not grp.empty else 0
    total_cikis = int(grp["Çıkış"].sum()) if not grp.empty else 0
    total_net = int(grp["Kümülatif"].iloc[-1]) if not grp.empty else 0
    summary_table.append({
        "Tarih": "Toplam",
        R_IN: tr_int_format(total_giris),
        R_OUT: tr_int_format(total_cikis),
        R_NET: tr_int_format(total_net),
    })

    return RPB, RPB_DISPLAY, df, summary_table, grp

def _save_tmp_file(binary: bytes, suffix: str) -> str:
    os.makedirs("/tmp", exist_ok=True)
    fid = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{fid}.{suffix}"
    with open(path, "wb") as f:
        f.write(binary)
    return path

# ================== Routes ==================
app = FastAPI(title="Cashflow API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"service": "cashflow-api", "version": APP_VERSION}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Detay CSV ----
@app.post("/api/cashflow/detail")
def cashflow_detail(req: CashflowRequest):
    RPB, RPB_DISPLAY, df, _, _ = _process(req)
    if isinstance(df, list):
        # boş rapor
        csv_bytes = "Tarih;Açıklama;Orijinal Tutar;Orijinal Para Birimi;Hareket;Rapor Tutarı;Net Nakit\n".encode("utf-8")
        path = _save_tmp_file(csv_bytes, "csv")
        return FileResponse(path, media_type="text/csv", filename="detay.csv")

    # CSV: binlik ayraçlı string yerine ham değerleri verelim mi? Talimatın ekran için binlik; CSV ham daha faydalı.
    df_csv = df[["Tarih","Açıklama","Orijinal Tutar","Orijinal Para Birimi","Hareket","Rapor Tutarı","Net Nakit"]].copy()
    df_csv["Orijinal Tutar"] = df_csv["Orijinal Tutar"].astype(int)
    df_csv.to_csv("/tmp/detay.csv", sep=";", index=False, encoding="utf-8-sig")
    return FileResponse("/tmp/detay.csv", media_type="text/csv", filename="detay.csv")

# ---- Özet JSON ----
@app.post("/api/cashflow/summary")
def cashflow_summary(req: CashflowRequest):
    RPB, RPB_DISPLAY, _, summary_table, _ = _process(req)
    return JSONResponse(content={
        "report_currency": RPB_DISPLAY,
        "summary_table": summary_table
    })

# ---- Grafik PNG ----
@app.post("/api/cashflow/chart")
def cashflow_chart(req: CashflowRequest):
    RPB, RPB_DISPLAY, _, _, grp = _process(req)
    # grp boşsa boş grafik döndürmeyelim
    if isinstance(grp, list) or grp.empty:
        # 1x1 boş PNG
        fig, ax = plt.subplots(figsize=(4,2))
        ax.text(0.5, 0.5, "Veri yok", ha="center", va="center")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        p = _save_tmp_file(buf.getvalue(), "png")
        plt.close(fig)
        return FileResponse(p, media_type="image/png", filename="nakit_akisi.png")

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
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    p = _save_tmp_file(buf.getvalue(), "png")
    plt.close(fig)
    return FileResponse(p, media_type="image/png", filename="nakit_akisi.png")

# ---- Full: Özet JSON + dosya linkleri ----
@app.post("/api/cashflow/full")
def cashflow_full(req: CashflowRequest, include_detail: bool = Query(True), include_chart: bool = Query(True)):
    RPB, RPB_DISPLAY, df, summary_table, grp = _process(req)

    detail_url = None
    if include_detail:
        # detay csv’i üret
        if isinstance(df, list):
            csv_bytes = "Tarih;Açıklama;Orijinal Tutar;Orijinal Para Birimi;Hareket;Rapor Tutarı;Net Nakit\n".encode("utf-8")
            detail_path = _save_tmp_file(csv_bytes, "csv")
        else:
            df_csv = df[["Tarih","Açıklama","Orijinal Tutar","Orijinal Para Birimi","Hareket","Rapor Tutarı","Net Nakit"]].copy()
            df_csv["Orijinal Tutar"] = df_csv["Orijinal Tutar"].astype(int)
            detail_path = f"/tmp/{uuid.uuid4().hex}.csv"
            df_csv.to_csv(detail_path, sep=";", index=False, encoding="utf-8-sig")
        # Render statik URL üretmez; FileResponse ile indirilir.
        # Burada sadece path döndürüyoruz; istemci bu path’i GET ile alamaz.
        # Bu yüzden full kullanırken ayrı çağrılar önerilir.
        detail_url = "/api/cashflow/detail"  # yol göstermek için

    chart_url = None
    if include_chart:
        chart_url = "/api/cashflow/chart"

    return JSONResponse(content={
        "report_currency": RPB_DISPLAY,
        "summary_table": summary_table,
        "detail_csv_endpoint": detail_url,
        "chart_png_endpoint": chart_url
    })

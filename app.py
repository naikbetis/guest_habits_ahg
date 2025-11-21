import os
import re
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import pdfplumber
import requests


# ---------- Helpers ----------

SEGMENT_RULES = {
    "CORPORATE": ["COR", "CORP", "COMPANY"],
    "SPECIAL_OCCASION": ["HONEYMOON", "ANNIVERSARY", "BIRTHDAY"],
    "LONG_STAY_MIN_NIGHTS": 7,
}


def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)


def load_pdf(file) -> pd.DataFrame:
    """Extract table-like data from an arrival-list style PDF.

    1. Coba beberapa strategi table extraction pdfplumber.
    2. Jika tetap gagal, fallback ke text parsing per baris
       dengan memisahkan kolom berdasarkan banyak spasi.
    """
    rows: List[List[str]] = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # 1) Coba default extract_tables
            tables = page.extract_tables()

            # 2) Jika kosong, coba dengan table_settings yang lebih ketat
            if not tables:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    }
                )

            for table in tables or []:
                for row in table:
                    if any(cell is not None for cell in row):
                        rows.append([
                            cell.strip() if isinstance(cell, str) else cell
                            for cell in row
                        ])

        # Fallback: tidak ada tabel sama sekali, coba parse text baris
        if not rows:
            text_rows: List[List[str]] = []
            header_candidates: List[str] = []

            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

                # Cari baris header yang berisi beberapa kata kunci umum
                for i, ln in enumerate(lines):
                    ln_u = ln.upper()
                    if "NAME" in ln_u and "ARRIVAL" in ln_u and "RATE" in ln_u:
                        header_candidates = re.split(r"\s{2,}", ln)
                        # Baris berikutnya sampai sebelum blank dianggap data
                        for data_ln in lines[i + 1 :]:
                            if not data_ln.strip():
                                break
                            cols = re.split(r"\s{2,}", data_ln.strip())
                            text_rows.append(cols)
                        break

            if header_candidates and text_rows:
                # Normalisasi panjang baris data ke jumlah kolom header
                max_cols = len(header_candidates)
                norm_rows: List[List[str]] = []
                for r in text_rows:
                    if len(r) < max_cols:
                        r = r + [""] * (max_cols - len(r))
                    elif len(r) > max_cols:
                        r = r[:max_cols]
                    norm_rows.append(r)

                header = header_candidates
                data_rows = norm_rows
                return pd.DataFrame(data_rows, columns=header)

    if not rows:
        # Masih gagal
        return pd.DataFrame()

    # Assume first non-empty row is header
    header = rows[0]
    data_rows = rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    return df


def detect_segments(row: pd.Series) -> List[str]:
    segments: List[str] = []

    # Mapping ke header aktual di report
    company = str(row.get("Customer", "") or "").upper()
    rate_code = str(row.get("Rate Cd", "") or "").upper()
    remark = str(row.get("Remark", "") or "").upper()
    nat = str(row.get("NAT", "") or "").upper()
    los = row.get("LOS", None)

    # Corporate
    if any(key in company for key in SEGMENT_RULES["CORPORATE"]) or "COR" in rate_code:
        segments.append("CORPORATE")

    # Special occasion
    if any(key in remark for key in SEGMENT_RULES["SPECIAL_OCCASION"]):
        segments.append("SPECIAL_OCCASION")

    # Long stay
    try:
        los_val = int(los) if pd.notnull(los) else 0
        if los_val >= SEGMENT_RULES["LONG_STAY_MIN_NIGHTS"]:
            segments.append("LONG_STAY")
    except Exception:
        pass

    # International
    if nat and nat not in ("ID", "IDN", "INDONESIA"):
        segments.append("INTERNATIONAL")

    return segments


PREFERENCE_KEYWORDS = {
    "ROOM_PREFERENCES": ["NON SMOKING", "NON-SMOKING", "HIGH FLOOR", "CITY VIEW", "TWIN", "KING"],
    "SPECIAL_OCCASION": ["HONEYMOON", "ANNIVERSARY", "BIRTHDAY", "BIRTHDAY DECORATION", "DECORATION"],
    "ACCESSIBILITY": ["SENIOR", "ELDERLY", "NOT FAR FROM LOBBY", "DISABILITY"],
    "TRANSPORT": ["AIRPORT PICKUP", "AIRPORT TRANSFER", "SHUTTLE"],
    "BUSINESS_NEEDS": ["MEETING ROOM", "MEETING", "CONFERENCE", "EARLY CHECK IN", "EARLY CHECK-IN", "LATE CHECK OUT", "LATE CHECK-OUT"],
}


def extract_preferences(text: str) -> Dict[str, List[str]]:
    text_u = (text or "").upper()
    prefs: Dict[str, List[str]] = {k: [] for k in PREFERENCE_KEYWORDS}
    for cat, kws in PREFERENCE_KEYWORDS.items():
        for kw in kws:
            if kw in text_u:
                prefs[cat].append(kw)
    # Remove empty categories
    return {k: v for k, v in prefs.items() if v}


def classify_rate_category(rate_code: str) -> str:
    code = (rate_code or "").upper()
    if code.startswith("COR") or "CORP" in code:
        return "CORPORATE_RATE"
    if code.startswith("BAR"):
        return "PUBLIC_RATE"
    if code.startswith("COMP") or "COMPL" in code:
        return "COMPLIMENTARY_RATE"
    if code.startswith("EMP") or "STAFF" in code:
        return "EMPLOYEE_RATE"
    if not code:
        return "UNKNOWN_RATE"
    return "OTHER_RATE"


def classify_booking_channel(customer: str) -> str:
    c = (customer or "").upper()
    if any(x in c for x in ["BOOKING.COM", "AGODA", "EXPEDIA", "TRIP.COM", "TRAVELOKA"]):
        return "OTA"
    if any(x in c for x in ["TRAVEL", "TOUR", "AGENT"]):
        return "TRAVEL_AGENT"
    if any(x in c for x in ["BANK", "CORP", "COMPANY", "INDONESIA", "SINERGIA", "ASIAN DEVELOPMENT BANK"]):
        return "CORPORATE"
    if not c:
        return "DIRECT_UNKNOWN"
    return "DIRECT_WALKIN"


def categorize_guest_profile(segments: List[str], booking_channel: str, rate_category: str, los: int) -> List[str]:
    profile: List[str] = []

    if "CORPORATE" in segments or rate_category == "CORPORATE_RATE" or booking_channel == "CORPORATE":
        profile.append("VIP_CORPORATE")

    if "SPECIAL_OCCASION" in segments:
        profile.append("SPECIAL_OCCASION")

    if "INTERNATIONAL" in segments:
        profile.append("INTERNATIONAL")

    if los >= SEGMENT_RULES["LONG_STAY_MIN_NIGHTS"]:
        profile.append("LONG_STAY")

    # Business solo vs leisure group (sederhana, bisa dikembangkan jika ada kolom Pax)
    if "CORPORATE" in segments or booking_channel in ("CORPORATE", "TRAVEL_AGENT"):
        profile.append("BUSINESS_SOLO")
    else:
        profile.append("LEISURE_GUEST")

    return sorted(list(set(profile)))


def build_guest_summary(row: pd.Series) -> Dict[str, Any]:
    # Gunakan nama header persis seperti di report, tetapi toleransi beberapa variasi
    remark = str(row.get("Remark", "") or "")

    # Beberapa kemungkinan nama kolom untuk instruksi pembayaran di Excel/PDF
    payment_keys = [
        "Payment instruction",
        "Payment instr",
        "Pay instr",
        "ment instruc",
    ]
    pay_instr_val = ""
    for key in payment_keys:
        if key in row.index:
            pay_instr_val = row.get(key, "") or ""
            break
    pay_instr = str(pay_instr_val)
    merged_text = " ".join([remark, pay_instr])

    company = str(row.get("Customer", "") or "").strip()
    rate_code = str(row.get("Rate Cd", "") or "").strip()
    los_val_str = str(row.get("LOS", "") or "").strip()
    try:
        los_int = int(float(los_val_str)) if los_val_str else 0
    except ValueError:
        los_int = 0

    rate_category = classify_rate_category(rate_code)
    booking_channel = classify_booking_channel(company)
    segments = detect_segments(row)
    guest_profile = categorize_guest_profile(segments, booking_channel, rate_category, los_int)

    return {
        "name": str(row.get("Name", "") or "").strip(),
        "company": company,
        "rate_code": rate_code,
        "room_type": str(row.get("Type", "") or "").strip(),
        "arrival": str(row.get("Arrival", "") or "").strip(),
        "departure": str(row.get("Departure", "") or "").strip(),
        "los": los_val_str,
        "nationality": str(row.get("NAT", "") or "").strip(),
        "remark": remark,
        "payment_instruction": pay_instr,
        "segments": segments,
        "guest_profile": guest_profile,
        "booking_channel": booking_channel,
        "rate_category": rate_category,
        "preferences": extract_preferences(merged_text),
    }


def get_openai_client(api_key: str | None):
    """Dummy helper for compatibility; returns True if API key is available.

    Kita tidak lagi memakai SDK OpenAI, tetapi HTTP langsung via requests.
    Fungsi ini hanya dipakai untuk cek apakah key tersedia.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return True


def generate_ai_recommendation(client, guest_summary: Dict[str, Any]) -> str:
    system_prompt = (
        "You are a guest experience specialist for a hotel. "
        "Given structured booking data, classify the guest type and generate concise, actionable "
        "operational and upsell suggestions. Respond in **bilingual format**: every line first in clear business English, "
        "then on the next line provide the Indonesian translation. "
        "Format output strictly as:\n\n"
        "GUEST: <Name / Company> (<Main Segment>)\n"
        "TAMU: <Nama / Perusahaan> (<Segmen Utama>)\n"
        "PRIORITY: <HIGH/MEDIUM/LOW> (<Reason in English>)\n"
        "PRIORITAS: <TINGGI/SEDANG/RENDAH> (<Alasan dalam Bahasa Indonesia>)\n"
        "SUGGESTIONS / SARAN:\n"
        "\u2713 <English point 1>\n   - <Terjemahan Indonesia poin 1>\n"
        "\u2713 <English point 2>\n   - <Terjemahan Indonesia poin 2>\n"
        "(3-7 points total)\n"
    )

    user_content = (
        "Data tamu:\n" +
        f"Name: {guest_summary['name']}\n" +
        f"Company/Group: {guest_summary['company']}\n" +
        f"Rate Code: {guest_summary['rate_code']}\n" +
        f"Rate Category: {guest_summary['rate_category']}\n" +
        f"Booking Channel: {guest_summary['booking_channel']}\n" +
        f"Room Type: {guest_summary['room_type']}\n" +
        f"Arrival: {guest_summary['arrival']}\n" +
        f"Departure: {guest_summary['departure']}\n" +
        f"Length of Stay: {guest_summary['los']} malam\n" +
        f"Nationality: {guest_summary['nationality']}\n" +
        f"Detected segments: {', '.join(guest_summary['segments'])}\n" +
        f"Guest profile categories: {', '.join(guest_summary['guest_profile'])}\n" +
        f"Preferences (from remarks/payment): {guest_summary['preferences']}\n" +
        f"Remark: {guest_summary['remark']}\n" +
        f"Payment Instruction: {guest_summary['payment_instruction']}\n" +
        "\nBuatkan kategori tamu utama, priority level, dan 3-7 saran tindakan.")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    url = base_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.4,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Guest Habits & Preparation AI", layout="wide")

# Header dengan logo + judul lebih kecil, responsif untuk mobile
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.image("https://ashleyhotelgroup.com/wp-content/themes/ashley/img/img-logo.png", width=120)
with header_col2:
    st.markdown(
        """
        <h2 style="margin-bottom: 0.2rem; font-weight: 600;">
            Guest Habits & Preparation AI
        </h2>
        <p style="margin-top: 0.2rem; font-size: 0.9rem; color: #666;">
            Analisis arrival list untuk rekomendasi treatment tamu secara otomatis.
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("Upload file **Expected Arrival** (PDF atau Excel)**, lalu sistem akan analisa pola tamu dan memberi rekomendasi treatment.")

# Kontrol utama di bagian atas (tanpa sidebar), lebih nyaman untuk mobile
top_col1, top_col2 = st.columns([1, 3])
with top_col1:
    max_rows = st.number_input(
        "Batas jumlah tamu yang dianalisa",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )
with top_col2:
    st.caption("Atur berapa banyak baris tamu yang akan dianalisa oleh sistem.")

uploaded_file = st.file_uploader("Upload PDF / Excel", type=["pdf", "xlsx", "xls"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    if file_name.endswith((".xlsx", ".xls")):
        df = load_excel(uploaded_file)
    else:
        df = load_pdf(uploaded_file)

    if df.empty:
        st.error("Gagal membaca tabel dari file. Mungkin struktur PDF perlu penyesuaian parser.")
    else:
        st.subheader("Preview Data Mentah")
        st.dataframe(df.head(20))

        # Debug: tampilkan nama-nama kolom yang terbaca
        st.caption("Debug: Kolom yang terbaca dari file")
        st.write(list(df.columns))

        # Normalisasi beberapa kolom nama umum kalau ada typo
        df_cols_upper = {c.upper(): c for c in df.columns}

        required_cols = ["NAME", "CUSTOMER", "RATE CD", "TYPE", "ARRIVAL", "DEPARTURE", "LOS", "NAT"]
        missing = [c for c in required_cols if c not in df_cols_upper]
        if missing:
            st.warning(f"Beberapa kolom tidak ditemukan dalam header: {missing}. Silakan cek format report.")

        client = get_openai_client(None)
        if not client:
            st.info("OPENAI_API_KEY tidak ditemukan di .Sistem hanya menjalankan rules-based segmentasi tanpa rekomendasi AI.")

        summaries: List[Dict[str, Any]] = []
        reco_texts: List[str] = []

        st.subheader("Analisis Per Tamu")
        limit = min(max_rows, len(df))

        for idx, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
            summary = build_guest_summary(row)
            summaries.append(summary)

            with st.expander(f"{idx}. {summary['name'] or summary['company'] or 'Guest'}"):
                st.write("**Segments (rules-based):**", ", ".join(summary["segments"]) or "-")
                st.write("**Preferences (keyword):**", summary["preferences"] or "-")

                # Debug: tampilkan row mentah untuk inspeksi jika parsing masih salah
                with st.expander("Lihat data mentah baris ini (debug)"):
                    st.write(row.to_dict())

                if client:
                    try:
                        reco = generate_ai_recommendation(client, summary)
                        reco_texts.append(reco)
                        st.text(reco)
                    except Exception as e:
                        st.error(f"Gagal memanggil OpenAI: {e}")
                else:
                    st.text("(AI recommendation disabled – set OPENAI_API_KEY  untuk mengaktifkan)")

else:
    st.info("Silakan upload file PDF/Excel terlebih dahulu.")

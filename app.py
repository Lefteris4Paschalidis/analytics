"""A small, explainable dashboard for uploaded sales datasets.

Run with: streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import unicodedata

import pandas as pd
import plotly.express as px
import streamlit as st


SAMPLE_PATH = Path("data/db.csv")
NO_COLUMN = "— Δεν υπάρχει —"

# The dashboard has internal names. Different CSV headers are mapped to them in the UI.
FIELD_LABELS = {
    "date": "Ημερομηνία παραγγελίας/πώλησης", "sales": "Αξία πωλήσεων / έσοδα",
    "profit": "Κέρδος", "cost": "Κόστος (αν δεν υπάρχει κέρδος)",
    "category": "Κατηγορία", "sub_category": "Υποκατηγορία", "region": "Περιοχή",
    "order_id": "ID παραγγελίας", "row_id": "ID γραμμής", "product_id": "ID προϊόντος",
    "product_name": "Όνομα προϊόντος", "quantity": "Ποσότητα", "discount": "Έκπτωση",
}
ALIASES = {
    "date": ("orderdate", "date", "invoicedate", "saledate", "transactiondate", "ημερομηνια"),
    "sales": ("sales", "revenue", "amount", "net sales", "net_sales", "turnover", "πωλησεις", "εσοδα"),
    "profit": ("profit", "netprofit", "net profit", "earnings", "κερδος"),
    "cost": ("cost", "cogs", "costofgoods", "cost of goods", "κοστος"),
    "category": ("category", "productcategory", "product category", "group", "κατηγορια"),
    "sub_category": ("subcategory", "sub category", "productgroup", "product group", "υποκατηγορια"),
    "region": ("region", "area", "territory", "περιοχη"),
    "order_id": ("orderid", "order id", "invoiceid", "invoice id", "transactionid", "παραγγελια"),
    "row_id": ("rowid", "row id", "lineid", "line id", "id"),
    "product_id": ("productid", "product id", "sku", "itemid", "item id"),
    "product_name": ("productname", "product name", "itemname", "item name", "description", "product"),
    "quantity": ("quantity", "qty", "units", "quantitysold", "ποσοτητα"),
    "discount": ("discount", "discountpercent", "discount percent", "εκπτωση"),
}


def normalized_name(value: object) -> str:
    """Compare headers safely despite spaces, punctuation, accents and case."""
    text = "".join(character for character in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(character))
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read common encodings and infer comma, semicolon or tab delimiters."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            frame = pd.read_csv(BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
            if frame.shape[1] < 2:
                raise ValueError("ανιχνεύθηκε μόνο μία στήλη")
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError("Δεν ήταν δυνατή η ανάγνωση του CSV. " + " | ".join(errors))


def suggested_column(columns: list[str], field: str) -> str:
    aliases = {normalized_name(value) for value in ALIASES[field]}
    for column in columns:
        if normalized_name(column) in aliases:
            return column
    return NO_COLUMN


def parse_number(values: pd.Series, decimal_comma: bool) -> pd.Series:
    """Convert currency-formatted values while leaving genuine numeric columns untouched."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    cleaned = values.astype("string").str.strip().str.replace(r"[€$£\s]", "", regex=True)
    if decimal_comma:
        cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_dataset(raw: pd.DataFrame, mapping: dict[str, str], decimal_comma: bool, day_first: bool) -> tuple[pd.DataFrame, list[str]]:
    """Create one validated internal table; raw uploaded data is never modified."""
    selected = {field: column for field, column in mapping.items() if column != NO_COLUMN}
    if "sales" not in selected:
        raise ValueError("Διάλεξε στήλη για «Αξία πωλήσεων / έσοδα».")
    if "profit" not in selected and "cost" not in selected:
        raise ValueError("Διάλεξε είτε «Κέρδος» είτε «Κόστος», ώστε να υπολογιστεί το κέρδος.")
    if len(set(selected.values())) != len(selected):
        raise ValueError("Μία στήλη δεν μπορεί να αντιστοιχεί σε δύο διαφορετικά πεδία.")

    data = pd.DataFrame(index=raw.index)
    data["Sales"] = parse_number(raw[selected["sales"]], decimal_comma)
    data["Profit"] = parse_number(raw[selected["profit"]], decimal_comma) if "profit" in selected else data["Sales"] - parse_number(raw[selected["cost"]], decimal_comma)
    warnings: list[str] = []

    if "date" in selected:
        data["Order Date"] = pd.to_datetime(raw[selected["date"]], format="mixed", dayfirst=day_first, errors="coerce")
    else:
        data["Order Date"] = pd.NaT
        warnings.append("Δεν αντιστοιχίστηκε ημερομηνία: η χρονική σειρά και το φίλτρο περιόδου δεν εμφανίζονται.")

    defaults = {"category": "Όλα τα προϊόντα", "sub_category": None, "region": "Όλες οι περιοχές", "order_id": None, "row_id": None, "product_id": "Άγνωστο προϊόν", "product_name": "Άγνωστο προϊόν", "quantity": None, "discount": None}
    for field, default in defaults.items():
        output_name = FIELD_LABELS[field]
        if field in selected:
            data[output_name] = raw[selected[field]]
        elif field == "sub_category":
            data[output_name] = data[FIELD_LABELS["category"]]
            warnings.append("Δεν αντιστοιχίστηκε υποκατηγορία: η ανάλυση γίνεται στο επίπεδο κατηγορίας.")
        elif field == "order_id":
            data[output_name] = "γραμμή-" + (raw.index + 1).astype(str)
            warnings.append("Δεν αντιστοιχίστηκε ID παραγγελίας: οι «μοναδικές παραγγελίες» ισούνται με τις γραμμές.")
        elif field == "row_id":
            data[output_name] = raw.index + 1
        elif field in ("quantity", "discount"):
            data[output_name] = pd.NA
        else:
            data[output_name] = default
    for field in ("quantity", "discount"):
        if field in selected:
            data[FIELD_LABELS[field]] = parse_number(raw[selected[field]], decimal_comma)

    before = len(data)
    data = data.dropna(subset=["Sales", "Profit"])
    if before - len(data):
        warnings.append(f"{before - len(data)} γραμμές με μη έγκυρες πωλήσεις ή κέρδος αποκλείστηκαν.")
    if data.empty:
        raise ValueError("Δεν έμεινε έγκυρη γραμμή μετά τον έλεγχο πωλήσεων και κέρδους.")
    if data["Order Date"].notna().any():
        invalid_dates = int(data["Order Date"].isna().sum())
        if invalid_dates:
            warnings.append(f"{invalid_dates} γραμμές έχουν μη έγκυρη ημερομηνία και δεν συμμετέχουν στη χρονική σειρά.")
    duplicate_ids = int(data[FIELD_LABELS["row_id"]].duplicated().sum())
    if duplicate_ids:
        warnings.append(f"Υπάρχουν {duplicate_ids} διπλότυπα ID γραμμής. Έλεγξε αν είναι αναμενόμενα.")
    if (data["Sales"] < 0).any() or (data["Ποσότητα"].dropna() < 0).any():
        warnings.append("Βρέθηκαν αρνητικές πωλήσεις ή ποσότητες· ίσως υπάρχουν επιστροφές ή διορθώσεις.")
    return data, warnings


def money(value: float) -> str:
    return f"${value:,.2f}"


def select_mapping(raw: pd.DataFrame) -> dict[str, str]:
    st.sidebar.subheader("Αντιστοίχιση στηλών")
    st.sidebar.caption("Οι προτάσεις είναι αυτόματες. Διόρθωσέ τες πριν την ανάλυση.")
    columns, options, mapping = list(raw.columns), None, {}
    options = [NO_COLUMN, *columns]
    for field, label in FIELD_LABELS.items():
        suggestion = suggested_column(columns, field)
        mapping[field] = st.sidebar.selectbox(label, options, index=options.index(suggestion), key=f"mapping_{field}")
    return mapping


def main() -> None:
    st.set_page_config(page_title="Ανάλυση πωλήσεων | Υποστήριξη Αποφάσεων", page_icon="📊", layout="wide")
    st.title("Ανάλυση Επιχειρηματικών Δεδομένων και Υποστήριξη Αποφάσεων")
    st.caption("Ανέβασε CSV πωλήσεων, αντιστοίχισε τις στήλες και διερεύνησε την κερδοφορία.")
    st.sidebar.header("Πηγή δεδομένων")
    source = st.sidebar.radio("Επίλεξε dataset", ["Ανέβασμα CSV", "Δείγμα Superstore του repository"])
    if source == "Ανέβασμα CSV":
        uploaded = st.sidebar.file_uploader("Επίλεξε αρχείο CSV", type=["csv", "CSV"])
        if uploaded is None:
            st.info("Ανέβασε ένα CSV από τη sidebar για να ξεκινήσει η ανάλυση.")
            st.stop()
        file_bytes, source_name = uploaded.getvalue(), uploaded.name
    else:
        if not SAMPLE_PATH.exists():
            st.error("Λείπει το προαιρετικό δείγμα `data/db.csv`. Ανέβασε δικό σου CSV.")
            st.stop()
        file_bytes, source_name = SAMPLE_PATH.read_bytes(), str(SAMPLE_PATH)
    try:
        raw = read_csv_bytes(file_bytes)
    except ValueError as error:
        st.error(f"Αδυναμία ανάγνωσης του «{source_name}»: {error}")
        st.info("Εξήγαγε το αρχείο ως CSV με επικεφαλίδες και δοκίμασε ξανά.")
        st.stop()

    st.sidebar.success(f"Φορτώθηκε: {source_name} ({len(raw):,} γραμμές, {len(raw.columns)} στήλες)")
    mapping = select_mapping(raw)
    decimal_comma = st.sidebar.toggle("Δεκαδικό κόμμα (π.χ. 1.234,56)", value=False)
    day_first = st.sidebar.toggle("Ημέρα πριν από μήνα στις ημερομηνίες", value=False)
    try:
        data, warnings = prepare_dataset(raw, mapping, decimal_comma, day_first)
    except ValueError as error:
        st.error(f"Το dataset δεν είναι ακόμη έτοιμο για αυτή την ανάλυση: {error}")
        st.info("Ελάχιστα χρειάζονται πωλήσεις και κέρδος (ή κόστος). Οι άλλες στήλες είναι προαιρετικές.")
        st.stop()
    for warning in warnings:
        st.warning(warning)

    st.sidebar.header("Φίλτρα")
    regions = sorted(data["Περιοχή"].dropna().astype(str).unique())
    categories = sorted(data["Κατηγορία"].dropna().astype(str).unique())
    selected_regions = st.sidebar.multiselect("Περιοχή", regions, default=regions)
    selected_categories = st.sidebar.multiselect("Κατηγορία", categories, default=categories)
    filtered = data.loc[data["Περιοχή"].astype(str).isin(selected_regions) & data["Κατηγορία"].astype(str).isin(selected_categories)].copy()
    has_dates = filtered["Order Date"].notna().any()
    if has_dates:
        dated = filtered.dropna(subset=["Order Date"])
        minimum_date, maximum_date = dated["Order Date"].min().date(), dated["Order Date"].max().date()
        selected_dates = st.sidebar.date_input("Περίοδος", (minimum_date, maximum_date), minimum_date, maximum_date)
        if len(selected_dates) != 2:
            st.info("Επίλεξε αρχική και τελική ημερομηνία.")
            st.stop()
        filtered = filtered.loc[filtered["Order Date"].between(pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1]))]
    if filtered.empty:
        st.warning("Δεν υπάρχουν γραμμές για αυτά τα φίλτρα.")
        st.stop()

    total_sales, total_profit = filtered["Sales"].sum(), filtered["Profit"].sum()
    margin = 100 * total_profit / total_sales if total_sales else 0
    metrics = st.columns(4)
    metrics[0].metric("Πωλήσεις", money(total_sales)); metrics[1].metric("Κέρδος", money(total_profit))
    metrics[2].metric("Περιθώριο κέρδους", f"{margin:.1f}%"); metrics[3].metric("Μοναδικές παραγγελίες", f"{filtered['ID παραγγελίας'].nunique():,}")
    st.subheader("Ερώτημα απόφασης: πού χρειάζεται έλεγχος κερδοφορίας;")
    st.write("Οι ομάδες με αρνητικό κέρδος είναι προτεραιότητες για έλεγχο τιμής, κόστους και εκπτώσεων. Η συσχέτιση δεν αποδεικνύει αιτιότητα.")
    by_group = filtered.groupby("Υποκατηγορία", dropna=False, as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Lines=("Sales", "size"))
    if filtered["Έκπτωση"].notna().any():
        by_group = by_group.merge(filtered.groupby("Υποκατηγορία", dropna=False)["Έκπτωση"].mean(), on="Υποκατηγορία")
    by_group["Profit Margin (%)"] = 100 * by_group["Profit"] / by_group["Sales"]
    by_group = by_group.sort_values("Profit")
    chart_col, table_col = st.columns((3, 2))
    with chart_col:
        chart = px.bar(by_group, x="Profit", y="Υποκατηγορία", orientation="h", color="Profit", color_continuous_scale=["#c0392b", "#f4d03f", "#1e8449"], title="Συνολικό κέρδος ανά ομάδα")
        chart.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(chart, use_container_width=True)
    with table_col:
        priorities = by_group.loc[by_group["Profit"] < 0].copy()
        if priorities.empty:
            st.success("Καμία ομάδα δεν έχει αρνητικό συνολικό κέρδος με τα τρέχοντα φίλτρα.")
        else:
            priorities["Sales"] = priorities["Sales"].map(money); priorities["Profit"] = priorities["Profit"].map(money)
            priorities["Profit Margin (%)"] = priorities["Profit Margin (%)"].map(lambda value: f"{value:.1f}%")
            if "Έκπτωση" in priorities: priorities["Έκπτωση"] = priorities["Έκπτωση"].map(lambda value: f"{value:.1%}")
            st.markdown("#### Προτεραιότητες διερεύνησης")
            st.dataframe(priorities.rename(columns={"Profit Margin (%)": "Περιθώριο"}), hide_index=True, use_container_width=True)
    if has_dates:
        st.subheader("Πορεία στον χρόνο")
        monthly = (filtered.dropna(subset=["Order Date"]).assign(Month=lambda frame: frame["Order Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")))
        timeline = px.line(monthly, x="Month", y=["Sales", "Profit"], markers=True, title="Μηνιαίες πωλήσεις και κέρδος")
        timeline.update_layout(yaxis_title="Ποσό", legend_title_text="Μετρική")
        st.plotly_chart(timeline, use_container_width=True)
    with st.expander("Προεπισκόπηση και έλεγχος δεδομένων"):
        st.write(f"{len(filtered):,} έγκυρες γραμμές μετά τα φίλτρα. Το αρχικό CSV δεν αλλάζει.")
        st.dataframe(filtered.sort_values("Order Date", ascending=False, na_position="last"), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

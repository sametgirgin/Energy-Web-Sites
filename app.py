from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple
from urllib.parse import urljoin

import feedparser
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


DEFAULT_WORKBOOK = Path(__file__).parent / "websites.xlsx"
DEFAULT_LIMIT = 8
REQUEST_HEADERS = {
    "User-Agent": "HeadlineFetcher/1.0 (+https://streamlit.io/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


st.set_page_config(page_title="Website Headline Fetcher", layout="wide")


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first column whose lowercase name contains any candidate token."""
    for col in df.columns:
        lower = str(col).lower()
        for token in candidates:
            if token in lower:
                return col
    return None


@st.cache_data
def load_sites_from_excel(source: str | Path | bytes) -> pd.DataFrame:
    """Normalize the Excel sheet into a dataframe with url and labels."""
    df = pd.read_excel(source)
    df.columns = [str(c).strip() for c in df.columns]

    url_col = None
    for col in reversed(df.columns):
        if df[col].astype(str).str.contains("http", na=False).any():
            url_col = col
            break
    if url_col is None:
        raise ValueError("Could not find a column with URLs in the uploaded sheet.")

    df = df.rename(columns={url_col: "url"})
    df = df[df["url"].notna()]
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.startswith("http")]

    name_col = _pick_column(df, ("source", "name", "title", "unnamed"))
    market_col = _pick_column(df, ("market", "industry"))
    region_col = _pick_column(df, ("region",))
    subregion_col = _pick_column(df, ("subregion",))

    df["label"] = df[name_col] if name_col else df["url"]
    df["market"] = df[market_col] if market_col else ""
    df["region"] = df[region_col] if region_col else ""
    df["subregion"] = df[subregion_col] if subregion_col else ""

    return df[["label", "url", "market", "region", "subregion"]].reset_index(drop=True)


def _from_feed(url: str, limit: int) -> List[Mapping[str, str]]:
    parsed = feedparser.parse(url)
    if not parsed.entries:
        return []

    headlines = []
    for entry in itertools.islice(parsed.entries, limit):
        title = entry.get("title") or ""
        link = entry.get("link") or url
        if title:
            headlines.append({"title": title.strip(), "link": link})
    return headlines


def _from_html(url: str, limit: int) -> List[Mapping[str, str]]:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
    except Exception:
        return []

    if not response.ok:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    seen = set()
    headlines: List[Mapping[str, str]] = []

    for tag in soup.select("h1, h2, h3"):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        text_key = text.lower()
        if text_key in seen:
            continue
        seen.add(text_key)

        link_tag = tag.find("a", href=True)
        link = urljoin(response.url, link_tag["href"]) if link_tag else response.url
        headlines.append({"title": text, "link": link})
        if len(headlines) >= limit:
            break

    if not headlines:
        page_title = soup.title.string if soup.title else None
        if page_title:
            headlines.append({"title": page_title.strip(), "link": response.url})

    return headlines


@st.cache_data(ttl=1800)
def fetch_headlines(url: str, limit: int = DEFAULT_LIMIT) -> Tuple[List[Mapping[str, str]], str]:
    """Try RSS/Atom first, then fall back to parsing top-level headings."""
    try:
        feed_items = _from_feed(url, limit)
        if feed_items:
            return feed_items, "rss/atom feed"

        html_items = _from_html(url, limit)
        if html_items:
            return html_items, "page headings"
    except Exception:
        return [], "error"

    return [], "unreachable"


def main() -> None:
    logo_path = Path(__file__).parent / "logo.png"
    header_left, header_right = st.columns([0.7, 0.3])
    with header_left:
        st.title("Energy Websites Headline Fetcher")

    with header_right:
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.caption("Place logo.png next to app.py to show a logo here.")

    overview_tab, headlines_tab = st.tabs(["Overview", "Headlines"])

    with overview_tab:
        st.subheader("Why this matters for the energy industry")
        st.markdown(
            "- Centralizes monitoring of gas, LNG, power, and transition-focused publishers.\n"
            "- Surfaces fresh headlines to support market intelligence, trading briefs, and policy tracking.\n"
            "- Uses RSS/Atom when available, and gracefully falls back to scanning site headings.\n"
            "- Filters by market/industry and region so analysts can focus on relevant geographies."
        )

    with headlines_tab:
        source_path = DEFAULT_WORKBOOK
        if not source_path.exists():
            st.error(
                "The bundled websites.xlsx file is missing. "
                "Place it next to app.py and restart the app."
            )
            return

        try:
            sites_df = load_sites_from_excel(source_path)
        except Exception as exc:
            st.error(f"Could not read the bundled Excel file: {exc}")
            return

        st.sidebar.header("Filters")
        markets = sorted(
            {str(m).strip() for m in sites_df["market"].dropna().unique() if str(m).strip()}
        )
        regions = sorted(
            {str(r).strip() for r in sites_df["region"].dropna().unique() if str(r).strip()}
        )
        subregions = sorted(
            {str(s).strip() for s in sites_df["subregion"].dropna().unique() if str(s).strip()}
        )

        market_filter = st.sidebar.multiselect("Market/Industry", markets)
        region_filter = st.sidebar.multiselect("Region", regions)
        subregion_filter = st.sidebar.multiselect("Subregion", subregions)
        item_limit = st.sidebar.slider("Headlines per site", 3, 15, DEFAULT_LIMIT)

        filtered_df = sites_df.copy()
        if market_filter:
            filtered_df = filtered_df[filtered_df["market"].isin(market_filter)]
        if region_filter:
            filtered_df = filtered_df[filtered_df["region"].isin(region_filter)]
        if subregion_filter:
            filtered_df = filtered_df[filtered_df["subregion"].isin(subregion_filter)]

        st.write(
            f"Loaded **{len(sites_df)}** sites from Excel "
            f"({len(filtered_df)} after filters)."
        )
        st.write(
            "Click **Fetch headlines** to pull updates. "
            "Network access is required when running this locally."
        )

        if st.button("Fetch headlines", type="primary"):
            for _, row in filtered_df.iterrows():
                label = row["label"]
                url = row["url"]
                with st.expander(f"{label} — {url}", expanded=False):
                    with st.spinner("Fetching..."):
                        headlines, source_type = fetch_headlines(url, limit=item_limit)

                    if not headlines:
                        st.info("No headlines found or site unreachable.")
                        continue

                    st.caption(f"Found via {source_type}")
                    for item in headlines:
                        title = item.get("title") or "Untitled"
                        link = item.get("link") or url
                        st.markdown(f"- [{title}]({link})")


if __name__ == "__main__":
    main()

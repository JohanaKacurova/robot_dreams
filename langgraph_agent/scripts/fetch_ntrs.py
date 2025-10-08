# scripts/fetch_ntrs.py
import os, time, json, pathlib, requests

API = "https://ntrs.nasa.gov/api"
OUT = os.getenv("NTRS_OUT", "data/ntrs")
QUERY = {
    # TUNE THIS to your interests (examples included):
    # "q": "guidance navigation control OR autonomy OR 'large language model' OR retrieval",
    "q": os.getenv("NTRS_QUERY", "autonomy OR guidance OR 'large language model'"),
    "disseminated": "DOCUMENT_AND_METADATA",  # ensure a public document exists
    "published.gte": os.getenv("NTRS_SINCE", "2018-01-01"),
    "page.size": os.getenv("NTRS_PAGE_SIZE", "30"),  # up to 100
}
os.makedirs(os.path.join(OUT, "pdfs"), exist_ok=True)
os.makedirs(os.path.join(OUT, "meta"), exist_ok=True)

def search(params):
    r = requests.get(f"{API}/citations/search", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])

def get_downloads(sti_id):
    r = requests.get(f"{API}/citations/{sti_id}/downloads", timeout=30)
    r.raise_for_status()
    return r.json().get("downloads", [])

def pick_pdf(downloads):
    for d in downloads:
        if d.get("mediaType") == "application/pdf":
            return d.get("url")
    for d in downloads:
        if str(d.get("url","")).lower().endswith(".pdf"):
            return d.get("url")
    return None

def main():
    results = search(QUERY)
    saved = 0
    for rec in results:
        sti_id = rec.get("id")
        if not sti_id:
            continue

        # Save metadata for later enrichment during ingest
        with open(os.path.join(OUT, "meta", f"{sti_id}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        # Pick a PDF to download
        try:
            url = pick_pdf(get_downloads(sti_id))
        except Exception:
            continue
        if not url:
            continue

        pdf_path = os.path.join(OUT, "pdfs", f"{sti_id}.pdf")
        if not pathlib.Path(pdf_path).exists():
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(pdf_path, "wb") as f:
                        for chunk in r.iter_content(1 << 14):
                            if chunk: f.write(chunk)
                saved += 1
                time.sleep(0.3)  # be gentle: API has rate limits
            except Exception:
                continue

    print(f"Saved PDFs: {saved} | Total records seen: {len(results)}")

if __name__ == "__main__":
    main()

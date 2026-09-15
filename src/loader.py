from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pypdf import PdfReader
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def load_html(filepath):            #Fn for SEC filing  (10K and 10Q)
    with open(filepath, "r", encoding="utf-8") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "lxml")

    for hidden in soup.find_all("ix:header"):
        hidden.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text


def load_earnings_transcript(filepath):      #Fn for Earning transcript
    with open(filepath, "r", encoding="utf-8") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "lxml")

    marker = soup.find("h2", id="full-conference-call-transcript")
    if marker is None:
        raise ValueError("Could not find transcript start marker — page structure may have changed")

    transcript_parts = []
    for sibling in marker.find_next_siblings():
        transcript_parts.append(sibling.get_text(separator=" ", strip=True))

    text = " ".join(transcript_parts)
    return text

def load_pdf(filepath):
    reader = PdfReader(filepath)

    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())

    text = " ".join(text_parts)
    return text

def load_document(filepath, doc_type):
    """
    doc_type must be one of: "sec_filing", "earnings_transcript", "pdf"
    """
    if doc_type == "sec_filing":
        return load_html(filepath)
    elif doc_type == "earnings_transcript":
        return load_earnings_transcript(filepath)
    elif doc_type == "pdf":
        return load_pdf(filepath)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")


if __name__ == "__main__":
    documents = [
        {"filepath": "data/raw/jpmc/jpmc_10k_2025.htm", "doc_type": "sec_filing"},
        {"filepath": "data/raw/jpmc/jpmc_10q_q2_2026.htm", "doc_type": "sec_filing"},
        {"filepath": "data/raw/jpmc/jpmc_earnings_call_q2_2026.htm", "doc_type": "earnings_transcript"},
    ]

    for doc in documents:
        text = load_document(doc["filepath"], doc["doc_type"])
        print(f"\n=== {doc['filepath']} ({doc['doc_type']}) ===")
        print(f"Total characters extracted: {len(text)}")
        print(text[:300])
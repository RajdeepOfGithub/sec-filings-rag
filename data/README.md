
# Corpus

Not committed. Three JPMorgan Chase documents, fetched to `data/raw/jpmc/`.

SEC requires a User-Agent header identifying the requester, or it rejects the request.

```
curl.exe -A "your.email@example.com" -o data/raw/jpmc/jpmc_10k_2025.htm "https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm"

curl.exe -A "your.email@example.com" -o data/raw/jpmc/jpmc_10q_q2_2026.htm "https://www.sec.gov/Archives/edgar/data/19617/000162828026054343/jpm-20260630.htm"

curl.exe -A "your.email@example.com" -o data/raw/jpmc/jpmc_earnings_call_q2_2026.htm "https://www.fool.com/earnings/call-transcripts/2026/07/22/jpmorgan-chase-jpm-q2-2026-earnings-call-transcript/"
```

Browser print-to-PDF silently truncates these filings. Use curl.

Expected sizes: 10-K ~12.9MB, 10-Q ~11.5MB, transcript ~580KB.

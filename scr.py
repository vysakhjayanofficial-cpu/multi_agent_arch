from weasyprint import HTML

HTML("https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm").write_pdf("webpage.pdf")
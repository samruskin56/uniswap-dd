# CLAUDE.md Template — Crypto VC

Copy this into your project folder and fill in what you can. Leave sections blank if you're unsure — Claude Code can help figure out the rest.

---

# Project Name

What are you trying to build? Describe it in plain English.

Example: "A dashboard that tracks our portfolio performance across all token positions"

## What I Want This To Do

List the main things you want this project to accomplish:

- 
- 
- 

## My Experience Level

- [ ] Brand new to coding
- [ ] Some experience (took a class, did some tutorials)
- [ ] Comfortable with basics but not advanced stuff

## My Preferences

- [ ] Keep it simple, even if it's not the "best" way
- [ ] Explain what the code does as you write it
- [ ] Avoid complex setups — I want something I can run easily
- [ ] I'm on Mac
- [ ] I'm on Windows (powershell)

## What Type of Project Is This?

- [ ] Portfolio monitoring / tracking
- [ ] Project research / due diligence
- [ ] Investment memo or report
- [ ] Liquid-token valuation + one-pager (use `liquid-valuation-template.ipynb` with `src/valuation` + `src/onepager`)
- [ ] Data analysis
- [ ] Other: _______________

## Data Sources I'll Use

- [ ] CSV or Excel files I have
- [ ] CoinGecko / CoinMarketCap (token prices)
- [ ] DefiLlama (TVL and protocol data)
- [ ] Dune Analytics (on-chain metrics)
- [ ] GitHub (developer activity)
- [ ] Google Sheets
- [ ] Messari (token prices and market data)
- [ ] Other: _______________

## Output I Need

- [ ] Charts and visualizations
- [ ] Word document (investment memo, report)
- [ ] Excel/CSV for further analysis
- [ ] Simple dashboard
- [ ] PDF
- [ ] Other: _______________

## Stuff I've Already Tried

If you've attempted this before or have existing files, mention them here:

## Questions I Have

List anything you're confused about or want Claude Code to explain:

- 
- 

---

## Context for Claude Code

I work at a crypto venture capital firm. My common workflows are:

**Portfolio Monitoring**
- Track token prices across our investments
- Calculate returns (how much we're up/down)
- Pull data from different sources into one view
- Create charts showing performance over time

**Project Research / Due Diligence**
- Analyze a project's GitHub activity (are developers active?)
- Look at on-chain data (users, transactions, TVL)
- Research team backgrounds and past investors
- Compare metrics across similar projects

**Investment Memos**
- Summarize research into a structured document
- Include relevant charts and data
- Standard sections: thesis, market, team, risks, terms
- Export to Word or PDF for sharing with partners

---

## Instructions for Claude Code

I'm new to coding. When working on this project:

1. **Prioritize simplicity** — Use straightforward approaches even if more elegant solutions exist. Avoid unnecessary abstractions.

2. **Explain as you go** — Add comments explaining what each section of code does and why.

3. **Minimize setup** — Avoid complex tooling, virtual environments, or configuration unless truly necessary. If setup is needed, provide step-by-step instructions.

4. **Use common libraries**:
   - `pandas` for any data work
   - `requests` for fetching data from APIs
   - `plotly` for interactive charts, `matplotlib` for simple static ones
   - `pycoingecko` for crypto price data (free, no API key needed)

5. **Make it runnable** — I should be able to run the code without extensive configuration. Prefer single-file scripts when possible.

6. **Handle errors gracefully** — Include basic error handling with clear messages that help me understand what went wrong.

7. **Add print statements** — Tell me what the code is doing as it runs so I can follow along and debug.

8. **Save intermediate data** — Export to CSV files so I can inspect the data and don't have to re-fetch everything if something breaks.

9. **Structure outputs for my work** — When analyzing projects, organize findings to map to our investment memo format (thesis, market size, competitive landscape, team, risks, terms).

10. **Suggest next steps** — After completing a task, briefly explain what I could do next to extend or improve the project.

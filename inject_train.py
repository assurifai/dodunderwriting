import streamlit as st


def inject_context(year):
    # inject context for fine-tuning
    ss = st.session_state
    ref_df = compute_financial_metrics(ss.ticker)

    if ref_df is not None:
        return f"Context: through an API, we have found these values for the given year: {ref_df[ref_df.index == year].iloc[0]}. If the figure you found was significantly different, return the value from the API."

    return ""


import requests
import pandas as pd
import streamlit as st
from apikeys import get_secret, ConfigKey

FMP_API_KEY = get_secret(ConfigKey.FMP)

BASE_URL = "https://financialmodelingprep.com/api/v3"

@st.cache_data
def fetch_fmp_financials(ticker):
    """Fetch financial statements from FMP API for a given ticker."""
    endpoints = {
        "income": f"{BASE_URL}/income-statement/{ticker}?apikey={FMP_API_KEY}&limit=6",
        "balance": f"{BASE_URL}/balance-sheet-statement/{ticker}?apikey={FMP_API_KEY}&limit=6",
        "cashflow": f"{BASE_URL}/cash-flow-statement/{ticker}?apikey={FMP_API_KEY}&limit=6"
    }
    
    financials = {}
    for key, url in endpoints.items():
        response = requests.get(url)
        if response.status_code == 200:
            financials[key] = response.json()
        else:
            st.error(f"Failed to fetch {key} data for {ticker}")
            return None
    
    return financials


@st.cache_data
def compute_financial_metrics(ticker):
    """Compute financial metrics and return a DataFrame."""
    data = fetch_fmp_financials(ticker)
    if not data:
        return None

    income_stmt = {x["date"][:4]: x for x in data["income"]}
    balance_sheet = {x["date"][:4]: x for x in data["balance"]}
    cashflow_stmt = {x["date"][:4]: x for x in data["cashflow"]}

    years = sorted(set(income_stmt.keys()) & set(balance_sheet.keys()) & set(cashflow_stmt.keys()), reverse=True)
    
    metrics = {
        "EBITDA": [],
        "EBITDA %": [],
        "Gross Margin %": [],
        "Net Income %": [],
        "Distributions": [],
        "Current Ratio": [],
        "Senior Funded Debt / EBITDA": [],
        "Total Debt / EBITDA": [],
        "Tangible Net Worth (TNW)": [],
        "B/S Leverage": [],
        "Debt Service Ratio": [],
        "Fixed Charge Ratio": [],
        "Capex Spend": [],
        "Asset Disposition": [],
        "Change in Debt": [],
        "Receivables Days (DSO)": [],
        "Inventory Days": [],
        "Payable Days (DPO)": [],
        "Cash Conversion Cycle": [],
        "Cash Burn Rate": [],
        "Days Cash On Hand": [],
    }

    for year in years:
        inc = income_stmt[year]
        bal = balance_sheet[year]
        cash = cashflow_stmt[year]

        # Fetch necessary values
        revenue = inc.get("revenue", 0)
        gross_profit = inc.get("grossProfit", 0)
        ebitda = inc.get("ebitda", 0)
        net_income = inc.get("netIncome", 0)
        total_assets = bal.get("totalAssets", 0)
        total_liabilities = bal.get("totalLiabilities", 0)
        total_equity = bal.get("totalStockholdersEquity", 0)
        total_debt = bal.get("shortTermDebt", 0) + bal.get("longTermDebt", 0)
        current_assets = bal.get("totalCurrentAssets", 0)
        current_liabilities = bal.get("totalCurrentLiabilities", 0)
        interest_expense = inc.get("interestExpense", 0)
        capex = cash.get("capitalExpenditure", 0)
        cash_flow_operations = cash.get("operatingCashFlow", 0)
        cash_on_hand = bal.get("cashAndCashEquivalents", 0)

        # Calculate financial ratios
        metrics["EBITDA"].append(ebitda)
        metrics["EBITDA %"].append((ebitda / revenue * 100) if revenue else None)
        metrics["Gross Margin %"].append((gross_profit / revenue * 100) if revenue else None)
        metrics["Net Income %"].append((net_income / revenue * 100) if revenue else None)
        metrics["Distributions"].append(cash.get("dividendsPaid", 0) + cash.get("commonStockRepurchased", 0))
        metrics["Current Ratio"].append(current_assets / current_liabilities if current_liabilities else None)
        metrics["Senior Funded Debt / EBITDA"].append(bal.get("longTermDebt", 0) / ebitda if ebitda else None)
        metrics["Total Debt / EBITDA"].append(total_debt / ebitda if ebitda else None)
        metrics["Tangible Net Worth (TNW)"].append(total_equity - bal.get("goodwill", 0) - bal.get("intangibleAssets", 0) - total_liabilities)
        metrics["B/S Leverage"].append(total_debt / total_equity if total_equity else None)
        metrics["Debt Service Ratio"].append(ebitda / (interest_expense + bal.get("currentPortionOfLongTermDebt", 0)) if interest_expense else None)
        metrics["Fixed Charge Ratio"].append(ebitda / (interest_expense + capex) if interest_expense else None)
        metrics["Capex Spend"].append(capex)
        metrics["Asset Disposition"].append(cash.get("saleOfPropertyPlantAndEquipment", 0))
        metrics["Change in Debt"].append(total_debt - (bal.get("shortTermDebtPrevious", 0) + bal.get("longTermDebtPrevious", 0)))
        
        # Working Capital Cycle
        receivables = bal.get("netReceivables", 0)
        inventory = bal.get("inventory", 0)
        payables = bal.get("accountPayables", 0)
        cogs = inc.get("costOfRevenue", 0)

        metrics["Receivables Days (DSO)"].append((receivables / revenue) * 365 if revenue else None)
        metrics["Inventory Days"].append((inventory / cogs) * 365 if cogs else None)
        metrics["Payable Days (DPO)"].append((payables / cogs) * 365 if cogs else None)
        metrics["Cash Conversion Cycle"].append(metrics["Receivables Days (DSO)"][-1] + metrics["Inventory Days"][-1] - metrics["Payable Days (DPO)"][-1])
        metrics["Cash Burn Rate"].append(-cash_flow_operations if cash_flow_operations < 0 else 0)
        metrics["Days Cash On Hand"].append((cash_on_hand / (cogs / 365)) if cogs else None)

    # Convert to DataFrame
    df = pd.DataFrame(metrics, index=years)
    df = df.sort_index(ascending=False)  # Ensure recent years are on top

    return df






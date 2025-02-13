from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .models import KeywordAnalysisList, RiskAnalysisList, ConfidencePair

financial_keywords = ["financial", "capital", "income", "billion", "million", "cash"]
value_parser = PydanticOutputParser(pydantic_object=ConfidencePair)
value_format_instructions = value_parser.get_format_instructions()

value_queries = [
    # Core Financials
    ("ebitda", "Income Statement, Notes", "Extract the operating income (EBITDA)."),
    (
        "revenue",
        "Income Statement, Notes",
        "Extract the net revenue or net sales globally.",
    ),
    (
        "cogs",
        "Income Statement, Notes",
        "Extract the total cost of sales or cost of goods sold.",
    ),
    (
        "net_income",
        "Income Statement, Notes",
        "Extract the net income or net earnings.",
    ),
    # Profitability Ratios
    (
        "ebitda_margin",
        "Income Statement, Notes",
        "Extract EBITDA percentage as a fraction of total revenue. Pull only from context.",
    ),
    (
        "gross_margin",
        "Income Statement, Notes",
        "Extract the gross margin percentage, which is (Gross Profit / Revenue) * 100.",
    ),
    (
        "net_income_margin",
        "Income Statement, Notes",
        "Extract net income percentage, which is (Net Income / Revenue) * 100.",
    ),
    # Capital Allocation
    (
        "distributions",
        "Cash Flow Statement (Financing)",
        "Extract the net distributions the company has done, including stock buybacks and dividends. If neither is mentioned, return 0.",
    ),
    # Liquidity Ratios
    (
        "current_ratio",
        "Balance Sheet",
        "Extract the current ratio, which is (Total Current Assets / Total Current Liabilities).",
    ),
    # Debt Ratios
    (
        "senior_funded_debt_to_ebitda",
        "Balance Sheet, Income Statement",
        "Extract the Senior Funded Debt to EBITDA ratio, calculated as (Long-Term Debt / EBITDA).",
    ),
    (
        "total_debt_to_ebitda",
        "Balance Sheet, Income Statement",
        "Extract the Total Debt to EBITDA ratio, calculated as (Total Debt / EBITDA).",
    ),
    (
        "tangible_net_worth",
        "Balance Sheet",
        "Extract Tangible Net Worth (TNW), calculated as (Total Stockholders' Equity - Intangible Assets - Goodwill).",
    ),
    (
        "balance_sheet_leverage",
        "Balance Sheet",
        "Extract the Balance Sheet Leverage ratio, calculated as (Total Assets / Total Equity).",
    ),
    # Coverage Ratios
    (
        "debt_service_ratio",
        "Income Statement, Balance Sheet",
        "Extract the Debt Service Ratio, calculated as (EBITDA / (Interest Expense + Current Portion of Long-Term Debt)).",
    ),
    (
        "fixed_charge_ratio",
        "Income Statement, Balance Sheet",
        "Extract the Fixed Charge Coverage Ratio, calculated as (EBITDA / (Interest Expense + Capital Expenditures)).",
    ),
    # Capital Expenditures & Debt Movement
    (
        "capex_spend",
        "Cash Flow Statement (Investing)",
        "Extract the capital expenditures (CapEx) spend.",
    ),
    (
        "asset_disposition",
        "Cash Flow Statement (Investing)",
        "Extract the amount received from the sale of assets (property, plant, equipment).",
    ),
    (
        "change_in_debt",
        "Balance Sheet",
        "Extract the change in total debt compared to the previous reporting period.",
    ),
    # Working Capital Cycle
    (
        "receivables_days",
        "Balance Sheet, Income Statement",
        "Extract Days Sales Outstanding (DSO), calculated as (Accounts Receivable / Revenue) * 365.",
    ),
    (
        "inventory_days",
        "Balance Sheet, Income Statement",
        "Extract Inventory Days, calculated as (Inventory / Cost of Goods Sold) * 365.",
    ),
    (
        "payable_days",
        "Balance Sheet, Income Statement",
        "Extract Days Payables Outstanding (DPO), calculated as (Accounts Payable / Cost of Goods Sold) * 365.",
    ),
    (
        "cash_conversion_cycle",
        "Balance Sheet, Income Statement",
        "Extract the Cash Conversion Cycle, calculated as (DSO + Inventory Days - DPO).",
    ),
    # Cash Flow & Liquidity
    (
        "cash_burn_rate",
        "Cash Flow Statement",
        "Extract the Cash Burn Rate, which is the negative operating cash flow (if cash flow from operations is negative).",
    ),
    (
        "days_cash_on_hand",
        "Balance Sheet, Income Statement",
        "Extract the Days Cash on Hand, calculated as (Cash and Cash Equivalents / (Cost of Goods Sold / 365)).",
    ),
]

value_template = PromptTemplate(
    template="Analyze the following financials-related excerpts from a 10-K annual report for a company for the year {year}. Make sure you only extract figures for the year {year}, not past years. Pay attention specifically to the following sections: {section}. Make sure you return dollar amounts in terms of millions of USD. {instruction}\n context: {context} \n Provide a structured response.\n{format_instructions}",
    input_variables=["year", "section", "instruction", "context"],
    partial_variables={"format_instructions": value_format_instructions},
)

get_value_queries = lambda year: {
    key: (
        ["financial_keywords"],
        value_template.format(
            year=year,
            section=section,
            instruction=instruction,
            context=reformat_context_prompt_for_train(year),
        ),
    )
    for [key, section, instruction] in value_queries
}


risk_parser = PydanticOutputParser(pydantic_object=RiskAnalysisList)
risk_format_instructions = risk_parser.get_format_instructions()

# Define query with explicit structure request
keyword_parser = PydanticOutputParser(pydantic_object=KeywordAnalysisList)
keyword_format_instructions = keyword_parser.get_format_instructions()

keywords = [
    "reduction",
    "investigation",
    "failure",
    "unfortunately",
    "going concern",
    "judgement",
    "litigation",
    "loss",
    "decline",
]

risk_queries = {
    "risk": (
        ["risk"],
        PromptTemplate(
            template="Analyze the following risk-related sections from the past 10-K reports of the company. Identify any key changes, trends, or new risks that emerged over time. Provide a summary of how the company's risk profile has evolved, citing specific years and paying attention to sections on Risk Factors (1A) and Market Risk (7A) if they are present. Provide a structured response.\n{format_instructions}",
            input_variables=[],
            partial_variables={"format_instructions": risk_format_instructions},
        ).format(),
    ),
    **{
        keyword: (
            [keyword],
            PromptTemplate(
                template=f"Analyze the following excerpts of a company's 10-K form containing the risk-related term {keyword}. Your task is to assess each excerpt and determine if it contains significant risks, negative sentiments, or noteworthy concerns. If it does, provide a brief analysis and summary, explaining why it is a concern for investors. If a trend or worsening risk is observed over multiple years, highlight it. Provide a structured response.\n{{format_instructions}}",
                input_variables=[],
                partial_variables={"format_instructions": keyword_format_instructions},
            ).format(),
        )
        for keyword in keywords
    },
}


def reformat_context_prompt_for_train(year):
    from inject_train import inject_context

    return inject_context(year)

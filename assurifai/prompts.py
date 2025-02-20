from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .models import KeywordAnalysisList, RiskAnalysisList, ConfidencePair

financial_keywords = ["financial", "capital", "income", "billion", "million", "cash"]
value_parser = PydanticOutputParser(pydantic_object=ConfidencePair)
value_format_instructions = value_parser.get_format_instructions()

value_queries = [
    # Core Financials
    ("ebitda", "Income Statement, Notes", "Extract the operating income (EBITDA)."),
    # Profitability Ratios
    (
        "ebitda_margin",
        "Income Statement, Notes",
        "Extract EBITDA percentage (EBITDA %).",
    ),
    (
        "gross_margin",
        "Income Statement, Notes",
        "Extract the gross margin percentage (Gross Margin %)."
    ),
    (
        "net_income_margin",
        "Income Statement, Notes",
        "Extract net income percentage (Net Income %).",
    ),
    # Capital Allocation
    (
        "distributions",
        "Cash Flow Statement (Financing)",
        "Extract the net distributions the company has done, including stock buybacks and dividends (Distributions).",
    ),
    # Liquidity Ratios
    (
        "current_ratio",
        "Balance Sheet",
        "Extract the current ratio: (Current Ratio).",
    ),
    # Debt Ratios
    (
        "senior_funded_debt_to_ebitda",
        "Balance Sheet, Income Statement",
        "Extract the Senior Funded Debt / EBITDA ratio.",
    ),
    (
        "total_debt_to_ebitda",
        "Balance Sheet, Income Statement",
        "Extract the Total Debt / EBITDA ratio.",
    ),
    (
        "tangible_net_worth",
        "Balance Sheet",
        "Extract Tangible Net Worth (TNW).",
    ),
    (
        "balance_sheet_leverage",
        "Balance Sheet",
        "Extract the Balance Sheet Leverage (B/S Leverage) ratio.",
    ),
    # Coverage Ratios
    (
        "debt_service_ratio",
        "Income Statement, Balance Sheet",
        "Extract the Debt Service Ratio.",
    ),
    (
        "fixed_charge_ratio",
        "Income Statement, Balance Sheet",
        "Extract the Fixed Charge Coverage Ratio (Fixed Charge Ratio).",
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
        "Extract the Asset Disposition.",
    ),
    (
        "change_in_debt",
        "Balance Sheet",
        "Extract the Change in Debt.",
    ),
    # Working Capital Cycle
    (
        "receivables_days",
        "Balance Sheet, Income Statement",
        "Extract Days Sales Outstanding (DSO), or Receivable Days (DSO).",
    ),
    (
        "inventory_days",
        "Balance Sheet, Income Statement",
        "Extract Inventory Days.",
    ),
    (
        "payable_days",
        "Balance Sheet, Income Statement",
        "Extract Payables Days Outstanding (DPO).",
    ),
    (
        "cash_conversion_cycle",
        "Balance Sheet, Income Statement",
        "Extract the Cash Conversion Cycle.",
    ),
    # Cash Flow & Liquidity
    (
        "cash_burn_rate",
        "Cash Flow Statement",
        "Extract the Cash Burn Rate.",
    ),
    (
        "days_cash_on_hand",
        "Balance Sheet, Income Statement",
        "Extract the Days Cash on Hand.",
    ),
]

value_template = PromptTemplate(
    template="Analyze the following financials-related excerpts from a 10-K annual report for a company for the year {year}. Make sure you only extract figures for the year {year}, not past years. Pay attention specifically to the following sections: {section}. Make sure you return dollar amounts in terms of USD. {instruction}\n context: {context} \n Provide a structured response.\n{format_instructions}",
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

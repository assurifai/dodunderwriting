from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .models import KeywordAnalysisList, RiskAnalysisList, ConfidencePair

financial_keywords = ["financial", "capital", "income", "billion", "million", "cash"]
value_parser = PydanticOutputParser(pydantic_object=ConfidencePair)
value_format_instructions = value_parser.get_format_instructions()

value_queries = [
    (
        "ebitda",
        "Income Statement, Notes",
        "Extract the operating income.",
    ),
    ("revenue", "Income Statement, Notes", "Extract the net revenue or net sales globally."),
    ("cogs", "Income Statement, Notes", "Extract the total cost of sales or cost of goods sold."),
    (
        "distributions",
        "Cash Flow Statement (Financing)",
        "Extract the net distributions the company has done, eg. buybacks/dividends. If neither is mentioned then return 0.",
    ),
]

value_template = PromptTemplate(
    template="Analyze the following financials-related excerpts from a 10-K annual report for a company for the year {year}. Make sure you only extract figures for the year {year}, not past years. Pay attention specifically to the following sections: {section}. Make sure you return dollar amounts in terms of millions of USD. {instruction}\n Provide a structured response.\n{format_instructions}",
    input_variables=["year", "section", "instruction"],
    partial_variables={"format_instructions": value_format_instructions},
)

get_value_queries = lambda year: {
    key: (
        [],
        value_template.format(year=year, section=section, instruction=instruction),
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

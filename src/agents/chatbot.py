"""
Modularized CRM chatbot agents originally authored in agents.ipynb.

Classes:
    - GenAIAgent: Base text-generation wrapper around DialoGPT-medium.
    - InsightGenerationAgent: Pipeline insight prompts.
    - AccountSummaryAgent: Account summary prompts with fallback text.
    - EmailDraftingAgent: Follow-up/proposal email prompts with fallback text.
    - CRMChatbot: Router that wires agents plus optional vector store + ML insights.

Notes:
    - No side effects on import; instantiate CRMChatbot in your script/notebook.
    - Vector store fallback expects ChromaDB client + SentenceTransformer embedder.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd
import torch
from chromadb.api.client import SharedSystemClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenAIAgent:
    """Base class for Gen AI agents with promptable text generation."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", max_length: int = 512) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def generate_response(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 256) -> str:
        """Generate a response with sampling controls."""
        try:
            inputs = self.tokenizer.encode(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_length
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            response = self.tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True)
            return response.strip()
        except Exception as exc:  # pragma: no cover - defensive
            return f"Error generating response: {exc}"


class InsightGenerationAgent(GenAIAgent):
    """Agent for generating business insights and recommendations."""

    def __init__(self) -> None:
        super().__init__()

    def generate_pipeline_insights(self, opportunities_data: pd.DataFrame) -> str:
        total_pipeline = opportunities_data["close_value"].sum()
        avg_deal_size = opportunities_data["close_value"].mean()
        conversion_rate = (
            len(opportunities_data[opportunities_data["deal_stage"] == "Won"])
            / len(opportunities_data)
            * 100
        )
        context = f"""
        PIPELINE ANALYSIS DATA:
        - Total Pipeline Value: ${total_pipeline:,}
        - Number of Opportunities: {len(opportunities_data)}
        - Average Deal Size: ${avg_deal_size:,.0f}
        - Conversion Rate: {conversion_rate:.1f}%
        - Top Industries: {opportunities_data['sector'].value_counts().head(3).to_dict()}
        - Stage Distribution: {opportunities_data['deal_stage'].value_counts().to_dict()}
        - Top Performing Sales Agents: {opportunities_data['sales_agent'].value_counts().head(3).to_dict()}
        - Regional Breakdown: {opportunities_data['regional_office'].value_counts().to_dict()}
        """

        prompt = f"""
        You are a senior sales analyst. Analyze the pipeline data and provide actionable insights:

        {context}

        Generate insights covering:
        1. Pipeline Health Assessment
        2. Deal Size and Conversion Trends
        3. Industry and Segment Performance
        4. Stage-Specific Recommendations
        5. Risk Factors and Mitigation Strategies
        6. Growth Opportunities

        Provide specific, actionable recommendations with data-driven reasoning.

        PIPELINE INSIGHTS:
        """
        return self.generate_response(prompt, temperature=0.2, max_new_tokens=350)


class AccountSummaryAgent(GenAIAgent):
    """Specialized agent for generating dynamic account summaries."""

    def __init__(self) -> None:
        super().__init__()

    def create_summary_prompt(
        self,
        account_data: pd.Series,
        opportunities: pd.DataFrame,
        activities: pd.DataFrame,
        ml_insights: Optional[str] = None,
    ) -> str:
        total_pipeline = opportunities["close_value"].sum() if not opportunities.empty else 0
        open_opps = len(opportunities[opportunities["close_date"].notna()])
        won_opps = len(opportunities[opportunities["deal_stage"] == "Won"])

        prompt = f"""
        You are an expert CRM analyst. Generate a comprehensive, professional account summary based on the following data:

        ACCOUNT INFORMATION:
        - Company: {account_data['account']}
        - Industry: {account_data['sector']}
        - Annual Revenue: ${account_data['revenue']:,}
        - Employees: {account_data['employees']}
        - Region: {account_data['office_location']}
        - Parent company: {account_data['subsidiary_of'] if account_data['subsidiary_of'] else "N/A"}

        SALES PERFORMANCE:
        - Total Pipeline Value: ${total_pipeline:,}
        - Open Opportunities: {open_opps}
        - Won Opportunities: {won_opps}

        OPPORTUNITY DETAILS:
        {opportunities[['opportunity_id', 'sales_agent', 'product', 'deal_stage', 'engage_date', 'close_date', 'close_value']].to_string(index=False) if not opportunities.empty else "No opportunities found"}

        RECENT ACTIVITIES:
        {activities[['activity_type', 'subject', 'outcome', 'activity_date']].head(5).to_string(index=False) if not activities.empty else "No recent activities"}

        ML INSIGHTS:
        {ml_insights or "N/A"}

        Generate a professional account summary that includes:
        1. Executive Overview (2-3 sentences about the account's status and potential)
        2. Key Metrics and Performance Indicators
        3. Opportunity Pipeline Analysis
        4. Engagement and Activity Summary
        5. Risk Assessment and Recommendations
        6. Next Steps and Action Items

        Format the response in clear sections with bullet points where appropriate. Be analytical, insights-driven, and actionable.

        ACCOUNT SUMMARY:
        """
        return prompt

    def generate_account_summary(
        self, account_data: pd.Series, opportunities: pd.DataFrame, activities: pd.DataFrame
    ) -> str:
        prompt = self.create_summary_prompt(account_data, opportunities, activities)
        summary = self.generate_response(prompt, temperature=0.3, max_new_tokens=400)

        if len(summary) < 50:
            return f"""
            **Account Summary: {account_data['account']}**

            **Executive Overview:** {account_data['account']} is a {account_data['sector'].lower()} company with ${account_data['revenue']:,} in annual revenue and {account_data['employees']} employees. This account is located the {account_data['office_location']} region and is a parent company of {account_data['subsidiary_of']}.

            **Key Metrics:**
            • Annual Revenue: ${account_data['revenue']:,}
            • Company Size: {account_data['employees']} employees

            **Pipeline Analysis:**
            • Total Opportunities: {len(opportunities)}
            • Pipeline Value: ${opportunities['close_value'].sum():,}
            • Open Deals: {len(opportunities[opportunities['close_date'].notna()])}

            **Recommendations:** Based on the account profile and activity level, focus on nurturing the relationship and identifying expansion opportunities.
            """
        return summary


class EmailDraftingAgent(GenAIAgent):
    """Specialized agent for generating contextual emails."""

    def __init__(self) -> None:
        super().__init__()

    def create_email_prompt(self, email_type: str, context_data: str, additional_context: str = "") -> str:
        base_prompt = f"""
        You are an expert sales professional writing personalized, engaging emails. Generate a professional email based on the context provided.

        EMAIL TYPE: {email_type}
        CONTEXT DATA: {context_data}
        ADDITIONAL CONTEXT: {additional_context}

        Email Guidelines:
        - Professional but warm tone
        - Personalized and specific to the recipient
        - Clear call-to-action
        - Appropriate length (not too long or short)
        - Include relevant business value
        - Use compelling subject line
        - Follow best practices for sales communication

        Generate a complete email with:
        1. Subject Line
        2. Professional greeting
        3. Body with clear purpose and value proposition
        4. Specific call-to-action
        5. Professional closing

        EMAIL:
        """
        return base_prompt

    def draft_follow_up_email(
        self, opportunity_data: pd.Series, account_data: pd.Series, last_activity: Optional[pd.Series] = None
    ) -> str:
        context = f""" OPPORTUNITY: {opportunity_data['opportunity_id']} ACCOUNT: {account_data['account']} INDUSTRY: {account_data['sector']} CURRENT STAGE: {opportunity_data['deal_stage']} DEAL VALUE: ${opportunity_data['close_value']:,} CLOSE DATE: {opportunity_data['close_date']} LAST CONTACT: {last_activity if last_activity else 'No recent activity recorded'}"""
        additional_context = f""" The opportunity is currently in {opportunity_data['deal_stage']} stage. Focus on moving the deal forward and addressing any potential concerns."""
        prompt = self.create_email_prompt("Follow-up", context, additional_context)
        email = self.generate_response(prompt, temperature=0.4, max_new_tokens=300)
        return self.format_email_output(email, opportunity_data, account_data)

    def draft_proposal_email(self, opportunity_data: pd.Series, account_data: pd.Series, proposal_details: str = "") -> str:
        context = f"""
        OPPORTUNITY INFORMATION:
        - Opportunity ID: {opportunity_data['opportunity_id']}
        - Product: {opportunity_data['product']}
        - Deal Stage: {opportunity_data['deal_stage']}
        - Sales Agent: {opportunity_data['sales_agent']}
        - Manager: {opportunity_data['manager']}
        - Regional Office: {opportunity_data['regional_office']}
        - Series: {opportunity_data['series']}
        - Engage Date: {opportunity_data['engage_date']}
        - Expected Close Date: {opportunity_data['close_date']}
        - Deal Value: ${opportunity_data['close_value']:,}
        - Sales Price: ${opportunity_data['sales_price']:,}

        ACCOUNT INFORMATION:
        - Account Name: {opportunity_data['account']}
        - Industry (Sector): {opportunity_data['sector']}
        - Year Established: {opportunity_data['year_established']}
        - Annual Revenue: ${opportunity_data['revenue']:,}
        - Employees: {opportunity_data['employees']}
        - Office Location: {opportunity_data['office_location']}
        - Subsidiary Of: {opportunity_data['subsidiary_of']}
        """
        additional_context = f"""
        The opportunity is ready for proposal presentation.
        Focus on scheduling a meeting to present the proposal.
        Highlight the business value and ROI for their {account_data['sector']} industry.
        Create excitement about the solution and next steps.
        """
        prompt = self.create_email_prompt("Proposal Presentation", context, additional_context)
        email = self.generate_response(prompt, temperature=0.3, max_new_tokens=280)
        return self.format_email_output(email, opportunity_data, account_data)

    def format_email_output(self, email_content: str, opportunity_data: pd.Series, account_data: pd.Series) -> str:
        if len(email_content.strip()) < 100:
            return f""" Subject: Following up on {opportunity_data['opportunity_id']} - Next Steps

Dear {account_data['account']} Team,

I hope this email finds you well. I wanted to follow up on opportunity {opportunity_data['opportunity_id']} that we've been discussing.

**Current Status:**
• Deal Stage: {opportunity_data['deal_stage']}
• Project Value: ${opportunity_data['close_value']:,}
• Target Timeline: {opportunity_data['close_date']}

**Next Steps:**
I'd like to schedule a brief call this week to discuss any questions you might have and outline the next steps in our process. This will help ensure we stay on track for your {opportunity_data['close_date']} timeline.

**Value Proposition:**
Our solution is specifically designed for {account_data['sector']} companies like {account_data['account']}, helping organizations achieve measurable results while reducing operational complexity.

Would you be available for a 30-minute call this week? I have openings on Tuesday and Thursday afternoons.

Best regards,
{opportunity_data['sales_agent']}

P.S. I've attached some relevant case studies from similar {account_data['sector']} implementations that you might find interesting.
"""
        return email_content


class CRMChatbot:
    """Router that orchestrates LLM agents, optional ML insights, and semantic search."""

    def __init__(
        self,
        client: Any,
        embedder: SentenceTransformer,
        ml_models: Optional[Dict[str, Any]],
        accounts: pd.DataFrame,
        opportunities: pd.DataFrame,
        leads: pd.DataFrame,
        activities: pd.DataFrame,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.ml_models = ml_models or {}
        self.accounts = accounts
        self.opportunities = opportunities
        self.leads = leads
        self.activities = activities

        self.summary_agent = AccountSummaryAgent()
        self.email_agent = EmailDraftingAgent()
        self.insight_agent = InsightGenerationAgent()

    def generate_account_summary(self, account_id: str) -> str:
        account = self.accounts[self.accounts["account_id"] == account_id]
        if account.empty:
            return f"Account {account_id} not found."

        account_data = account.iloc[0]
        related_opps = self.opportunities[self.opportunities["account_id"] == account_id]
        related_activities = (
            self.activities[self.activities["account_id"] == account_id]
            if not self.activities.empty
            else pd.DataFrame()
        )
        return self.summary_agent.generate_account_summary(account_data, related_opps, related_activities)

    def draft_email(self, opportunity_id: str, email_type: str = "follow_up") -> str:
        opp = self.opportunities[self.opportunities["opportunity_id"] == opportunity_id]
        if opp.empty:
            return f"Opportunity {opportunity_id} not found."

        opp_data = opp.iloc[0]
        account = self.accounts[self.accounts["account_id"] == opp_data["account_id"]].iloc[0]
        last_activity = (
            self.activities[self.activities["account_id"] == opp_data["account_id"]]
            .sort_values("activity_date", ascending=False)
            .iloc[0]
            if not self.activities.empty
            else None
        )

        if email_type == "proposal":
            return self.email_agent.draft_proposal_email(opp_data, account)
        return self.email_agent.draft_follow_up_email(opp_data, account, last_activity)

    def generate_insights(self, insight_type: str = "pipeline") -> str:
        if insight_type == "pipeline":
            return self.insight_agent.generate_pipeline_insights(self.opportunities)
        return "Insight type not supported yet."

    def get_ml_insights(self, query: str) -> str:
        insights = []
        q = query.lower()

        if "lead" in q and "score" in q and not self.leads.empty:
            high_score_leads = self.leads[self.leads.get("lead_score", 0) > 70]
            insights.append(f"Found {len(high_score_leads)} high-scoring leads (>70)")

        if "opportunity" in q and ("win" in q or "probability" in q):
            if "probability" in self.opportunities.columns:
                high_prob_opps = self.opportunities[self.opportunities["probability"] > 80]
                insights.append(f"Found {len(high_prob_opps)} high-probability opportunities (>80%)")
            elif "opp_win" in self.ml_models:
                opp_match = re.search(r"opp-[\\da-z]+", q)
                if opp_match:
                    opp_id = opp_match.group().upper()
                    bundle = self.ml_models["opp_win"]
                    feats = bundle.get("features")
                    model = bundle.get("model")
                    if feats is not None and model is not None:
                        row = feats[feats["opportunity_id"] == opp_id]
                        if not row.empty:
                            X = row.drop(columns=["opportunity_id"])
                            try:
                                prob = model.predict_proba(X)[:, 1][0]
                                insights.append(f"Win probability for {opp_id}: {prob:.2%}")
                            except Exception as exc:  # pragma: no cover - defensive
                                insights.append(f"Error scoring {opp_id}: {exc}")
                        else:
                            insights.append(f"No preprocessed features found for {opp_id}")

        if "account" in q and "health" in q:
            insights.append("Account health scores calculated using ML model based on activities and engagement")

        return "\n".join(insights) if insights else "No specific ML insights found for your query."

    def process_natural_language_query(self, query: str) -> str:
        query_lower = query.lower()

        if any(word in query_lower for word in ["account", "company"]) and "summary" in query_lower:
            account_match = re.search(r"acc-\\d+", query_lower)
            if account_match:
                account_id = account_match.group().upper()
                return self.generate_account_summary(account_id)
            return "Please specify an account ID (e.g., ACC-00001) for the summary."

        if "email" in query_lower or "draft" in query_lower:
            opp_match = re.search(r"opp-[\\da-zA-Z]+", query_lower)
            if opp_match:
                opp_id = opp_match.group().upper()
                email_type = "proposal" if "proposal" in query_lower else "follow_up"
                return self.draft_email(opp_id, email_type)
            return "Please specify an opportunity ID (e.g., OPP-00001) for email drafting."

        if any(word in query_lower for word in ["insights", "analysis", "pipeline", "forecast"]):
            insight_type = "pipeline" if "pipeline" in query_lower else "general"
            return self.generate_insights(insight_type)

        if any(word in query_lower for word in ["predict", "score", "probability", "ml"]):
            return self.get_ml_insights(query)

        # Semantic search fallback using ChromaDB collections
        acc_collection = self.client.get_collection("sales_accounts")
        opp_collection = self.client.get_collection("sales_opportunities")

        query_embedding = self.embedder.encode([query]).tolist()
        acc_results = acc_collection.query(query_embeddings=query_embedding, n_results=3)
        opp_results = opp_collection.query(query_embeddings=query_embedding, n_results=3)

        response = "Based on your query, here are the most relevant results:\n\n"

        if acc_results["documents"] and acc_results["documents"][0]:
            response += "**Relevant Accounts:**\n"
            for i, doc in enumerate(acc_results["documents"][0][:2]):
                meta = acc_results["metadatas"][0][i]
                meta_text = ", ".join(f"{k}: {v}" for k, v in meta.items())
                response += f"- {doc}\n  _({meta_text})_\n"
            response += "\n"

        if opp_results["documents"] and opp_results["documents"][0]:
            response += "**Relevant Opportunities:**\n"
            for i, doc in enumerate(opp_results["documents"][0][:2]):
                meta = opp_results["metadatas"][0][i]
                meta_text = ", ".join(f"{k}: {v}" for k, v in meta.items())
                response += f"- {doc}\n  _({meta_text})_\n"

        if (not acc_results["documents"] or not acc_results["documents"][0]) and (
            not opp_results["documents"] or not opp_results["documents"][0]
        ):
            response += "No relevant results found."
        return response


__all__ = [
    "GenAIAgent",
    "InsightGenerationAgent",
    "AccountSummaryAgent",
    "EmailDraftingAgent",
    "CRMChatbot",
]

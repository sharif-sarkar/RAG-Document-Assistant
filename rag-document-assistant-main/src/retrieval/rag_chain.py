import logging
from typing import Optional

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import MessagesPlaceholder

from src.config.settings import Settings


class RAGChainFactory:
    """
    Factory class for creating RAG (Retrieval-Augmented Generation) chains.
    Supports customizable prompts and chain configurations.
    """

    # Default RAG prompt template
    DEFAULT_RAG_TEMPLATE = """Answer the question based ONLY on the following context:

{context}

Question: {question}
"""

    # Conversational RAG prompt template with chat history
    CONVERSATIONAL_RAG_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

    # Final answer template for conversational mode
    ANSWER_TEMPLATE = """Answer the question based ONLY on the following context:

{context}

Question: {question}
"""

    def __init__(self, settings: Settings):
        """
        Initialize RAG chain factory with configuration.

        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def create_basic_rag_chain(
        self, retriever, llm, custom_template: Optional[str] = None
    ):
        """
        Create a basic RAG chain.

        Args:
            retriever: Document retriever
            llm: Language model
            custom_template: Optional custom prompt template

        Returns:
            RAG chain
        """
        template = custom_template or self.DEFAULT_RAG_TEMPLATE
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        self.logger.info("Created basic RAG chain")
        return chain

    def create_conversational_rag_chain(
        self,
        retriever,
        llm,
        question_rephrasing_llm,
        custom_template: Optional[str] = None,
    ):
        """
        Create a conversational RAG chain with chat history support.

        Args:
            retriever: Document retriever
            llm: Language model for answering
            question_rephrasing_llm: Language model for rephrasing questions
            custom_template: Optional custom prompt template

        Returns:
            Conversational RAG chain
        """
        template = custom_template or self.ANSWER_TEMPLATE
        answer_prompt = ChatPromptTemplate.from_template(template)

        question_rephrasing_prompt = ChatPromptTemplate.from_template(
            self.CONVERSATIONAL_RAG_TEMPLATE
        )

        rephrase_chain = (
            question_rephrasing_prompt | question_rephrasing_llm | StrOutputParser()
        )

        def retrieve_and_answer(inputs):
            question = inputs.get("question")
            chat_history = inputs.get("chat_history", "")

            rephrased_question = rephrase_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )

            context = retriever.invoke(rephrased_question)
            context_str = "\n\n".join([doc.page_content for doc in context])

            return answer_prompt.invoke({"context": context_str, "question": question})

        chain = retrieve_and_answer | llm | StrOutputParser()

        self.logger.info("Created conversational RAG chain")
        return chain

    def create_rag_chain(
        self,
        retriever,
        llm,
        mode: str = "basic",
        question_rephrasing_llm=None,
        custom_template: Optional[str] = None,
    ):
        """
        Create a RAG chain based on the specified mode.

        Args:
            retriever: Document retriever
            llm: Language model
            mode: Chain mode ('basic' or 'conversational')
            question_rephrasing_llm: LLM for rephrasing (required for conversational)
            custom_template: Optional custom prompt template

        Returns:
            RAG chain
        """
        if mode == "conversational":
            if question_rephrasing_llm is None:
                self.logger.warning(
                    "No question_rephrasing_llm provided, using main LLM"
                )
                question_rephrasing_llm = llm
            return self.create_conversational_rag_chain(
                retriever, llm, question_rephrasing_llm, custom_template
            )
        else:
            return self.create_basic_rag_chain(retriever, llm, custom_template)

    def update_template(self, template: str) -> None:
        """
        Update the default RAG template.

        Args:
            template: New template string
        """
        self.DEFAULT_RAG_TEMPLATE = template
        self.logger.info("RAG template updated")

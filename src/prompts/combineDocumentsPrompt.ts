import {
  AIMessagePromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
} from "langchain/prompts";

// Combine documents prompt
export const combineDocumentsPromptTemplate = ChatPromptTemplate.fromMessages([
  AIMessagePromptTemplate.fromTemplate(
    "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\n"
  ),
  new MessagesPlaceholder("chat_history"),
  HumanMessagePromptTemplate.fromTemplate("Question: {question}"),
]);

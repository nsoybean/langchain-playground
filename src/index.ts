import "dotenv/config";

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RetrievalQAChain } from "langchain/chains";
import { pino } from "pino";
import { formatDocumentsAsString } from "langchain/util/document";
import {
  AIMessagePromptTemplate,
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "langchain/schema/runnable";
import { StringOutputParser } from "langchain/schema/output_parser";
import { BufferMemory } from "langchain/memory";
import { PromptTemplate } from "langchain/prompts";
import { MongoDBChatMessageHistory } from "langchain/stores/message/mongodb";
// import { serializeChatHistory } from "../lib/helper";
import { MongoClient, ObjectId } from "mongodb";
import { combineDocumentsPromptTemplate } from "./prompts/combineDocumentsPrompt";

const logger = pino({
  level: "debug",
});

// CONSTANTS
const INPUT_QUESTION = process.env.INPUT_QUESTION;

// CONFIG
// openAI API key
const OPENAIAPIKEY = process.env.OPENAI_API_KEY;
// supabase
const PRIVATEKEY = process.env.SUPABASE_PRIVATE_KEY;
const URL = process.env.SUPABASE_URL;

const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");
const collection = client.db("langchain").collection("memory");
// generate a new sessionId string
const sessionId = "65643daabcc584e5fa4a5c88"; // hardcoded, to test retrieval and update

chat();

// main chat implementation
async function chat() {
  // DB
  await client.connect();

  // Init
  if (!INPUT_QUESTION) {
    logger.fatal(`Expected env var INPUT_QUESTION`);
    throw new Error(`Expected env var INPUT_QUESTION`);
  }

  if (!PRIVATEKEY) {
    logger.fatal(`Expected env var SUPABASE_PRIVATE_KEY`);
    throw new Error(`Expected env var SUPABASE_PRIVATE_KEY`);
  }
  if (!URL) {
    logger.fatal(`Expected env var SUPABASE_URL`);
    throw new Error(`Expected env var SUPABASE_URL`);
  }
  if (!OPENAIAPIKEY) {
    logger.fatal(`Expected env var OPENAI_API_KEY`);
    throw new Error(`Expected env var OPENAI_API_KEY`);
  }

  // init vector store
  const supabaseClient = createClient(URL, PRIVATEKEY);
  const vectorStore = new SupabaseVectorStore(
    new OpenAIEmbeddings({ openAIApiKey: OPENAIAPIKEY }),
    {
      client: supabaseClient,
      tableName: "documents",
      queryName: "match_documents",
    }
  );

  // Initialize a retriever wrapper around the vector store
  const vectorStoreRetriever = vectorStore.asRetriever({
    searchType: "mmr", // Use max marginal relevance search
    searchKwargs: { fetchK: 5 },
  });

  const memory = new BufferMemory({
    memoryKey: "chat_history",
    chatHistory: new MongoDBChatMessageHistory({
      collection,
      sessionId,
    }),
  });

  // Initialize the LLM to use to answer the question.
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: OPENAIAPIKEY,
    verbose: true, // debugging purposes
  }).pipe(new StringOutputParser());

  const conversationalQaChain = RunnableSequence.from([
    {
      question: (input: { question: string }) => input.question,
      chat_history: async () => {
        const dbMsgs = await memory.chatHistory.getMessages();
        return dbMsgs;
      },
      context: async (input: { question: string }) => {
        const relevantDocs = await vectorStoreRetriever.getRelevantDocuments(
          input.question
        );
        return formatDocumentsAsString(relevantDocs);
      },
    },
    combineDocumentsPromptTemplate,
    model,
    new StringOutputParser(),
  ]);

  // call
  const result = await conversationalQaChain.invoke({
    question: INPUT_QUESTION,
  });

  // persist memory
  await memory.saveContext(
    {
      input: INPUT_QUESTION,
    },
    {
      output: result,
    }
  );

  logger.debug("🚀 LLM output:", result);
  logger.debug(`🚀 program completed`);
  return;
}

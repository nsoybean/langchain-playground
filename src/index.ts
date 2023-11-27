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

const logger = pino({
  level: "debug",
});

// CONSTANTS
const PDF_PATH = process.env.PDF_PATH;
const INPUT_QUESTION = process.env.INPUT_QUESTION;

// CONFIG
// openAI API key
const OPENAIAPIKEY = process.env.OPENAI_API_KEY;
// supabase
const PRIVATEKEY = process.env.SUPABASE_PRIVATE_KEY;
const URL = process.env.SUPABASE_URL;

// const memory = new BufferMemory({
//   returnMessages: true,
//   memoryKey: "chat_history",
// });

const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");
const collection = client.db("langchain").collection("memory");
// generate a new sessionId string
// const sessionId = new ObjectId().toString();
const sessionId = "65643daabcc584e5fa4a5c88";

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

chat();

// main chat implementation
async function chat() {
  // DB
  await client.connect();

  // Init
  if (!PDF_PATH && !INPUT_QUESTION) {
    logger.fatal(`Expected at least one of PDF_PATH or INPUT_QUESTION`);
    throw new Error(`Expected at least one of PDF_PATH or INPUT_QUESTION`);
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

  // parse and embed doc if PDF is provided
  if (PDF_PATH) {
    const loader = new PDFLoader(PDF_PATH);
    const docs = await loader.load();
    logger.debug(`docs length:, ${docs.length}`);

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 200,
      chunkOverlap: 50,
    });
    const docsChunks = await textSplitter.splitDocuments(docs);
    logger.debug(`docsChunks length: ${docsChunks.length}`);

    vectorStore.addDocuments(docsChunks);
  }

  // Initialize a retriever wrapper around the vector store
  const vectorStoreRetriever = vectorStore.asRetriever({
    searchType: "mmr", // Use max marginal relevance search
    searchKwargs: { fetchK: 5 },
  });

  /**
   * Create a prompt template for generating an answer based on context and
   * a question.
   *
   * Chat history will be an empty string if it's the first question.
   *
   * inputVariables: ["chatHistory", "context", "question"]
   */
  const questionGeneratorTemplate = ChatPromptTemplate.fromMessages([
    AIMessagePromptTemplate.fromTemplate(
      "Given the following chat history and a follow up question, rephrase the follow up question to be a standalone question"
    ),
    new MessagesPlaceholder("chat_history"),
    AIMessagePromptTemplate.fromTemplate(`Follow Up Input: {question}
  Standalone question:`),
  ]);

  // Combine documents prompt:
  const combineDocumentsPrompt = ChatPromptTemplate.fromMessages([
    AIMessagePromptTemplate.fromTemplate(
      "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\n"
    ),
    new MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.fromTemplate("Question: {question}"),
  ]);

  const combineDocumentsChain = RunnableSequence.from([
    {
      question: (output: string) => output,
      chat_history: async () => {
        const dbMsgs = await memory.chatHistory.getMessages();
        console.log(
          "🚀 ~ file: index.ts:170 ~ chat_history: ~ dbMsgs1:",
          dbMsgs
        );
        return dbMsgs;
      },
      context: async (output: string) => {
        const relevantDocs = await vectorStoreRetriever.getRelevantDocuments(
          output
        );
        return formatDocumentsAsString(relevantDocs);
      },
    },
    combineDocumentsPrompt,
    model,
    new StringOutputParser(),
  ]);

  const conversationalQaChain = RunnableSequence.from([
    {
      question: (i: { question: string }) => i.question,
      chat_history: async () => {
        const dbMsgs = await memory.chatHistory.getMessages();
        return dbMsgs;
      },
    },
    questionGeneratorTemplate,
    model,
    new StringOutputParser(),
    combineDocumentsChain,
  ]);

  if (INPUT_QUESTION) {
    const result = await conversationalQaChain.invoke({
      question: INPUT_QUESTION,
    });

    await memory.saveContext(
      {
        input: INPUT_QUESTION,
      },
      {
        output: result,
      }
    );

    console.log("🚀 ~ file: index.ts:179 ~ chat ~ result:", result);
  }

  logger.debug(`🚀 program compeleted`);
  return;
}

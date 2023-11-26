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
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "langchain/schema/runnable";
import { StringOutputParser } from "langchain/schema/output_parser";
import { BufferMemory } from "langchain/memory";
// import { serializeChatHistory } from "../lib/helper";

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

const memory = new BufferMemory({
  memoryKey: "chatHistory",
});

chat();

// main chat implementation
async function chat() {
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
  const vectorStoreRetriever = vectorStore.asRetriever();

  // Create a system & human prompt for the chat model
  const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  ----------------
  {context}`;

  const messages = [
    SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.fromTemplate("{question}"),
  ];
  const prompt = ChatPromptTemplate.fromMessages(messages);

  // Initialize the LLM to use to answer the question.
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: OPENAIAPIKEY,
    verbose: true, // debugging purposes
  });

  // main QA sequence
  const chain = RunnableSequence.from([
    {
      context: vectorStoreRetriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);

  // ask qn
  const response = await chain.invoke(INPUT_QUESTION);
  logger.debug(`response: ${JSON.stringify(response)}`);

  logger.debug(`🚀 program compeleted`);
}

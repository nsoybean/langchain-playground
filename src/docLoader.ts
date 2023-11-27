import "dotenv/config";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { pino } from "pino";
import { StringOutputParser } from "langchain/schema/output_parser";

const logger = pino({
  level: "debug",
});

// CONSTANTS
const PDF_PATH = process.env.PDF_PATH;

// CONFIG
// openAI API key
const OPENAIAPIKEY = process.env.OPENAI_API_KEY;
// supabase
const PRIVATEKEY = process.env.SUPABASE_PRIVATE_KEY;
const URL = process.env.SUPABASE_URL;

// Initialize the LLM to use to answer the question.
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: OPENAIAPIKEY,
  verbose: true, // debugging purposes
}).pipe(new StringOutputParser());

loadDoc();

// main chat implementation
async function loadDoc() {
  // Init
  if (!PDF_PATH) {
    logger.fatal(`Expected at least one of PDF_PATH`);
    throw new Error(`Expected at least one of PDF_PATH`);
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

  logger.debug(`🚀 program completed`);
  return;
}

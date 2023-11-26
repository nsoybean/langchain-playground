// Document loader
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { pino } from "pino";

const WEB_URL = "";
const logger = pino({
  level: "debug",
});

webLoader();

async function webLoader() {
  const loader = new CheerioWebBaseLoader(WEB_URL);
  const data = await loader.load();
  logger.debug(`data: ${JSON.stringify(data)}`);
}

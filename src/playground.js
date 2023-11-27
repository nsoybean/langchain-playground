const { HumanMessage, AIMessage } = require("langchain/schema");
const { BufferMemory, ChatMessageHistory } = require("langchain/memory");

const my_chat = [
  {
    schema: "AIMessage",
    content: "Hi, how can i help you? What is your name?",
  },
  { schema: "HumanMessage", content: "my name is Jonas" },
];

async function playground() {
  const humanMsg = new HumanMessage("My name's Jonas");
  console.log("🚀 ~ file: playground.js:5 ~ playground ~ humanMsg:", humanMsg);
  console.log(
    "🚀 ~ file: playground.js:5 ~ playground ~ humanMsg:",
    JSON.parse(JSON.stringify(humanMsg))
  );
  const AIMsg = new AIMessage("Nice to meet you, Jonas!");
  console.log(
    "🚀 ~ file: playground.js:7 ~ playground ~ AIMsg:",
    JSON.stringify(AIMsg)
  );

  const pastMessages = [
    JSON.parse(JSON.stringify(humanMsg)),
    JSON.parse(JSON.stringify(AIMsg)),
  ];
  const memory = new BufferMemory({
    chatHistory: new ChatMessageHistory(pastMessages),
  });

  const res = await memory.loadMemoryVariables({'chatHi});
  console.log("🚀 ~ file: playground.js:40 ~ playground ~ chat_history:", res);
}

playground();

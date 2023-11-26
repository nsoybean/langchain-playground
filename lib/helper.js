// helper function
export const serializeChatHistory = (chatHistory) => {
  if (Array.isArray(chatHistory)) {
    return chatHistory.join("\n");
  }
  return chatHistory;
};

import MessageBubble from "../MessageBubble";

export default function MessageBubbleExample() {
  return (
    <div className="space-y-4 p-6 bg-background">
      <MessageBubble 
        type="user" 
        content="Please summarize this long paragraph about artificial intelligence and machine learning..." 
      />
      <MessageBubble 
        type="ai" 
        content="Here's a concise summary: AI and ML are transforming how we process information, enabling computers to learn from data and make intelligent decisions without explicit programming."
        onCopy={() => console.log("Copied to clipboard")}
      />
    </div>
  );
}

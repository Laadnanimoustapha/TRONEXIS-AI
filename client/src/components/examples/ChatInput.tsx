import ChatInput from "../ChatInput";

export default function ChatInputExample() {
  return (
    <div className="min-h-[200px] flex items-end">
      <ChatInput onSend={(text) => console.log("Sent:", text)} />
    </div>
  );
}

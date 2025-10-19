import { useState, useRef, useEffect } from "react";
import ModeToggle from "@/components/ModeToggle";
import MessageBubble from "@/components/MessageBubble";
import ChatInput from "@/components/ChatInput";
import EmptyState from "@/components/EmptyState";
import ThemeToggle from "@/components/ThemeToggle";
import { useToast } from "@/hooks/use-toast";

type Mode = "summarize" | "reformulate";

interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("summarize");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsProcessing(true);

    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content:
          mode === "summarize"
            ? "This is a demo summary. In the full version, this will be powered by Gemini AI to provide intelligent text summarization."
            : "This is a demo reformulation. In the full version, this will be powered by Gemini AI to reformulate your text in different styles.",
      };
      setMessages((prev) => [...prev, aiMessage]);
      setIsProcessing(false);
    }, 1500);
  };

  const handleExampleClick = (text: string) => {
    handleSend(text);
  };

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content);
    toast({
      title: "Copied to clipboard",
      description: "The message has been copied successfully.",
    });
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      <header className="flex items-center justify-between p-4 border-b bg-background/80 backdrop-blur-lg sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">AI</span>
          </div>
          <div>
            <h1 className="font-semibold text-lg" data-testid="text-app-title">
              Text Assistant
            </h1>
            <p className="text-xs text-muted-foreground">Powered by Gemini AI</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <ModeToggle mode={mode} onModeChange={setMode} />
          <ThemeToggle />
        </div>
      </header>

      {messages.length === 0 ? (
        <EmptyState mode={mode} onExampleClick={handleExampleClick} />
      ) : (
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto p-6 space-y-6">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                type={message.type}
                content={message.content}
                onCopy={message.type === "ai" ? () => handleCopy(message.content) : undefined}
              />
            ))}
            {isProcessing && (
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-chart-1 flex items-center justify-center">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                </div>
                <div className="px-4 py-3 bg-card rounded-2xl rounded-bl-md">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      <ChatInput
        onSend={handleSend}
        disabled={isProcessing}
        placeholder={
          mode === "summarize"
            ? "Enter text to summarize..."
            : "Enter text to reformulate..."
        }
      />
    </div>
  );
}

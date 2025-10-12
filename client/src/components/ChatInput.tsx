import { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSend, disabled = false, placeholder = "Enter your text here..." }: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [text]);

  const handleSend = () => {
    if (text.trim() && !disabled) {
      onSend(text);
      setText("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="sticky bottom-0 bg-background/80 backdrop-blur-lg border-t p-4">
      <div className="max-w-4xl mx-auto relative">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={placeholder}
          data-testid="input-chat"
          className="w-full min-h-[56px] max-h-[200px] px-4 py-3 pr-14 bg-card border border-input rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground text-card-foreground leading-relaxed"
          rows={1}
        />
        <Button
          onClick={handleSend}
          disabled={!text.trim() || disabled}
          size="icon"
          data-testid="button-send"
          className="absolute right-2 bottom-2 rounded-full"
        >
          <Send className="w-4 h-4" />
        </Button>
        <div className="mt-2 text-xs text-muted-foreground text-right">
          {text.length} characters · Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}

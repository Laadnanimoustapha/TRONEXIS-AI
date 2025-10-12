import { Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Copy } from "lucide-react";

interface MessageBubbleProps {
  type: "user" | "ai";
  content: string;
  onCopy?: () => void;
}

export default function MessageBubble({ type, content, onCopy }: MessageBubbleProps) {
  if (type === "user") {
    return (
      <div className="flex justify-end gap-3">
        <div 
          className="max-w-[85%] px-4 py-3 bg-primary text-primary-foreground rounded-2xl rounded-br-md"
          data-testid="message-user"
        >
          <p className="leading-relaxed whitespace-pre-wrap">{content}</p>
        </div>
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
          <User className="w-4 h-4 text-muted-foreground" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 group">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-chart-1 flex items-center justify-center">
        <Bot className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 max-w-[85%]">
        <div 
          className="px-4 py-3 bg-card rounded-2xl rounded-bl-md"
          data-testid="message-ai"
        >
          <p className="leading-relaxed whitespace-pre-wrap text-card-foreground">{content}</p>
        </div>
        {onCopy && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onCopy}
            data-testid="button-copy-message"
            className="mt-2 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Copy className="w-3 h-3 mr-2" />
            Copy
          </Button>
        )}
      </div>
    </div>
  );
}

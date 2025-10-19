import { Sparkles, RefreshCw } from "lucide-react";

type Mode = "summarize" | "reformulate";

interface ModeToggleProps {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
}

export default function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
  return (
    <div className="inline-flex items-center gap-1 p-1 bg-muted rounded-lg">
      <button
        onClick={() => onModeChange("summarize")}
        data-testid="button-mode-summarize"
        className={`inline-flex items-center gap-2 px-4 min-h-9 rounded-md font-medium transition-all ${
          mode === "summarize"
            ? "bg-chart-2 text-white"
            : "text-muted-foreground hover-elevate"
        }`}
      >
        <Sparkles className="w-4 h-4" />
        Summarize
      </button>
      <button
        onClick={() => onModeChange("reformulate")}
        data-testid="button-mode-reformulate"
        className={`inline-flex items-center gap-2 px-4 min-h-9 rounded-md font-medium transition-all ${
          mode === "reformulate"
            ? "bg-chart-3 text-white"
            : "text-muted-foreground hover-elevate"
        }`}
      >
        <RefreshCw className="w-4 h-4" />
        Reformulate
      </button>
    </div>
  );
}

import { Sparkles, RefreshCw, FileText, Wand2 } from "lucide-react";
import { Card } from "@/components/ui/card";

type Mode = "summarize" | "reformulate";

interface EmptyStateProps {
  mode: Mode;
  onExampleClick: (text: string) => void;
}

const examples = {
  summarize: [
    {
      icon: FileText,
      title: "Long Article",
      text: "Artificial intelligence has revolutionized the way we approach complex problems. From healthcare to finance, AI systems are being deployed to analyze vast amounts of data, identify patterns, and make predictions that were previously impossible for humans to achieve alone.",
    },
    {
      icon: Wand2,
      title: "Meeting Notes",
      text: "Today's team meeting covered Q4 goals, budget allocation, new hiring plans, marketing strategy updates, and the product roadmap. Sarah presented the sales figures, John discussed technical challenges, and Maria outlined the customer feedback summary.",
    },
  ],
  reformulate: [
    {
      icon: FileText,
      title: "Make it Formal",
      text: "Hey, just wanted to let you know that the project is going great and we should be done soon!",
    },
    {
      icon: Wand2,
      title: "Simplify",
      text: "The implementation of sophisticated algorithmic paradigms necessitates a comprehensive reevaluation of our existing technological infrastructure.",
    },
  ],
};

export default function EmptyState({ mode, onExampleClick }: EmptyStateProps) {
  const currentExamples = examples[mode];
  const Icon = mode === "summarize" ? Sparkles : RefreshCw;
  const title = mode === "summarize" ? "Summarize Your Text" : "Reformulate Your Text";
  const description = mode === "summarize" 
    ? "Enter any text and get a concise, intelligent summary powered by AI"
    : "Transform your text into different styles - formal, casual, professional, or simplified";

  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="max-w-3xl w-full text-center space-y-8">
        <div className="space-y-3">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10">
            <Icon className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-3xl font-semibold tracking-tight" data-testid="text-empty-title">
            {title}
          </h1>
          <p className="text-lg text-muted-foreground max-w-xl mx-auto">
            {description}
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {currentExamples.map((example, index) => {
            const ExampleIcon = example.icon;
            return (
              <Card
                key={index}
                onClick={() => onExampleClick(example.text)}
                data-testid={`card-example-${index}`}
                className="p-4 text-left cursor-pointer hover-elevate active-elevate-2 transition-all"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-muted flex items-center justify-center">
                    <ExampleIcon className="w-5 h-5 text-muted-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium mb-2">{example.title}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {example.text}
                    </p>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}

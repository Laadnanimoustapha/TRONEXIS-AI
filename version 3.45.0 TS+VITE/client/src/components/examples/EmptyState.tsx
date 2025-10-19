import { useState } from "react";
import EmptyState from "../EmptyState";

export default function EmptyStateExample() {
  const [mode] = useState<"summarize" | "reformulate">("summarize");
  
  return (
    <div className="h-screen bg-background">
      <EmptyState 
        mode={mode} 
        onExampleClick={(text) => console.log("Example clicked:", text)} 
      />
    </div>
  );
}

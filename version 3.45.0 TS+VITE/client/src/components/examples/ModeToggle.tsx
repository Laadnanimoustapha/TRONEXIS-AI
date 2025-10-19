import { useState } from "react";
import ModeToggle from "../ModeToggle";

export default function ModeToggleExample() {
  const [mode, setMode] = useState<"summarize" | "reformulate">("summarize");
  
  return <ModeToggle mode={mode} onModeChange={setMode} />;
}

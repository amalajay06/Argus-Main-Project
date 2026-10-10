import { FormEvent, useState } from "react";

interface ChatInputProps {
  placeholder: string;
  onSend: (message: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ placeholder, onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-end gap-3 bg-argus-panel border border-argus-border rounded-xl px-4 py-3 focus-within:border-argus-accent/60 transition-colors"
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
          }
        }}
        placeholder={placeholder}
        rows={1}
        className="flex-1 bg-transparent resize-none text-sm placeholder:text-argus-muted focus:outline-none max-h-40"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="shrink-0 text-xs font-medium bg-argus-accent text-argus-black rounded-lg px-4 py-2 disabled:opacity-40 hover:bg-argus-accent/90 transition-colors"
      >
        Send
      </button>
    </form>
  );
}

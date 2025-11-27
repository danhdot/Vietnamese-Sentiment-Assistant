import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { classifySentiment, fetchHistory } from "./api";
import type { SentimentResponse } from "./types";
import { HistoryList } from "./components/HistoryList";

const HISTORY_LIMIT = 20;
const MIN_TEXT_LENGTH = 4;

export default function App() {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const historyQuery = useQuery({
    queryKey: ["history"],
    queryFn: () => fetchHistory(HISTORY_LIMIT)
  });

  const classifyMutation = useMutation({
    mutationFn: classifySentiment,
    onSuccess: (result) => {
      queryClient.setQueryData<SentimentResponse[]>(["history"], (old) => {
        const current = old ?? [];
        return [result, ...current].slice(0, HISTORY_LIMIT);
      });
      setText("");
      setError(null);
    },
    onError: (err: any) => {
      const apiMessage = err?.response?.data?.detail;
      const message = apiMessage || "Không thể phân loại ngay lúc này.";
      window.alert(message);
      setError(message);
    }
  });

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = text.trim();
    if (trimmed.length < MIN_TEXT_LENGTH) {
      const message = `Câu quá ngắn! (>= ${MIN_TEXT_LENGTH} ký tự)`;
      window.alert(message);
      setError(message);
      return;
    }
    classifyMutation.mutate(trimmed);
  };

  return (
    <main className="app-shell">
      <section className="panel">
        <header>
          <h1>Trợ lý phân loại cảm xúc tiếng Việt</h1>
          <p>Dựa trên mô hình Transformer (distilbert-base-multilingual-cased).</p>
        </header>

        <form onSubmit={handleSubmit} className="input-form">
          <textarea
            value={text}
            placeholder="Nhập câu tiếng Việt (ví dụ: \"Hôm nay tôi rất vui\")"
            onChange={(e) => setText(e.target.value)}
            rows={4}
            disabled={classifyMutation.isPending}
          />
          <button type="submit" disabled={classifyMutation.isPending}>
            {classifyMutation.isPending ? "Đang phân loại..." : "Phân loại cảm xúc"}
          </button>
        </form>

        {error && <p className="error-text">{error}</p>}
      </section>

      <section className="panel">
        <div className="history-header">
          <h2>Lịch sử phân loại</h2>
          <button
            type="button"
            className="refresh"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["history"] })}
            disabled={historyQuery.isRefetching}
          >
            Làm mới
          </button>
        </div>
        {historyQuery.isLoading ? (
          <p>Đang tải...</p>
        ) : historyQuery.isError ? (
          <p className="error-text">Không thể tải lịch sử.</p>
        ) : (
          <HistoryList entries={historyQuery.data ?? []} />
        )}
      </section>
    </main>
  );
}

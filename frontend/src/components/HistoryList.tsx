import type { SentimentResponse } from "../types";
import "./HistoryList.css";

interface Props {
  entries: SentimentResponse[];
}

const LABEL_MAP: Record<SentimentResponse["sentiment"], string> = {
  POSITIVE: "Tích cực",
  NEUTRAL: "Trung tính",
  NEGATIVE: "Tiêu cực"
};

export function HistoryList({ entries }: Props) {
  if (!entries.length) {
    return <p className="history-empty">Chưa có lịch sử phân loại.</p>;
  }

  return (
    <div className="history-wrapper">
      <table>
        <thead>
          <tr>
            <th>Câu</th>
            <th>Nhãn</th>
            <th>Độ tin cậy</th>
            <th>Thời gian</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={`${entry.created_at}-${entry.text}`}>
              <td>{entry.text}</td>
              <td>
                <span className={`badge badge-${entry.sentiment.toLowerCase()}`}>
                  {LABEL_MAP[entry.sentiment]}
                </span>
              </td>
              <td>{(entry.confidence * 100).toFixed(1)}%</td>
              <td>{new Date(entry.created_at).toLocaleString("vi-VN")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

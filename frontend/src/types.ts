export type SentimentLabel = "POSITIVE" | "NEUTRAL" | "NEGATIVE";

export interface SentimentResponse {
  text: string;
  sentiment: SentimentLabel;
  confidence: number;
  created_at: string;
}

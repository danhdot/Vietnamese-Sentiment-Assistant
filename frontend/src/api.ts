import axios from "axios";
import type { SentimentResponse } from "./types";

const client = axios.create({
  baseURL: "/api"
});

export async function classifySentiment(text: string): Promise<SentimentResponse> {
  const { data } = await client.post<SentimentResponse>("/sentiment", { text });
  return data;
}

export async function fetchHistory(limit = 20): Promise<SentimentResponse[]> {
  const { data } = await client.get<SentimentResponse[]>("/history", { params: { limit } });
  return data;
}

// types/index.ts
export interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  recommendations?: Laptop[];
  intent?: string;
}

export interface Laptop {
  brand: string;
  model_name: string;
  price_inr: number;
  core: string;
  ram_size: string;
  graphics_processor: string;
  display_size: string;
  storage_type: string;
  special_features: string;
}

export interface ChatRequest {
  message: string;
  session_id: string;
}

export interface ChatResponse {
  response: string;
  recommendations?: Laptop[];
  similar_products?: Laptop[];
  intent: string;
  confidence_score: number;
}
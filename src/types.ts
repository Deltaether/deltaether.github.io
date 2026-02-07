export interface MethodResult {
  data: number[];
  ai_probability: number;  // 0.0 - 1.0
  metrics: [string, number][];
}

export interface AnalysisResult {
  score: number;
  max_score: number;
  verdict: string;
  confidence: string;
  verdict_class: 'high' | 'medium' | 'low';
  noise: MethodResult;
  high_pass: MethodResult;
  ela: MethodResult;
  posterize: MethodResult;
  channels: MethodResult;
  fft: MethodResult;
  fft_size: number;
  noise_patterns: MethodResult;
  gradients: MethodResult;
  clip_space: MethodResult;
}

export interface AnalyzerWasm {
  Analyzer: new (data: Uint8Array, width: number, height: number) => {
    analyze(): AnalysisResult;
  };
  default: () => Promise<void>;
}

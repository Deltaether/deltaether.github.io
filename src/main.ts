import { applyColormap } from './colormap';
import type { AnalysisResult, MethodResult } from './types';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

interface PanelConfig {
  canvasId: string;
  resultId: string;
  colormap: string;
  getResult: (r: AnalysisResult) => MethodResult;
  formatMetrics: (m: MethodResult) => string;
  size?: (r: AnalysisResult) => number;
}

const panelConfigs: PanelConfig[] = [
  {
    canvasId: 'noiseCanvas',
    resultId: 'noiseResult',
    colormap: 'jet',
    getResult: r => r.noise,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)} | STD: ${m.metrics[1][1].toFixed(2)}`,
  },
  {
    canvasId: 'highpassCanvas',
    resultId: 'highpassResult',
    colormap: 'hot',
    getResult: r => r.high_pass,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)} | STD: ${m.metrics[1][1].toFixed(2)}`,
  },
  {
    canvasId: 'elaCanvas',
    resultId: 'elaResult',
    colormap: 'hot',
    getResult: r => r.ela,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)}`,
  },
  {
    canvasId: 'posterizeCanvas',
    resultId: 'posterizeResult',
    colormap: 'viridis',
    getResult: r => r.posterize,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)}`,
  },
  {
    canvasId: 'channelCanvas',
    resultId: 'channelResult',
    colormap: 'plasma',
    getResult: r => r.channels,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)} | STD: ${m.metrics[1][1].toFixed(2)}`,
  },
  {
    canvasId: 'fftCanvas',
    resultId: 'fftResult',
    colormap: 'bone',
    getResult: r => r.fft,
    formatMetrics: m => `Cross Score: ${m.metrics[0][1].toFixed(3)}`,
    size: r => r.fft_size,
  },
  {
    canvasId: 'noisePatternsCanvas',
    resultId: 'noisePatternsResult',
    colormap: 'plasma',
    getResult: r => r.noise_patterns,
    formatMetrics: m => {
      const dir = m.metrics.find(x => x[0] === 'directional_bias')?.[1] ?? 0;
      return `Type: ${m.is_ai ? 'AI' : 'Brush'} | Dir: ${dir.toFixed(3)}`;
    },
  },
  {
    canvasId: 'gradientsCanvas',
    resultId: 'gradientsResult',
    colormap: 'hot',
    getResult: r => r.gradients,
    formatMetrics: m => {
      const ratio = m.metrics.find(x => x[0] === 'perfect_ratio')?.[1] ?? 0;
      return `Perfect: ${(ratio * 100).toFixed(1)}% of gradients`;
    },
  },
  {
    canvasId: 'clipCanvas',
    resultId: 'clipResult',
    colormap: 'viridis',
    getResult: r => r.clip_space,
    formatMetrics: m => {
      const coherence = m.metrics.find(x => x[0] === 'coherence')?.[1] ?? 0;
      return `Coherence: ${(coherence * 100).toFixed(1)}%`;
    },
  },
];

const indicatorNames = [
  'Noise', 'High Pass', 'ELA', 'Banding', 'Channels',
  'FFT Cross', 'Brush', 'Gradients', 'Style',
];

function getIndicatorClass(probability: number): string {
  if (probability >= 0.65) return 'detected';
  if (probability >= 0.35) return 'uncertain';
  return 'clear';
}

let currentWidth = 0;
let currentHeight = 0;
let lastResult: AnalysisResult | null = null;
let worker: Worker | null = null;
let workerReady = false;
let userPrediction: 'human' | 'unsure' | 'ai' | null = null;
let userFeedback: 'human' | 'unsure' | 'ai' | null = null;

function initWorker(): Promise<void> {
  return new Promise((resolve, reject) => {
    // Worker is pre-built to js/worker.js (absolute path from root)
    worker = new Worker('/js/worker.js', { type: 'module' });
    
    const timeout = setTimeout(() => {
      reject(new Error('Worker initialization timeout'));
    }, 10000);

    worker.onmessage = (e) => {
      if (e.data.type === 'ready') {
        clearTimeout(timeout);
        workerReady = true;
        resolve();
      }
    };

    worker.onerror = (err) => {
      clearTimeout(timeout);
      reject(err);
    };
  });
}

function handleFile(file: File): void {
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file.');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    alert('Image too large. Max 10MB.');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => displayPreview(img, file);
    img.src = e.target?.result as string;
  };
  reader.readAsDataURL(file);
}

function displayPreview(img: HTMLImageElement, file: File): void {
  // No size limit - let's see what it can handle
  currentWidth = img.width;
  currentHeight = img.height;

  const canvas = $<HTMLCanvasElement>('originalCanvas');
  canvas.width = currentWidth;
  canvas.height = currentHeight;

  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, currentWidth, currentHeight);

  const fileSize = (file.size / 1024).toFixed(1);
  $('imageInfo').textContent = `${img.width}x${img.height} | ${file.type} | ${fileSize} KB`;

  // Reset prediction state
  userPrediction = null;
  userFeedback = null;
  $('predictionPrompt').classList.remove('hidden');
  $('analyzeBtn').classList.add('hidden');
  document.querySelectorAll('.predict-btn').forEach(btn => btn.classList.remove('selected'));

  $('upload-section').classList.add('hidden');
  $('preview-section').classList.remove('hidden');
}

function updateProgress(step: string): void {
  $('loaderStep').textContent = step;
}

async function runAnalysis(): Promise<void> {
  if (!worker || !workerReady) {
    alert('Analyzer not ready. Please refresh the page.');
    return;
  }

  $('preview-section').classList.add('hidden');
  $('loading-section').classList.remove('hidden');
  $('progressFill').style.width = '0%';
  updateProgress('Preparing image...');

  const canvas = $<HTMLCanvasElement>('originalCanvas');
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.getImageData(0, 0, currentWidth, currentHeight);

  // Send to worker (runs in background thread)
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Analysis timeout'));
      $('loading-section').classList.add('hidden');
      $('preview-section').classList.remove('hidden');
      alert('Analysis took too long. Try a smaller image.');
    }, 60000);

    worker!.onmessage = (e) => {
      const { type, step, result, error } = e.data;

      switch (type) {
        case 'progress':
          updateProgress(step);
          $('progressFill').style.width = '50%';
          break;

        case 'result':
          clearTimeout(timeout);
          $('progressFill').style.width = '100%';
          updateProgress('Rendering...');
          lastResult = result;
          setTimeout(() => {
            displayResults(result);
            sendAnalytics(result); // Send anonymous metrics
            resolve();
          }, 50);
          break;

        case 'error':
          clearTimeout(timeout);
          $('loading-section').classList.add('hidden');
          $('preview-section').classList.remove('hidden');
          alert(`Analysis failed: ${error}`);
          reject(new Error(error));
          break;
      }
    };

    // Transfer the buffer for better performance
    const buffer = imgData.data.buffer.slice(0);
    worker!.postMessage(
      { type: 'analyze', data: buffer, width: currentWidth, height: currentHeight },
      [buffer]
    );
  });
}

function displayResults(result: AnalysisResult): void {
  $('loading-section').classList.add('hidden');
  $('results-section').classList.remove('hidden');

  // Reset feedback UI
  document.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('selected'));
  $('feedbackThanks').classList.add('hidden');

  $('scoreValue').textContent = result.score.toString();
  $('scoreCircle').className = `score-circle ${result.verdict_class}`;
  $('verdictText').textContent = result.verdict;
  $('confidenceText').textContent = `Confidence: ${result.confidence}`;

  // Indicators with probability percentages
  const indicatorsList = $('indicatorsList');
  indicatorsList.innerHTML = '';

  const results = [
    result.noise, result.high_pass, result.ela, result.posterize,
    result.channels, result.fft, result.noise_patterns, result.gradients,
    result.clip_space,
  ];

  indicatorNames.forEach((name, i) => {
    const prob = results[i].ai_probability;
    const pct = Math.round(prob * 100);
    const tag = document.createElement('span');
    tag.className = `indicator-tag ${getIndicatorClass(prob)}`;
    tag.innerHTML = `${pct}% ${name}`;
    indicatorsList.appendChild(tag);
  });

  // Render panels
  panelConfigs.forEach((panel) => {
    const canvas = $<HTMLCanvasElement>(panel.canvasId);
    const resultEl = $(panel.resultId);
    const methodResult = panel.getResult(result);
    const size = panel.size ? panel.size(result) : currentWidth;
    const w = panel.size ? size : currentWidth;
    const h = panel.size ? size : currentHeight;

    const prob = methodResult.ai_probability;
    const pct = Math.round(prob * 100);
    
    renderCanvas(canvas, methodResult.data, w, h, panel.colormap);
    resultEl.textContent = `${pct}% AI | ${panel.formatMetrics(methodResult)}`;
    resultEl.className = `panel-result ${getIndicatorClass(prob)}`;
  });
}

function renderCanvas(
  canvas: HTMLCanvasElement,
  grayData: number[],
  width: number,
  height: number,
  colormapName: string
): void {
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d')!;
  const coloredData = applyColormap(grayData, width, height, colormapName);
  const imgData = new ImageData(coloredData, width, height);
  ctx.putImageData(imgData, 0, 0);
}

const ANALYTICS_URL = 'https://script.google.com/macros/s/AKfycbwMJcZuj9hxCrGIKRqeZoE_ctpF4gtBpiqBXPhN3PvgbHqjFh1M3I1YohpGC7qXPFfbFQ/exec';

function buildExportData(result: AnalysisResult, includeFeedback = false) {
  const data: Record<string, unknown> = {
    timestamp: new Date().toISOString(),
    dimensions: { width: currentWidth, height: currentHeight },
    score: result.score,
    verdict: result.verdict,
    confidence: result.confidence,
    methods: {
      noise: {
        ai_probability: result.noise.ai_probability,
        metrics: Object.fromEntries(result.noise.metrics)
      },
      high_pass: {
        ai_probability: result.high_pass.ai_probability,
        metrics: Object.fromEntries(result.high_pass.metrics)
      },
      ela: {
        ai_probability: result.ela.ai_probability,
        metrics: Object.fromEntries(result.ela.metrics)
      },
      posterize: {
        ai_probability: result.posterize.ai_probability,
        metrics: Object.fromEntries(result.posterize.metrics)
      },
      channels: {
        ai_probability: result.channels.ai_probability,
        metrics: Object.fromEntries(result.channels.metrics)
      },
      fft: {
        ai_probability: result.fft.ai_probability,
        metrics: Object.fromEntries(result.fft.metrics)
      },
      noise_patterns: {
        ai_probability: result.noise_patterns.ai_probability,
        metrics: Object.fromEntries(result.noise_patterns.metrics)
      },
      gradients: {
        ai_probability: result.gradients.ai_probability,
        metrics: Object.fromEntries(result.gradients.metrics)
      },
      clip_space: {
        ai_probability: result.clip_space.ai_probability,
        metrics: Object.fromEntries(result.clip_space.metrics)
      }
    }
  };
  
  if (includeFeedback) {
    data.user_prediction = userPrediction;
    data.user_feedback = userFeedback;
  }
  
  return data;
}

async function sendAnalytics(result: AnalysisResult, withFeedback = false): Promise<void> {
  try {
    const data = buildExportData(result, withFeedback);
    await fetch(ANALYTICS_URL, {
      method: 'POST',
      mode: 'no-cors', // Apps Script doesn't support CORS preflight
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  } catch (e) {
    // Silent fail - don't interrupt user experience
    console.debug('Analytics send failed:', e);
  }
}

function exportResults(): void {
  if (!lastResult) return;

  const exportData = buildExportData(lastResult);

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ai-detector-results-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function reset(): void {
  $('results-section').classList.add('hidden');
  $('preview-section').classList.add('hidden');
  $('upload-section').classList.remove('hidden');
  lastResult = null;
  userPrediction = null;
  userFeedback = null;

  // Reset UI state
  const input = $<HTMLInputElement>('imageInput');
  input.value = '';
  document.querySelectorAll('.predict-btn, .feedback-btn').forEach(b => b.classList.remove('selected'));
  $('feedbackThanks').classList.add('hidden');
}

async function init(): Promise<void> {
  const imageInput = $<HTMLInputElement>('imageInput');
  const selectBtn = $('selectBtn');
  const dropzone = $('dropzone');
  const analyzeBtn = $('analyzeBtn');
  const resetBtn = $('resetBtn');
  const exportBtn = $('exportBtn');

  const openFilePicker = () => {
    imageInput.click();
  };

  selectBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    openFilePicker();
  });

  dropzone.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    if (target.id === 'selectBtn' || target.closest('#selectBtn')) return;
    openFilePicker();
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  imageInput.addEventListener('change', (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  analyzeBtn.addEventListener('click', runAnalysis);
  resetBtn.addEventListener('click', reset);
  exportBtn.addEventListener('click', exportResults);

  // Prediction buttons (before analysis)
  document.querySelectorAll('.predict-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const prediction = target.dataset.prediction as 'human' | 'unsure' | 'ai';
      userPrediction = prediction;
      
      // Update UI
      document.querySelectorAll('.predict-btn').forEach(b => b.classList.remove('selected'));
      target.classList.add('selected');
      
      // Show analyze button
      $('analyzeBtn').classList.remove('hidden');
    });
  });

  // Feedback buttons (after analysis)
  document.querySelectorAll('.feedback-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const feedback = target.dataset.feedback as 'human' | 'unsure' | 'ai';
      userFeedback = feedback;
      
      // Update UI
      document.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('selected'));
      target.classList.add('selected');
      $('feedbackThanks').classList.remove('hidden');
      
      // Send analytics with feedback
      if (lastResult) {
        sendAnalytics(lastResult, true);
      }
    });
  });

  // Initialize worker
  try {
    await initWorker();
  } catch (err) {
    console.error('Failed to initialize worker:', err);
    alert('Failed to initialize analyzer. Please refresh the page.');
  }
}

document.addEventListener('DOMContentLoaded', init);

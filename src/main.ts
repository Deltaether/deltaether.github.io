import { applyColormap } from './colormap';
import type { AnalysisResult, MethodResult } from './types';

// Will be loaded dynamically
let Analyzer: any;

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

interface PanelConfig {
  canvas: HTMLCanvasElement;
  result: HTMLElement;
  colormap: string;
  getResult: (r: AnalysisResult) => MethodResult;
  formatMetrics: (m: MethodResult) => string;
  size?: (r: AnalysisResult) => number;
}

const panels: PanelConfig[] = [
  {
    canvas: $('noiseCanvas'),
    result: $('noiseResult'),
    colormap: 'jet',
    getResult: r => r.noise,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)} | STD: ${m.metrics[1][1].toFixed(2)}`,
  },
  {
    canvas: $('highpassCanvas'),
    result: $('highpassResult'),
    colormap: 'hot',
    getResult: r => r.high_pass,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)} | STD: ${m.metrics[1][1].toFixed(2)}`,
  },
  {
    canvas: $('elaCanvas'),
    result: $('elaResult'),
    colormap: 'hot',
    getResult: r => r.ela,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)}`,
  },
  {
    canvas: $('posterizeCanvas'),
    result: $('posterizeResult'),
    colormap: 'viridis',
    getResult: r => r.posterize,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)}`,
  },
  {
    canvas: $('channelCanvas'),
    result: $('channelResult'),
    colormap: 'plasma',
    getResult: r => r.channels,
    formatMetrics: m => `Mean: ${m.metrics[0][1].toFixed(2)} | STD: ${m.metrics[1][1].toFixed(2)}`,
  },
  {
    canvas: $('fftCanvas'),
    result: $('fftResult'),
    colormap: 'bone',
    getResult: r => r.fft,
    formatMetrics: m => `Cross Score: ${m.metrics[0][1].toFixed(3)}`,
    size: r => r.fft_size,
  },
  {
    canvas: $('noisePatternsCanvas'),
    result: $('noisePatternsResult'),
    colormap: 'plasma',
    getResult: r => r.noise_patterns,
    formatMetrics: m => {
      const uniformity = m.metrics.find(x => x[0] === 'uniformity')?.[1] ?? 0;
      const dir = m.metrics.find(x => x[0] === 'directional_bias')?.[1] ?? 0;
      return `Type: ${m.is_ai ? 'AI' : 'Brush'} | Dir: ${dir.toFixed(3)}`;
    },
  },
  {
    canvas: $('gradientsCanvas'),
    result: $('gradientsResult'),
    colormap: 'hot',
    getResult: r => r.gradients,
    formatMetrics: m => {
      const ratio = m.metrics.find(x => x[0] === 'perfect_ratio')?.[1] ?? 0;
      return `Perfect: ${(ratio * 100).toFixed(1)}% of gradients`;
    },
  },
  {
    canvas: $('clipCanvas'),
    result: $('clipResult'),
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

let currentWidth = 0;
let currentHeight = 0;

async function loadWasm(): Promise<void> {
  const wasm = await import('../pkg/ai_art_analyzer.js');
  await wasm.default();
  Analyzer = wasm.Analyzer;
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
  const maxSize = 800;
  let width = img.width;
  let height = img.height;

  if (width > maxSize || height > maxSize) {
    if (width > height) {
      height = (height / width) * maxSize;
      width = maxSize;
    } else {
      width = (width / height) * maxSize;
      height = maxSize;
    }
  }

  currentWidth = Math.floor(width);
  currentHeight = Math.floor(height);

  const canvas = $<HTMLCanvasElement>('originalCanvas');
  canvas.width = currentWidth;
  canvas.height = currentHeight;

  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, currentWidth, currentHeight);

  const fileSize = (file.size / 1024).toFixed(1);
  $('imageInfo').textContent = `${img.width}x${img.height} | ${file.type} | ${fileSize} KB`;

  $('upload-section').classList.add('hidden');
  $('preview-section').classList.remove('hidden');
}

async function runAnalysis(): Promise<void> {
  $('preview-section').classList.add('hidden');
  $('loading-section').classList.remove('hidden');

  await new Promise(r => setTimeout(r, 50));

  const canvas = $<HTMLCanvasElement>('originalCanvas');
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.getImageData(0, 0, currentWidth, currentHeight);

  $('loaderStep').textContent = 'Running Rust analyzer...';
  await new Promise(r => setTimeout(r, 50));

  const analyzer = new Analyzer(
    new Uint8Array(imgData.data.buffer),
    currentWidth,
    currentHeight
  );

  const result: AnalysisResult = analyzer.analyze();

  displayResults(result);
}

function displayResults(result: AnalysisResult): void {
  $('loading-section').classList.add('hidden');
  $('results-section').classList.remove('hidden');

  $('scoreValue').textContent = result.score.toString();
  $('scoreCircle').className = `score-circle ${result.verdict_class}`;
  $('verdictText').textContent = result.verdict;
  $('confidenceText').textContent = `Confidence: ${result.confidence}`;

  // Indicators
  const indicatorsList = $('indicatorsList');
  indicatorsList.innerHTML = '';

  const results = [
    result.noise, result.high_pass, result.ela, result.posterize,
    result.channels, result.fft, result.noise_patterns, result.gradients,
    result.clip_space,
  ];

  indicatorNames.forEach((name, i) => {
    const tag = document.createElement('span');
    tag.className = `indicator-tag ${results[i].is_ai ? 'detected' : 'clear'}`;
    tag.innerHTML = `${results[i].is_ai ? '!' : '+'} ${name}`;
    indicatorsList.appendChild(tag);
  });

  // Render panels
  panels.forEach((panel, i) => {
    const methodResult = panel.getResult(result);
    const size = panel.size ? panel.size(result) : currentWidth;
    const w = panel.size ? size : currentWidth;
    const h = panel.size ? size : currentHeight;

    renderCanvas(panel.canvas, methodResult.data, w, h, panel.colormap);
    panel.result.textContent = panel.formatMetrics(methodResult);
    panel.result.className = `panel-result ${methodResult.is_ai ? 'detected' : 'clear'}`;
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

function reset(): void {
  $('results-section').classList.add('hidden');
  $('preview-section').classList.add('hidden');
  $('upload-section').classList.remove('hidden');

  const input = $<HTMLInputElement>('imageInput');
  input.value = '';
}

async function init(): Promise<void> {
  await loadWasm();

  const imageInput = $<HTMLInputElement>('imageInput');
  const selectBtn = $('selectBtn');
  const dropzone = $('dropzone');
  const analyzeBtn = $('analyzeBtn');
  const resetBtn = $('resetBtn');

  selectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    imageInput.click();
  });

  dropzone.addEventListener('click', () => imageInput.click());

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
}

document.addEventListener('DOMContentLoaded', init);

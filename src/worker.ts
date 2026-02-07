// Web Worker for WASM image analysis
// Runs in separate thread to avoid blocking UI

import init, { Analyzer } from '../pkg/ai_art_analyzer.js';

let wasmReady = false;

// Initialize WASM when worker starts
async function initWasm() {
  try {
    // Absolute path from root
    await init('/pkg/ai_art_analyzer_bg.wasm');
    wasmReady = true;
    self.postMessage({ type: 'ready' });
  } catch (err) {
    console.error('WASM init error:', err);
    self.postMessage({ type: 'error', error: 'Failed to initialize WASM: ' + String(err) });
  }
}

// Handle messages from main thread
self.onmessage = async (e: MessageEvent) => {
  const { type, data, width, height } = e.data;

  console.log('[Worker] Received message:', type, 'size:', width, 'x', height);

  if (type === 'analyze') {
    if (!wasmReady) {
      self.postMessage({ type: 'error', error: 'WASM not ready' });
      return;
    }

    try {
      self.postMessage({ type: 'progress', step: 'Creating analyzer...' });
      console.log('[Worker] Creating analyzer with', data.byteLength, 'bytes');
      
      const analyzer = new Analyzer(
        new Uint8Array(data),
        width,
        height
      );

      self.postMessage({ type: 'progress', step: 'Running 9 detection methods...' });
      console.log('[Worker] Starting analysis...');
      
      const startTime = performance.now();
      const result = analyzer.analyze();
      const elapsed = performance.now() - startTime;
      
      console.log('[Worker] Analysis completed in', elapsed.toFixed(0), 'ms');
      
      self.postMessage({ type: 'result', result });
      
      // Clean up
      analyzer.free();
    } catch (err) {
      console.error('[Worker] Error:', err);
      self.postMessage({ type: 'error', error: String(err) });
    }
  }
};

// Start initialization
initWasm();

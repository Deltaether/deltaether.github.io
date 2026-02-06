type RGB = [number, number, number];
type ColormapFn = (value: number) => RGB;

const clamp = (v: number): number => Math.max(0, Math.min(1, v));

const jet: ColormapFn = (value) => {
  value = clamp(value);
  let r: number, g: number, b: number;

  if (value < 0.125) {
    r = 0; g = 0; b = 0.5 + value * 4;
  } else if (value < 0.375) {
    r = 0; g = (value - 0.125) * 4; b = 1;
  } else if (value < 0.625) {
    r = (value - 0.375) * 4; g = 1; b = 1 - (value - 0.375) * 4;
  } else if (value < 0.875) {
    r = 1; g = 1 - (value - 0.625) * 4; b = 0;
  } else {
    r = 1 - (value - 0.875) * 4; g = 0; b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
};

const hot: ColormapFn = (value) => {
  value = clamp(value);
  let r: number, g: number, b: number;

  if (value < 0.33) {
    r = value * 3; g = 0; b = 0;
  } else if (value < 0.67) {
    r = 1; g = (value - 0.33) * 3; b = 0;
  } else {
    r = 1; g = 1; b = (value - 0.67) * 3;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
};

const interpolateColors = (colors: RGB[], value: number): RGB => {
  value = clamp(value);
  const idx = value * (colors.length - 1);
  const low = Math.floor(idx);
  const high = Math.min(low + 1, colors.length - 1);
  const t = idx - low;

  return [
    Math.round(colors[low][0] + t * (colors[high][0] - colors[low][0])),
    Math.round(colors[low][1] + t * (colors[high][1] - colors[low][1])),
    Math.round(colors[low][2] + t * (colors[high][2] - colors[low][2])),
  ];
};

const viridis: ColormapFn = (value) => {
  const colors: RGB[] = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
  ];
  return interpolateColors(colors, value);
};

const plasma: ColormapFn = (value) => {
  const colors: RGB[] = [
    [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40],
    [240, 249, 33],
  ];
  return interpolateColors(colors, value);
};

const bone: ColormapFn = (value) => {
  value = clamp(value);
  let r: number, g: number, b: number;

  if (value < 0.375) {
    r = value * 7 / 8;
    g = value * 7 / 8;
    b = value * 7 / 8 + value / 3;
  } else if (value < 0.75) {
    r = value * 7 / 8;
    g = value * 7 / 8 + (value - 0.375) / 3;
    b = 0.375 * 7 / 8 + 0.375 / 3 + (value - 0.375) * 7 / 8;
  } else {
    r = 0.75 * 7 / 8 + (value - 0.75) / 3 + (value - 0.75) * 7 / 8;
    g = 0.75 * 7 / 8 + 0.375 / 3 + (value - 0.75) * 7 / 8;
    b = 0.375 * 7 / 8 + 0.375 / 3 + 0.375 * 7 / 8 + (value - 0.75) * 7 / 8;
  }

  return [
    Math.round(Math.min(1, r) * 255),
    Math.round(Math.min(1, g) * 255),
    Math.round(Math.min(1, b) * 255),
  ];
};

const colormaps: Record<string, ColormapFn> = { jet, hot, viridis, plasma, bone };

export function applyColormap(
  grayData: number[],
  width: number,
  height: number,
  colormapName: string
): Uint8ClampedArray {
  const colormap = colormaps[colormapName] || jet;
  const result = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < grayData.length; i++) {
    const value = grayData[i] / 255;
    const [r, g, b] = colormap(value);
    const idx = i * 4;
    result[idx] = r;
    result[idx + 1] = g;
    result[idx + 2] = b;
    result[idx + 3] = 255;
  }

  return result;
}

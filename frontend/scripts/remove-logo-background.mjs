import sharp from 'sharp';
import { fileURLToPath } from 'node:url';

const path = fileURLToPath(new URL('../public/statics/logo.png', import.meta.url));
const { data, info } = await sharp(path).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const seen = new Uint8Array(info.width * info.height);
const queue = [];

const isBackground = index => {
  const offset = index * 4;
  const r = data[offset], g = data[offset + 1], b = data[offset + 2];
  return Math.min(r, g, b) >= 218 && Math.max(r, g, b) - Math.min(r, g, b) <= 22;
};
const add = index => {
  if (!seen[index] && isBackground(index)) {
    seen[index] = 1;
    queue.push(index);
  }
};
for (let x = 0; x < info.width; x++) { add(x); add((info.height - 1) * info.width + x); }
for (let y = 0; y < info.height; y++) { add(y * info.width); add(y * info.width + info.width - 1); }

for (let cursor = 0; cursor < queue.length; cursor++) {
  const index = queue[cursor];
  const x = index % info.width, y = Math.floor(index / info.width);
  data[index * 4 + 3] = 0;
  if (x > 0) add(index - 1);
  if (x + 1 < info.width) add(index + 1);
  if (y > 0) add(index - info.width);
  if (y + 1 < info.height) add(index + info.width);
}

await sharp(data, { raw: info }).trim({ background: { r: 0, g: 0, b: 0, alpha: 0 } }).png().toFile(fileURLToPath(new URL('../public/statics/logo-clean.png', import.meta.url)));

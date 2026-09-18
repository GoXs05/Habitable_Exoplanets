/* Procedural planet viewport.
 *
 * Renders a shaded sphere into a 2D canvas, one pixel at a time, and lets the
 * user drag it to rotate. anime.js drives every transition: parameter tweens
 * when a slider moves, inertial spin-down on release, and a morph pulse when
 * the planet type changes.
 *
 * Performance shape: the expensive part (fractal noise) is baked once into a
 * scalar equirectangular map and only regenerated when the planet TYPE changes.
 * Everything else -- temperature, mass, radius, habitability -- is a cheap
 * 256-entry colour lookup rebuilt per frame, so dragging stays at 60fps.
 */

(function (global) {
  'use strict';

  const TEX_W = 320, TEX_H = 160;
  // The per-pixel loop only walks the sphere's bounding box, so cost tracks the
  // planet's on-screen diameter, not the canvas area. Cap the SHORTER canvas
  // dimension (which is what bounds the sphere) and let the long side follow
  // the aspect ratio -- the extra width is just starfield, which is cheap.
  const MAX_MIN_DIM = 360;
  const MAX_PIXELS = 480000;       // backstop for extreme aspect ratios
  const TAU = Math.PI * 2;

  /* ───────────────────────── noise ───────────────────────── */

  function hash3(i, j, k) {
    let n = (i * 374761393 + j * 668265263 + k * 1274126177) | 0;
    n = (n ^ (n >>> 13)) * 1274126177;
    n = (n ^ (n >>> 16)) >>> 0;
    return n / 4294967295;
  }

  const smooth = (t) => t * t * (3 - 2 * t);

  function valueNoise3(x, y, z) {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    const xf = smooth(x - xi), yf = smooth(y - yi), zf = smooth(z - zi);

    const c000 = hash3(xi, yi, zi), c100 = hash3(xi + 1, yi, zi);
    const c010 = hash3(xi, yi + 1, zi), c110 = hash3(xi + 1, yi + 1, zi);
    const c001 = hash3(xi, yi, zi + 1), c101 = hash3(xi + 1, yi, zi + 1);
    const c011 = hash3(xi, yi + 1, zi + 1), c111 = hash3(xi + 1, yi + 1, zi + 1);

    const x00 = c000 + (c100 - c000) * xf, x10 = c010 + (c110 - c010) * xf;
    const x01 = c001 + (c101 - c001) * xf, x11 = c011 + (c111 - c011) * xf;
    const y0 = x00 + (x10 - x00) * yf, y1 = x01 + (x11 - x01) * yf;
    return y0 + (y1 - y0) * zf;
  }

  function fbm(x, y, z, octaves, seed) {
    let sum = 0, amp = 0.5, freq = 1, norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * valueNoise3(x * freq + seed, y * freq + seed * 1.7, z * freq + seed * 2.3);
      norm += amp;
      amp *= 0.5;
      freq *= 2.02;
    }
    return sum / norm;
  }

  /* ───────────────────────── palettes ───────────────────────── */

  // [position, r, g, b] stops; positions ascending 0..1.
  const P = {
    frozen: [[0, 14, 34, 60], [0.40, 30, 70, 108], [0.48, 96, 150, 186],
             [0.56, 190, 220, 236], [0.74, 232, 242, 248], [1, 255, 255, 255]],
    temperate: [[0, 8, 28, 58], [0.36, 16, 68, 120], [0.47, 30, 108, 160],
                [0.50, 198, 182, 134], [0.56, 62, 112, 58], [0.72, 96, 122, 60],
                [0.86, 122, 106, 84], [1, 242, 246, 250]],
    arid: [[0, 52, 30, 20], [0.34, 98, 58, 34], [0.55, 148, 94, 48],
           [0.76, 188, 134, 76], [1, 216, 180, 130]],
    molten: [[0, 16, 10, 10], [0.34, 46, 20, 16], [0.60, 96, 30, 16],
             [0.78, 198, 70, 18], [0.90, 248, 150, 32], [1, 255, 228, 142]],
    jovian: [[0, 116, 76, 46], [0.24, 168, 118, 74], [0.44, 216, 180, 132],
             [0.60, 238, 216, 180], [0.78, 206, 148, 94], [1, 246, 236, 212]],
    jovianCold: [[0, 118, 134, 152], [0.28, 168, 186, 200], [0.5, 212, 226, 236],
                 [0.72, 236, 244, 250], [1, 255, 255, 255]],
    neptunian: [[0, 10, 38, 90], [0.30, 26, 72, 142], [0.55, 48, 118, 190],
                [0.76, 112, 172, 222], [1, 200, 228, 244]],
  };

  function buildLUT(stops) {
    const lut = new Uint8ClampedArray(768);
    let s = 0;
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      while (s < stops.length - 2 && t > stops[s + 1][0]) s++;
      const a = stops[s], b = stops[s + 1];
      const span = b[0] - a[0] || 1;
      const k = Math.max(0, Math.min(1, (t - a[0]) / span));
      lut[i * 3] = a[1] + (b[1] - a[1]) * k;
      lut[i * 3 + 1] = a[2] + (b[2] - a[2]) * k;
      lut[i * 3 + 2] = a[3] + (b[3] - a[3]) * k;
    }
    return lut;
  }

  const LUT = {};
  for (const key in P) LUT[key] = buildLUT(P[key]);

  function mixLUT(out, a, b, t) {
    for (let i = 0; i < 768; i++) out[i] = a[i] + (b[i] - a[i]) * t;
    return out;
  }

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const smoothstep = (e0, e1, v) => clamp01((v - e0) / (e1 - e0));

  const GAS = { Jovian: 1, Neptunian: 1 };

  /* Which two palettes the current temperature sits between, and how far. */
  function regimeFor(type, temp) {
    if (type === 'Jovian') {
      if (temp > 700) return { a: LUT.jovian, b: LUT.molten, t: smoothstep(700, 1400, temp), molten: smoothstep(700, 1400, temp) };
      return { a: LUT.jovianCold, b: LUT.jovian, t: smoothstep(120, 320, temp), molten: 0 };
    }
    if (type === 'Neptunian') {
      if (temp > 700) return { a: LUT.neptunian, b: LUT.molten, t: smoothstep(700, 1500, temp), molten: smoothstep(700, 1500, temp) };
      return { a: LUT.jovianCold, b: LUT.neptunian, t: smoothstep(100, 260, temp), molten: 0 };
    }
    // Rocky worlds walk frozen -> temperate -> arid -> molten.
    if (temp < 245) return { a: LUT.frozen, b: LUT.temperate, t: smoothstep(190, 245, temp), molten: 0 };
    if (temp < 340) return { a: LUT.temperate, b: LUT.temperate, t: 0, molten: 0 };
    if (temp < 800) return { a: LUT.temperate, b: LUT.arid, t: smoothstep(340, 560, temp), molten: 0 };
    return { a: LUT.arid, b: LUT.molten, t: smoothstep(800, 1300, temp), molten: smoothstep(850, 1400, temp) };
  }

  /* ───────────────────────── terrain maps ───────────────────────── */

  function generateMaps(type, seed) {
    const height = new Float32Array(TEX_W * TEX_H);
    const cloud = new Float32Array(TEX_W * TEX_H);
    const gas = !!GAS[type];

    // Smaller worlds get busier, higher-frequency terrain.
    const freq = gas ? 1.6 : 2.5;
    const octaves = gas ? 4 : 5;
    const bands = type === 'Jovian' ? 9 : 6;

    let i = 0;
    for (let v = 0; v < TEX_H; v++) {
      const lat = (0.5 - (v + 0.5) / TEX_H) * Math.PI;   // +pi/2 .. -pi/2
      const cy = Math.cos(lat), sy = Math.sin(lat);
      for (let u = 0; u < TEX_W; u++, i++) {
        const lon = ((u + 0.5) / TEX_W - 0.5) * TAU;
        const px = cy * Math.sin(lon), py = sy, pz = cy * Math.cos(lon);

        let h;
        if (gas) {
          // Latitude bands, warped by turbulence so they swirl rather than stripe.
          const warp = fbm(px * 2.2, py * 3.4, pz * 2.2, octaves, seed) - 0.5;
          const b = Math.sin(lat * bands + warp * 5.2);
          const detail = fbm(px * freq * 3, py * freq * 3, pz * freq * 3, 3, seed + 11) - 0.5;
          h = clamp01(0.5 + b * 0.34 + detail * 0.26);
        } else {
          const n = fbm(px * freq, py * freq, pz * freq, octaves, seed);
          const ridge = 1 - Math.abs(fbm(px * freq * 2, py * freq * 2, pz * freq * 2, 3, seed + 7) - 0.5) * 2;
          h = n * 0.78 + ridge * 0.22;
          // Summed octaves cluster hard around 0.5, which leaves almost no
          // ocean and flattens the land into one colour. Expand the contrast
          // and bias low so temperate worlds read as water worlds with
          // continents, the way Earth does.
          h = clamp01((h - 0.5) * 2.3 + 0.43);
          // Polar caps read as "planet", not "noise ball".
          h = clamp01(h + smoothstep(0.68, 1, Math.abs(sy)) * 0.26);
        }
        height[i] = h;

        const c = fbm(px * 3.1, py * 3.1, pz * 3.1, 4, seed + 31);
        cloud[i] = clamp01((c - 0.48) * 3.1);
      }
    }
    return { height, cloud };
  }

  /* ───────────────────────── bilinear sampling ───────────────────────── */

  function sample(map, u, v) {
    let x0 = Math.floor(u), y0 = Math.floor(v);
    const fx = u - x0, fy = v - y0;
    let x1 = x0 + 1, y1 = y0 + 1;
    x0 = ((x0 % TEX_W) + TEX_W) % TEX_W;
    x1 = ((x1 % TEX_W) + TEX_W) % TEX_W;
    if (y0 < 0) y0 = 0; else if (y0 >= TEX_H) y0 = TEX_H - 1;
    if (y1 < 0) y1 = 0; else if (y1 >= TEX_H) y1 = TEX_H - 1;
    const r0 = y0 * TEX_W, r1 = y1 * TEX_W;
    const a = map[r0 + x0], b = map[r0 + x1], c = map[r1 + x0], d = map[r1 + x1];
    const top = a + (b - a) * fx, bot = c + (d - c) * fx;
    return top + (bot - top) * fy;
  }

  /* ───────────────────────── the view ───────────────────────── */

  function create(canvas, options) {
    const opts = options || {};
    const ctx = canvas.getContext('2d');
    const reduceMotion = global.matchMedia
      && global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const dur = (ms) => (reduceMotion ? 0 : ms);

    const view = {
      // Tweened visual state. These live on SEPARATE objects on purpose:
      // anime.remove() cancels every animation on a target, with no
      // per-property granularity, so sharing one object means the morph pulse
      // would cancel the intro fade and leave the planet at alpha 0.
      p: {
        size: 0.6, flat: 0.02, temp: 288, atmo: 0.12, spin: 0.18,
        water: 0, cloudAmt: 0.35, ring: 0,
      },
      intro: { v: 0 },
      wobble: { v: 0 },
      life: { v: 0 },
      type: null,
      seed: 7,
      maps: null,
      yaw: 0.6, pitch: -0.22,
      spinVel: 0, dragging: false,
      lut: new Uint8ClampedArray(768),
      emis: new Float32Array(256),
      lutTemp: null, lutType: null, lutWater: null,
      buffer: null, bufW: 0, bufH: 0, img: null,
      cssW: 0, cssH: 0,
      stars: null,
      cloudPhase: 0,
      visible: true,
      raf: 0,
      lastT: performance.now(),
    };

    /* ── palette ── */
    function rebuildLUT() {
      const type = view.type || 'Terran';
      const r = regimeFor(type, view.p.temp);
      mixLUT(view.lut, r.a, r.b, r.t);

      // Habitable-looking worlds get a faint biosphere tint on the mid-band.
      const life = view.life.v;
      if (life > 0.01 && !GAS[type]) {
        for (let i = 128; i < 224; i++) {
          const w = life * 0.30 * Math.sin(((i - 128) / 96) * Math.PI);
          view.lut[i * 3] += (58 - view.lut[i * 3]) * w;
          view.lut[i * 3 + 1] += (128 - view.lut[i * 3 + 1]) * w;
          view.lut[i * 3 + 2] += (62 - view.lut[i * 3 + 2]) * w;
        }
      }

      for (let i = 0; i < 256; i++) {
        view.emis[i] = r.molten * smoothstep(0.42, 1, i / 255) * 1.25;
      }
      view.lutTemp = view.p.temp;
      view.lutType = type;
      view.lutWater = view.p.water;
      view.lutLife = view.life.v;
    }

    /* ── sizing ── */
    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      view.cssW = rect.width;
      view.cssH = rect.height;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);

      const scale = Math.min(1, MAX_MIN_DIM / Math.min(canvas.width, canvas.height));
      let bw = Math.max(2, Math.round(canvas.width * scale));
      let bh = Math.max(2, Math.round(canvas.height * scale));
      if (bw * bh > MAX_PIXELS) {
        const k = Math.sqrt(MAX_PIXELS / (bw * bh));
        bw = Math.max(2, Math.round(bw * k));
        bh = Math.max(2, Math.round(bh * k));
      }
      view.bufW = bw;
      view.bufH = bh;
      view.img = ctx.createImageData(view.bufW, view.bufH);
      view.buffer = view.img.data;
      view.stars = makeStars(view.bufW, view.bufH);
    }

    function makeStars(w, h) {
      const n = Math.round((w * h) / 2600);
      const out = new Float32Array(n * 3);
      let s = 1337;
      const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
      for (let i = 0; i < n; i++) {
        out[i * 3] = rnd() * w;
        out[i * 3 + 1] = rnd() * h;
        out[i * 3 + 2] = 0.25 + rnd() * 0.75;
      }
      return out;
    }

    /* ── the per-pixel render ── */
    function render(dt) {
      if (!view.buffer || !view.maps) return;

      if (view.p.temp !== view.lutTemp || view.type !== view.lutType
          || view.p.water !== view.lutWater || view.life.v !== view.lutLife) rebuildLUT();

      const W = view.bufW, H = view.bufH, buf = view.buffer;
      buf.fill(0);

      // Starfield behind everything.
      const stars = view.stars;
      for (let i = 0; i < stars.length; i += 3) {
        const x = stars[i] | 0, y = stars[i + 1] | 0, b = stars[i + 2] * 205;
        const o = (y * W + x) * 4;
        buf[o] = b; buf[o + 1] = b; buf[o + 2] = b * 1.02; buf[o + 3] = 255;
      }

      const cx = W / 2, cy = H / 2;
      const intro = view.intro.v;
      const wob = view.wobble.v;
      const baseR = Math.min(W, H) * 0.5 * view.p.size * (0.6 + 0.4 * intro);
      const R = baseR * (1 + wob * 0.05);
      const flat = view.p.flat;
      const Ry = R * (1 - flat) * (1 - wob * 0.09);
      const atmo = view.p.atmo;
      const outer = 1 + atmo;

      // Light and half-vector (view direction is +z).
      const lx = -0.46, ly = 0.42, lz = 0.78;
      const ll = Math.hypot(lx, ly, lz);
      const Lx = lx / ll, Ly = ly / ll, Lz = lz / ll;
      let hx = Lx, hy = Ly, hz = Lz + 1;
      const hl = Math.hypot(hx, hy, hz);
      hx /= hl; hy /= hl; hz /= hl;

      const cosY = Math.cos(-view.yaw), sinY = Math.sin(-view.yaw);
      const cosP = Math.cos(-view.pitch), sinP = Math.sin(-view.pitch);

      const lut = view.lut, emis = view.emis;
      const cloudAmt = view.p.cloudAmt;
      const cloudShift = view.cloudPhase;
      const waterLevel = view.p.water;
      const alphaScale = intro;

      // Atmosphere colour tracks the regime: cool blue when temperate, hot
      // orange when molten, teal when the model likes the planet.
      const molten = smoothstep(800, 1400, view.p.temp);
      const aR = 120 + molten * 135 + view.life.v * -20;
      const aG = 170 - molten * 60 + view.life.v * 20;
      const aB = 255 - molten * 150 - view.life.v * 40;

      const y0 = Math.max(0, Math.floor(cy - Ry * outer - 2));
      const y1 = Math.min(H, Math.ceil(cy + Ry * outer + 2));
      const x0 = Math.max(0, Math.floor(cx - R * outer - 2));
      const x1 = Math.min(W, Math.ceil(cx + R * outer + 2));

      for (let py = y0; py < y1; py++) {
        const sy = (py + 0.5 - cy) / Ry;
        for (let px = x0; px < x1; px++) {
          const sx = (px + 0.5 - cx) / R;
          const d2 = sx * sx + sy * sy;
          const o = (py * W + px) * 4;

          if (d2 > outer * outer) continue;

          if (d2 > 1) {
            // Atmospheric halo outside the disc.
            const d = Math.sqrt(d2);
            let a = 1 - (d - 1) / atmo;
            a = a * a * 0.62;
            const lit = Math.max(0.16, (sx * Lx + sy * Ly) / d * 0.5 + 0.5);
            const al = a * lit * alphaScale;
            if (al <= 0.004) continue;
            blend(buf, o, aR, aG, aB, al);
            continue;
          }

          const z = Math.sqrt(1 - d2);

          // View-space normal -> model space (undo pitch, then yaw).
          const y1r = sy * cosP - z * sinP;
          const z1r = sy * sinP + z * cosP;
          const mx = sx * cosY + z1r * sinY;
          const mz = -sx * sinY + z1r * cosY;
          const my = y1r;

          const lat = Math.asin(my < -1 ? -1 : my > 1 ? 1 : my);
          const lon = Math.atan2(mx, mz);
          const u = (lon / TAU + 0.5) * TEX_W;
          const v = (0.5 - lat / Math.PI) * TEX_H;

          const h = sample(view.maps.height, u, v);
          const idx = (h * 255) | 0;

          const diff = Math.max(0, sx * Lx + sy * Ly + z * Lz);
          const shade = 0.11 + diff * 0.95;

          let r = lut[idx * 3] * shade;
          let g = lut[idx * 3 + 1] * shade;
          let b = lut[idx * 3 + 2] * shade;

          // Emissive lava ignores the lighting term.
          const e = emis[idx];
          if (e > 0) {
            r += lut[idx * 3] * e;
            g += lut[idx * 3 + 1] * e;
            b += lut[idx * 3 + 2] * e;
          }

          // Specular glint on open water.
          if (waterLevel > 0 && h < waterLevel) {
            const sp = sx * hx + sy * hy + z * hz;
            if (sp > 0) {
              const s = Math.pow(sp, 46) * 190 * waterLevel * diff;
              r += s; g += s; b += s;
            }
          }

          // Clouds, drifting slowly in longitude.
          if (cloudAmt > 0.01) {
            const c = sample(view.maps.cloud, u + cloudShift, v) * cloudAmt;
            if (c > 0.004) {
              const cl = 236 * shade;
              r += (cl - r) * c; g += (cl - g) * c; b += (cl + 6 - b) * c;
            }
          }

          // Fresnel rim -- the atmosphere seen edge-on.
          const fres = Math.pow(1 - z, 3.2) * (0.35 + diff * 0.9);
          r += aR * fres * 0.55; g += aG * fres * 0.55; b += aB * fres * 0.55;

          buf[o] = r; buf[o + 1] = g; buf[o + 2] = b;
          buf[o + 3] = 255 * alphaScale;
        }
      }

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Rings sit behind the planet on the far side, in front on the near side.
      const ring = view.p.ring;
      const sx2 = canvas.width / W;
      if (ring > 0.01) drawRing(ctx, canvas.width / 2, canvas.height / 2,
                                R * sx2, Ry * sx2, ring * alphaScale, true);
      pushBuffer();
      if (ring > 0.01) drawRing(ctx, canvas.width / 2, canvas.height / 2,
                                R * sx2, Ry * sx2, ring * alphaScale, false);
    }

    const scratch = document.createElement('canvas');
    let scratchCtx = null;
    function pushBuffer() {
      // Assigning width/height clears and reallocates, so only do it on resize.
      if (scratch.width !== view.bufW || scratch.height !== view.bufH) {
        scratch.width = view.bufW;
        scratch.height = view.bufH;
      }
      if (!scratchCtx) scratchCtx = scratch.getContext('2d');
      scratchCtx.putImageData(view.img, 0, 0);
      ctx.drawImage(scratch, 0, 0, canvas.width, canvas.height);
    }

    function blend(buf, o, r, g, b, a) {
      const ia = buf[o + 3] / 255;
      const na = a + ia * (1 - a);
      buf[o] = (r * a + buf[o] * ia * (1 - a)) / (na || 1);
      buf[o + 1] = (g * a + buf[o + 1] * ia * (1 - a)) / (na || 1);
      buf[o + 2] = (b * a + buf[o + 2] * ia * (1 - a)) / (na || 1);
      buf[o + 3] = na * 255;
    }

    function drawRing(c, cx, cy, R, Ry, alpha, back) {
      const tilt = Math.max(0.06, Math.abs(Math.sin(view.pitch)) * 0.9 + 0.07);
      c.save();
      c.beginPath();
      if (back) c.rect(0, 0, c.canvas.width, cy);
      else c.rect(0, cy, c.canvas.width, c.canvas.height - cy);
      c.clip();
      const bands = [[1.32, 0.30], [1.50, 0.52], [1.62, 0.22], [1.76, 0.40], [1.88, 0.16]];
      for (const [rr, op] of bands) {
        c.beginPath();
        c.ellipse(cx, cy, R * rr, Ry * rr * tilt, 0, 0, TAU);
        c.strokeStyle = `rgba(226, 214, 190, ${op * alpha})`;
        c.lineWidth = Math.max(1, R * 0.055);
        c.stroke();
      }
      c.restore();
    }

    /* ── animation loop ── */
    function frame(now) {
      const dt = Math.min(0.05, (now - view.lastT) / 1000);
      view.lastT = now;
      if (view.visible) {
        if (!view.dragging) view.yaw += (view.spinVel || view.p.spin) * dt;
        view.cloudPhase += dt * 1.6;
        render(dt);
      }
      view.raf = requestAnimationFrame(frame);
    }

    /* ── drag to rotate ── */
    let lastX = 0, lastY = 0, lastMove = 0, vx = 0;

    canvas.addEventListener('pointerdown', (e) => {
      view.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('grabbing');
      lastX = e.clientX; lastY = e.clientY; lastMove = performance.now();
      if (global.anime) anime.remove(view);
      view.spinVel = 0;
      if (opts.onFirstDrag) { opts.onFirstDrag(); opts.onFirstDrag = null; }
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!view.dragging) return;
      const now = performance.now();
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      const k = 0.0095;
      view.yaw += dx * k;
      view.pitch = Math.max(-1.25, Math.min(1.25, view.pitch + dy * k));
      const dt = Math.max(8, now - lastMove);
      vx = (dx * k) / (dt / 1000);
      lastX = e.clientX; lastY = e.clientY; lastMove = now;
    });

    function endDrag(e) {
      if (!view.dragging) return;
      view.dragging = false;
      canvas.classList.remove('grabbing');
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* already gone */ }

      // Inertia: anime.js decays the flung velocity back to the idle spin.
      if (global.anime && !reduceMotion) {
        view.spinVel = Math.max(-9, Math.min(9, vx));
        anime.remove(view);
        anime({
          targets: view,
          spinVel: view.p.spin,
          duration: 1500,
          easing: 'easeOutQuart',
          complete: () => { view.spinVel = 0; },
        });
      } else {
        view.spinVel = 0;
      }
    }
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    /* ── parameter mapping ── */
    const logNorm = (v, lo, hi) =>
      clamp01((Math.log10(Math.max(v, 1e-6)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)));

    function derive(input) {
      const type = input.planet_type;
      const gas = !!GAS[type];
      const temp = input.surface_temperature;

      const rt = logNorm(input.radius, 0.76, 18.2);
      const size = 0.56 + rt * 0.40;

      // Oblateness comes from a planet's ROTATION, which this dataset does not
      // record -- orbital period says nothing about it. So it is a per-type
      // constant: gas giants are visibly oblate, rocky worlds essentially not.
      const flat = gas ? 0.085 : 0.015;

      // Heavier worlds hold a thicker atmosphere.
      const atmo = 0.055 + logNorm(input.mass, 0.4, 6301) * 0.20;

      // Constant, slow idle rotation. Deliberately NOT tied to orbital period:
      // a planet's day and its year are independent quantities, and the archive
      // has no rotation-period column. This spin is a viewing aid only; orbital
      // period drives the orbit diagram instead.
      const spin = 0.14;

      const liquid = gas ? 0 : smoothstep(200, 250, temp) * (1 - smoothstep(330, 420, temp));
      const water = liquid * 0.50;
      const cloudAmt = gas ? 0.20 : 0.10 + liquid * 0.34;

      return {
        size, flat, temp, atmo, spin, water, cloudAmt,
        ring: type === 'Jovian' ? 1 : 0,
      };
    }

    /* ── public API ── */
    function setParams(input, immediate) {
      const next = derive(input);
      const type = input.planet_type;

      if (type !== view.type) {
        view.type = type;
        view.maps = generateMaps(type, view.seed);
        rebuildLUT();
        // Morph pulse so a type change reads as a transformation.
        if (global.anime && !reduceMotion) {
          anime.remove(view.wobble);
          view.wobble.v = 1;
          anime({
            targets: view.wobble, v: 0,
            duration: 1100, easing: 'easeOutElastic(1, 0.4)',
          });
        }
      }

      if (global.anime && !immediate && !reduceMotion) {
        // Cancel only the previous parameter tween, so rapid slider drags
        // don't stack competing animations on the same properties.
        anime.remove(view.p);
        anime({
          targets: view.p,
          size: next.size, flat: next.flat, temp: next.temp, atmo: next.atmo,
          spin: next.spin, water: next.water, cloudAmt: next.cloudAmt,
          ring: next.ring,
          duration: dur(620),
          easing: 'easeOutCubic',
        });
      } else {
        Object.assign(view.p, next);
      }
    }

    function setProbability(p) {
      if (global.anime && !reduceMotion) {
        anime.remove(view.life);
        anime({ targets: view.life, v: p, duration: dur(700), easing: 'easeOutCubic' });
      } else {
        view.life.v = p;
      }
    }

    function start() {
      resize();
      view.type = null;
      if (global.anime && !reduceMotion) {
        anime({ targets: view.intro, v: 1, duration: 900, easing: 'easeOutExpo' });
      } else {
        view.intro.v = 1;
      }
      view.lastT = performance.now();
      view.raf = requestAnimationFrame(frame);
    }

    new ResizeObserver(() => resize()).observe(canvas.parentElement || canvas);
    document.addEventListener('visibilitychange', () => { view.visible = !document.hidden; });
    if (global.IntersectionObserver) {
      new IntersectionObserver((entries) => {
        view.visible = entries[0].isIntersecting && !document.hidden;
      }, { threshold: 0.02 }).observe(canvas);
    }

    return { setParams, setProbability, start, resize };
  }

  /* ───────────────────────── orbit diagram ─────────────────────────
   * Orbital period's own visual, kept separate from the globe's rotation.
   * The orbit's SIZE is derived, not invented: Kepler's third law gives
   * a = (P / 365.25 d)^(2/3) AU for a 1 solar-mass host, so the period slider
   * moves the planet in and out exactly as the physics says it should.
   * Only the lap TIME is compressed, since real periods span 0.67 to 14,000
   * days and nothing would be watchable otherwise.
   */
  function createOrbit(canvas, options) {
    const opts = options || {};
    const ctx = canvas.getContext('2d');
    const reduceMotion = global.matchMedia
      && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const o = {
      period: 365.25, aAU: 1, angle: -0.6, lap: 7,
      cssW: 0, cssH: 0, visible: true, lastT: performance.now(), raf: 0,
    };
    const TILT = 0.34;   // viewing angle of the orbital plane

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      o.cssW = rect.width; o.cssH = rect.height;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function setPeriod(days) {
      o.period = days;
      o.aAU = Math.pow(days / 365.25, 2 / 3);
      // Log-compress 0.67 d .. 14,006 d into a 3.5 s .. 14 s lap.
      const t = clamp01((Math.log10(Math.max(days, 1e-3)) - Math.log10(0.67))
                        / (Math.log10(14006) - Math.log10(0.67)));
      o.lap = 3.5 + t * 10.5;
      if (opts.onChange) opts.onChange(o.period, o.aAU, o.lap);
    }

    function draw() {
      if (!o.cssW) return;
      const W = o.cssW, H = o.cssH;
      ctx.clearRect(0, 0, W, H);

      const cx = W / 2, cy = H / 2;
      const maxRx = Math.min(W / 2 - 14, (H / 2 - 8) / TILT);
      // Orbit radius on a log scale so 0.01 AU and 20 AU both stay on screen.
      const t = clamp01((Math.log10(Math.max(o.aAU, 1e-4)) - Math.log10(0.01))
                        / (Math.log10(30) - Math.log10(0.01)));
      const rx = 20 + t * Math.max(1, maxRx - 20);
      const ry = rx * TILT;

      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, TAU);
      ctx.strokeStyle = 'rgba(255,255,255,0.16)';
      ctx.lineWidth = 1;
      ctx.stroke();

      const px = cx + rx * Math.cos(o.angle);
      const py = cy + ry * Math.sin(o.angle);
      const far = Math.sin(o.angle) < 0;     // behind the star

      if (far) drawPlanet(px, py, far);
      drawStar(cx, cy);
      if (!far) drawPlanet(px, py, far);
    }

    function drawStar(cx, cy) {
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 17);
      g.addColorStop(0, 'rgba(255,238,190,0.95)');
      g.addColorStop(0.35, 'rgba(255,205,110,0.42)');
      g.addColorStop(1, 'rgba(255,190,90,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(cx, cy, 17, 0, TAU); ctx.fill();

      ctx.beginPath(); ctx.arc(cx, cy, 4.6, 0, TAU);
      ctx.fillStyle = '#fff4d6'; ctx.fill();
    }

    function drawPlanet(px, py, far) {
      const r = far ? 4.2 : 5.4;
      ctx.beginPath(); ctx.arc(px, py, r + 2, 0, TAU);
      ctx.fillStyle = '#0b0b0d';                 // 2px surface ring
      ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, r, 0, TAU);
      ctx.fillStyle = far ? 'rgba(157,197,244,0.72)' : '#cde2fb';
      ctx.fill();
    }

    function frame(now) {
      const dt = Math.min(0.05, (now - o.lastT) / 1000);
      o.lastT = now;
      if (o.visible) {
        if (!reduceMotion) o.angle += (TAU / o.lap) * dt;
        draw();
      }
      o.raf = requestAnimationFrame(frame);
    }

    function start() { resize(); o.lastT = performance.now(); o.raf = requestAnimationFrame(frame); }

    new ResizeObserver(() => { resize(); draw(); }).observe(canvas.parentElement || canvas);
    document.addEventListener('visibilitychange', () => { o.visible = !document.hidden; });
    if (global.IntersectionObserver) {
      new IntersectionObserver((e) => { o.visible = e[0].isIntersecting && !document.hidden; },
                               { threshold: 0.02 }).observe(canvas);
    }

    return { setPeriod, start, resize };
  }

  global.PlanetView = { create };
  global.OrbitView = { create: createOrbit };
})(window);

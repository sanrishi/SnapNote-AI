#!/usr/bin/env node
// Engine D: Satori (HTML/CSS -> SVG) + Resvg (SVG -> PNG)
// Vercel OG stack. No browser.
const fs = require('fs');
const path = require('path');
const satori = require('satori').default;
const { Resvg } = require('@resvg/resvg-js');

async function renderSatori(title, heroSvg, callouts, result) {
  const t0 = performance.now();
  // Satori expects a React-like element tree (JSX via satori's h)
  // We build a simple HTML/CSS infographic shell and let Satori lay it out.
  const html = {
    type: 'div',
    props: {
      style: {
        width: '800px',
        background: 'white',
        padding: '16px',
        fontFamily: 'Inter, sans-serif',
        color: '#1e293b',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      },
      children: [
        { type: 'h1', props: { style: { fontSize: '16px', fontWeight: 800, textAlign: 'center' }, children: title } },
        { type: 'div', props: { style: { width: '100%', height: '420px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }, children: heroSvg } },
        { type: 'div', props: { style: { display: 'flex', gap: '12px' }, children: callouts.map(c => ({ type: 'div', props: { style: { flex: 1, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '8px', fontSize: '11px' }, children: c } })) } },
        { type: 'div', props: { style: { background: '#fef9c3', border: '1px solid #facc15', borderRadius: '8px', padding: '10px', fontWeight: 700, textAlign: 'center' }, children: result } },
      ]
    }
  };
  // Satori needs fonts
  const fontPath = require.resolve('@fontsource/inter/files/inter-latin-400-normal.woff');
  const fontData = fs.readFileSync(fontPath);
  const svg = await satori(html, {
    width: 800,
    height: 900,
    fonts: [{ name: 'Inter', data: fontData, weight: 400, style: 'normal' }],
  });
  const resvg = new Resvg(svg, { fitTo: { mode: 'width', value: 800 } });
  const pngData = resvg.render();
  const png = pngData.asPng();
  const dt = performance.now() - t0;
  return { svg, png, ms: dt };
}

(async () => {
  const out = path.join(__dirname, 'outputs');
  require('fs').mkdirSync(out, { recursive: true });
  const hero = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="#ddd"/></svg>';
  for (const [name, title] of [['torque', 'Torque'], ['argand', 'Square on Argand Plane']]) {
    const t0 = performance.now();
    try {
      const { svg, png, ms } = await renderSatori(title, hero, ['callout 1', 'callout 2'], 'Area = 4');
      fs.writeFileSync(path.join(out, `d_satori_${name}.svg`), svg);
      fs.writeFileSync(path.join(out, `d_satori_${name}.png`), png);
      console.log(`D satori ${name}: ${svg.length} chars SVG, ${png.length} bytes PNG, ${ms.toFixed(1)}ms`);
    } catch (e) {
      console.log(`D satori ${name} failed:`, e.message);
    }
  }
})();

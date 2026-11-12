function setupQuantizationLab() {
  const slider = document.querySelector('#bit-slider');
  const value = document.querySelector('#bit-value');
  const scale = document.querySelector('#quant-scale');
  const error = document.querySelector('#quant-error');
  const chart = document.querySelector('#quant-vectors');
  if (!slider || !value || !scale || !error || !chart) return;

  const samples = [-1.2, -0.8, -0.3, 0.1, 0.4, 0.8, 1.2];
  const maxAbs = Math.max(...samples.map((x) => Math.abs(x)));
  const render = () => {
    const bits = Number(slider.value);
    const qmax = 2 ** (bits - 1) - 1;
    const step = maxAbs / qmax;
    const restored = samples.map((x) =>
      Math.max(-qmax, Math.min(qmax, Math.round(x / step))) * step
    );
    const mse = samples.reduce((sum, x, index) =>
      sum + (x - restored[index]) ** 2, 0) / samples.length;
    value.textContent = `INT${bits}`;
    scale.textContent = step.toFixed(4);
    error.textContent = mse.toFixed(6);
    chart.replaceChildren();
    samples.forEach((x, index) => {
      const item = document.createElement('div');
      item.className = 'quant-bin';
      const bar = document.createElement('i');
      bar.style.height = `${Math.max(6, Math.abs(restored[index]) / maxAbs * 80)}px`;
      bar.title = `${x.toFixed(2)} → ${restored[index].toFixed(3)}`;
      const label = document.createElement('span');
      label.textContent = restored[index].toFixed(2);
      const original = document.createElement('small');
      original.textContent = x.toFixed(1);
      item.append(bar, label, original);
      chart.append(item);
    });
  };
  slider.addEventListener('input', render);
  render();
}

setupQuantizationLab();


// Fit code cells to their contents so they remain readable on a livestream.
document.querySelectorAll('.cell textarea').forEach((area) => {
  const fit = () => { area.style.height = 'auto'; area.style.height = Math.max(180, area.scrollHeight + 4) + 'px'; };
  area.addEventListener('input', fit); fit();
});

const motionToggle = document.querySelector('.motion-toggle');
if (motionToggle) {
  motionToggle.addEventListener('click', () => {
    const paused = document.body.classList.toggle('motion-paused');
    motionToggle.setAttribute('aria-pressed', String(paused));
    motionToggle.textContent = paused ? 'Play animations' : 'Pause animations';
  });
}

// Step-by-step, speaker-controlled explanations. Values are illustrative.
const lessons = {
 'contrast-lesson': [
  {title:'1. Encode two paired examples', visual:'<div class="viz-pairs"><div>🐕 Photo 1 <b>↔</b> “a dog”</div><div>🚲 Photo 2 <b>↔</b> “a bicycle”</div></div>',text:'A training batch contains matched image–caption pairs. The image and text encoders have separate parameters.'},
  {title:'2. Normalize image and text vectors', visual:'<div class="viz-row"><b>Image encoder → I₁, I₂</b><i>↓</i><b>L2 normalize</b><i>↑</i><b>Text encoder → T₁, T₂</b></div>',text:'Both branches project into the same feature dimension. L2 normalization makes dot products equal cosine similarities.'},
  {title:'3. Compare every image with every caption', visual:'<div class="viz-matrix"><span></span><b>dog</b><b>bike</b><b>dog</b><strong>0.8 ✓</strong><span>0.1</span><b>bike</b><span>0.2</span><strong>0.9 ✓</strong></div>',text:'The 2 × 2 matrix contains all pairwise scores. Diagonal entries are matched pairs; the others are negatives for this batch.'},
  {title:'4. Compute loss in both directions', visual:'<div class="viz-formula">Image → text: cross-entropy over each row<br>Text → image: cross-entropy over each column<br><strong>CLIP loss = average of the two</strong></div>',text:'A learned positive logit scale multiplies the similarities before cross-entropy. The objective encourages each correct match to score higher than alternatives.'},
  {title:'5. Send gradients back through both branches', visual:'<div class="viz-row"><b>Image encoder</b><i>←</i><b>Loss</b><i>→</i><b>Text encoder</b></div><div class="viz-label">backpropagation updates both sets of weights</div>',text:'The optimizer uses gradients from both directions to update the encoders, projections and learned logit scale. Retrieval after training needs only encoding, normalization and ranking.'}
 ],
 'quant-lesson': [
  {title:'1. Start with a floating-point activation', visual:'<div class="viz-formula"><strong>x = 0.37</strong><br>scale = 0.1 &nbsp; zero point = 0</div>',text:'This example uses signed INT8, with an illustrative fixed scale. Real PTQ selects parameters using the tensor and representative calibration data.'},
  {title:'2. Express the value in quantization steps', visual:'<div class="viz-formula">x / scale = 0.37 / 0.1<br><strong>= 3.7</strong></div>',text:'Scale is the spacing between adjacent reconstructed values. Here, 0.37 lies between the codes for 0.3 and 0.4.'},
  {title:'3. Round to an integer code', visual:'<div class="viz-numberline"><span>3 → 0.30</span><strong>3.7 → 4</strong><span>4 → 0.40</span></div>',text:'Round 3.7 to 4, then add zero point 0. Multiple floating-point values can map to the same integer, so rounding loses information.'},
  {title:'4. Clamp to the integer range', visual:'<div class="viz-numberline"><span>−128</span><strong>q = 4 ✓</strong><span>127</span></div>',text:'The result is inside the INT8 range. A value outside the representable range would be clipped to an endpoint, introducing saturation error.'},
  {title:'5. Dequantize and measure the error', visual:'<div class="viz-formula">x̂ = (4 − 0) × 0.1 = <strong>0.40</strong><br>error = 0.40 − 0.37 = <strong>+0.03</strong></div>',text:'Dequantization returns an approximation on the original scale; it cannot restore discarded detail. In a model, errors can accumulate and affect similarity ranking.'}
 ]
};
document.querySelectorAll('[data-lesson]').forEach(root => {
 const steps=lessons[root.dataset.lesson]; if (!steps) return;
 const nodes=[...root.querySelectorAll('.lesson-node')], dots=root.querySelector('.lesson-dots');
 const visual=root.querySelector('.lesson-visual'), copy=root.querySelector('.lesson-copy'), play=root.querySelector('[data-action="play"]');
 let index=0, timer=null;
 steps.forEach((_,i)=>{const b=document.createElement('button'); b.type='button'; b.setAttribute('aria-label',`Go to step ${i+1}`); b.addEventListener('click',()=>{stop();show(i)}); dots.append(b)});
 function show(n){index=Math.max(0,Math.min(steps.length-1,n)); const st=steps[index]; nodes.forEach((v,i)=>v.classList.toggle('active',i===index)); [...dots.children].forEach((v,i)=>v.classList.toggle('active',i===index)); visual.innerHTML=st.visual; copy.querySelector('small').textContent=`STEP ${index+1} / ${steps.length}`; copy.querySelector('h4').textContent=st.title; copy.querySelector('p').textContent=st.text;}
 function stop(){if(timer) clearInterval(timer);timer=null;play.textContent='▶ Play';}
 root.querySelector('[data-action="prev"]').addEventListener('click',()=>{stop();show(index-1)});
 root.querySelector('[data-action="next"]').addEventListener('click',()=>{stop();show(index+1)});
 play.addEventListener('click',()=>{if(timer){stop();return} if(index===steps.length-1)show(0);play.textContent='❚❚ Pause';timer=setInterval(()=>{if(index>=steps.length-1){stop();return}show(index+1)},4500)});
 show(0);
});

// Copy each terminal command or Python snippet without wrapping it in another language.
document.querySelectorAll('.copy-command').forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.closest('.cal-command, .command-step').querySelector('pre code').textContent);
      button.textContent = 'Copied';
      setTimeout(() => button.textContent = 'Copy', 1500);
    } catch (error) { button.textContent = 'Select and copy'; }
  });
});



document.querySelectorAll('.cell').forEach(cell => {
  const button = cell.querySelector('.run');
  if (!button) return;
  button.disabled = false;
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(cell.querySelector('textarea').value);
      button.textContent = 'Copied';
      setTimeout(() => button.textContent = 'Copy code', 1500);
    } catch (_) { button.textContent = 'Select and copy'; }
  });
});
(async () => {
  const select = document.querySelector('#public-query');
  const results = document.querySelector('#public-results');
  const note = document.querySelector('#public-query-note');
  if (!select || !results) return;
  try {
    const response = await fetch('assets/attention/trace.json');
    if (!response.ok) throw new Error('Saved results unavailable');
    const data = await response.json();
    data.queries.forEach((query, i) => {
      const option = document.createElement('option');
      option.value = i; option.textContent = query.query; select.append(option);
    });
    function render() {
      const query = data.queries[Number(select.value)];
      results.replaceChildren();
      query.photos.forEach((photo, i) => {
        const figure = document.createElement('figure');
        const image = document.createElement('img');
        image.src = photo.url; image.alt = `Result ${i+1} for ${query.query}`;
        const caption = document.createElement('figcaption');
        caption.textContent = `#${i+1} · cosine ${photo.score.toFixed(4)}`;
        figure.append(image, caption); results.append(figure);
      });
      note.textContent = `Recorded ranking for “${query.query}” among 512 COCO images. Scores are similarities, not confidence percentages.`;
    }
    select.addEventListener('change', render); render();
  } catch (_) { note.textContent = 'Unable to load the saved search examples. Please refresh the page.'; }
})();

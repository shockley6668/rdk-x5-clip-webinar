/* A measured last-layer text-attention trace, followed all the way to retrieval. */
(async function () {
  const root = document.querySelector('.attention-walkthrough');
  if (!root) return;
  const scene = root.querySelector('#aw-scene');
  const querySelect = root.querySelector('#aw-query');
  const headSelect = root.querySelector('#aw-head');
  const play = root.querySelector('#aw-play');
  const previous = root.querySelector('#aw-previous');
  const next = root.querySelector('#aw-next');
  const status = root.querySelector('#aw-play-status');
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (v, digits = 3) => Number(v).toFixed(digits);
  const label = t => t === '<|endoftext|>' ? 'EOT' : t === '<|startoftext|>' ? 'SOT' : t;
  const vector = (values, className = '') => `<div class="aw-vector ${className}">${values.map(v => `<span style="--magnitude:${Math.min(1,Math.abs(v)/3)}" class="${v<0?'negative':'positive'}">${fmt(v)}</span>`).join('')}<i>…</i></div>`;
  let data;
  try {
    const res = await fetch('assets/attention/trace.json');
    if (!res.ok) throw new Error('Trace unavailable');
    data = await res.json();
  } catch (error) {
    scene.textContent = 'The saved attention trace could not be loaded. Please refresh the page.';
    return;
  }
  querySelect.replaceChildren(...data.queries.map((q, i) => {
    const o = document.createElement('option'); o.value = i; o.textContent = q.query; return o;
  }));
  headSelect.replaceChildren(...Array.from({length:8}, (_, i) => {
    const o = document.createElement('option'); o.value = i; o.textContent = `Head ${i+1} of 8`; return o;
  }));
  let step = 0, head = 0, query = 0, selectedToken = 0, timer = null;
  const steps = [
    ['Find the position that will represent the sentence', 'CLIP uses the final EOT position to produce its text feature. That position starts as a special token, then gathers context as the sequence passes through the Transformer.', 'We will follow EOT. Each earlier word still has its own representation; CLIP selects EOT only after the final layer.'],
    ['Let the EOT query compare with every available key', 'We zoom into layer 12. After LayerNorm, learned linear projections create Q, K and V. This EOT query compares with each key using all 64 coordinates. Select a token to inspect its score.', 'A score decides how much to read from a position. It is not yet the information that gets added to EOT.'],
    ['Convert the scores into routing weights', 'Softmax exponentiates and normalizes the scores across positions. Higher scores get larger weights, and the weights add up to one. The causal mask excludes later padding positions.', 'These weights will multiply the V vectors. A large weight is not a class probability or proof that one word explains the prediction.'],
    ['Multiply the V vectors, then add them', 'Each token contributes weight × value vector. Add those contributions coordinate by coordinate. The result is one 64-dimensional head output at the EOT position.', 'This head output is a context update for EOT. It is not a word, an image label, or the final CLIP text feature.'],
    ['Feed that result into the rest of the final block', 'All eight head outputs are concatenated, projected and added to the incoming EOT state. The MLP adds a second update. This is layer 12, so the updated EOT now goes to final LayerNorm and text projection.', 'The head output is actually consumed: concatenate → output projection → residual addition → MLP → final EOT state.'],
    ['Use the final text feature to rank photos', 'Final LayerNorm and text_projection turn the 512-dimensional EOT state into a 1,024-dimensional text feature. L2-normalize it, dot it with each cached image feature, and sort.', 'Attention changes contextual features inside the encoder. Photo search uses the final feature vector, not the attention weights.']
  ];

  function stop() {
    if (timer) clearInterval(timer);
    timer = null; play.textContent = '▶ Play walkthrough';
    status.textContent = 'Advance one step at a time, or play.';
  }
  function route(q, h, weighted = false) {
    const n = q.tokens.length;
    return `<div class="aw-query-vector"><small>EOT QUERY · HEAD ${head+1}</small>${vector(h.q_preview)}<span>64 coordinates · preview of first 6</span></div>
      <svg class="aw-fan" viewBox="0 0 1000 70" preserveAspectRatio="none" aria-hidden="true">${q.tokens.map((_, i) => `<path class="${i === selectedToken?'selected':''}" d="M500 0 C500 35 ${(i+.5)*1000/n} 30 ${(i+.5)*1000/n} 69" style="--aw-order:${i};stroke-width:${weighted?2+8*h.weights[i]:3}"></path>`).join('')}</svg>
      <div class="aw-key-row" style="--token-count:${n}">${q.tokens.map((t,i) => `<button type="button" data-token="${i}" class="${i===selectedToken?'selected':''}"><small>POSITION ${i}</small><b>${esc(label(t))}</b><span>${weighted?fmt(h.weights[i]*100,1)+'%':fmt(h.scores[i])}</span></button>`).join('')}</div>`;
  }
  function show() {
    const q = data.queries[query], h = q.heads[head];
    root.querySelector('#attention-title').textContent = `How does “${q.query}” become a search?`;
    root.classList.toggle('aw-started', step > 0);
    selectedToken = Math.min(selectedToken, q.tokens.length-1);
    root.querySelector('#aw-step-number').textContent = `STEP ${step+1} / 6`;
    root.querySelector('#aw-step-title').textContent = steps[step][0];
    root.querySelector('#aw-step-explanation').textContent = steps[step][1];
    root.querySelector('#aw-takeaway').textContent = steps[step][2];
    root.querySelectorAll('[data-aw-step]').forEach((b,i) => {
      b.classList.toggle('active', i===step); b.setAttribute('aria-current', i===step?'step':'false');
    });
    previous.disabled = step === 0;
    next.textContent = step === 5 ? 'Replay from start ↻' : 'Next →';
    headSelect.disabled = step === 5;
    root.querySelector('#aw-reconstruction').textContent = `Reconstructed forward versus model.encode_text: maximum absolute difference ${q.max_abs_reconstruction_error.toExponential(2)}. Checkpoint SHA-256: ${data.checkpoint_sha256}.`;
    if (step === 0) {
      scene.innerHTML = `<div class="aw-token-ribbon">${q.tokens.map((t,i)=>`<div class="${i===q.eot_index?'eot':''}"><small>${i===q.eot_index?'FOLLOW THIS POSITION':'TOKEN '+i}</small><b>${esc(label(t))}</b></div>`).join('')}</div>
        <div class="aw-layer-map"><div><small>EMBEDDING + POSITIONS</small><b>77 × 512</b></div><i>→</i><div><small>ALREADY COMPUTED</small><div class="aw-layer-stack">${Array.from({length:11},(_,i)=>`<span>${i+1}</span>`).join('')}</div><b>11 Transformer blocks</b></div><i>→</i><div class="focus"><small>OUR ZOOM</small><b>Layer 12</b><span>EOT state: 512 numbers</span></div></div>
        <div class="aw-preview"><b>Incoming EOT state at layer 12</b>${vector(q.eot_input_preview)}<small>First 6 of 512 values. This state already contains context from layers 1–11.</small></div>`;
    } else if (step === 1) {
      const t = selectedToken;
      scene.innerHTML = route(q,h)+`<div class="aw-score-calculation"><div><small>KEY FOR “${esc(label(q.tokens[t]))}”</small>${vector(h.k_preview[t])}</div><div class="aw-score-result"><small>FULL 64-D DOT PRODUCT / √64</small><b>${fmt(h.scores[t]*8)} ÷ 8 = <em>${fmt(h.scores[t])}</em></b></div></div><p class="aw-footnote">Click a token above to follow another comparison. We show one query row, not the entire 77 × 77 attention matrix.</p>`;
    } else if (step === 2) {
      scene.innerHTML = `<div class="aw-weight-table"><div class="aw-weight-heading"><b>Position</b><b>Q · K / √64</b><b>Softmax weight</b></div>${q.tokens.map((t,i)=>`<div class="aw-weight-row" style="--row:${i}"><b>${esc(label(t))}</b><span>${fmt(h.scores[i])}</span><div><i style="--weight:${h.weights[i]*100}%"></i><strong>${fmt(h.weights[i]*100,1)}%</strong></div></div>`).join('')}</div><div class="aw-normalization"><b>Σ weights = ${fmt(h.weight_sum,4)}</b><span>All non-padding positions shown, including SOT and EOT.</span></div><p class="aw-footnote">SOT can receive a large share of attention. These are measured weights; they are not hand-designed “word importance” scores.</p>`;
    } else if (step === 3) {
      scene.innerHTML = `<p class="aw-footnote">Rounded display only; sums use full precision. On a narrow screen, scroll the table sideways.</p><div class="aw-sum-scroll"><table class="aw-sum-table"><caption>Head ${head+1}: add token contributions · first 6 of 64 coordinates</caption><thead><tr><th>Token</th><th>Weight × V</th>${[0,1,2,3,4,5].map(i=>`<th>dim ${i}</th>`).join('')}</tr></thead><tbody>${q.tokens.map((t,i)=>`<tr style="--row:${i}"><th>${esc(label(t))}</th><td>${fmt(h.weights[i],3)} × V<sub>${i}</sub></td>${h.v_preview[i].map(v=>`<td class="${v<0?'aw-negative':'aw-positive'}">${fmt(v*h.weights[i])}</td>`).join('')}</tr>`).join('')}</tbody><tfoot><tr><th>Σ = h${head+1}</th><td>64-d output</td>${h.output_preview.map(v=>`<td>${fmt(v)}</td>`).join('')}</tr></tfoot></table></div><div class="aw-used-next"><b>WHERE DOES h${head+1} GO?</b><span>Into concatenation with the other 7 heads. Continue to see the actual next operation.</span></div>`;
    } else if (step === 4) {
      scene.innerHTML = `<div class="aw-head-assembly"><div class="aw-eight-heads">${q.heads.map((v,i)=>`<div class="${i===head?'selected':''}"><b>h${i+1}</b><span>64 values</span></div>`).join('')}</div><div class="aw-merge-arrow">→</div><div class="aw-merge"><small>CONCATENATE</small><b>8 × 64 = 512</b><p>No averaging of heads.</p></div><div class="aw-merge-arrow">→</div><div class="aw-merge"><small>OUTPUT PROJECTION</small><b>W<sub>O</sub> + bias</b><p>512 → 512</p></div></div>
        <div class="aw-residual"><div><small>INCOMING EOT · x</small>${vector(q.eot_input_preview)}</div><b>+</b><div><small>ATTENTION UPDATE · a</small>${vector(q.projected_preview)}</div><b>=</b><div class="aw-result"><small>x₁ = x + a</small>${vector(q.residual_preview)}</div></div>
        <div class="aw-mlp"><b>x₂ = x₁ + MLP(LN(x₁))</b>${vector(q.updated_preview)}<span>Updated EOT after the final block · 512 values</span></div>`;
    } else {
      scene.innerHTML = `<div class="aw-final-path"><div><small>FINAL EOT STATE</small><b>512-d</b></div><i>→</i><div><small>LayerNorm + text_projection</small><b>512 → 1024</b></div><i>→</i><div><small>L2 NORMALIZE</small><b>text feature t</b></div><i>→</i><div class="aw-result"><small>CACHED IMAGE FEATURES</small><b>scoreᵢ = iᵢ · t</b></div></div>
        <div class="aw-preview"><b>t · first 6 of 1,024 values</b>${vector(q.text_feature_preview)}</div><div class="aw-photo-results">${q.photos.map((p,i)=>`<figure><img src="${p.url}" alt="COCO result ${i+1} for ${esc(q.query)}"><figcaption><b>#${i+1}</b><span>cosine ${fmt(p.score,4)}</span></figcaption></figure>`).join('')}</div><p class="aw-footnote">Real FP32 ranking among 512 COCO images for “${esc(q.query)}”. Switch the query to see both its features and its retrieved photos change. <a href="#demo">Explore the search example →</a></p>`;
    }
    scene.querySelectorAll('[data-token]').forEach(button => button.addEventListener('click', () => {
      stop(); selectedToken = Number(button.dataset.token); show();
    }));
  }
  previous.addEventListener('click', () => { stop(); step = Math.max(0,step-1); show(); });
  next.addEventListener('click', () => { stop(); step = (step+1)%6; show(); });
  play.addEventListener('click', () => {
    if (timer) { stop(); return; }
    if (step === 5) step = 0;
    show(); play.textContent = '❚❚ Pause'; status.textContent = 'One operation every 8 seconds. Pause to discuss.';
    timer = setInterval(() => {
      if (step === 5) { stop(); return; }
      step++; show();
    }, 8000);
  });
  querySelect.addEventListener('change', () => { stop(); query = Number(querySelect.value); selectedToken=0; show(); });
  headSelect.addEventListener('change', () => { stop(); head = Number(headSelect.value); show(); });
  root.querySelectorAll('[data-aw-step]').forEach(button => button.addEventListener('click', () => {
    stop(); step = Number(button.dataset.awStep); show();
  }));
  document.querySelector('.motion-toggle')?.addEventListener('click', () => {
    if (document.body.classList.contains('motion-paused')) stop();
  });
  const focusButton = root.querySelector('#aw-focus');
  function focusView(enabled) {
    root.classList.toggle('aw-presenting', enabled);
    document.body.classList.toggle('aw-open', enabled);
    focusButton.setAttribute('aria-pressed', String(enabled));
    focusButton.textContent = enabled ? 'Exit presentation view ✕' : 'Presentation view ⤢';
    if (enabled) {
      root.setAttribute('role', 'dialog'); root.setAttribute('aria-modal', 'true');
      root.scrollTop = 0;
    } else {
      root.removeAttribute('role'); root.removeAttribute('aria-modal');
    }
    focusButton.focus();
  }
  focusButton.addEventListener('click', () => focusView(!root.classList.contains('aw-presenting')));
  root.addEventListener('keydown', event => {
    if (!root.classList.contains('aw-presenting')) return;
    if (event.key === 'Escape') { focusView(false); return; }
    if (event.key !== 'Tab') return;
    const all = [...root.querySelectorAll('button:not(:disabled),select:not(:disabled),summary,a[href]')].filter(el => el.getClientRects().length);
    const first = all[0], last = all[all.length-1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  show();
})();


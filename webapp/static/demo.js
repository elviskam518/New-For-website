const form = document.getElementById('job-form');
const jobCard = document.getElementById('job-card');
let pollHandle = null;

function setState(job) {
  document.getElementById('status').innerText = job.status;
  document.getElementById('progress').innerText = job.progress;
  document.getElementById('progress-text').innerText = job.progress_text;
  document.getElementById('bar').style.width = `${job.progress}%`;
  const logs = job.logs.join('\n');
  const logEl = document.getElementById('logs');
  logEl.textContent = logs;
  logEl.scrollTop = logEl.scrollHeight;

  const resultPanel = document.getElementById('result-panel');
  if (job.status === 'completed' && job.summary) {
    const result = job.summary.result || {};
    resultPanel.innerHTML = `
      <h4>Completed Result</h4>
      <p><b>Model:</b> ${result.model || '-'}</p>
      <p><b>Accuracy:</b> ${result.accuracy ?? '-'}</p>
      <p><b>F1:</b> ${result.f1 ?? '-'}</p>
      <p><b>Min DI:</b> ${result.min_di ?? '-'}</p>
      <p><a href="/api/jobs/${job.id}">Full job JSON</a></p>
      <p><a href="/results">View all completed runs</a></p>
    `;
  } else if (job.status === 'failed') {
    resultPanel.innerHTML = `<h4>Job failed</h4><p>${job.error || ''}</p>`;
  }
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  setState(job);
  if (['completed', 'failed'].includes(job.status)) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = new FormData(form);
  data.set('include_latent_vis', form.include_latent_vis.checked ? 'true' : 'false');

  const resp = await fetch('/api/jobs/start', { method: 'POST', body: data });
  const payload = await resp.json();
  if (!resp.ok) {
    alert(payload.detail || 'Failed to start job');
    return;
  }

  jobCard.style.display = 'block';
  document.getElementById('job-id').innerText = payload.job_id;
  if (pollHandle) clearInterval(pollHandle);
  await pollJob(payload.job_id);
  pollHandle = setInterval(() => pollJob(payload.job_id), 2000);
});

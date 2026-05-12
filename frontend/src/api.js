const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '';

class ApiClient {
  constructor() {
    this.token = null;
  }

  setToken(token) { this.token = token; }

  async _fetch(path, opts = {}) {
    const res = await fetch(`${BACKEND_URL}${path}`, {
      ...opts,
      headers: {
        'Content-Type': 'application/json',
        ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
        ...(opts.headers || {}),
      },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async startProcessing(data) {
    return this._fetch('/api/process-video', { method: 'POST', body: JSON.stringify(data) });
  }

  async getJobStatus(jobId) {
    return this._fetch(`/api/job/${jobId}`);
  }

  async getSubmissions() {
    return this._fetch('/api/submissions');
  }

  async healthCheck() {
    return this._fetch('/health');
  }

  waitForCompletion(jobId, onProgress = () => {}, interval = 3000) {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const s = await this.getJobStatus(jobId);
          onProgress(s);
          if (s.status === 'completed') resolve(s);
          else if (s.status === 'failed') reject(new Error(s.error || 'Processing failed'));
          else setTimeout(poll, interval);
        } catch (e) { reject(e); }
      };
      poll();
    });
  }
}

export const api = new ApiClient();

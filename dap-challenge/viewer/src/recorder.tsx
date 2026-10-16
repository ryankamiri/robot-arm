import { useEffect, useState, type FormEvent } from 'react';

type Take = {
  episode_id: string;
  status: string;
  row_count: number;
  error: string | null;
  diagnostics?: { missed_deadlines: number; repeated_frames: number };
};

type Status = {
  phase: string;
  connected: boolean;
  error: string | null;
  active_episode_id: string | null;
  recorded_rows: number;
  pending: Take[];
  demo: boolean;
  diagnostics: { missed_deadlines: number; max_cycle_s: number; camera_age_s: number };
};

async function api(path: string, body?: unknown) {
  const response = await fetch(`/api/${path}`, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}

export function Recorder() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState('');
  const [videoError, setVideoError] = useState(false);
  const [ports, setPorts] = useState<{ port: string; description: string }[]>([]);
  const [profile, setProfile] = useState('');
  const [form, setForm] = useState({
    leader_port: '', follower_port: '', leader_id: '', follower_id: '',
    calibration_dir: '', camera_index: '0', width: '', height: '', camera_fps: '30',
  });

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api('status');
        if (active) setStatus(next);
      } catch (failure) {
        if (active) setError(String(failure));
      } finally {
        if (active) timer = setTimeout(poll, 500);
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, []);

  const takeId = selected || status?.active_episode_id || '';
  const take = status?.pending.find(entry => entry.episode_id === takeId);
  useEffect(() => setVideoError(false), [takeId]);

  async function action(path: string, body: unknown = {}) {
    setBusy(true);
    setError('');
    try {
      const next = await api(path, body);
      setStatus(previous => ({ ...next, demo: previous?.demo ?? next.demo }));
      if (path.endsWith('/keep') || path.endsWith('/discard')) setSelected('');
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }

  async function connect(event: FormEvent) {
    event.preventDefault();
    await action('connect', {
      ...form, calibration_dir: form.calibration_dir || null,
      camera_index: Number(form.camera_index),
      width: Number(form.width || (status?.demo ? 320 : 0)),
      height: Number(form.height || (status?.demo ? 240 : 0)), camera_fps: Number(form.camera_fps),
    });
  }

  async function inspect() {
    setBusy(true);
    setError('');
    try {
      const next = await api('camera/inspect', { camera_index: Number(form.camera_index) });
      setProfile(JSON.stringify(next.camera_profile, null, 2));
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }

  return <main>
    <h1>BC recorder</h1>
    {status?.demo && <p role="status"><strong>Demo: synthetic devices. No physical robot or camera is connected.</strong></p>}
    <p>Session: <strong>{status?.phase ?? 'Connecting to backend…'}</strong></p>
    {(error || status?.error) && <p role="alert">{error || status?.error}</p>}

    {!status?.connected && <form onSubmit={connect}>
      <fieldset disabled={busy || !status}>
        <legend>Connect session</legend>
        <button type="button" onClick={() => {
          void api('ports').then(setPorts).catch(failure => setError(String(failure)));
        }}>List USB ports</button>
        {ports.map(port => <p key={port.port}>{port.port} · {port.description}</p>)}
        <datalist id="ports">{ports.map(port => <option key={port.port} value={port.port} />)}</datalist>
        {(['leader_port', 'follower_port', 'leader_id', 'follower_id', 'calibration_dir'] as const).map(key =>
          <label key={key}>{key.replaceAll('_', ' ')}
            <input value={form[key]} list={key.endsWith('port') ? 'ports' : undefined}
              required={!status?.demo && key !== 'calibration_dir'}
              onChange={event => setForm({ ...form, [key]: event.target.value })} />
          </label>,
        )}
        <p>Use the IDs you calibrated with LeRobot. Existing calibration is required.</p>
        {(['camera_index', 'width', 'height', 'camera_fps'] as const).map(key =>
          <label key={key}>{key === 'camera_fps' ? 'Camera images per second' : key.replaceAll('_', ' ')}
            <input type="number" value={form[key]} min={key === 'camera_index' ? 0 : 1}
              required={!status?.demo} onChange={event => setForm({ ...form, [key]: event.target.value })} />
          </label>,
        )}
        <button type="button" onClick={() => void inspect()} disabled={status?.demo}>Inspect selected camera</button>
        {profile && <pre>{profile}</pre>}
        <p>Choose capture size after inspection. Model input remains 224 × 168.</p>
        <button type="submit">{status?.demo ? 'Connect demo session' : 'Connect arms and camera'}</button>
      </fieldset>
    </form>}

    {status?.connected && <section>
      <p>Follower follows the leader during recording, review, and reset.</p>
      <div className="actions">
        <button disabled={busy || status.phase !== 'ready'} onClick={() => {
          setSelected(''); void action('recording/start');
        }}>Start recording</button>
        <button disabled={busy || status.phase !== 'recording'} onClick={() => void action('recording/stop')}>Stop recording</button>
        <button disabled={busy} onClick={() => void action('disconnect')}>End session / release motors</button>
      </div>
      <p>Support the follower before ending the session: motor torque will be released.</p>
      <p>Recording target: 30 iterations per second. Rows in latest take: {status.recorded_rows}.</p>
      <p>Session missed deadlines: {status.diagnostics.missed_deadlines} · Slowest cycle: {(status.diagnostics.max_cycle_s * 1000).toFixed(1)} ms · Latest frame age: {(status.diagnostics.camera_age_s * 1000).toFixed(1)} ms</p>
      {status.diagnostics.missed_deadlines > 0 && <p role="status">Timing warning: recording continues. Inspect the take before keeping it.</p>}
    </section>}

    {status && status.phase !== 'recording' && status.phase !== 'finalizing' && <section>
      <h2>Review</h2>
      <select aria-label="Pending take" value={takeId} disabled={busy} onChange={event => setSelected(event.target.value)}>
        <option value="">Select a pending take</option>
        {status.pending.map(entry => <option key={entry.episode_id} value={entry.episode_id}>{entry.episode_id} · {entry.status}</option>)}
      </select>
      {take?.status === 'recording' && <p>Finalizing pending take…</p>}
      {take && take.status !== 'recording' && <>
        <p>{take.status} · {take.row_count ?? 0} observations</p>
        {take.error && <p role="alert">{take.error}</p>}
        <video key={takeId} controls preload="metadata" src={`/api/episodes/${encodeURIComponent(takeId)}/video`}
          onError={() => setVideoError(true)} />
        {videoError && <p role="alert">Video could not be played. The pending files remain preserved.</p>}
        <p>Take missed deadlines: {take.diagnostics?.missed_deadlines ?? 'unavailable'} · Repeated camera frames: {take.diagnostics?.repeated_frames ?? 'unavailable'}</p>
        <div className="actions">
          <button disabled={busy || take.status !== 'review' || videoError} onClick={() => void action('episodes/keep', { episode_id: takeId })}>Keep</button>
          <button disabled={busy} onClick={() => void action('episodes/discard', { episode_id: takeId })}>Discard</button>
        </div>
        {take.status === 'incomplete' && <p>Incomplete takes are preserved for inspection and cannot enter training data.</p>}
      </>}
      {status.phase === 'reset' && <>
        <p>Move the leader to reset the follower. Reset movements are not recorded.</p>
        <button disabled={busy} onClick={() => void action('reset/ready')}>Reset done / ready</button>
      </>}
    </section>}
  </main>;
}

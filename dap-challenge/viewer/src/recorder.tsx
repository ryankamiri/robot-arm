import { useEffect, useState } from 'react';
import { api } from './api';
import { ConnectionPanel } from './components/connection-panel';
import { CameraPanel } from './components/camera-panel';
import { SessionStatus } from './components/session-status';

type Take = {
  episode_id: string;
  name?: string;
  notes?: string;
  mac_camera?: { index: number } | null;
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

export function Recorder() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState('');
  const [episodeName, setEpisodeName] = useState('');
  const [episodeNotes, setEpisodeNotes] = useState('');
  const [lastKept, setLastKept] = useState('');
  const [videoError, setVideoError] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api('status');
        if (active) setStatus(next);
      } catch (failure) {
        if (active) { setError(String(failure)); setCountdown(null); }
      } finally {
        if (active) timer = setTimeout(poll, 500);
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, []);

  const takeId = selected || status?.active_episode_id || '';
  const take = status?.pending.find(entry => entry.episode_id === takeId);
  useEffect(() => {
    setVideoError(false);
    setEpisodeName(take?.name ?? '');
    setEpisodeNotes(take?.notes ?? '');
  }, [takeId]);

  async function action(path: string, body: unknown = {}) {
    setBusy(true);
    setError('');
    try {
      const next = await api(path, body);
      setStatus(previous => ({ ...next, demo: previous?.demo ?? next.demo }));
      if (path.endsWith('/keep')) setLastKept(episodeName.trim() || takeId);
      if (path.endsWith('/keep') || path.endsWith('/discard')) setSelected('');
    } catch (failure) {
      setError(String(failure));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (countdown === null) return;
    if (!status?.connected || status.phase !== 'ready') {
      setCountdown(null);
      return;
    }

    const timer = setTimeout(() => {
      if (countdown > 1) setCountdown(countdown - 1);
      else {
        setCountdown(null);
        setSelected('');
        void action('recording/start');
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown, status?.connected, status?.phase]);

  return <main>
    <h1>BC recorder</h1>
    <SessionStatus phase={status?.phase} demo={status?.demo} error={error || status?.error} />
    {status && !status.connected && <ConnectionPanel demo={status.demo} busy={busy}
      onConnect={settings => action('connect', settings)} />}

    {status?.connected && <section>
      <p>Follower follows the leader during recording, review, and reset.</p>
      <div className="actions">
        <button disabled={busy || countdown !== null || status.phase !== 'ready'} onClick={() => {
          setError(''); setCountdown(3);
        }}>Start recording</button>
        {countdown !== null && <button onClick={() => setCountdown(null)}>Cancel countdown</button>}
        <button disabled={busy || status.phase !== 'recording'} onClick={() => void action('recording/stop')}>Stop recording</button>
        <button disabled={busy} onClick={() => { setCountdown(null); void action('disconnect'); }}>End session / release motors</button>
      </div>
      {countdown !== null && <p className="countdown" role="status" aria-live="assertive">Recording starts in {countdown}…</p>}
      <p>Support the follower before ending the session: motor torque will be released.</p>
      <p>Recording target: 30 iterations per second. Rows in latest take: {status.recorded_rows}.</p>
      <p>Session missed deadlines: {status.diagnostics.missed_deadlines} · Slowest cycle: {(status.diagnostics.max_cycle_s * 1000).toFixed(1)} ms · Latest frame age: {(status.diagnostics.camera_age_s * 1000).toFixed(1)} ms</p>
      {status.diagnostics.missed_deadlines > 0 && <p role="status">Timing warning: recording continues. Inspect the take before keeping it.</p>}
    </section>}

    {status && status.phase !== 'recording' && status.phase !== 'finalizing' && <section>
      <h2>Review</h2>
      {lastKept && <p role="status">Saved episode: {lastKept}</p>}
      <select aria-label="Pending take" value={takeId} disabled={busy} onChange={event => setSelected(event.target.value)}>
        <option value="">Select a pending take</option>
        {status.pending.map(entry => <option key={entry.episode_id} value={entry.episode_id}>{entry.name || entry.episode_id} · {entry.status}</option>)}
      </select>
      {take?.status === 'recording' && <p>Finalizing pending take…</p>}
      {take && take.status !== 'recording' && <>
        <p>{take.status} · {take.row_count ?? 0} observations</p>
        {take.error && <p role="alert">{take.error}</p>}
        <CameraPanel key={takeId} title="Arm camera" src={`/api/episodes/${encodeURIComponent(takeId)}/video`}
          onError={() => setVideoError(true)} />
        {take.mac_camera && <CameraPanel key={`${takeId}-mac`} title="Mac camera"
          src={`/api/episodes/${encodeURIComponent(takeId)}/video?camera=mac`} onError={() => setVideoError(true)} />}
        {videoError && <p role="alert">Video could not be played. The pending files remain preserved.</p>}
        <p>Take missed deadlines: {take.diagnostics?.missed_deadlines ?? 'unavailable'} · Repeated camera frames: {take.diagnostics?.repeated_frames ?? 'unavailable'}</p>
        <label>Episode name (optional)
          <input value={episodeName} disabled={busy} onChange={event => setEpisodeName(event.target.value)} />
        </label>
        <label>Notes (optional)
          <textarea value={episodeNotes} disabled={busy} rows={3} onChange={event => setEpisodeNotes(event.target.value)} />
        </label>
        <div className="actions">
          <button disabled={busy || take.status !== 'review' || videoError} onClick={() => void action('episodes/keep', { episode_id: takeId, name: episodeName, notes: episodeNotes })}>Keep</button>
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

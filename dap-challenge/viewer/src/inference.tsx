import { useEffect, useState } from 'react';
import { api } from './api';
import { LiveCameras } from './components/live-cameras';
import { ConnectionPanel } from './components/connection-panel';
import { SessionStatus } from './components/session-status';

type Checkpoint = { path: string; label: string };
type Status = {
  demo: boolean; connected: boolean; phase: string; error: string | null; owned_by_you: boolean;
  policy_version: string | null; pending_policy_version: string | null; hold_confirmed: boolean; action_mode: string;
  starting_pose: { positions: number[] } | null; positions: number[] | null;
  requested_target: number[] | null; sent_target: number[] | null;
  settings: { control: Record<string, number> };
  diagnostics: { iterations: number; missed_deadlines: number; cycle_ms: number; prediction_ms: number; camera_age_s: number; limited_commands: number };
};

export function Inference() {
  const [clientId] = useState(() => {
    let id = sessionStorage.getItem('inference-client');
    if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('inference-client', id); }
    return id;
  });
  const [status, setStatus] = useState<Status | null>(null);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selected, setSelected] = useState('');
  const [mode, setMode] = useState('');
  const [error, setError] = useState('');
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const moving = status?.phase === 'running' || status?.phase === 'resetting';
  const owned = status?.connected && status.owned_by_you;
  const loaded = Boolean(status?.policy_version || status?.pending_policy_version);

  useEffect(() => { if (status && !mode) setMode(status.action_mode); }, [status?.action_mode, mode]);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api(`inference/status?client_id=${encodeURIComponent(clientId)}`);
        if (active) setStatus(next);
      } catch (failure) { if (active) setError(String(failure)); }
      finally { if (active) timer = setTimeout(poll, 500); }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [clientId]);

  useEffect(() => {
    if (!owned) return;
    const timer = setInterval(() => {
      void api('inference/heartbeat', { client_id: clientId }).catch(failure => setError(String(failure)));
    }, 500);
    function stopOnLeave() {
      navigator.sendBeacon('/api/inference/stop', new Blob([JSON.stringify({ client_id: clientId })], { type: 'application/json' }));
    }
    window.addEventListener('pagehide', stopOnLeave);
    return () => { clearInterval(timer); window.removeEventListener('pagehide', stopOnLeave); };
  }, [owned, clientId]);

  async function refresh() {
    try {
      const result = await api('inference/checkpoints');
      setCheckpoints(result.items); setWarnings(result.warnings);
      setSelected(previous => result.items.some((item: Checkpoint) => item.path === previous) ? previous : result.items[0]?.path ?? '');
    } catch (failure) { setError(String(failure)); }
  }
  useEffect(() => { void refresh(); }, []);

  async function action(path: string, body: Record<string, unknown> = {}) {
    setBusy(true); setError('');
    try {
      const result = await api(`inference/${path}`, { ...body, client_id: clientId });
      setStatus(previous => ({ ...previous, ...result }));
    } catch (failure) { setError(String(failure)); }
    finally { setBusy(false); }
  }

  return <main>
    <h1>Policy inference</h1>
    <SessionStatus phase={status?.phase} demo={status?.demo} error={error || status?.error} />
    <section>
      <label>Local checkpoint
        <select aria-label="Local checkpoint" value={selected} disabled={busy || moving} onChange={event => setSelected(event.target.value)}>
          <option value="">Select a checkpoint</option>
          {checkpoints.map(checkpoint => <option key={checkpoint.path} value={checkpoint.path}>{checkpoint.label}</option>)}
        </select>
      </label>
      <div className="actions">
        <button disabled={busy || moving} onClick={() => void refresh()}>Refresh checkpoints</button>
        <button disabled={busy || moving || !selected || (status?.connected && !owned)} onClick={() => void action('load', { checkpoint: selected })}>Load checkpoint</button>
      </div>
      {warnings.map((warning, index) => <p key={index} role="status">{warning}</p>)}
      <p>Loaded policy: {status?.policy_version ?? 'None'}</p>
      {status?.pending_policy_version && <p>Next attempt: {status.pending_policy_version}</p>}
      <label>Action selection
        <select aria-label="Action selection" value={mode} disabled={busy || moving} onChange={event => setMode(event.target.value)}>
          <option value="mean">Mean action — tanh(mean)</option>
          <option value="sample">Sample Gaussian, then tanh</option>
        </select>
      </label>
    </section>
    {status && !status.connected && <ConnectionPanel inference demo={status.demo} busy={busy} onConnect={settings => action('connect', { settings })} />}
    {status?.connected && <>
      {!owned && <p role="alert">Another browser owns this session. Return to that browser to stop or disconnect it.</p>}
      <section>
        <div className="actions">
          <button disabled={busy || !owned || moving || !loaded || !status.starting_pose} onClick={() => void action('start', { mode })}>Start policy</button>
          <button disabled={!owned} onClick={() => void action('stop')}>Stop policy / hold</button>
          <button disabled={busy || !owned || moving} onClick={() => void action('pose/capture')}>Capture starting pose</button>
          <button disabled={busy || !owned || status.phase === 'resetting' || !status.starting_pose} onClick={() => void action('pose/return')}>{status.phase === 'resetting' ? 'Returning to start…' : 'Return to start'}</button>
          <button disabled={!owned} onClick={() => void action('disconnect')}>Disconnect / release motors</button>
        </div>
        <p>{moving ? 'Runtime is sending position targets.' : status.hold_confirmed ? 'Hold command sent.' : 'Hold command not confirmed.'} Support the follower before Disconnect.</p>
        <p>Return to start stops the policy and moves the follower to the captured pose, then holds there.</p>
        <p>Starting pose: {status.starting_pose?.positions.map(value => value.toFixed(2)).join(', ') ?? 'Not captured'}</p>
        <p>Current positions: {status.positions?.map(value => value.toFixed(2)).join(', ') ?? 'Unavailable'}</p>
        <p>Requested target: {status.requested_target?.map(value => value.toFixed(2)).join(', ') ?? 'None'}</p>
        <p>Sent target: {status.sent_target?.map(value => value.toFixed(2)).join(', ') ?? 'None'}</p>
        <p>Control target: {status.settings.control.iterations_per_second} iterations per second. Latest cycle: {status.diagnostics.cycle_ms.toFixed(1)} ms · Prediction: {status.diagnostics.prediction_ms.toFixed(1)} ms.</p>
        <p>Missed deadlines: {status.diagnostics.missed_deadlines} · Limited commands: {status.diagnostics.limited_commands} · Oldest camera frame: {(status.diagnostics.camera_age_s * 1000).toFixed(0)} ms.</p>
        <p>No inference episodes or videos are saved. Browser loss stops policy motion after {status.settings.control.browser_timeout_s} seconds; devices remain connected and held.</p>
      </section>
      {owned && <LiveCameras clientId={clientId} />}
    </>}
  </main>;
}

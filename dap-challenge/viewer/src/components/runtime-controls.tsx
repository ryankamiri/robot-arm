import { useEffect, useState } from 'react';
import { api } from '../api';

export type RuntimeStatus = {
  demo: boolean; connected: boolean; phase: string; error: string | null; owned_by_you: boolean;
  policy_version: string | null; pending_policy_version: string | null; hold_confirmed: boolean; action_mode: string;
  starting_pose: { positions: number[] } | null; positions: number[] | null;
  requested_target: number[] | null; sent_target: number[] | null;
  settings: { control: Record<string, number> };
  diagnostics: { iterations: number; missed_deadlines: number; cycle_ms: number; prediction_ms: number; camera_age_s: number; limited_commands: number };
};

export function useRuntime(leavePath = 'inference/stop') {
  const [clientId] = useState(() => {
    let id = sessionStorage.getItem('inference-client');
    if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('inference-client', id); }
    return id;
  });
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const owned = Boolean(status?.connected && status.owned_by_you);
  const moving = status?.phase === 'running' || status?.phase === 'resetting';
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try { const next = await api(`inference/status?client_id=${encodeURIComponent(clientId)}`); if (active) setStatus(next); }
      catch (failure) { if (active) setError(String(failure)); }
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
    function leave() {
      navigator.sendBeacon(`/api/${leavePath}`, new Blob([JSON.stringify({ client_id: clientId })], { type: 'application/json' }));
    }
    window.addEventListener('pagehide', leave);
    return () => {
      clearInterval(timer);
      window.removeEventListener('pagehide', leave);
      leave();
    };
  }, [owned, clientId, leavePath]);
  async function action(path: string, body: Record<string, unknown> = {}) {
    setBusy(true); setError('');
    try {
      const result = await api(`inference/${path}`, { ...body, client_id: clientId });
      setStatus(previous => ({ ...previous, ...result, owned_by_you: path === 'connect' ? true : previous?.owned_by_you ?? false }));
    } catch (failure) { setError(String(failure)); }
    finally { setBusy(false); }
  }
  return { clientId, status, error, setError, busy, owned, moving, action };
}

export function RuntimeControls({ status, owned, busy, moving, action, children, collecting = false, showDisconnect = true }: {
  status: RuntimeStatus; owned: boolean; busy: boolean; moving: boolean; collecting?: boolean; showDisconnect?: boolean;
  action: (path: string, body?: Record<string, unknown>) => Promise<void>; children?: React.ReactNode;
}) {
  return <section>
    {!owned && <p role="alert">Another browser owns this session.</p>}
    <p>Active policy: {status.policy_version ?? 'None'}</p>
    {status.pending_policy_version && <p>Next attempt: {status.pending_policy_version}</p>}
    <div className="actions">
      {children}
      <button disabled={busy || !owned || moving} onClick={() => void action('pose/capture')}>Capture starting pose</button>
      <button disabled={busy || !owned || collecting || status.phase === 'resetting' || !status.starting_pose} onClick={() => void action('pose/return')}>{status.phase === 'resetting' ? 'Returning to start…' : 'Return to start'}</button>
      {showDisconnect && <button disabled={!owned} onClick={() => void action('disconnect')}>Disconnect / release motors</button>}
    </div>
    <p>{moving ? 'Runtime is sending position targets.' : status.hold_confirmed ? 'Hold command sent.' : 'Hold command not confirmed.'} Support the follower before Disconnect.</p>
    <details><summary>Positions and timing</summary>
    <p>Starting pose: {status.starting_pose?.positions.map(value => value.toFixed(2)).join(', ') ?? 'Not captured'}</p>
    <p>Current positions: {status.positions?.map(value => value.toFixed(2)).join(', ') ?? 'Unavailable'}</p>
    <p>Requested target: {status.requested_target?.map(value => value.toFixed(2)).join(', ') ?? 'None'}</p>
    <p>Sent target: {status.sent_target?.map(value => value.toFixed(2)).join(', ') ?? 'None'}</p>
    <p>Latest cycle: {status.diagnostics.cycle_ms.toFixed(1)} ms | Prediction: {status.diagnostics.prediction_ms.toFixed(1)} ms | Missed deadlines: {status.diagnostics.missed_deadlines} | Limited commands: {status.diagnostics.limited_commands}</p>
    </details>
  </section>;
}

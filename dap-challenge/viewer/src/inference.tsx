import { useEffect, useState } from 'react';
import { api } from './api';
import { LiveCameras } from './components/live-cameras';
import { ConnectionPanel } from './components/connection-panel';
import { useRuntime, RuntimeControls } from './components/runtime-controls';
import { SessionStatus } from './components/session-status';

type Checkpoint = { path: string; label: string };
export function Inference() {
  const { clientId, status, error, setError, busy, owned, moving, action } = useRuntime();
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selected, setSelected] = useState('');
  const [mode, setMode] = useState('');
  const [warnings, setWarnings] = useState<string[]>([]);
  const loaded = Boolean(status?.policy_version || status?.pending_policy_version);
  useEffect(() => { if (status && !mode) setMode(status.action_mode); }, [status?.action_mode, mode]);

  async function refresh() {
    try {
      const result = await api('inference/checkpoints');
      setCheckpoints(result.items); setWarnings(result.warnings);
      setSelected(previous => result.items.some((item: Checkpoint) => item.path === previous) ? previous : result.items[0]?.path ?? '');
    } catch (failure) { setError(String(failure)); }
  }
  useEffect(() => { void refresh(); }, []);

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
      <RuntimeControls {...{ status, owned, busy, moving, action }}>
        <button disabled={busy || !owned || moving || !loaded || !status.starting_pose} onClick={() => void action('start', { mode })}>Start policy</button>
        <button disabled={!owned} onClick={() => void action('stop')}>Stop policy / hold</button>
      </RuntimeControls>
      <p>No inference episodes or videos are saved. Browser loss stops motion and holds the arm.</p>
      {owned && <LiveCameras clientId={clientId} />}
    </>}
  </main>;
}

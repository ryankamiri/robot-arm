import { useEffect, useState } from 'react';
import { api } from './api';
import { useRuntime, RuntimeControls } from './components/runtime-controls';
import { ConnectionPanel } from './components/connection-panel';
import { LiveCameras } from './components/live-cameras';
import { SessionStatus } from './components/session-status';

type Criterion = { label: string; weight: number; optional?: boolean; gated?: boolean; anchors: string[] };
type Task = { label: string; gates: string[]; criteria: Record<string, Criterion> };
type Episode = { episode_id: string; status: string; selected_tasks: string[]; transitions: number; policy_version: string; reward?: number; error?: string };
type RLStatus = {
  run_id: string | null; last_run_id: string | null; owned_by_you: boolean; runs: string[]; active_attempt: string | null; maximum: number;
  rubric: { version: string; tasks: Record<string, Task> } | null; episodes: Episode[];
  learner: { state: string; alive?: boolean; critic_updates?: number; actor_updates?: number; earned_updates?: number; error?: string; exit_code?: number };
};
type Answers = Record<string, Record<string, number | boolean | null>>;

function GradeForm({ episode, tasks, busy, submit, discard, videoUrl }: {
  episode: Episode; tasks: Record<string, Task>; busy: boolean; videoUrl: (view: string) => string;
  submit: (cued: string[], answers: Answers, notes: string) => void; discard: () => void;
}) {
  const [cued, setCued] = useState(episode.selected_tasks);
  const [answers, setAnswers] = useState<Answers>(() => Object.fromEntries(episode.selected_tasks.map(name => [name,
    Object.fromEntries(Object.keys(tasks[name].criteria).map(key => [key, tasks[name].gates.includes(key) ? false : 0]))])));
  const [tracking, setTracking] = useState<number | null>(0);
  const trackingCriterion = tasks[episode.selected_tasks[0]].criteria.tracking;
  const gradedAnswers: Answers = Object.fromEntries(cued.map(name => [name, { ...answers[name], tracking }]));
  const [notes, setNotes] = useState('');
  function setAnswer(name: string, key: string, value: number | boolean | null) {
    setAnswers(previous => ({ ...previous, [name]: { ...previous[name], [key]: value } }));
  }
  const scores = cued.map(name => Object.entries(tasks[name].criteria).reduce((total, [key, criterion]) =>
    total + (criterion.gated === false || tasks[name].gates.every(gate => answers[name][gate] === true)
      ? criterion.weight * Number(gradedAnswers[name][key] ?? 0) : 0), 0));
  const preview = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  return <section>
    <h2>Review rollout</h2><p>{episode.episode_id} | {episode.transitions} transitions | {episode.policy_version}</p>
    <div className="actions">{['arm', 'mac'].map(view => <label key={view}>{view === 'arm' ? 'Arm camera' : 'Mac camera'}<video controls preload="metadata" src={videoUrl(view)} style={{ width: '100%', maxWidth: 480 }} /></label>)}</div>
    {episode.selected_tasks.map(name => <fieldset key={name} disabled={busy}>
      <legend>{tasks[name].label}</legend>
      <label><input type="checkbox" checked={cued.includes(name)} onChange={event => setCued(previous => event.target.checked ? [...previous, name] : previous.filter(item => item !== name))} /> I actually cued this behavior</label>
      {cued.includes(name) && Object.entries(tasks[name].criteria).filter(([key]) => key !== 'tracking').map(([key, criterion]) => <label key={key}>
        {criterion.label} | weight {criterion.weight}
        <select value={answers[name][key] === null ? 'na' : String(answers[name][key])} disabled={criterion.gated !== false && !tasks[name].gates.includes(key) && !tasks[name].gates.every(gate => answers[name][gate] === true)}
          onChange={event => setAnswer(name, key, event.target.value === 'na' ? null : tasks[name].gates.includes(key) ? event.target.value === 'true' : Number(event.target.value))}>
          {criterion.optional && <option value="na">Not tested</option>}
          {tasks[name].gates.includes(key) ? <><option value="false">No | {criterion.anchors[0]}</option><option value="true">Yes | {criterion.anchors[1]}</option></>
            : criterion.anchors.map((anchor, index) => <option key={index} value={index / 2}>{index / 2} | {anchor}</option>)}
        </select>
      </label>)}
    </fieldset>)}
    <fieldset disabled={busy || !cued.length}>
      <legend>Rollout bonus</legend>
      <label>{trackingCriterion.label} | weight {trackingCriterion.weight}
        <select value={tracking === null ? 'na' : String(tracking)} onChange={event => setTracking(event.target.value === 'na' ? null : Number(event.target.value))}>
          {trackingCriterion.optional && <option value="na">Not tested</option>}
          {trackingCriterion.anchors.map((anchor, index) => <option key={index} value={index / 2}>{index / 2} | {anchor}</option>)}
        </select>
      </label>
      <p>Grade once for the entire rollout. This bonus applies independently of either behavior succeeding.</p>
    </fieldset>
    <p>Reward preview: {preview?.toFixed(3) ?? 'No cued task'}. The backend recomputes it before replay admission. Untested tracking earns no bonus.</p>
    <label>Notes<textarea maxLength={5000} value={notes} onChange={event => setNotes(event.target.value)} /></label>
    <div className="actions"><button disabled={busy || !cued.length} onClick={() => submit(cued, gradedAnswers, notes)}>Submit score and continue</button><button disabled={busy} onClick={discard}>Abort / exclude rollout</button></div>
  </section>;
}

export function RLRecorder() {
  const live = useRuntime('rl/attempt/abort');
  const { clientId, status, busy: runtimeBusy, owned, moving, action } = live;
  const [rl, setRL] = useState<RLStatus | null>(null);
  const [name, setName] = useState(() => `sac-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}`);
  const [tasks, setTasks] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [countdown, setCountdown] = useState<number | null>(null);
  const activeRun = Boolean(rl?.run_id && rl.owned_by_you);
  const pending = rl?.episodes.find(episode => episode.status === 'review');
  const finalizing = activeRun && !rl?.active_attempt && Boolean(rl?.episodes.some(episode => episode.status === 'recording'));
  const step = activeRun && pending ? 'review' : finalizing ? 'saving' : activeRun && owned ? 'rollout' : 'setup';
  const rollout = rl?.episodes.find(episode => episode.episode_id === rl.active_attempt) ?? pending
    ?? (finalizing ? rl?.episodes.find(episode => episode.status === 'recording') : undefined);
  const policyVersion = rollout?.policy_version ?? status?.policy_version;
  const working = busy || runtimeBusy;
  const counting = countdown !== null;
  const canStartRollout = activeRun && owned && status?.phase === 'holding' && !working && !pending
    && !finalizing && !rl?.active_attempt && tasks.length > 0 && Boolean(status?.starting_pose);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try { const next = await api(`rl/status?client_id=${encodeURIComponent(clientId)}`); if (active) setRL(next); }
      catch (failure) { if (active) setError(String(failure)); }
      finally { if (active) timer = setTimeout(poll, 1000); }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [clientId]);
  async function command(path: string, body: Record<string, unknown> = {}) {
    setCountdown(null);
    setBusy(true); setError('');
    try {
      const next = await api(`rl/${path}`, { client_id: clientId, ...body });
      setRL(next);
      setNotice(path === 'attempt/score' ? 'Score saved. Return to start when you are ready for the next rollout.'
        : path === 'attempt/discard' ? 'Rollout excluded. Return to start for the next rollout.' : '');
    }
    catch (failure) { setError(String(failure)); }
    finally { setBusy(false); }
  }
  useEffect(() => {
    if (countdown === null) return;
    if (!canStartRollout || error || live.error) {
      setCountdown(null);
      return;
    }

    const timer = setTimeout(() => {
      if (countdown > 1) setCountdown(countdown - 1);
      else void command('attempt/start', { tasks });
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown, canStartRollout, error, live.error, tasks]);

  useEffect(() => { window.scrollTo(0, 0); }, [step]);

  return <main>
    <h1>RL rollouts</h1>
    <SessionStatus phase={status?.phase} demo={status?.demo} error={error || live.error || status?.error} />
    <section aria-label="Learner status">
      <h2>Learner status</h2>
      <details open={!rl?.run_id}>
        <summary>RL run: {rl?.run_id ?? 'Choose a run'}</summary>
        <label>RL run name<input value={name} disabled={working || Boolean(rl?.learner.alive) || moving || counting} onChange={event => setName(event.target.value)} /></label>
        <label>Resume local run<select value={rl?.runs.includes(name) ? name : ''} disabled={working || Boolean(rl?.learner.alive) || moving || counting} onChange={event => setName(event.target.value)}><option value="">Select a saved run</option>{rl?.runs.map(run => <option key={run} value={run}>{run}</option>)}</select></label>
        <button disabled={working || !name || Boolean(rl?.learner.alive) || moving || counting || Boolean(rl?.active_attempt) || (Boolean(status?.connected) && !owned)} onClick={() => void command('run/load', { run_id: name })}>Load run</button>
      </details>
      <p><strong>{rollout ? 'Rollout policy' : 'Current policy'}:</strong> {policyVersion ?? 'Not loaded'}</p>
      {status?.pending_policy_version && <p><strong>Next rollout policy:</strong> {status.pending_policy_version}</p>}
      <p>{rl?.learner.state ?? 'Not started'} ({rl?.learner.alive ? 'process running' : 'process stopped'}) | Critic updates: {rl?.learner.critic_updates ?? 0} / {rl?.learner.earned_updates ?? 0} earned | Actor updates: {rl?.learner.actor_updates ?? 0}</p>
      <div className="actions">
        <button disabled={working || !activeRun || Boolean(rl?.learner.alive)} onClick={() => void command('training/start')}>Start training</button>
        <button disabled={working || !activeRun || !rl?.learner.alive || rl.learner.state === 'stopping'} onClick={() => void command('training/stop')}>Stop training</button>
      </div>
      <p>Training and the arm connection are independent. You can collect and score rollouts while training is stopped.</p>
    </section>
    {rl?.learner.error && <pre role="alert">{rl.learner.error}</pre>}
    <ol aria-label="Collection steps">
      <li aria-current={step === 'setup' ? 'step' : undefined}>Set up session</li>
      <li aria-current={step === 'rollout' ? 'step' : undefined}>Collect rollout</li>
      <li aria-current={step === 'review' || step === 'saving' ? 'step' : undefined}>Review and score</li>
    </ol>
    {activeRun && <section>
      <p>{rl?.run_id} | Scored rollouts: {rl?.episodes.filter(episode => episode.status === 'scored').length ?? 0} / {rl?.maximum ?? 50}</p>

    </section>}
    {owned && <button disabled={runtimeBusy} onClick={() => { setCountdown(null); void action('disconnect'); }}>End session</button>}
    {notice && <p role="status">{notice}</p>}
    {step === 'setup' && <section>
      <h2>Set up session</h2>
      {!activeRun && <p>Load an RL run above to choose its policy and rollout storage. Start training separately whenever you want.</p>}
      {status && !status.connected && <ConnectionPanel inference demo={status.demo} busy={working} connectLabel="Start session" onConnect={settings => action('connect', { settings })} />}
      {status?.connected && !owned && <p>Connect from the browser that owns the follower.</p>}
      {owned && <LiveCameras clientId={clientId} />}
    </section>}
    {step === 'saving' && <section aria-live="polite"><h2>Preparing review…</h2><p>The arm is holding while both videos finish saving.</p></section>}
    {step === 'rollout' && status && <>
      <h2>Collect rollout</h2>
      <fieldset disabled={working || counting || moving || Boolean(pending)}><legend>Behaviors to test this rollout</legend>{Object.entries(rl?.rubric?.tasks ?? {}).map(([key, task]) => <label key={key}><input type="checkbox" checked={tasks.includes(key)} onChange={event => setTasks(previous => event.target.checked ? [...previous, key] : previous.filter(name => name !== key))} />{task.label}</label>)}</fieldset>
      <RuntimeControls {...{ status, owned, busy: working || counting, moving }} action={(path, body) => { setCountdown(null); return action(path, body); }} collecting={Boolean(rl?.active_attempt)} showDisconnect={false}>
        <button disabled={!canStartRollout || counting} onClick={() => { setError(''); live.setError(''); setCountdown(3); }}>Start rollout</button>
        {counting && <button onClick={() => setCountdown(null)}>Cancel countdown</button>}
        <button disabled={!owned || !rl?.active_attempt} onClick={() => void command('attempt/stop')}>End rollout</button>
        <button disabled={!owned || !rl?.active_attempt} onClick={() => void command('attempt/abort')}>Abort rollout</button>
      </RuntimeControls>
      {counting && <p className="countdown" role="status" aria-live="assertive">Rollout starts in {countdown}…</p>}
      <p>Rollouts sample policy actions. End rollout holds the arm for review and scoring. End session releases the motors.</p>
      {owned && <LiveCameras clientId={clientId} />}
    </>}
    {step === 'review' && pending && rl?.rubric && <>
      <p>The arm is holding. Review both videos and submit a score to return to the rollout controls.</p>
      <GradeForm key={pending.episode_id} episode={pending} tasks={rl.rubric.tasks} busy={working}
      videoUrl={view => `/api/rl/attempt/${pending.episode_id}/video?client_id=${encodeURIComponent(clientId)}&camera=${view}`}
      submit={(cued, answers, notes) => void command('attempt/score', { episode_id: pending.episode_id, cued_tasks: cued, answers, notes })}
      discard={() => void command('attempt/discard', { episode_id: pending.episode_id })} />
    </>}
    <details><summary>Rollout history</summary><p>{rl?.last_run_id}</p>{rl?.episodes.map(episode => <p key={episode.episode_id}>{episode.episode_id} | {episode.status} | {episode.transitions} transitions{episode.reward !== undefined ? ` | reward ${episode.reward.toFixed(3)}` : ''}{episode.error ? ` | ${episode.error}` : ''}</p>)}</details>
  </main>;
}

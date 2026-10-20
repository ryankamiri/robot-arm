import { NavLink } from 'react-router';

type Props = { phase?: string; demo?: boolean; error?: string | null };

export function SessionStatus({ phase, demo, error }: Props) {
  return <>
    <nav><NavLink to="/record/bc">BC recorder</NavLink> | <NavLink to="/inference">Policy inference</NavLink> | <NavLink to="/record/rl">RL rollouts</NavLink></nav>
    {demo && <p role="status"><strong>Demo: synthetic devices. No physical robot or camera is connected.</strong></p>}
    <p>Session: <strong>{phase ?? 'Connecting to backend…'}</strong></p>
    {error && <p role="alert">{error}</p>}
  </>;
}

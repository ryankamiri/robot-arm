type Props = { phase?: string; demo?: boolean; error?: string | null };

export function SessionStatus({ phase, demo, error }: Props) {
  return <>
    <nav><a href="/record/bc">BC recorder</a> · <a href="/inference">Policy inference</a></nav>
    {demo && <p role="status"><strong>Demo: synthetic devices. No physical robot or camera is connected.</strong></p>}
    <p>Session: <strong>{phase ?? 'Connecting to backend…'}</strong></p>
    {error && <p role="alert">{error}</p>}
  </>;
}

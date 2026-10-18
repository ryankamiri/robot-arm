import { useEffect, useState, type FormEvent } from 'react';
import { api } from '../api';

type Props = { demo: boolean; busy: boolean; inference?: boolean; onConnect: (settings: Record<string, unknown>) => Promise<void> };

export function ConnectionPanel({ demo, busy, inference = false, onConnect }: Props) {
  const [ports, setPorts] = useState<{ port: string; description: string; serial?: string | null }[]>([]);
  const [cameras, setCameras] = useState<{ index: number; name: string; device_id: string }[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [profile, setProfile] = useState('');
  const [form, setForm] = useState({ leader_port: '', follower_port: '', leader_id: '', follower_id: '',
    calibration_dir: '', camera_index: '', mac_camera_index: '', width: '640', height: '480', camera_fps: '30' });

  async function discover() {
    setDiscovering(true);
    try {
      const devices = await api('devices');
      setPorts(devices.ports); setCameras(devices.cameras);
      setWarnings(devices.warnings.filter((warning: string) => !inference || !warning.startsWith('Leader')));
      setForm(previous => {
        const next = { ...previous };
        for (const key of Object.keys(next) as (keyof typeof next)[]) {
          if (key in devices.settings) next[key] = String(devices.settings[key] ?? '');
        }
        return next;
      });
    } catch (error) { setWarnings([String(error)]); }
    finally { setDiscovering(false); }
  }

  useEffect(() => { if (!demo) void discover(); }, [demo]);

  async function connect(event: FormEvent) {
    event.preventDefault();
    await onConnect({ ...form, calibration_dir: form.calibration_dir || null,
      leader_serial: ports.find(port => port.port === form.leader_port)?.serial ?? null,
      follower_serial: ports.find(port => port.port === form.follower_port)?.serial ?? null,
      camera_device_id: cameras.find(camera => camera.index === Number(form.camera_index))?.device_id ?? null,
      mac_camera_device_id: cameras.find(camera => camera.index === Number(form.mac_camera_index))?.device_id ?? null,
      mac_camera_index: form.mac_camera_index === '' ? null : Number(form.mac_camera_index),
      camera_index: Number(form.camera_index), width: Number(form.width), height: Number(form.height), camera_fps: Number(form.camera_fps),
    });
  }

  async function inspect() {
    setDiscovering(true);
    try { const result = await api('camera/inspect', { camera_index: Number(form.camera_index) }); setProfile(JSON.stringify(result.camera_profile, null, 2)); }
    catch (error) { setWarnings([String(error)]); }
    finally { setDiscovering(false); }
  }

  const fields = inference ? ['follower_port', 'follower_id', 'calibration_dir'] as const
    : ['leader_port', 'follower_port', 'leader_id', 'follower_id', 'calibration_dir'] as const;
  return <form onSubmit={connect}>
    <fieldset disabled={busy || discovering}>
      <legend>Connect session</legend>
      <button type="button" onClick={() => void discover()}>{discovering ? 'Finding devices…' : 'Rediscover saved setup'}</button>
      <p>{inference ? 'Connect holds the follower at its measured position. The leader is not used.' : 'Devices are detected automatically. Connect starts teleoperation.'}</p>
      {warnings.map((warning, index) => <p role="status" key={index}>{warning}</p>)}
      <datalist id="ports">{ports.map(port => <option key={port.port} value={port.port} />)}</datalist>
      {fields.map(key => <label key={key}>{key.replaceAll('_', ' ')}
        <input value={form[key]} list={key.endsWith('port') ? 'ports' : undefined} required={!demo && key !== 'calibration_dir'}
          onChange={event => setForm({ ...form, [key]: event.target.value })} />
      </label>)}
      <p>Use existing calibrated arm IDs. Support the follower while connecting or disconnecting.</p>
      <label>Arm camera
        <select aria-label="Camera" required={!demo} value={form.camera_index} onChange={event => setForm({ ...form, camera_index: event.target.value })}>
          <option value="">Select a camera</option>
          {cameras.map(camera => <option key={camera.device_id} value={camera.index}>{camera.name} · index {camera.index}</option>)}
        </select>
      </label>
      <label>Mac camera
        <select aria-label="Mac camera" required={!demo} value={form.mac_camera_index} onChange={event => setForm({ ...form, mac_camera_index: event.target.value })}>
          <option value="">Select the Mac camera</option>
          {cameras.filter(camera => String(camera.index) !== form.camera_index).map(camera => <option key={camera.device_id} value={camera.index}>{camera.name} · index {camera.index}</option>)}
        </select>
      </label>
      {(['width', 'height', 'camera_fps'] as const).map(key => <label key={key}>{key === 'camera_fps' ? 'Camera images per second' : key}
        <input type="number" min={1} required value={form[key]} onChange={event => setForm({ ...form, [key]: event.target.value })} />
      </label>)}
      {!inference && <button type="button" disabled={demo} onClick={() => void inspect()}>Inspect selected camera</button>}
      {profile && <pre>{profile}</pre>}
      <p>Model input: two views at 224 × 168. {inference && 'Capture: 640 × 480, 30 images per second.'}</p>
      <button type="submit">{demo ? 'Connect demo session' : inference ? 'Connect follower and cameras' : 'Connect arms and camera'}</button>
    </fieldset>
  </form>;
}

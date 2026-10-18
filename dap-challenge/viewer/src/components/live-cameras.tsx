import { useEffect, useState } from 'react';
import { CameraPanel } from './camera-panel';

type View = 'arm' | 'mac';
type Answer = { sdp: string; type: 'answer'; views: Record<string, View>; peer_id: string };

async function gather(peer: RTCPeerConnection, signal: AbortSignal) {
  if (peer.iceGatheringState === 'complete') return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      peer.removeEventListener('icegatheringstatechange', changed);
      signal.removeEventListener('abort', aborted);
    };
    const changed = () => { if (peer.iceGatheringState === 'complete') { cleanup(); resolve(); } };
    const aborted = () => { cleanup(); reject(new Error('Camera connection cancelled.')); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('Camera network discovery timed out.')); }, 10000);
    peer.addEventListener('icegatheringstatechange', changed);
    signal.addEventListener('abort', aborted, { once: true });
    if (signal.aborted) aborted();
    else changed();
  });
}

export function LiveCameras({ clientId }: { clientId: string }) {
  const [streams, setStreams] = useState<Partial<Record<View, MediaStream>>>({});
  const [state, setState] = useState('Connecting');
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const peer = new RTCPeerConnection({ iceServers: [], bundlePolicy: 'max-bundle' });
    let views: Record<string, View> = {};
    let peerId = '';
    setStreams({}); setState('Connecting'); setError('');
    const release = () => {
      if (peerId) {
        navigator.sendBeacon('/api/inference/preview/close', new Blob([JSON.stringify({ client_id: clientId, peer_id: peerId })], { type: 'application/json' }));
        peerId = '';
      }
    };
    peer.ontrack = event => {
      const view = views[event.transceiver.mid ?? ''];
      if (!view || controller.signal.aborted) return;
      setStreams(previous => ({ ...previous, [view]: new MediaStream([event.track]) }));
    };
    peer.onconnectionstatechange = () => {
      if (controller.signal.aborted) return;
      setState(peer.connectionState);
      if (['failed', 'disconnected', 'closed'].includes(peer.connectionState)) setError('Video connection lost. Retry the previews.');
    };
    const timeout = setTimeout(() => {
      if (peer.connectionState !== 'connected') {
        setError('Camera connection timed out. Retry the previews.');
        controller.abort(); peer.close(); release();
      }
    }, 20000);
    async function connect() {
      peer.addTransceiver('video', { direction: 'recvonly' });
      peer.addTransceiver('video', { direction: 'recvonly' });
      await peer.setLocalDescription(await peer.createOffer());
      await gather(peer, controller.signal);
      const response = await fetch('/api/inference/preview/offer', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ client_id: clientId, sdp: peer.localDescription!.sdp, type: 'offer' }),
      });
      const answer: Answer & { detail?: string } = await response.json();
      if (!response.ok) throw new Error(answer.detail ?? 'Could not connect camera previews.');
      views = answer.views; peerId = answer.peer_id;
      if (controller.signal.aborted) { release(); return; }
      await peer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    }
    void connect().catch(failure => {
      if (!controller.signal.aborted) setError(String(failure));
      peer.close(); release();
    });
    window.addEventListener('pagehide', release);
    return () => {
      clearTimeout(timeout); controller.abort();
      peer.getReceivers().forEach(receiver => receiver.track.stop());
      peer.close(); release(); window.removeEventListener('pagehide', release);
    };
  }, [clientId, attempt]);
  return <section>
    <p role="status">Live video: {state}</p>
    {error && <p role="alert">{error} <button onClick={() => setAttempt(value => value + 1)}>Retry video</button></p>}
    <CameraPanel title="Arm camera" stream={streams.arm} />
    <CameraPanel title="Mac camera" stream={streams.mac} />
  </section>;
}

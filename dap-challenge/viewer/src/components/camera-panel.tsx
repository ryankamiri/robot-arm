import { useEffect, useRef } from 'react';

type Props = { title: string; src?: string; stream?: MediaStream; onError?: () => void };

export function CameraPanel({ title, src, stream, onError }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const element = video.current;
    if (!element) return;
    element.srcObject = stream ?? null;
    return () => { element.srcObject = null; };
  }, [stream]);
  return <section>
    <h3>{title}</h3>
    <video ref={video} aria-label={title} controls={!stream} autoPlay={!!stream} muted={!!stream}
      playsInline preload="metadata" src={src} onError={onError} />
  </section>;
}

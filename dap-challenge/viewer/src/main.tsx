import { createRoot } from 'react-dom/client';
import { Recorder } from './recorder';
import { Inference } from './inference';
import './style.css';

createRoot(document.getElementById('root')!).render(
  window.location.pathname === '/record/bc'
    ? <Recorder />
    : window.location.pathname === '/inference'
      ? <Inference />
      : <main><a href="/record/bc">BC recorder</a> · <a href="/inference">Policy inference</a></main>,
);

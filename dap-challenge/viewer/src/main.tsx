import { createRoot } from 'react-dom/client';
import { Recorder } from './recorder';
import './style.css';

createRoot(document.getElementById('root')!).render(
  window.location.pathname === '/record/bc'
    ? <Recorder />
    : <main><a href="/record/bc">Open BC recorder</a></main>,
);

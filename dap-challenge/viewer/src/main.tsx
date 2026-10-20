import { createRoot } from 'react-dom/client';
import { BrowserRouter, NavLink, Route, Routes } from 'react-router';
import { Recorder } from './recorder';
import { RLRecorder } from './rl';
import { Inference } from './inference';
import './style.css';

function Home() {
  return <main>
    <NavLink to="/record/bc">BC recorder</NavLink> | <NavLink to="/inference">Policy inference</NavLink> | <NavLink to="/record/rl">RL rollouts</NavLink>
  </main>;
}

createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/record/bc" element={<Recorder />} />
      <Route path="/record/rl" element={<RLRecorder />} />
      <Route path="/inference" element={<Inference />} />
      <Route path="*" element={<Home />} />
    </Routes>
  </BrowserRouter>,
);

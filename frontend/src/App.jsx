import { useState } from 'react';
import WebThreads from './components/WebThreads';
import SpecularButton from './components/SpecularButton';

const Icon = ({ children }) => <span className="menu-icon" aria-hidden="true">{children}</span>;
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function ChatInterface({ onHome }) {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const send = async event => {
    event.preventDefault();
    const text = message.trim();
    if (!text || isSending) return;

    setMessages(current => [...current, { role: 'user', text }]);
    setMessage('');
    setIsSending(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setMessages(current => [...current, { role: 'assistant', text: data.answer }]);
    } catch (error) {
      setMessages(current => [
        ...current,
        { role: 'error', text: 'MinionAI could not reach the backend. Please try again.' },
      ]);
      console.error(error);
    } finally {
      setIsSending(false);
    }
  };
  const recents = ['Plan my product launch', 'Build a memory system', 'Explain neural networks', 'Write a launch announcement', 'Design an AI workflow'];
  return (
    <main className="chat-app">
      <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar--closed'}`}>
        <div className="sidebar-head">
          <button className="chat-brand" onClick={onHome}><img src="/statics/logo.png" alt=""/><b>MinionAI</b></button>
          <button className="icon-button" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">◧</button>
        </div>
        <nav className="side-menu">
          <button className="active"><Icon>✎</Icon>New chat</button>
          <button><Icon>▥</Icon>Library</button>
          <button><Icon>□</Icon>Projects</button>
          <button><Icon>◷</Icon>Scheduled</button>
          <button><Icon>✦</Icon>Agents</button>
          <button><Icon>•••</Icon>More</button>
        </nav>
        <div className="recents"><p>Recents</p>{recents.map(item => <button key={item}>{item}</button>)}</div>
        <button className="profile"><span>SS</span><span><b>Savio Sunny</b><small>Free plan</small></span><i>•••</i></button>
      </aside>

      <section className="chat-main">
        {!sidebarOpen && <button className="open-sidebar icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">☰</button>}
        <button className="help-button" aria-label="Help">?</button>
        <div className={`chat-stage ${messages.length ? 'has-messages' : ''}`}>
          {messages.length === 0 ? (
            <div className="welcome">
              <img src="/statics/logo.png" alt="MinionAI" />
              <h1>What can I help with?</h1>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((item, index) => (
                <div className={`${item.role}-message`} key={`${item.role}-${index}`}>{item.text}</div>
              ))}
              {isSending && <div className="assistant-message assistant-message--loading">MinionAI is thinking…</div>}
            </div>
          )}
          <form className="composer" onSubmit={send}>
            <button type="button" className="add-button" aria-label="Add attachment">＋</button>
            <input value={message} onChange={e => setMessage(e.target.value)} placeholder="Ask MinionAI" aria-label="Message MinionAI" disabled={isSending} />
            <button type="button" className="model-button">Minion <span>⌄</span></button>
            <button type="button" className="mic-button" aria-label="Voice input">♩</button>
            <button className="send-button" aria-label="Send message" disabled={isSending || !message.trim()}>↑</button>
          </form>
          {messages.length === 0 && <div className="quick-actions">
            <button><Icon>▧</Icon>Create an image</button>
            <button><Icon>✎</Icon>Write or edit</button>
            <button><Icon>◎</Icon>Search the web</button>
          </div>}
        </div>
      </section>
    </main>
  );
}

export default function App() {
  const [started, setStarted] = useState(false);
  if (started) return <ChatInterface onHome={() => setStarted(false)} />;
  return (
    <main className="page">
      <div className="ambient" aria-hidden="true">
        <WebThreads
          color1="#5252ff" color2="#f17dff" color3="#ffffff"
          speed={0.2} threadCount={6} frequency={5} spread={0.18}
          taper={1} position={0.49} fanMode="center" glow={0.02}
          falloff={0.6} thickness={1.1} brightness={0.7} opacity={1}
          mirror shimmer={false} grain grainIntensity={0.045}
          mouseInteraction mouseStrength={0.3}
        />
      </div>

      <nav className="nav" aria-label="Main navigation">
        <div className="banner-left">
          <a className="brand" href="#top" aria-label="Minion AI home">
            <img src="/statics/logo.png" alt="" />
            <span className="brand-copy"><b>MinionAI</b><small>Aonz AI Studio · 2026</small></span>
          </a>
        </div>
        <div className="nav-links">
          <a href="#work">Work</a><a href="#about">About</a><a href="#contact">Contact</a>
        </div>
        <SpecularButton onClick={() => setStarted(true)} className="nav-cta" size="lg" radius={24} lineColor="#cf9cff" baseColor="#61556d" intensity={1.3} shineSize={12} shineFade={45} followMouse proximity={220}>
          Get Started
        </SpecularButton>
      </nav>

      <section className="hero" id="top">
        <div className="hero-top"><span>Chat</span><span>with</span></div>
        <div className="hero-bottom"><h1>Minion</h1><p>An AI that remembers.</p></div>
      </section>
    </main>
  );
}

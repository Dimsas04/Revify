import React, { useState } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const { signIn, signUp, signInWithGoogle } = useAuth();
  const [mode, setMode] = useState('sign-in');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setMessage('');
    setSubmitting(true);
    const { error: authError, data } = mode === 'sign-in'
      ? await signIn(email, password)
      : await signUp(email, password, name);
    setSubmitting(false);

    if (authError) {
      setError(authError.message);
      return;
    }
    if (mode === 'sign-up' && !data.session) {
      setMessage('Account created. Check your email to confirm your address.');
      return;
    }
    navigate(location.state?.from?.pathname || '/analysis', { replace: true });
  };

  const handleGoogle = async () => {
    setError('');
    const { error: authError } = await signInWithGoogle();
    if (authError) setError(authError.message);
  };

  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <div className="rounded-2xl bg-white p-8 shadow-xl">
        <h1 className="mb-2 text-3xl font-bold text-gray-900">
          {mode === 'sign-in' ? 'Welcome back' : 'Create your account'}
        </h1>
        <p className="mb-6 text-gray-600">Sign in to run and save Revify analyses.</p>
        <button type="button" onClick={handleGoogle} className="mb-4 w-full rounded-lg border px-4 py-3 font-medium hover:bg-gray-50">
          Continue with Google
        </button>
        <div className="mb-4 text-center text-sm text-gray-500">or use email</div>
        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === 'sign-up' && (
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" className="w-full rounded-lg border px-4 py-3" />
          )}
          <input required type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" className="w-full rounded-lg border px-4 py-3" />
          <input required minLength={6} type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" className="w-full rounded-lg border px-4 py-3" />
          {error && <p className="text-sm text-red-600">{error}</p>}
          {message && <p className="text-sm text-green-600">{message}</p>}
          <button disabled={submitting} className="w-full rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white disabled:opacity-50">
            {submitting ? 'Please wait...' : mode === 'sign-in' ? 'Sign in' : 'Sign up'}
          </button>
        </form>
        <button type="button" onClick={() => setMode(mode === 'sign-in' ? 'sign-up' : 'sign-in')} className="mt-5 w-full text-sm text-blue-600 hover:underline">
          {mode === 'sign-in' ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
        </button>
        <Link to="/" className="mt-4 block text-center text-sm text-gray-500 hover:text-blue-600">Back to home</Link>
      </div>
    </div>
  );
};

export default Login;

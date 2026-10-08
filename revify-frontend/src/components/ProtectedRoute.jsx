import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return <div className="flex min-h-[50vh] items-center justify-center text-gray-600">Loading...</div>;
  }

  return user ? children : <Navigate to="/login" replace state={{ from: location }} />;
};

export default ProtectedRoute;

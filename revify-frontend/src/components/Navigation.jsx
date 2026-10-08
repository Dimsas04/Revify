import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline';
import { SparklesIcon } from '@heroicons/react/24/solid';
import { useAuth } from '../context/AuthContext';

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const { user, signOut } = useAuth();

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Analysis', href: '/analysis' },
    { name: 'Results', href: '/results' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-gray-200/20 shadow-lg"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 group">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg"
            >
              <SparklesIcon className="h-6 w-6 text-white" />
            </motion.div>
            <motion.span
              className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
              whileHover={{ scale: 1.05 }}
            >
              Revify
            </motion.span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navigation.filter((item) => item.href === '/' || user).map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`relative px-4 py-2 text-sm font-medium transition-colors duration-200 ${
                  isActive(item.href)
                    ? 'text-blue-600'
                    : 'text-gray-600 hover:text-blue-600'
                }`}
              >
                {item.name}
                {isActive(item.href) && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-600 to-purple-600"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
              </Link>
            ))}
          </div>
          <div className="hidden items-center gap-3 md:flex">
            {user ? (
              <>
                <span className="max-w-32 truncate text-sm text-gray-600">{user.email}</span>
                <button onClick={signOut} className="text-sm font-medium text-gray-600 hover:text-blue-600">Sign out</button>
              </>
            ) : (
              <Link to="/login" className="text-sm font-medium text-blue-600">Sign in</Link>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 rounded-lg text-gray-600 hover:text-blue-600 hover:bg-gray-100 transition-colors"
            >
              {isMenuOpen ? (
                <XMarkIcon className="h-6 w-6" />
              ) : (
                <Bars3Icon className="h-6 w-6" />
              )}
            </motion.button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{
          opacity: isMenuOpen ? 1 : 0,
          height: isMenuOpen ? 'auto' : 0,
        }}
        transition={{ duration: 0.3 }}
        className="md:hidden overflow-hidden bg-white/95 backdrop-blur-xl border-t border-gray-200/20"
      >
        <div className="px-4 py-2 space-y-1">
          {navigation.filter((item) => item.href === '/' || user).map((item) => (
            <Link
              key={item.name}
              to={item.href}
              onClick={() => setIsMenuOpen(false)}
              className={`block px-4 py-3 text-base font-medium rounded-lg transition-colors ${
                isActive(item.href)
                  ? 'text-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
              }`}
            >
              {item.name}
            </Link>
          ))}
          {user ? (
            <button onClick={signOut} className="block px-4 py-3 text-base font-medium text-gray-600">Sign out</button>
          ) : (
            <Link to="/login" onClick={() => setIsMenuOpen(false)} className="block px-4 py-3 text-base font-medium text-blue-600">Sign in</Link>
          )}
        </div>
      </motion.div>
    </motion.nav>
  );
};

export default Navigation;
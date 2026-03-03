import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  MagnifyingGlassIcon,
  ChartBarIcon,
  DocumentTextIcon,
  SparklesIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  ArrowRightIcon,
  ArrowLeftIcon
} from '@heroicons/react/24/outline';
import { CheckCircleIcon as CheckCircleSolid } from '@heroicons/react/24/solid';
import { revifyAPI } from '../services/api';

const Analysis = () => {
  // Flow state
  const [currentStep, setCurrentStep] = useState('input'); // 'input', 'features', 'analyzing'
  
  // Form state
  const [productUrl, setProductUrl] = useState('');
  const [productName, setProductName] = useState('');
  
  // Feature extraction state
  const [extracting, setExtracting] = useState(false);
  const [extractError, setExtractError] = useState(null);
  const [extractedFeatures, setExtractedFeatures] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState(new Set());
  const [reviewsReady, setReviewsReady] = useState(false);
  
  // Analysis state
  const [status, setStatus] = useState({
    is_running: false,
    progress: 0,
    current_phase: '',
    error: null,
    result: null,
    start_time: null
  });
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const location = useLocation();
  const navigate = useNavigate();

  // Initialize from location state if coming from Home page
  useEffect(() => {
    if (location.state?.productUrl) {
      setProductUrl(location.state.productUrl);
      setProductName(location.state.productName || '');
      // Don't auto-start - let user click "Extract Features" button
    }
  }, []);

  // Step 1: Handle feature extraction
  const handleExtractFeatures = async (e, urlOverride = null) => {
    if (e) e.preventDefault();
    const url = urlOverride || productUrl;
    if (!url.trim()) return;
    
    setExtracting(true);
    setExtractError(null);
    
    try {
      // Start async feature extraction
      await revifyAPI.extractFeatures(url, productName);
      
      let featuresSet = false; // Track if we've already set features
      
      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await revifyAPI.getFeatureStatus();
          
          if (statusResponse.error) {
            setExtractError(statusResponse.error);
            setExtracting(false);
            clearInterval(pollInterval);
            return;
          }
          
          // Show features as soon as they're ready (even if reviews still scraping)
          if (statusResponse.completed && statusResponse.features && !featuresSet) {
            // Set features ONLY ONCE
            setExtractedFeatures(statusResponse.features);
            setSelectedFeatures(new Set(statusResponse.features));
            setCurrentStep('features');
            setExtracting(false);
            featuresSet = true; // Mark as set
          }
          
          // Update reviews_ready status independently (doesn't affect selection)
          if (statusResponse.reviews_ready !== undefined) {
            setReviewsReady(statusResponse.reviews_ready);
            
            if (statusResponse.reviews_ready) {
              clearInterval(pollInterval);
            }
          }
        } catch (pollError) {
          console.error('Polling error:', pollError);
        }
      }, 2000); // Poll every 2 seconds
      
    } catch (err) {
      setExtractError(err.message);
      setExtracting(false);
    }
  };

  // Step 2: Toggle feature selection
  const toggleFeature = (feature) => {
    const newSelected = new Set(selectedFeatures);
    if (newSelected.has(feature)) {
      newSelected.delete(feature);
    } else {
      newSelected.add(feature);
    }
    setSelectedFeatures(newSelected);
  };

  const toggleAllFeatures = () => {
    if (selectedFeatures.size === extractedFeatures.length) {
      setSelectedFeatures(new Set());
    } else {
      setSelectedFeatures(new Set(extractedFeatures));
    }
  };

  // Step 3: Start analysis with selected features
  const handleStartAnalysis = async () => {
    if (selectedFeatures.size === 0) {
      alert('Please select at least one feature to analyze');
      return;
    }

    // Clear any old status/results before starting new analysis
    setStatus({
      is_running: false,
      progress: 0,
      current_phase: '',
      error: null,
      result: null,
      start_time: null
    });
    setElapsedTime(0);

    setCurrentStep('analyzing');
    
    try {
      await revifyAPI.startAnalysis(
        productUrl,
        productName,
        Array.from(selectedFeatures) // Pass selected features
      );
    } catch (err) {
      setStatus(prev => ({
        ...prev,
        error: err.message,
        is_running: false
      }));
    }
  };

  // Poll for analysis progress when in analyzing step
  useEffect(() => {
    if (currentStep !== 'analyzing') return;
    
    const pollStatus = async () => {
      try {
        const statusData = await revifyAPI.getAnalysisStatus();
        setStatus(statusData);

        // Calculate elapsed time if analysis is running
        if (statusData.start_time) {
          const startTime = new Date(statusData.start_time);
          const now = new Date();
          const elapsed = Math.floor((now - startTime) / 1000);
          setElapsedTime(elapsed);
        }

        // If analysis is complete, navigate to results
        if (statusData.result && !statusData.is_running && !statusData.error) {
          setTimeout(() => {
            navigate('/results', { 
              state: { 
                result: statusData.result,
                productUrl,
                productName
              } 
            });
          }, 2000); // Give user time to see completion
        }
      } catch (error) {
        console.error('Error polling status:', error);
        setStatus(prev => ({
          ...prev,
          error: 'Failed to get analysis status',
          is_running: false
        }));
      }
    };

    // Poll immediately and then every 2 seconds
    pollStatus();
    const interval = setInterval(pollStatus, 2000);

    return () => clearInterval(interval);
  }, [currentStep, productUrl, productName, navigate]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getPhaseIcon = (phase) => {
    if (phase.toLowerCase().includes('feature')) {
      return MagnifyingGlassIcon;
    } else if (phase.toLowerCase().includes('review') || phase.toLowerCase().includes('scrap')) {
      return DocumentTextIcon;
    } else if (phase.toLowerCase().includes('analyz')) {
      return ChartBarIcon;
    } else {
      return SparklesIcon;
    }
  };

  const analysisSteps = [
    { 
      id: 'initialize',
      title: 'Initializing',
      description: 'Setting up analysis environment',
      range: [0, 20]
    },
    { 
      id: 'extract',
      title: 'Feature Extraction',
      description: 'Identifying key product features',
      range: [20, 40]
    },
    { 
      id: 'scrape',
      title: 'Review Scraping',
      description: 'Gathering customer reviews',
      range: [40, 70]
    },
    { 
      id: 'analyze',
      title: 'AI Analysis',
      description: 'Processing sentiment and insights',
      range: [70, 95]
    },
    { 
      id: 'complete',
      title: 'Finalizing',
      description: 'Preparing comprehensive report',
      range: [95, 100]
    }
  ];

  const getCurrentProgressStep = () => {
    return analysisSteps.find(step => 
      status.progress >= step.range[0] && status.progress < step.range[1]
    ) || analysisSteps[0];
  };

  const progressStep = getCurrentProgressStep();
  const PhaseIcon = getPhaseIcon(status.current_phase);

  // Step 1: URL Input Form
  if (currentStep === 'input') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-2xl"
        >
          <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 md:p-12">
            <div className="text-center mb-8">
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Start Your Analysis
              </h1>
              <p className="text-gray-600 text-lg">
                Enter the Amazon product URL to begin analyzing customer reviews
              </p>
            </div>

            <form onSubmit={handleExtractFeatures} className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Product URL *
                </label>
                <input
                  type="url"
                  value={productUrl}
                  onChange={(e) => setProductUrl(e.target.value)}
                  placeholder="https://www.amazon.in/product-name/dp/B0XXXXXXXX"
                  required
                  className="w-full px-4 py-3 rounded-xl border-2 border-gray-300 focus:border-blue-500 focus:outline-none transition-colors"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Product Name (Optional)
                </label>
                <input
                  type="text"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  placeholder="e.g., Wireless Headphones"
                  className="w-full px-4 py-3 rounded-xl border-2 border-gray-300 focus:border-blue-500 focus:outline-none transition-colors"
                />
              </div>

              {extractError && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-red-50 border-2 border-red-200 rounded-xl flex items-start gap-3"
                >
                  <ExclamationTriangleIcon className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  <p className="text-red-700 text-sm">{extractError}</p>
                </motion.div>
              )}

              <button
                type="submit"
                disabled={extracting || !productUrl.trim()}
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-[1.02] flex items-center justify-center gap-2"
              >
                {extracting ? (
                  <>
                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                    Extracting Features and Reviews....
                  </>
                ) : (
                  <>
                    Extract Features and Reviews
                    <ArrowRightIcon className="w-5 h-5" />
                  </>
                )}
              </button>
            </form>

            <div className="mt-6 p-4 bg-blue-50 rounded-xl">
              <p className="text-sm text-blue-800">
                💡 <strong>Tip:</strong> We'll first extract the key features from this product, 
                then you can choose which ones to analyze in detail.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  // Step 2: Feature Selection
  if (currentStep === 'features') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-3">
              Select Features to Analyze
            </h2>
            <p className="text-gray-600 text-lg">
              We found <span className="font-semibold text-blue-600">{extractedFeatures.length} features</span>. 
              Choose which ones you'd like to analyze in detail.
            </p>
          </motion.div>

          {/* Reviews Status Banner */}
          {!reviewsReady && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <ArrowPathIcon className="w-5 h-5 text-blue-600 animate-spin" />
                <div>
                  <p className="text-blue-800 font-medium">Reviews are being collected in the background</p>
                  <p className="text-blue-600 text-sm">Feel free to select your features while we gather customer feedback</p>
                </div>
              </div>
            </motion.div>
          )}

          {reviewsReady && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-green-50 border-l-4 border-green-500 p-4 mb-6 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <CheckCircleIcon className="w-5 h-5 text-green-600" />
                <p className="text-green-800 font-medium">Reviews ready! You can start the analysis whenever you're ready.</p>
              </div>
            </motion.div>
          )}

          {/* Selection Controls */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-6 mb-6"
          >
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center gap-4">
                <button
                  onClick={toggleAllFeatures}
                  className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:shadow-lg transition-all"
                >
                  {selectedFeatures.size === extractedFeatures.length ? 'Deselect All' : 'Select All'}
                </button>
                <span className="text-gray-600">
                  <span className="font-semibold text-blue-600">{selectedFeatures.size}</span> of {extractedFeatures.length} selected
                </span>
              </div>
              
              <button
                onClick={() => setCurrentStep('input')}
                className="px-6 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg transition-colors flex items-center gap-2"
              >
                <ArrowLeftIcon className="w-4 h-4" />
                Back to URL
              </button>
            </div>
          </motion.div>

          {/* Feature Grid */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-4"
          >
            <AnimatePresence>
              {extractedFeatures.map((feature, index) => {
                const isSelected = selectedFeatures.has(feature);
                return (
                  <motion.div
                    key={feature}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => toggleFeature(feature)}
                    className={`
                      relative cursor-pointer p-6 rounded-xl border-2 transition-all duration-300
                      ${isSelected 
                        ? 'bg-gradient-to-br from-blue-50 to-purple-50 border-blue-400 shadow-lg' 
                        : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-md'
                      }
                    `}
                  >
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 mt-1">
                        {isSelected ? (
                          <CheckCircleSolid className="w-6 h-6 text-blue-600" />
                        ) : (
                          <div className="w-6 h-6 rounded-full border-2 border-gray-400" />
                        )}
                      </div>
                      <div className="flex-1">
                        <h3 className={`font-semibold text-lg ${isSelected ? 'text-blue-900' : 'text-gray-800'}`}>
                          {feature}
                        </h3>
                      </div>
                    </div>

                    {isSelected && (
                      <motion.div
                        layoutId={`selected-${feature}`}
                        className="absolute inset-0 bg-blue-400/10 rounded-xl pointer-events-none"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                      />
                    )}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </motion.div>

          {/* Bottom Sticky Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-8 bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-6 sticky bottom-4"
          >
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="text-sm text-gray-600">
                {!reviewsReady ? (
                  <>⏳ <strong>Meanwhile:</strong> Select the features you want to focus on while reviews are being collected</>
                ) : (
                  <>💡 <strong>Tip:</strong> Selecting fewer features will speed up the analysis</>
                )}
              </div>
              <button
                onClick={handleStartAnalysis}
                disabled={selectedFeatures.size === 0 || !reviewsReady}
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 flex items-center justify-center gap-2"
              >
                {!reviewsReady ? (
                  <>
                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                    Preparing reviews...
                  </>
                ) : (
                  <>
                    Start Analysis ({selectedFeatures.size} Feature{selectedFeatures.size !== 1 ? 's' : ''})
                    <ArrowRightIcon className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  // Step 3: Analysis Progress (existing code with minor tweaks)
  if (status.error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md w-full text-center"
        >
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-red-200">
            <ExclamationTriangleIcon className="h-16 w-16 text-red-500 mx-auto mb-6" />
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Analysis Failed</h2>
            <p className="text-gray-600 mb-6">{status.error}</p>
            <div className="space-y-3">
              <button
                onClick={() => window.location.reload()}
                className="w-full btn-primary flex items-center justify-center space-x-2"
              >
                <ArrowPathIcon className="h-5 w-5" />
                <span>Retry Analysis</span>
              </button>
              <button
                onClick={() => navigate('/')}
                className="w-full btn-secondary"
              >
                Back to Home
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 pt-10">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Analyzing Your{' '}
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Product
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-6">{productName}</p>
          <div className="inline-flex items-center space-x-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full border border-gray-200">
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-gray-700">
              Analysis in Progress • {formatTime(elapsedTime)}
            </span>
          </div>
        </motion.div>

        {/* Progress Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-200 p-8 mb-8"
        >
          {/* Current Phase */}
          <div className="text-center mb-8">
            <motion.div
              key={status.current_phase}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-3xl mb-6 shadow-lg"
            >
              <PhaseIcon className="h-10 w-10 text-white" />
            </motion.div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">{progressStep.title}</h2>
            <p className="text-gray-600">{status.current_phase || progressStep.description}</p>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">Progress</span>
              <span className="text-sm font-bold text-blue-600">{status.progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full shadow-sm"
                initial={{ width: 0 }}
                animate={{ width: `${status.progress}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Steps */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {analysisSteps.map((step, index) => {
              const isCompleted = status.progress > step.range[1];
              const isActive = status.progress >= step.range[0] && status.progress < step.range[1];
              const isPending = status.progress < step.range[0];

              return (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`text-center p-4 rounded-2xl transition-all duration-300 ${
                    isCompleted
                      ? 'bg-green-50 border border-green-200'
                      : isActive
                      ? 'bg-blue-50 border border-blue-200 ring-2 ring-blue-100'
                      : 'bg-gray-50 border border-gray-200'
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-full mx-auto mb-3 flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                      isCompleted
                        ? 'bg-green-500 text-white'
                        : isActive
                        ? 'bg-blue-500 text-white animate-pulse'
                        : 'bg-gray-300 text-gray-600'
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircleIcon className="h-5 w-5" />
                    ) : (
                      index + 1
                    )}
                  </div>
                  <h3
                    className={`font-semibold text-sm mb-1 ${
                      isCompleted
                        ? 'text-green-800'
                        : isActive
                        ? 'text-blue-800'
                        : 'text-gray-600'
                    }`}
                  >
                    {step.title}
                  </h3>
                  <p
                    className={`text-xs ${
                      isCompleted
                        ? 'text-green-600'
                        : isActive
                        ? 'text-blue-600'
                        : 'text-gray-500'
                    }`}
                  >
                    {step.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Fun Facts */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-gray-200"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
            Did You Know? 🤔
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600 mb-1">95%</div>
              <p className="text-sm text-gray-600">of people read reviews before purchasing</p>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600 mb-1">13+</div>
              <p className="text-sm text-gray-600">reviews needed for trust</p>
            </div>
            <div>
              <div className="text-2xl font-bold text-pink-600 mb-1">68%</div>
              <p className="text-sm text-gray-600">trust positive reviews more</p>
            </div>
          </div>
        </motion.div>

        {/* Loading Animation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-center mt-12 mb-8"
        >
          <div className="flex items-center justify-center space-x-2 text-gray-500">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
          <p className="mt-4 text-gray-600">
            {status.is_running ? 'Processing your request...' : 'Preparing results...'}
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default Analysis;
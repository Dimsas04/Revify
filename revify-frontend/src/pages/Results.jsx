import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  ArrowDownTrayIcon,
  ShareIcon,
  ChartBarIcon,
  DocumentTextIcon,
  StarIcon,
  TrophyIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClipboardDocumentIcon
} from '@heroicons/react/24/outline';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { revifyAPI } from '../services/api';
import {
  getSentimentColor,
  getSentimentIcon,
  normalizeSentiment,
  downloadJSON,
  copyToClipboard
} from '../utils/helpers';

const Results = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [copied, setCopied] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();

  const productUrl = location.state?.productUrl || '';
  const productName = location.state?.productName || 'Product Analysis';
  const passedResult = location.state?.result;
  const analysisRequestId = location.state?.analysisRequestId;

  const normalizeResult = (result) => {
    if (!result || typeof result !== 'object') {
      return result;
    }

    const analysis = result.analysis;
    if (analysis && !Array.isArray(analysis) && Array.isArray(analysis.analyses)) {
      return {
        ...result,
        analysis: analysis.analyses,
      };
    }

    return result;
  };

  useEffect(() => {
    const fetchResults = async () => {
      try {
        if (passedResult) {
          setResults(normalizeResult(passedResult));
          setLoading(false);
          return;
        }

        // Try to fetch results from API
        const data = await revifyAPI.getResults(analysisRequestId);
        setResults(normalizeResult(data.result || data));
      } catch (error) {
        console.error('Error fetching results:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [passedResult]);

  const handleDownload = () => {
    if (results) {
      downloadJSON(results, `revify-analysis-${Date.now()}.json`);
    }
  };

  const handleShare = async () => {
    const shareData = {
      title: `Revify Analysis - ${productName}`,
      text: `Check out this comprehensive product analysis by Revify!`,
      url: window.location.href
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        // Fallback to copying URL
        await copyToClipboard(window.location.href);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const getSentimentData = () => {
    if (!results?.analysis) return [];

    const sentimentCounts = results.analysis.reduce((acc, item) => {
      const sentiment = item.sentiment || 'Unknown';
      acc[sentiment] = (acc[sentiment] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(sentimentCounts).map(([sentiment, count]) => ({
      name: sentiment,
      value: count,
      color: getSentimentColorHex(sentiment)
    }));
  };

  const getSentimentColorHex = (sentiment) => {
    switch (normalizeSentiment(sentiment)) {
      case 'positive': return '#22c55e';
      case 'negative': return '#ef4444';
      case 'mixed': return '#f59e0b';
      case 'neutral': return '#6b7280';
      default: return '#9ca3af';
    }
  };

  const getFeatureScoreData = () => {
    if (!results?.analysis) return [];

    return results.analysis.map(item => ({
      feature: item.feature?.substring(0, 15) + (item.feature?.length > 15 ? '...' : ''),
      fullFeature: item.feature,
      score: getSentimentScore(item.sentiment),
      sentiment: item.sentiment
    })).sort((a, b) => b.score - a.score);
  };

  const getSentimentScore = (sentiment) => {
    switch (normalizeSentiment(sentiment)) {
      case 'positive': return 100;
      case 'mixed': return 60;
      case 'neutral': return 50;
      case 'negative': return 20;
      default: return 0;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center"
        >
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analysis results...</p>
        </motion.div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md w-full text-center"
        >
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-red-200">
            <ExclamationTriangleIcon className="h-16 w-16 text-red-500 mx-auto mb-6" />
            <h2 className="text-2xl font-bold text-gray-900 mb-4">No Results Found</h2>
            <p className="text-gray-600 mb-6">
              {error || 'Unable to load analysis results. Please try running the analysis again.'}
            </p>
            <button
              onClick={() => navigate('/')}
              className="w-full btn-primary"
            >
              Start New Analysis
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  const sentimentData = getSentimentData();
  const featureScoreData = getFeatureScoreData();
  const topFeature = featureScoreData[0];
  const bottomFeature = featureScoreData[featureScoreData.length - 1];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 pt-10 pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center space-x-2 px-4 py-2 bg-green-100 text-green-800 rounded-full mb-6">
            <CheckCircleIcon className="h-5 w-5" />
            <span className="font-medium">Analysis Complete</span>
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Analysis{' '}
            <span className="bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
              Results
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-6">{productName}</p>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleDownload}
              className="inline-flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:bg-blue-700 transition-all duration-200"
            >
              <ArrowDownTrayIcon className="h-5 w-5" />
              <span>Download Report</span>
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleShare}
              className={`inline-flex items-center space-x-2 px-6 py-3 font-semibold rounded-xl shadow-lg transition-all duration-200 ${
                copied
                  ? 'bg-green-600 text-white'
                  : 'bg-white text-gray-700 hover:shadow-xl border border-gray-200'
              }`}
            >
              {copied ? (
                <>
                  <CheckCircleIcon className="h-5 w-5" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <ShareIcon className="h-5 w-5" />
                  <span>Share Results</span>
                </>
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-lg"
          >
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-blue-100 rounded-xl">
                <ChartBarIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Features Analyzed</p>
                <p className="text-2xl font-bold text-gray-900">{results.analysis?.length || 0}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-lg"
          >
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-purple-100 rounded-xl">
                <DocumentTextIcon className="h-6 w-6 text-purple-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Reviews Processed</p>
                <p className="text-2xl font-bold text-gray-900">{results.total_reviews || 0}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-lg"
          >
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-green-100 rounded-xl">
                <TrophyIcon className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Top Feature</p>
                <p className="text-lg font-bold text-gray-900 truncate" title={topFeature?.fullFeature}>
                  {topFeature?.feature || 'N/A'}
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-lg"
          >
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-yellow-100 rounded-xl">
                <StarIcon className="h-6 w-6 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Overall Sentiment</p>
                <p className="text-lg font-bold text-gray-900">
                  {sentimentData.length > 0 ? sentimentData[0].name : 'Mixed'}
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Sentiment Distribution Chart */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-lg"
          >
            <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
              <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                <ChartBarIcon className="h-5 w-5 text-white" />
              </div>
              <span>Sentiment Distribution</span>
            </h3>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sentimentData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {sentimentData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Feature Scores Chart */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-lg"
          >
            <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg">
                <TrophyIcon className="h-5 w-5 text-white" />
              </div>
              <span>Feature Scores</span>
            </h3>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureScoreData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name, props) => [value, 'Score']}
                    labelFormatter={(label, payload) => payload?.[0]?.payload?.fullFeature || label}
                  />
                  <Bar 
                    dataKey="score" 
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </div>

        {/* Detailed Analysis */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200 shadow-lg overflow-hidden"
        >
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
              <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
                <DocumentTextIcon className="h-5 w-5 text-white" />
              </div>
              <span>Detailed Feature Analysis</span>
            </h3>
            <p className="text-gray-600 mt-2">Click on any feature to see detailed insights</p>
          </div>

          <div className="divide-y divide-gray-200">
            {results.analysis?.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + index * 0.1 }}
                className={`p-6 hover:bg-gray-50 cursor-pointer transition-colors ${
                  selectedFeature === index ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                }`}
                onClick={() => setSelectedFeature(selectedFeature === index ? null : index)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-3">
                      <h4 className="text-lg font-semibold text-gray-900">
                        {feature.feature}
                      </h4>
                      <span className={`inline-flex items-center space-x-1 px-3 py-1 rounded-full text-sm font-medium border ${getSentimentColor(feature.sentiment)}`}>
                        <span>{getSentimentIcon(feature.sentiment)}</span>
                        <span>{feature.sentiment}</span>
                      </span>
                    </div>
                    <p className="text-gray-700 leading-relaxed">{feature.verdict}</p>

                    {selectedFeature === index && feature.key_points && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4"
                      >
                        <h5 className="font-semibold text-gray-900 mb-2">Key Points:</h5>
                        <div className="space-y-2">
                          {feature.key_points.map((point, pointIndex) => (
                            <div key={pointIndex} className="flex items-start space-x-2">
                              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                              <span className="text-gray-700">
                                {typeof point === 'string' 
                                  ? point 
                                  : Object.entries(point)[0]?.[0] || 'N/A'
                                }
                                {typeof point === 'object' && Object.entries(point)[0]?.[1] && (
                                  <span className="ml-2 text-sm text-gray-500">
                                    ({Object.entries(point)[0][1]} mentions)
                                  </span>
                                )}
                              </span>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </div>
                  
                  <div className={`ml-4 px-3 py-1 rounded-full text-sm font-bold ${
                    getSentimentScore(feature.sentiment) >= 80 ? 'bg-green-100 text-green-800' :
                    getSentimentScore(feature.sentiment) >= 60 ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {getSentimentScore(feature.sentiment)}%
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
          className="text-center mt-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white"
        >
          <h3 className="text-2xl font-bold mb-4">Analyze Another Product?</h3>
          <p className="text-blue-100 mb-6">
            Get instant insights on any Amazon product with our AI-powered analysis
          </p>
          <button
            onClick={() => navigate('/')}
            className="btn-primary bg-white text-blue-600 hover:bg-gray-100"
          >
            Start New Analysis
          </button>
        </motion.div>
      </div>
    </div>
  );
};

export default Results;
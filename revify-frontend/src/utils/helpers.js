// URL validation utility
export const isValidURL = (string) => {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
};

// Check if URL is an Amazon product URL
export const isAmazonURL = (url) => {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname.includes('amazon.');
  } catch (_) {
    return false;
  }
};

// Extract product name from URL
export const extractProductNameFromURL = (url) => {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    
    // For Amazon URLs, try to extract product name from path
    if (isAmazonURL(url)) {
      const parts = pathname.split('/');
      const productPart = parts.find(part => part.includes('-') && part.length > 10);
      if (productPart) {
        return productPart.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      }
    }
    
    return '';
  } catch (_) {
    return '';
  }
};

// Format sentiment color
export const normalizeSentiment = (sentiment) => {
  const value = String(sentiment || '').trim().toLowerCase();
  if ("mixed" in value) {
    return 'mixed';
  }
  if ("positive" in value) {
    return 'positive';
  }
  if ("negative" in value) {
    return 'negative';
  }
  // if (
  //   "positive" in value 
  // ) {
  //   return 'positive';
  // }
  // if (value.includes('mixed')) {
  //   return 'mixed';
  // }
  // if (value.includes('neutral') || value.includes('no data')) {
  //   return 'neutral';
  // }
  // return 'unknown';
  if ("neutral" in value || "no data" in value) {
    return 'neutral';
  }

};

export const getSentimentColor = (sentiment) => {
  switch (normalizeSentiment(sentiment)) {
    case 'positive':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'negative':
      return 'text-red-600 bg-red-50 border-red-200';
    case 'mixed':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    case 'neutral':
      return 'text-gray-600 bg-gray-50 border-gray-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

// Format sentiment icon
export const getSentimentIcon = (sentiment) => {
  switch (normalizeSentiment(sentiment)) {
    case 'positive':
      return '😊';
    case 'negative':
      return '😞';
    case 'mixed':
      return '😐';
    case 'neutral':
      return '😶';
    default:
      return '❓';
  }
};

// Format time ago
export const timeAgo = (date) => {
  const now = new Date();
  const diffTime = Math.abs(now - date);
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
  const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
  const diffMinutes = Math.floor(diffTime / (1000 * 60));
  
  if (diffDays > 0) {
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  } else if (diffHours > 0) {
    return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  } else if (diffMinutes > 0) {
    return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
  } else {
    return 'Just now';
  }
};

// Truncate text
export const truncateText = (text, maxLength = 100) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

// Debounce function for search inputs
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

// Generate random ID
export const generateId = () => {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

// Copy text to clipboard
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy text: ', err);
    return false;
  }
};

// Download JSON as file
export const downloadJSON = (data, filename = 'data.json') => {
  const dataStr = JSON.stringify(data, null, 2);
  const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
  
  const exportFileDefaultName = filename;
  
  const linkElement = document.createElement('a');
  linkElement.setAttribute('href', dataUri);
  linkElement.setAttribute('download', exportFileDefaultName);
  linkElement.click();
};
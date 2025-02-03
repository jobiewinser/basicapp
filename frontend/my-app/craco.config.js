const path = require('path');

module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.resolve.modules = [
        path.resolve(__dirname, 'my-app/src'),  // Ensure this path is correct
        'node_modules',
      ];
      return webpackConfig;
    },
  },
};
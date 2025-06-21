using System;

namespace MuseTalk.API
{
    /// <summary>
    /// Configuration for MuseTalk inference
    /// </summary>
    [Serializable]
    public class MuseTalkConfig
    {
        public string ModelPath = "MuseTalk";
        public string Version = "v15"; // only v15 is supported
        public string Device = "cpu"; // "cpu" or "cuda"
        public int BatchSize = 4;
        public float ExtraMargin = 10f; // Additional margin for v15
        public bool UseINT8 = true; // Enable INT8 quantization (CPU-optimized, default for Mac)
        
        // Disk caching configuration
        public bool EnableDiskCache = true; // Enable persistent disk caching for avatar processing
        public string CacheDirectory = ""; // Cache directory path (empty = auto-detect)
        public int MaxCacheEntriesPerAvatar = 1000; // Maximum cache entries per avatar hash
        public long MaxCacheSizeMB = 1024; // Maximum total cache size in MB (1GB default)
        public int CacheVersionNumber = 1; // Cache version for invalidation on format changes
        public bool CacheLatentsOnly = false; // Cache only latents (faster) vs full avatar data (slower but complete)
        
        public MuseTalkConfig()
        {
        }
        
        public MuseTalkConfig(string modelPath, string version = "v15")
        {
            if (version != "v15")
            {
                throw new NotSupportedException("Only v15 is supported");
            }
            ModelPath = modelPath;
            Version = version;
        }
        
        /// <summary>
        /// Create configuration optimized for performance with disk caching
        /// </summary>
        public static MuseTalkConfig CreateOptimized(string modelPath = "MuseTalk")
        {
            return new MuseTalkConfig(modelPath)
            {
                EnableDiskCache = true,
                CacheLatentsOnly = false,
                MaxCacheSizeMB = 2048,
                UseINT8 = true
            };
        }
        
        /// <summary>
        /// Create configuration for development/debugging with full texture caching
        /// </summary>
        public static MuseTalkConfig CreateForDevelopment(string modelPath = "MuseTalk")
        {
            return new MuseTalkConfig(modelPath)
            {
                EnableDiskCache = true,
                CacheLatentsOnly = false, // Full texture caching for debugging
                MaxCacheSizeMB = 512, // Smaller cache for development
                UseINT8 = false // Full precision for better quality debugging
            };
        }
    }
}